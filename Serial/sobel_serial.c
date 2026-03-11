/*
 * ============================================================
 *  EE7218 - High Performance Computing
 *  Group 02 - Parallel Sobel Edge Detector
 *  Phase 2: Serial Implementation (C | stb_image)
 * ============================================================
 *
 *  Processing Pipeline:
 *    1. Load image      (BMP or PNG — color or grayscale)
 *    2. Grayscale       (done automatically by stb_image)
 *    3. Gaussian Blur   (5x5 kernel — reduces noise)
 *    4. Sobel Filter    (Gx + Gy kernels — detects edges)
 *    5. Save output     (BMP or PNG via stb_image_write)
 *
 *  Required header files (place in same folder as this .c file):
 *    stb_image.h        — https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
 *    stb_image_write.h  — https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
 *
 *  Compile:
 *    gcc -O2 -o sobel_serial sobel_serial.c -lm
 *
 *  Run:
 *    ./sobel_serial <input.[bmp|png]> <output.[bmp|png]> [threshold]
 *
 *  Examples:
 *    ./sobel_serial photo.png   edges.png
 *    ./sobel_serial image.bmp   result.bmp  30
 *    ./sobel_serial input.png   output.bmp      (cross-format works too)
 *
 *  NOTE: No -lpng or -ljpeg needed! stb_image is header-only.
 * ============================================================
 */

#define _POSIX_C_SOURCE 200809L   /* Required for clock_gettime / CLOCK_MONOTONIC */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/*
 * stb_image setup:
 *   STB_IMAGE_IMPLEMENTATION must be defined ONCE before including
 *   stb_image.h — this tells stb to insert the actual function bodies
 *   here. Without this define, you get only the declarations.
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/*
 * stb_image_write setup:
 *   Same idea — STB_IMAGE_WRITE_IMPLEMENTATION must be defined ONCE
 *   before including stb_image_write.h
 */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/* ============================================================
 *  IMAGE STRUCTURE
 *
 *  Holds a flat grayscale pixel buffer + dimensions.
 *  Storage : row-major  →  index = row * width + col
 *  Values  : 0 (black) to 255 (white)
 * ============================================================ */
typedef struct {
    int      width;
    int      height;
    uint8_t* pixels;    /* heap-allocated, size = width * height */
} Image;


/* ============================================================
 *  IMAGE HELPERS
 * ============================================================ */

/* Allocate a new zero-filled Image. Returns 1 on success, 0 on failure. */
int image_alloc(Image* img, int width, int height) {
    img->width  = width;
    img->height = height;
    img->pixels = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    return (img->pixels != NULL);
}

/* Free the pixel buffer and reset all fields. */
void image_free(Image* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width  = 0;
    img->height = 0;
}

/* Pixel access macro: PIXEL(img, row, col) */
#define PIXEL(img, row, col)  ((img).pixels[(row) * (img).width + (col)])

/* Total pixel count */
#define TOTAL(img)  ((img).width * (img).height)


/* ============================================================
 *  TIMING UTILITY
 *  Returns current time in milliseconds (high resolution).
 * ============================================================ */
double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}


/* ============================================================
 *  UTILITY: Extract lowercase file extension
 *  e.g.  "photo.PNG"  →  "png"
 *        "image.BMP"  →  "bmp"
 * ============================================================ */
void get_extension(const char* path, char* out_ext, int max_len) {
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) { out_ext[0] = '\0'; return; }
    int i = 0;
    dot++;  /* skip the dot itself */
    while (*dot && i < max_len - 1)
        out_ext[i++] = (char)tolower((unsigned char)*dot++);
    out_ext[i] = '\0';
}


/* ============================================================
 *  STEP 1+2: LOAD IMAGE + GRAYSCALE CONVERSION
 *
 *  Uses stb_image to load BMP or PNG.
 *  stbi_load() with channels=1 automatically converts any
 *  color image (RGB/RGBA) to grayscale in one call.
 *
 *  No manual grayscale conversion code needed — stb handles it.
 *
 *  Supported input formats: BMP, PNG (and also JPG, TGA, GIF
 *  if those files are provided — stb supports them all)
 * ============================================================ */
int load_image(const char* path, Image* img) {
    int w, h, channels;

    /*
     * stbi_load() parameters:
     *   path          — file path
     *   &w, &h        — output width and height
     *   &channels     — original number of channels in file (1=gray, 3=RGB, 4=RGBA)
     *   1             — FORCE output to 1 channel (grayscale)
     *
     * Returns a malloc'd pixel buffer, or NULL on failure.
     * We must call stbi_image_free() on it when done.
     */
    uint8_t* data = stbi_load(path, &w, &h, &channels, 1);

    if (!data) {
        fprintf(stderr, "ERROR: Failed to load image: %s\n", path);
        fprintf(stderr, "       stb reason: %s\n", stbi_failure_reason());
        return 0;
    }

    /*
     * Copy stb's buffer into our Image struct.
     * We copy instead of using stb's buffer directly so we can
     * call image_free() consistently using our own free() later.
     */
    if (!image_alloc(img, w, h)) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        stbi_image_free(data);
        return 0;
    }

    memcpy(img->pixels, data, w * h * sizeof(uint8_t));

    /* Always free stb's buffer after copying */
    stbi_image_free(data);

    printf("      Done — %d x %d px  (original channels: %d → converted to grayscale)\n",
           w, h, channels);
    return 1;
}


/* ============================================================
 *  STEP 5: SAVE IMAGE
 *
 *  Uses stb_image_write to save as BMP or PNG.
 *  Automatically picks the right format from the file extension.
 *
 *  stbi_write_png() — lossless, best for edge images
 *  stbi_write_bmp() — uncompressed, universally compatible
 * ============================================================ */
int save_image(const char* path, const Image* img) {
    char ext[16];
    get_extension(path, ext, sizeof(ext));

    int result = 0;

    if (strcmp(ext, "png") == 0) {
        /*
         * stbi_write_png() parameters:
         *   path          — output file path
         *   width, height — image dimensions
         *   1             — number of channels (1 = grayscale)
         *   pixels        — pixel data buffer
         *   width * 1     — stride in bytes (bytes per row)
         */
        result = stbi_write_png(path, img->width, img->height,
                                1, img->pixels, img->width * 1);

    } else if (strcmp(ext, "bmp") == 0) {
        /*
         * stbi_write_bmp() parameters:
         *   path          — output file path
         *   width, height — image dimensions
         *   1             — number of channels (1 = grayscale)
         *   pixels        — pixel data buffer
         */
        result = stbi_write_bmp(path, img->width, img->height,
                                1, img->pixels);

    } else {
        fprintf(stderr, "ERROR: Unsupported output format '.%s'\n"
                        "       Supported: bmp, png\n", ext);
        return 0;
    }

    if (!result) {
        fprintf(stderr, "ERROR: Failed to save image: %s\n", path);
        return 0;
    }

    return 1;
}


/* ============================================================
 *  STEP 3: GAUSSIAN BLUR  (5x5 kernel)
 *
 *  Purpose:
 *    Smooths the grayscale image before Sobel to reduce noise.
 *    Without this, Sobel detects random pixel noise as false edges.
 *
 *  How it works:
 *    Convolves the image with a 5x5 Gaussian kernel.
 *    Each output pixel = weighted average of its 5x5 neighborhood.
 *    Pixels near the center get higher weights (bell-curve shape).
 *
 *  5x5 Gaussian kernel (unnormalized):
 *    [  1,  4,  7,  4,  1 ]
 *    [  4, 16, 26, 16,  4 ]
 *    [  7, 26, 41, 26,  7 ]
 *    [  4, 16, 26, 16,  4 ]
 *    [  1,  4,  7,  4,  1 ]
 *    Sum = 273  →  divide each result by 273 to normalize
 *
 *  Border handling:
 *    5x5 kernel needs 2 pixels of border on each side.
 *    Border pixels are copied unchanged from input.
 *
 *  Parameters:
 *    input  — grayscale image (output of load_image)
 *    output — blurred result  (pre-allocated, same size as input)
 * ============================================================ */
void gaussian_blur(const Image* input, Image* output) {

    const int K[5][5] = {
        {  1,  4,  7,  4,  1 },
        {  4, 16, 26, 16,  4 },
        {  7, 26, 41, 26,  7 },
        {  4, 16, 26, 16,  4 },
        {  1,  4,  7,  4,  1 }
    };
    const int K_SUM = 273;   /* Sum of all kernel values for normalization */

    int W = input->width;
    int H = input->height;

    /* Copy input to output so border pixels are preserved */
    memcpy(output->pixels, input->pixels, W * H * sizeof(uint8_t));

    int row, col, kr, kc;

    /* Skip 2-pixel border — 5x5 kernel needs 2 neighbors on each side */
    for (row = 2; row < H - 2; row++) {
        for (col = 2; col < W - 2; col++) {

            int sum = 0;

            /* Convolve 5x5 neighborhood with Gaussian kernel */
            for (kr = -2; kr <= 2; kr++) {
                for (kc = -2; kc <= 2; kc++) {
                    sum += K[kr + 2][kc + 2] * PIXEL(*input, row + kr, col + kc);
                }
            }

            /* Normalize by K_SUM to keep result in [0, 255] */
            PIXEL(*output, row, col) = (uint8_t)(sum / K_SUM);
        }
    }
}


/* ============================================================
 *  STEP 4: SOBEL EDGE DETECTION
 *
 *  Applies two 3x3 Sobel kernels to every interior pixel
 *  of the blurred image. Border pixels are set to 0 (black).
 *
 *  Gx kernel — detects vertical edges (left-right brightness change):
 *    [ -1   0  +1 ]
 *    [ -2   0  +2 ]
 *    [ -1   0  +1 ]
 *
 *  Gy kernel — detects horizontal edges (top-bottom brightness change):
 *    [ -1  -2  -1 ]
 *    [  0   0   0 ]
 *    [ +1  +2  +1 ]
 *
 *  Gradient magnitude (accurate formula):
 *    G = sqrt(Gx^2 + Gy^2)
 *
 *  Parameters:
 *    input     — blurred grayscale image (output of gaussian_blur)
 *    output    — edge image (pre-allocated, same size as input)
 *    threshold — if > 0: pixels above threshold → 255, others → 0
 *                if = 0: output shows full gradient values (0-255)
 * ============================================================ */
void sobel_serial(const Image* input, Image* output, int threshold) {

    const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    int W = input->width;
    int H = input->height;

    /* Initialize all output to zero — border pixels stay black */
    memset(output->pixels, 0, W * H * sizeof(uint8_t));

    int row, col, kr, kc;
    for (row = 1; row < H - 1; row++) {
        for (col = 1; col < W - 1; col++) {

            int gx = 0, gy = 0;

            /* Apply both Sobel kernels to the 3x3 neighborhood */
            for (kr = -1; kr <= 1; kr++) {
                for (kc = -1; kc <= 1; kc++) {
                    int px = PIXEL(*input, row + kr, col + kc);
                    gx += Kx[kr + 1][kc + 1] * px;
                    gy += Ky[kr + 1][kc + 1] * px;
                }
            }

            /* Gradient magnitude — accurate sqrt formula */
            int mag = (int)sqrt((double)(gx * gx + gy * gy));

            /* Clamp to [0, 255] */
            if (mag > 255) mag = 255;

            /* Apply threshold if set */
            if (threshold > 0)
                PIXEL(*output, row, col) = (mag > threshold) ? 255 : 0;
            else
                PIXEL(*output, row, col) = (uint8_t)mag;
        }
    }
}


/* ============================================================
 *  RMSE — Accuracy Validation Metric
 *
 *  Compares two images pixel-by-pixel.
 *  RMSE = sqrt( sum((a[i] - b[i])^2) / N )
 *
 *  RMSE = 0.0  →  pixel-perfect match
 *  Used in Phase 3/4/5 to validate parallel output vs serial.
 * ============================================================ */
double compute_rmse(const Image* a, const Image* b) {
    if (a->width != b->width || a->height != b->height) {
        fprintf(stderr, "ERROR: RMSE — image sizes do not match\n");
        return -1.0;
    }
    double sum = 0.0;
    int i, total = a->width * a->height;
    for (i = 0; i < total; i++) {
        double d = (double)a->pixels[i] - (double)b->pixels[i];
        sum += d * d;
    }
    return sqrt(sum / total);
}


/* ============================================================
 *  MAIN
 * ============================================================ */
int main(int argc, char* argv[]) {

    if (argc < 3) {
        printf("\nUsage: %s <input> <output> [threshold]\n\n"
               "  input     : BMP or PNG image (color or grayscale)\n"
               "  output    : BMP or PNG (can differ from input format)\n"
               "  threshold : 0-255, default=0 (show full gradient)\n\n"
               "Examples:\n"
               "  %s photo.png   edges.png\n"
               "  %s image.bmp   result.bmp  30\n"
               "  %s input.png   output.bmp\n\n",
               argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int threshold = (argc >= 4) ? atoi(argv[3]) : 0;

    printf("\n========================================\n"
           "  EE7218 HPC - Serial Sobel Detector\n"
           "  Group 02 | C | stb_image\n"
           "========================================\n"
           "  Input     : %s\n"
           "  Output    : %s\n"
           "  Threshold : %d\n"
           "========================================\n",
           input_path, output_path, threshold);

    /* --------------------------------------------------
     *  STEP 1+2: Load image + Grayscale conversion
     *  stb_image handles both in a single call
     * -------------------------------------------------- */
    Image gray;
    printf("\n[1/4] Loading & converting to grayscale...\n");
    double t_load = now_ms();
    if (!load_image(input_path, &gray)) return 1;
    double load_ms = now_ms() - t_load;
    printf("      Load time: %.2f ms\n", load_ms);

    /* --------------------------------------------------
     *  STEP 3: Gaussian Blur
     *  Smooths image to reduce noise before Sobel
     * -------------------------------------------------- */
    Image blurred;
    if (!image_alloc(&blurred, gray.width, gray.height)) {
        fprintf(stderr, "ERROR: Failed to allocate blur buffer\n");
        image_free(&gray); return 1;
    }

    printf("[2/4] Applying Gaussian Blur (5x5 kernel)...\n");
    double t_blur = now_ms();
    gaussian_blur(&gray, &blurred);
    double blur_ms = now_ms() - t_blur;
    printf("      Done  (%.3f ms)\n", blur_ms);

    /* --------------------------------------------------
     *  STEP 4: Sobel Edge Detection
     *  Reads blurred image, detects edges
     * -------------------------------------------------- */
    Image edges;
    if (!image_alloc(&edges, gray.width, gray.height)) {
        fprintf(stderr, "ERROR: Failed to allocate edge buffer\n");
        image_free(&gray); image_free(&blurred); return 1;
    }

    printf("[3/4] Applying Sobel Filter...\n");
    double t_sobel = now_ms();
    sobel_serial(&blurred, &edges, threshold);
    double sobel_ms = now_ms() - t_sobel;
    printf("      Done  (%.3f ms)  <-- benchmark this\n", sobel_ms);

    /* --------------------------------------------------
     *  STEP 5: Save output image
     * -------------------------------------------------- */
    printf("[4/4] Saving edge image...\n");
    double t_save = now_ms();
    if (!save_image(output_path, &edges)) {
        image_free(&gray); image_free(&blurred); image_free(&edges);
        return 1;
    }
    double save_ms = now_ms() - t_save;
    printf("      Done  (%.2f ms)\n", save_ms);

    /* --------------------------------------------------
     *  RESULTS SUMMARY
     * -------------------------------------------------- */
    double total_ms = load_ms + blur_ms + sobel_ms + save_ms;

    printf("\n========================================\n"
           "  RESULTS SUMMARY\n"
           "========================================\n"
           "  Image Size       : %d x %d px\n"
           "  Total Pixels     : %d\n"
           "----------------------------------------\n"
           "  [1] Load + Gray  : %8.3f ms\n"
           "  [2] Gaussian Blur: %8.3f ms\n"
           "  [3] Sobel Filter : %8.3f ms  <-- parallelize this\n"
           "  [4] Save         : %8.3f ms\n"
           "  ─────────────────────────────────────\n"
           "  Total Pipeline   : %8.3f ms\n"
           "  Sobel Throughput : %8.2f Mpixels/sec\n"
           "========================================\n"
           "  Done.\n"
           "========================================\n\n",
           gray.width, gray.height,
           TOTAL(gray),
           load_ms,
           blur_ms,
           sobel_ms,
           save_ms,
           total_ms,
           (TOTAL(gray) / 1.0e6) / (sobel_ms / 1000.0));

    /* Cleanup */
    image_free(&gray);
    image_free(&blurred);
    image_free(&edges);

    return 0;
}