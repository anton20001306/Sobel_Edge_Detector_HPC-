/*
 * ============================================================
 *  EE7218 - High Performance Computing
 *  Group 02 - Parallel Sobel Edge Detector
 *  Phase 2: Serial Implementation (C | Multi-Format)
 * ============================================================
 *
 *  Supported Formats : BMP (native), PNG (libpng), JPG (libjpeg)
 
 * ============================================================
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* PNG support */
#include <png.h>

/* JPEG support */
#ifdef WITH_JPEG
#include <jpeglib.h>
#endif


/* ============================================================
 *  IMAGE STRUCTURE
 *  Flat grayscale pixel buffer + dimensions.
 *  Pixels stored row-major: index = row * width + col
 *  Values range: 0 (black) to 255 (white)
 * ============================================================ */
typedef struct {
    int      width;
    int      height;
    uint8_t* pixels;   /* heap-allocated array of size width * height */
} Image;


/* ============================================================
 *  IMAGE HELPERS
 * ============================================================ */

/* Allocate a new Image filled with zeros. Returns 1 on success, 0 on failure */
int image_alloc(Image* img, int width, int height) {
    img->width  = width;
    img->height = height;
    img->pixels = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    return (img->pixels != NULL);
}

/* Free image pixel buffer */
void image_free(Image* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width  = 0;
    img->height = 0;
}

/* Access pixel at (row, col) */
#define PIXEL(img, row, col) ((img).pixels[(row) * (img).width + (col)])

/* Total pixel count */
#define TOTAL(img) ((img).width * (img).height)


/* ============================================================
 *  TIMING UTILITY
 *  Returns current time in milliseconds using POSIX monotonic clock.
 * ============================================================ */
double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}


/* ============================================================
 *  UTILITY: Get lowercase file extension from path
 *  Writes result into out_ext buffer (caller provides).
 *  e.g. "photo.JPG" -> "jpg"
 * ============================================================ */
void get_extension(const char* path, char* out_ext, int max_len) {
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) {
        out_ext[0] = '\0';
        return;
    }
    int i = 0;
    dot++; /* skip the dot itself */
    while (*dot && i < max_len - 1) {
        out_ext[i++] = (char)tolower((unsigned char)*dot++);
    }
    out_ext[i] = '\0';
}


/* ============================================================
 *  BMP I/O  (native — no external library required)
 *
 *  BMP file layout:
 *   [14 bytes] File Header  — magic number, file size, pixel offset
 *   [40 bytes] Info Header  — width, height, bit depth, compression
 *   [optional] Color Palette — 256 x 4 bytes for 8-bit images
 *   [n bytes]  Pixel Data   — stored bottom-to-top, rows padded to 4 bytes
 * ============================================================ */
#pragma pack(push, 1)
typedef struct {
    uint16_t type;          /* Must be 0x4D42 ("BM") */
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t pixel_offset;  /* Byte offset to start of pixel data */
} BMPFileHeader;

typedef struct {
    uint32_t header_size;   /* Always 40 */
    int32_t  width;
    int32_t  height;        /* Negative = top-down storage order */
    uint16_t planes;        /* Always 1 */
    uint16_t bit_count;     /* 8 = grayscale, 24 = RGB */
    uint32_t compression;   /* 0 = uncompressed (BI_RGB) */
    uint32_t image_size;
    int32_t  x_ppm;
    int32_t  y_ppm;
    uint32_t clr_used;
    uint32_t clr_important;
} BMPInfoHeader;
#pragma pack(pop)

/* Load a BMP file (8-bit grayscale or 24-bit RGB) into an Image struct */
int load_bmp(const char* path, Image* img) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open BMP: %s\n", path);
        return 0;
    }

    BMPFileHeader fh;
    BMPInfoHeader ih;
    if (fread(&fh, sizeof(BMPFileHeader), 1, fp) != 1) { fprintf(stderr, "ERROR: Failed reading BMP file header\n"); fclose(fp); return 0; }
    if (fread(&ih, sizeof(BMPInfoHeader), 1, fp) != 1) { fprintf(stderr, "ERROR: Failed reading BMP info header\n"); fclose(fp); return 0; }

    if (fh.type != 0x4D42) {
        fprintf(stderr, "ERROR: Not a valid BMP file: %s\n", path);
        fclose(fp); return 0;
    }
    if (ih.compression != 0) {
        fprintf(stderr, "ERROR: Compressed BMP not supported. Use uncompressed BMP.\n");
        fclose(fp); return 0;
    }
    if (ih.bit_count != 8 && ih.bit_count != 24) {
        fprintf(stderr, "ERROR: Only 8-bit and 24-bit BMP supported.\n");
        fclose(fp); return 0;
    }

    int w        = ih.width;
    int h        = abs(ih.height);
    int top_down = (ih.height < 0);    /* Negative height = top-down row order */
    int bpp      = ih.bit_count / 8;
    int stride   = (w * bpp + 3) & ~3; /* BMP rows padded to 4-byte boundary */

    if (!image_alloc(img, w, h)) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        fclose(fp); return 0;
    }

    uint8_t* row_buf = (uint8_t*)malloc(stride);
    if (!row_buf) {
        fprintf(stderr, "ERROR: Row buffer allocation failed\n");
        image_free(img); fclose(fp); return 0;
    }

    fseek(fp, fh.pixel_offset, SEEK_SET);

    int r;
    for (r = 0; r < h; r++) {
        int dest_row = top_down ? r : (h - 1 - r); /* BMP default = bottom-up */
        if (fread(row_buf, 1, stride, fp) != (size_t)stride) { fprintf(stderr, "ERROR: Unexpected end of BMP pixel data\n"); free(row_buf); image_free(img); fclose(fp); return 0; }

        int c;
        for (c = 0; c < w; c++) {
            if (bpp == 1) {
                /* 8-bit grayscale: one byte per pixel */
                PIXEL(*img, dest_row, c) = row_buf[c];
            } else {
                /* 24-bit RGB stored as BGR — convert using ITU-R BT.601 luminance */
                uint8_t B = row_buf[c * 3 + 0];
                uint8_t G = row_buf[c * 3 + 1];
                uint8_t R = row_buf[c * 3 + 2];
                PIXEL(*img, dest_row, c) = (uint8_t)(0.299*R + 0.587*G + 0.114*B);
            }
        }
    }

    free(row_buf);
    fclose(fp);
    return 1;
}

/* Save an Image as an 8-bit grayscale BMP file */
int save_bmp(const char* path, const Image* img) {
    int stride        = (img->width + 3) & ~3;
    int palette_bytes = 256 * 4;
    uint32_t pixel_offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + palette_bytes;
    uint32_t file_size    = pixel_offset + stride * img->height;

    BMPFileHeader fh;
    memset(&fh, 0, sizeof(fh));
    fh.type         = 0x4D42;
    fh.file_size    = file_size;
    fh.pixel_offset = pixel_offset;

    BMPInfoHeader ih;
    memset(&ih, 0, sizeof(ih));
    ih.header_size = 40;
    ih.width       = img->width;
    ih.height      = img->height;  /* Positive = bottom-up (standard BMP) */
    ih.planes      = 1;
    ih.bit_count   = 8;

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot write BMP: %s\n", path);
        return 0;
    }

    fwrite(&fh, sizeof(BMPFileHeader), 1, fp);
    fwrite(&ih, sizeof(BMPInfoHeader), 1, fp);

    /* Write grayscale color palette: entry i = (B=i, G=i, R=i, pad=0) */
    int i;
    for (i = 0; i < 256; i++) {
        uint8_t entry[4] = { (uint8_t)i, (uint8_t)i, (uint8_t)i, 0 };
        fwrite(entry, 1, 4, fp);
    }

    /* Write pixel data bottom-to-top (standard BMP row order) */
    uint8_t* row_buf = (uint8_t*)calloc(stride, 1);
    int r;
    for (r = img->height - 1; r >= 0; r--) {
        memcpy(row_buf, &img->pixels[r * img->width], img->width);
        fwrite(row_buf, 1, stride, fp);
    }

    free(row_buf);
    fclose(fp);
    return 1;
}


/* ============================================================
 *  PNG I/O  (via libpng — compile with -lpng)
 * ============================================================ */

/* Load a PNG file into an Image struct (auto-converts to 8-bit grayscale) */
int load_png(const char* path, Image* img) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open PNG: %s\n", path);
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                             NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "ERROR: png_create_read_struct failed\n");
        fclose(fp); return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "ERROR: png_create_info_struct failed\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp); return 0;
    }

    /* libpng error recovery jump point */
    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "ERROR: Failed reading PNG: %s\n", path);
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp); return 0;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int w          = (int)png_get_image_width(png, info);
    int h          = (int)png_get_image_height(png, info);
    png_byte color = png_get_color_type(png, info);
    png_byte depth = png_get_bit_depth(png, info);

    /* Normalize all PNG variants to 8-bit grayscale */
    if (depth == 16)
        png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_RGB || color == PNG_COLOR_TYPE_RGB_ALPHA)
        png_set_rgb_to_gray(png, 1, 0.299, 0.587);
    if (color & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(png);
    if (depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    png_read_update_info(png, info);

    if (!image_alloc(img, w, h)) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp); return 0;
    }

    /* Build row pointer array pointing into our pixel buffer */
    png_bytep* rows = (png_bytep*)malloc(h * sizeof(png_bytep));
    if (!rows) {
        fprintf(stderr, "ERROR: Row pointer allocation failed\n");
        image_free(img);
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp); return 0;
    }

    int r;
    for (r = 0; r < h; r++)
        rows[r] = &img->pixels[r * w];

    png_read_image(png, rows);

    free(rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return 1;
}

/* Save an Image as an 8-bit grayscale PNG file */
int save_png(const char* path, const Image* img) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot write PNG: %s\n", path);
        return 0;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                              NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "ERROR: png_create_write_struct failed\n");
        fclose(fp); return 0;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "ERROR: png_create_info_struct failed\n");
        png_destroy_write_struct(&png, NULL);
        fclose(fp); return 0;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "ERROR: Failed writing PNG: %s\n", path);
        png_destroy_write_struct(&png, &info);
        fclose(fp); return 0;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info,
                 img->width, img->height, 8,
                 PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    int r;
    for (r = 0; r < img->height; r++)
        png_write_row(png, &img->pixels[r * img->width]);

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 1;
}


/* ============================================================
 *  JPEG I/O  (via libjpeg — compile with -DWITH_JPEG -ljpeg)
 *  If not compiled with JPEG support, a clear install message shown.
 * ============================================================ */
#ifdef WITH_JPEG

int load_jpg(const char* path, Image* img) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open JPG: %s\n", path);
        return 0;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);

    /* Force grayscale output regardless of source color space */
    cinfo.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&cinfo);

    int w = (int)cinfo.output_width;
    int h = (int)cinfo.output_height;

    if (!image_alloc(img, w, h)) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(fp); return 0;
    }

    while ((int)cinfo.output_scanline < h) {
        uint8_t* row_ptr = &img->pixels[cinfo.output_scanline * w];
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    return 1;
}

int save_jpg(const char* path, const Image* img) {
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot write JPG: %s\n", path);
        return 0;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width      = img->width;
    cinfo.image_height     = img->height;
    cinfo.input_components = 1;
    cinfo.in_color_space   = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while ((int)cinfo.next_scanline < img->height) {
        uint8_t* row = &img->pixels[cinfo.next_scanline * img->width];
        jpeg_write_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
    return 1;
}

#else

/* Stubs shown when compiled without -DWITH_JPEG */
int load_jpg(const char* path, Image* img) {
    (void)path; (void)img;
    fprintf(stderr,
        "ERROR: JPEG support not compiled in.\n"
        "  Install : sudo apt install libjpeg-dev\n"
        "  Recompile: gcc -O2 -DWITH_JPEG -o sobel_serial sobel_serial.c -lm -lpng -ljpeg\n");
    return 0;
}
int save_jpg(const char* path, const Image* img) {
    (void)path; (void)img;
    fprintf(stderr,
        "ERROR: JPEG support not compiled in.\n"
        "  Install : sudo apt install libjpeg-dev\n"
        "  Recompile: gcc -O2 -DWITH_JPEG -o sobel_serial sobel_serial.c -lm -lpng -ljpeg\n");
    return 0;
}

#endif /* WITH_JPEG */


/* ============================================================
 *  FORMAT DISPATCHER
 *  Routes load/save calls to the correct handler by file extension.
 * ============================================================ */
int load_image(const char* path, Image* img) {
    char ext[16];
    get_extension(path, ext, sizeof(ext));

    if (strcmp(ext, "bmp") == 0)
        return load_bmp(path, img);
    if (strcmp(ext, "png") == 0)
        return load_png(path, img);
    if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0)
        return load_jpg(path, img);

    fprintf(stderr,
        "ERROR: Unsupported input format '.%s'\n"
        "  Supported: bmp, png, jpg, jpeg\n", ext);
    return 0;
}

int save_image(const char* path, const Image* img) {
    char ext[16];
    get_extension(path, ext, sizeof(ext));

    if (strcmp(ext, "bmp") == 0)
        return save_bmp(path, img);
    if (strcmp(ext, "png") == 0)
        return save_png(path, img);
    if (strcmp(ext, "jpg") == 0 || strcmp(ext, "jpeg") == 0)
        return save_jpg(path, img);

    fprintf(stderr,
        "ERROR: Unsupported output format '.%s'\n"
        "  Supported: bmp, png, jpg, jpeg\n", ext);
    return 0;
}


/* ============================================================
 *  SOBEL EDGE DETECTION — SERIAL
 *
 *  Applies two 3x3 Sobel kernels to every interior pixel.
 *  Border pixels (1-pixel ring on all sides) stay 0 (black).
 *
 *  Gx kernel — detects vertical edges (horizontal gradient):
 *    [ -1   0  +1 ]
 *    [ -2   0  +2 ]
 *    [ -1   0  +1 ]
 *
 *  Gy kernel — detects horizontal edges (vertical gradient):
 *    [ -1  -2  -1 ]
 *    [  0   0   0 ]
 *    [ +1  +2  +1 ]
 *
 *  Gradient magnitude (accurate formula):
 *    G = sqrt(Gx^2 + Gy^2)
 *
 *  Parameters:
 *    input     — source grayscale Image
 *    output    — destination Image (must be pre-allocated same size)
 *    threshold — pixels with G > threshold -> 255, others keep G value
 *                Use 0 to show full gradient without hard threshold
 * ============================================================ */
void sobel_serial(const Image* input, Image* output, int threshold) {

    /* Sobel kernel coefficients */
    const int Kx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int Ky[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

    int W = input->width;
    int H = input->height;

    /* Initialize output to zero — border pixels remain 0 */
    memset(output->pixels, 0, W * H * sizeof(uint8_t));

    int row, col, kr, kc;
    for (row = 1; row < H - 1; row++) {
        for (col = 1; col < W - 1; col++) {

            int gx = 0, gy = 0;

            /* Convolve 3x3 neighborhood with both Sobel kernels */
            for (kr = -1; kr <= 1; kr++) {
                for (kc = -1; kc <= 1; kc++) {
                    int px = PIXEL(*input, row + kr, col + kc);
                    gx += Kx[kr + 1][kc + 1] * px;
                    gy += Ky[kr + 1][kc + 1] * px;
                }
            }

            /* Gradient magnitude — accurate sqrt formula */
            int mag = (int)sqrt((double)(gx * gx + gy * gy));

            /* Clamp to valid pixel range [0, 255] */
            if (mag > 255) mag = 255;

            /* Apply threshold: strong edges -> 255, weak -> 0 */
            if (threshold > 0)
                PIXEL(*output, row, col) = (mag > threshold) ? 255 : 0;
            else
                PIXEL(*output, row, col) = (uint8_t)mag;
        }
    }
}


/* ============================================================
 *  RMSE — Accuracy Metric
 *
 *  Compares two images pixel-by-pixel.
 *  RMSE = sqrt( sum((a[i] - b[i])^2) / N )
 *
 *  RMSE = 0.0  -> pixel-perfect match (ideal for parallel validation)
 *  Used in Phase 3/4/5 to confirm parallel output matches serial.
 * ============================================================ */
double compute_rmse(const Image* a, const Image* b) {
    if (a->width != b->width || a->height != b->height) {
        fprintf(stderr, "ERROR: RMSE — image sizes do not match\n");
        return -1.0;
    }
    double sum = 0.0;
    int i;
    int total = a->width * a->height;
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
               "  input     : BMP, PNG, or JPG image\n"
               "  output    : BMP, PNG, or JPG (can differ from input format)\n"
               "  threshold : 0-255, default=0 (show full gradient)\n\n"
               "Examples:\n"
               "  %s photo.jpg  edges.png  30\n"
               "  %s image.bmp  result.bmp\n"
               "  %s input.png  output.bmp\n\n",
               argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int threshold = (argc >= 4) ? atoi(argv[3]) : 0;

    printf("\n========================================\n"
           "  EE7218 HPC - Serial Sobel Detector\n"
           "  Group 02 | C | Multi-Format\n"
           "========================================\n"
           "  Input     : %s\n"
           "  Output    : %s\n"
           "  Threshold : %d\n"
           "----------------------------------------\n",
           input_path, output_path, threshold);

    /* --- Load image --- */
    Image input;
    double t0 = now_ms();
    if (!load_image(input_path, &input)) return 1;
    double load_ms = now_ms() - t0;

    printf("  Loaded    : %d x %d px  (%.2f ms)\n",
           input.width, input.height, load_ms);

    /* --- Allocate output buffer --- */
    Image output;
    if (!image_alloc(&output, input.width, input.height)) {
        fprintf(stderr, "ERROR: Failed to allocate output image\n");
        image_free(&input);
        return 1;
    }

    /* --- Run Sobel and measure time --- */
    printf("----------------------------------------\n"
           "  Running Sobel edge detection...\n");

    double t1 = now_ms();
    sobel_serial(&input, &output, threshold);
    double sobel_ms = now_ms() - t1;

    /* --- Save output --- */
    double t2 = now_ms();
    if (!save_image(output_path, &output)) {
        image_free(&input);
        image_free(&output);
        return 1;
    }
    double save_ms = now_ms() - t2;

    double total_ms = now_ms() - t0;

    /* --- Print results --- */
    printf("----------------------------------------\n"
           "  Image Size   : %d x %d\n"
           "  Total Pixels : %d\n"
           "----------------------------------------\n"
           "  Load Time    : %.3f ms\n"
           "  Sobel Time   : %.3f ms   <-- benchmark this\n"
           "  Save Time    : %.3f ms\n"
           "  Total Time   : %.3f ms\n"
           "  Throughput   : %.2f Mpixels/sec\n"
           "========================================\n"
           "  Done.\n"
           "========================================\n\n",
           input.width, input.height,
           TOTAL(input),
           load_ms,
           sobel_ms,
           save_ms,
           total_ms,
           (TOTAL(input) / 1.0e6) / (sobel_ms / 1000.0));

    /* --- Cleanup --- */
    image_free(&input);
    image_free(&output);

    return 0;
}
