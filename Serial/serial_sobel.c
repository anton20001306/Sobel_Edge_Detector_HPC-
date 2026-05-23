/****************************************************************************
 * FILE: serial_sobel.c
 * DESCRIPTION:
 *   Serial Sobel Edge Detection with Gaussian Blur
 *   Baseline implementation for performance comparison.
 *   Pipeline: Load PGM → Gaussian Blur → Sobel Edge Detection → Save PGM

 * COMPILE:  gcc -O2 -o serial_sobel serial_sobel.c -lm -fopenmp
 * RUN:      ./serial_sobel input.pgm output.pgm
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* -------------------
 * sobel_magnitude
 * ------------------- */
static int sobel_magnitude(int gx, int gy)
{
    int val = (int)sqrt((double)(gx * gx + gy * gy));
    if (val > 255) val = 255;
    if (val < 0)   val = 0;
    return val;
}

/* -------------------
 * skip_pgm_comments
 * ------------------- */
static void skip_pgm_comments(FILE *fp)
{
    int c;
    while ((c = fgetc(fp)) != EOF)
    {
        if (c == '#')
        {
            while ((c = fgetc(fp)) != EOF && c != '\n')
                ;
        }
        else
        {
            ungetc(c, fp);
            break;
        }
    }
}

/* ===================================
 * MAIN
 * =================================== */
int main(int argc, char *argv[])
{
    const char *in_path  = (argc >= 2) ? argv[1] : "input.pgm";
    const char *out_path = (argc >= 3) ? argv[2] : "output_serial.pgm";

    /* ---- Open input PGM ---- */
    FILE *fp = fopen(in_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "ERROR: Cannot open '%s'\n", in_path);
        return 1;
    }

    /* ---- Read PGM header (handles comment lines) ---- */
    char format[8];
    int  width, height, maxval;

    skip_pgm_comments(fp);
    fscanf(fp, "%7s", format);

    if (strcmp(format, "P5") != 0)
    {
        fprintf(stderr, "ERROR: Only binary PGM (P5) is supported.\n");
        fclose(fp);
        return 1;
    }

    skip_pgm_comments(fp);
    fscanf(fp, "%d %d", &width, &height);
    skip_pgm_comments(fp);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);

    printf("Image: %s  [%d x %d], maxval=%d\n", in_path, width, height, maxval);

    /* ---- Allocate image buffers ---- */
    unsigned char *image = (unsigned char *)malloc(width * height);
    unsigned char *blur  = (unsigned char *)malloc(width * height);
    unsigned char *edge  = (unsigned char *)malloc(width * height);

    if (!image || !blur || !edge)
    {
        fprintf(stderr, "ERROR: malloc failed\n");
        fclose(fp);
        return 1;
    }

    /* ---- Initialise border pixels to zero ---- */
    memset(blur, 0, width * height);
    memset(edge, 0, width * height);

    /* ---- Read raw pixel data ---- */
    if (fread(image, 1, width * height, fp) != (size_t)(width * height))
    {
        fprintf(stderr, "ERROR: Could not read full image data\n");
        fclose(fp);
        free(image); free(blur); free(edge);
        return 1;
    }
    fclose(fp);

    /* ---- Kernels ---- */

    /* 3×3 Gaussian kernel (sum = 16) for noise reduction */
    const int G[3][3] =
    {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    /* Horizontal Sobel kernel */
    const int Gx[3][3] =
    {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    /* Vertical Sobel kernel */
    const int Gy[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    /* ===================================
     * START TIMING
     * =================================== */
    double t_start = omp_get_wtime();

    /* ---- Stage 1: Gaussian Blur ---- */
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int sum = 0;
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    sum += image[(i + x) * width + (j + y)] * G[x + 1][y + 1];

            blur[i * width + j] = (unsigned char)(sum / 16);
        }
    }

    /* ---- Stage 2: Sobel Edge Detection ---- */
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int gx = 0, gy = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    int pixel = blur[(i + x) * width + (j + y)];
                    gx += pixel * Gx[x + 1][y + 1];
                    gy += pixel * Gy[x + 1][y + 1];
                }
            }
            edge[i * width + j] = (unsigned char)sobel_magnitude(gx, gy);
        }
    }

    /* ===================================
     * END TIMING
     * =================================== */
    double t_end = omp_get_wtime();

    printf("Serial execution time : %.6f seconds\n", t_end - t_start);

    /* ---- Write output PGM ---- */
    FILE *out = fopen(out_path, "wb");
    if (!out)
    {
        fprintf(stderr, "ERROR: Cannot open '%s' for writing\n", out_path);
        free(image); free(blur); free(edge);
        return 1;
    }
    fprintf(out, "P5\n%d %d\n255\n", width, height);
    fwrite(edge, 1, width * height, out);
    fclose(out);

    printf("Output saved to       : %s\n", out_path);
    printf("Edge detection finished.\n");

    free(image);
    free(blur);
    free(edge);
    return 0;
}
