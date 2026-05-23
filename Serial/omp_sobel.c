/****************************************************************************
 * FILE: omp_sobel.c
 * DESCRIPTION:
 *   OpenMP Parallel Sobel Edge Detection with Gaussian Blur.
 *
 *   Pipeline:
 *     Load PGM → [OMP] Gaussian Blur → [OMP] Sobel Edge Detection → Save PGM
 *
 * COMPILE:  gcc -O2 -fopenmp -o omp_sobel omp_sobel.c -lm
 * RUN:      ./omp_sobel [input.pgm] [output.pgm] [num_threads]
 ****************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * sobel_magnitude
 * ---------------------------------------------------------------------- */
static inline int sobel_magnitude(int gx, int gy)
{
    int val = (int)sqrt((double)(gx * gx + gy * gy));
    if (val > 255) val = 255;
    if (val < 0)   val = 0;
    return val;
}

/* -------------------------------------------------------------------------
 * skip_pgm_comments
 * ---------------------------------------------------------------------- */
static void skip_pgm_comments(FILE *fp)
{
    int c;
    while ((c = fgetc(fp)) != EOF)
    {
        if (c == '#')
            while ((c = fgetc(fp)) != EOF && c != '\n')
                ;
        else
        {
            ungetc(c, fp);
            break;
        }
    }
}

/* =========================================================================
 * MAIN
 * ====================================================================== */
int main(int argc, char *argv[])
{
    const char *in_path  = (argc >= 2) ? argv[1] : "input.pgm";
    const char *out_path = (argc >= 3) ? argv[2] : "output_omp.pgm";
    int num_threads      = (argc >= 4) ? atoi(argv[3]) : 0;

    if (num_threads > 0)
        omp_set_num_threads(num_threads);

    /* ---- Open and parse PGM header ---- */
    FILE *fp = fopen(in_path, "rb");
    if (!fp)
    {
        fprintf(stderr, "ERROR: Cannot open '%s'\n", in_path);
        return 1;
    }

    char format[8];
    int  width, height, maxval;

    skip_pgm_comments(fp);
    fscanf(fp, "%7s", format);
    skip_pgm_comments(fp);
    fscanf(fp, "%d %d", &width, &height);
    skip_pgm_comments(fp);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);

    if (format[0] != 'P' || format[1] != '5')
    {
        fprintf(stderr, "ERROR: Only binary PGM (P5) is supported.\n");
        fclose(fp); return 1;
    }

    printf("Image   : %s  [%d x %d]\n", in_path, width, height);

    /* ---- Allocate buffers ---- */
    unsigned char *image = (unsigned char *)malloc(width * height);
    unsigned char *blur  = (unsigned char *)malloc(width * height);
    unsigned char *edge  = (unsigned char *)malloc(width * height);

    if (!image || !blur || !edge)
    {
        fprintf(stderr, "ERROR: malloc failed\n");
        fclose(fp); return 1;
    }

    /* Border pixels */
    memset(blur, 0, width * height);
    memset(edge, 0, width * height);

    if (fread(image, 1, width * height, fp) != (size_t)(width * height))
    {
        fprintf(stderr, "ERROR: Incomplete image read\n");
        fclose(fp); free(image); free(blur); free(edge); return 1;
    }
    fclose(fp);

    /* ---- Kernels---- */

    /* Gaussian blur kernel, sum = 16 */
    const int G[3][3] =
    {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    const int Gx[3][3] =   /* horizontal Sobel */
    {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    const int Gy[3][3] =   /* vertical Sobel */
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    /*
     * Setting chunk = height / (4 * nthreads)
     */
    int nthreads = 0;
    int chunk    = 1;

    /* ============================
     * PARALLEL REGION 
     * ============================ */
    double t_start = omp_get_wtime();

#pragma omp parallel shared(image, blur, edge, width, height, \
                             G, Gx, Gy, nthreads, chunk)
    {
        /* ---- Discover thread count and set chunk on first entry ---- */
#pragma omp single
        {
            nthreads = omp_get_num_threads();
            /* 4 chunks per thread keeps load well balanced */
            chunk = (height > 4 * nthreads) ? height / (4 * nthreads) : 1;
            printf("Threads : %d  |  chunk = %d rows\n", nthreads, chunk);
        }

        /* ============================
         * STAGE 1: Gaussian Blur
         * ============================ */
#pragma omp for schedule(static, chunk)
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

        /* ============================
         * STAGE 2: Sobel Edge Detection
         * ============================ */
#pragma omp for schedule(static, chunk)
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

    } /* ---- end parallel region ---- */

    double t_end = omp_get_wtime();

    printf("OpenMP execution time : %.6f seconds  (%d threads)\n",
           t_end - t_start, nthreads);

    /* ---- Write output PGM ---- */
    FILE *out = fopen(out_path, "wb");
    if (!out)
    {
        fprintf(stderr, "ERROR: Cannot open '%s' for writing\n", out_path);
        free(image); free(blur); free(edge); return 1;
    }
    fprintf(out, "P5\n%d %d\n255\n", width, height);
    fwrite(edge, 1, width * height, out);
    fclose(out);

    printf("Output saved to       : %s\n", out_path);
    printf("Parallel edge detection finished.\n");

    free(image);
    free(blur);
    free(edge);
    return 0;
}
