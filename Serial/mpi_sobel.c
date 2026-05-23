/****************************************************************************
 * FILE: mpi_sobel.c
 * DESCRIPTION:
 *   Distributed-Memory Sobel Edge Detection using MPI.
 * 
 * COMPILE:  mpicc -O2 -o mpi_sobel mpi_sobel.c -lm
 * RUN:      mpirun -np 4 ./mpi_sobel input.pgm output_mpi.pgm
 *           mpirun -np 1 ./mpi_sobel input.pgm output_mpi.pgm   (baseline)
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- Sobel gradient magnitude ---- */
static inline int sobel_magnitude(int gx, int gy)
{
    int val = (int)sqrt((double)(gx * gx + gy * gy));
    if (val > 255) val = 255;
    if (val < 0)   val = 0;
    return val;
}

/* ---- Skip comment lines in PGM headers ---- */
static void skip_pgm_comments(FILE *fp)
{
    int c;
    while ((c = fgetc(fp)) != EOF)
    {
        if (c == '#')
            while ((c = fgetc(fp)) != EOF && c != '\n')
                ;
        else { ungetc(c, fp); break; }
    }
}

/* =====================
 * MAIN
 * ===================== */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *in_path  = (argc >= 2) ? argv[1] : "input.pgm";
    const char *out_path = (argc >= 3) ? argv[2] : "output_mpi.pgm";

    int width = 0, height = 0, maxval = 0;

    /* ---- Read-only image buffer ---- */
    unsigned char *full_image = NULL;
    unsigned char *full_edge  = NULL;

    /* =====================================
     * RANK 0: Read PGM header and pixel data
     * ===================================== */
    if (rank == 0)
    {
        FILE *fp = fopen(in_path, "rb");
        if (!fp)
        {
            fprintf(stderr, "ERROR: Cannot open '%s'\n", in_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char fmt[8];
        skip_pgm_comments(fp);  fscanf(fp, "%7s", fmt);
        skip_pgm_comments(fp);  fscanf(fp, "%d %d", &width, &height);
        skip_pgm_comments(fp);  fscanf(fp, "%d", &maxval);
        fgetc(fp);

        if (fmt[0] != 'P' || fmt[1] != '5')
        {
            fprintf(stderr, "ERROR: Only binary PGM (P5) supported.\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("[Rank 0] Image: %s  [%d x %d]\n", in_path, width, height);

        full_image = (unsigned char *)malloc(width * height);
        full_edge  = (unsigned char *)calloc(width * height, 1);

        if (!full_image || !full_edge)
        {
            fprintf(stderr, "ERROR: malloc failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (fread(full_image, 1, width * height, fp) != (size_t)(width * height))
        {
            fprintf(stderr, "ERROR: Incomplete image read\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(fp);
    }

    /* =====================================
     * Broadcast image dimensions to all ranks
     * ===================================== */
    MPI_Bcast(&width,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* =====================================
     * Compute row-slab distribution
     * ===================================== */
    int base_rows = height / size;
    int rem       = height % size;

    /* sendcounts[r] = number of pixels rank r receives (rows × width) */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int *row_counts = (int *)malloc(size * sizeof(int)); /* rows per rank */

    int offset = 0;
    for (int r = 0; r < size; r++)
    {
        row_counts[r]  = base_rows + (r < rem ? 1 : 0);
        sendcounts[r]  = row_counts[r] * width;
        displs[r]      = offset;
        offset        += sendcounts[r];
    }

    int local_rows   = row_counts[rank];
    int local_pixels = local_rows * width;

    /* =====================================
     * Allocate local buffers
     * ===================================== */
    int buf_rows   = local_rows + 2;
    int buf_pixels = buf_rows * width;

    unsigned char *local_image = (unsigned char *)calloc(buf_pixels, 1);
    unsigned char *local_blur  = (unsigned char *)calloc(buf_pixels, 1);
    unsigned char *local_edge  = (unsigned char *)calloc(buf_pixels, 1);

    if (!local_image || !local_blur || !local_edge)
    {
        fprintf(stderr, "ERROR: malloc failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* =====================================
     * Scatter image rows
     * ===================================== */
    MPI_Scatterv(
        full_image,              /* send buffer (rank 0 only) */
        sendcounts, displs,      /* counts and offsets         */
        MPI_UNSIGNED_CHAR,
        local_image + width,     /* receive at row 1 (after ghost) */
        local_pixels,
        MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD
    );

    /* Free the full image on rank 0 */
    if (rank == 0) { free(full_image); full_image = NULL; }

    /* =====================================
     * Exchange ghost rows for the image buffer
     * ===================================== */
    int up   = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Call 1: */
    MPI_Sendrecv(
        local_image + local_rows * width,       /* send: my last real row  */
        width, MPI_UNSIGNED_CHAR, down, 0,
        local_image,                             /* recv: ghost row at top  */
        width, MPI_UNSIGNED_CHAR, up,   0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    /* Call 2: */
    MPI_Sendrecv(
        local_image + width,                     /* send: my first real row */
        width, MPI_UNSIGNED_CHAR, up,   1,
        local_image + (local_rows + 1) * width,  /* recv: ghost row at end  */
        width, MPI_UNSIGNED_CHAR, down, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    /* ================================================================
     * Kernels (shared constants, identical across all ranks)
     * ============================================================== */
    const int G[3][3]  = {{1,2,1},{2,4,2},{1,2,1}};
    const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    /* ================================================================
     * START TIMING (after all communication setup)
     * ============================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* =====================================
     * STAGE 1: Gaussian Blur
     * ===================================== */
    /*
     * BORDER FIX: serial code skips global row 0 and row (height-1).
     * Rank 0 owns global row 0 at buffer row 1  → start from ro    w 2.
     * Last rank owns global row (height-1) at buffer row local_rows → end at local_rows-1.
     * Buffers are calloc'd (zero), so skipped rows stay 0 — matching serial exactly.
     */
    int i_start = (rank == 0)        ? 2             : 1;
    int i_end   = (rank == size - 1) ? local_rows - 1 : local_rows;

    for (int i = i_start; i <= i_end; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int sum = 0;
            for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                    sum += local_image[(i + x) * width + (j + y)]
                           * G[x + 1][y + 1];
            local_blur[i * width + j] = (unsigned char)(sum / 16);
        }
    }

    /* ================================================================
     * Exchange ghost rows for the blur buffer before Stage 2
     * Same correct pattern: send down/recv from up, then send up/recv from down
     * ============================================================== */
    MPI_Sendrecv(
        local_blur + local_rows * width,          /* send: my last blur row   */
        width, MPI_UNSIGNED_CHAR, down, 2,
        local_blur,                                /* recv: top ghost          */
        width, MPI_UNSIGNED_CHAR, up,   2,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        local_blur + width,                        /* send: my first blur row  */
        width, MPI_UNSIGNED_CHAR, up,   3,
        local_blur + (local_rows + 1) * width,    /* recv: bottom ghost       */
        width, MPI_UNSIGNED_CHAR, down, 3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    /* ================================================================
     * STAGE 2: Sobel Edge Detection (on local slab)
     * ============================================================== */
    /* Same i_start / i_end: skip global border rows in Sobel pass too */
    for (int i = i_start; i <= i_end; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int gx = 0, gy = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    int pixel = local_blur[(i + x) * width + (j + y)];
                    gx += pixel * Gx[x + 1][y + 1];
                    gy += pixel * Gy[x + 1][y + 1];
                }
            }
            local_edge[i * width + j] = (unsigned char)sobel_magnitude(gx, gy);
        }
    }

    /* ================================================================
     * END TIMING – use MPI_Reduce to get the maximum time across ranks
     * ============================================================== */
    double t_end     = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ================================================================
     * Gather edge results back to rank 0
     * (gather from row 1 of local_edge, skipping the ghost row)
     * ============================================================== */
    MPI_Gatherv(
        local_edge + width,    /* skip ghost row 0 */
        local_pixels,
        MPI_UNSIGNED_CHAR,
        full_edge,             /* destination on rank 0 */
        sendcounts, displs,
        MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD
    );

    /* ================================================================
     * RANK 0: Write output PGM
     * ============================================================== */
    if (rank == 0)
    {
        printf("MPI execution time    : %.6f seconds  (%d processes)\n",
               max_time, size);

        FILE *out = fopen(out_path, "wb");
        if (!out)
        {
            fprintf(stderr, "ERROR: Cannot open '%s' for writing\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(out, "P5\n%d %d\n255\n", width, height);
        fwrite(full_edge, 1, width * height, out);
        fclose(out);

        printf("Output saved to       : %s\n", out_path);
        printf("MPI edge detection finished.\n");
        free(full_edge);
    }

    free(local_image);
    free(local_blur);
    free(local_edge);
    free(sendcounts);
    free(displs);
    free(row_counts);

    MPI_Finalize();
    return 0;
}