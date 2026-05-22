/****************************************************************************
 * FILE: hybrid_sobel.cu
 * DESCRIPTION:
 *   Hybrid MPI + CUDA Sobel Edge Detection.
 *
 *   Parallelisation Strategy:
 *     MPI distributes row-slabs across nodes/processes (same decomposition
 *     as mpi_sobel.c).  Within each MPI rank, the Gaussian Blur and Sobel
 *     Edge Detection kernels are offloaded to the GPU via CUDA.  Each rank
 *     selects a GPU using (rank % deviceCount) to support multi-GPU nodes.
 *
 *   Memory Flow (per rank):
 *     Host slab → cudaMemcpy H2D → Gaussian kernel → Sobel kernel
 *     → cudaMemcpy D2H → MPI_Gatherv → Rank 0 writes PGM
 *
 *   CUDA Kernel Design:
 *     Both kernels use 2-D thread blocks (BLOCK_W × BLOCK_H = 16×16 = 256
 *     threads/block).  Each thread computes exactly one output pixel.
 *     Thread indices map directly to (row, col) in the local slab, so no
 *     atomic operations or shared-memory coordination are required.
 *
 *   Timing:
 *     cudaEvent_t records GPU kernel time separately from H2D/D2H transfer
 *     time, so transfer overhead can be reported independently.
 *     MPI_Reduce(MPI_MAX) gives the bottleneck time across all ranks.
 *
 * COURSE:   EE7218 – High Performance Computing
 * GROUP:    02 (Electrical & Information Engineering)
 * MEMBERS:  EG/2021/4512 – W.A.P.N Fernando
 *           EG/2021/4654 – S.W.M Madhusan
 *
 * COMPILE:  nvcc -O2 -o hybrid_sobel hybrid_sobel.cu -lm \
 *               $(mpicc --showme:compile) $(mpicc --showme:link)
 *
 *   OR, if mpicxx wrapper supports CUDA:
 *           mpicxx -O2 -o hybrid_sobel hybrid_sobel.cu -lm -lcudart
 *
 * RUN:      mpirun -np 4 ./hybrid_sobel input.pgm output_hybrid.pgm
 *
 * REQUIRES: CUDA-capable GPU, CUDA Toolkit, Open MPI built with CUDA support
 *           (or any MPI implementation – GPU-direct not required here).
 *
 * ACCURACY: Expected RMSE = 0 vs serial output (bit-identical computation).
 ****************************************************************************/

#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- Block dimensions for CUDA kernels ---- */
#define BLOCK_W 16
#define BLOCK_H 16

/* ==========================================================================
 * CUDA error-checking macro
 * ======================================================================== */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

/* ==========================================================================
 * CUDA KERNEL – Gaussian Blur
 *
 * Each thread computes one output pixel in 'blur', reading from 'image'.
 * The 3×3 Gaussian kernel (sum=16) is embedded as device constants.
 * Boundary threads (row/col == 0 or == edge) write nothing and exit early.
 * ======================================================================== */
__global__ void gaussianBlurKernel(
    const unsigned char * __restrict__ image,
          unsigned char * __restrict__ blur,
    int width, int buf_rows, int row_first, int row_last)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    /* Skip ghost rows, out-of-bounds, and global image border rows */
    if (row < row_first || row > row_last || col < 1 || col >= width - 1)
        return;

    const int G[3][3] = {{1,2,1},{2,4,2},{1,2,1}};

    int sum = 0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            sum += (int)image[(row + x) * width + (col + y)]
                   * G[x + 1][y + 1];

    blur[row * width + col] = (unsigned char)(sum / 16);
}

/* ==========================================================================
 * CUDA KERNEL – Sobel Edge Detection
 *
 * Reads from 'blur' (fully written by gaussianBlurKernel + sync).
 * Writes gradient magnitude to 'edge'.
 * ======================================================================== */
__global__ void sobelKernel(
    const unsigned char * __restrict__ blur,
          unsigned char * __restrict__ edge,
    int width, int buf_rows, int row_first, int row_last)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < row_first || row > row_last || col < 1 || col >= width - 1)
        return;

    const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

    int gx = 0, gy = 0;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            int pixel = (int)blur[(row + x) * width + (col + y)];
            gx += pixel * Gx[x + 1][y + 1];
            gy += pixel * Gy[x + 1][y + 1];
        }
    }

    int val = (int)sqrt((double)(gx * gx + gy * gy));
    if (val > 255) val = 255;
    if (val < 0)   val = 0;

    edge[row * width + col] = (unsigned char)val;
}

/* ==========================================================================
 * Helper: skip PGM comment lines
 * ======================================================================== */
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

/* ==========================================================================
 * MAIN
 * ======================================================================== */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Assign GPU: round-robin over available devices ---- */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        if (rank == 0)
            fprintf(stderr, "ERROR: No CUDA-capable GPU found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    const char *in_path  = (argc >= 2) ? argv[1] : "input.pgm";
    const char *out_path = (argc >= 3) ? argv[2] : "output_hybrid.pgm";

    int width = 0, height = 0, maxval = 0;

    unsigned char *full_image = NULL;
    unsigned char *full_edge  = NULL;

    /* ================================================================
     * RANK 0: Read PGM
     * ============================================================== */
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

        printf("[Rank 0] Image: %s  [%d x %d]  | GPUs available: %d\n",
               in_path, width, height, device_count);

        full_image = (unsigned char *)malloc(width * height);
        full_edge  = (unsigned char *)calloc(width * height, 1);
        fread(full_image, 1, width * height, fp);
        fclose(fp);
    }

    MPI_Bcast(&width,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ================================================================
     * Row-slab decomposition (same as MPI-only version)
     * ============================================================== */
    int  base_rows  = height / size;
    int  rem        = height % size;
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int *row_counts = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int r = 0; r < size; r++)
    {
        row_counts[r] = base_rows + (r < rem ? 1 : 0);
        sendcounts[r] = row_counts[r] * width;
        displs[r]     = offset;
        offset       += sendcounts[r];
    }

    int local_rows   = row_counts[rank];
    int local_pixels = local_rows * width;

    /* ================================================================
     * Allocate host buffers with +2 ghost rows
     * ============================================================== */
    int buf_rows   = local_rows + 2;
    int buf_pixels = buf_rows * width;

    unsigned char *h_image = (unsigned char *)calloc(buf_pixels, 1);
    unsigned char *h_edge  = (unsigned char *)calloc(buf_pixels, 1);

    /* ================================================================
     * Scatter row slabs to all ranks
     * ============================================================== */
    MPI_Scatterv(
        full_image, sendcounts, displs, MPI_UNSIGNED_CHAR,
        h_image + width,   /* skip ghost row 0 */
        local_pixels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD
    );
    if (rank == 0) { free(full_image); full_image = NULL; }

    /* ================================================================
     * Exchange ghost rows (host buffers, same pattern as MPI version)
     * ============================================================== */
    int up   = (rank > 0)       ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Correct pattern: send bottom→down recv top ghost←up (call 1)
     *                  send top→up    recv bot ghost←down (call 2) */
    MPI_Sendrecv(h_image + local_rows * width,   width, MPI_UNSIGNED_CHAR, down, 0,
                 h_image,                          width, MPI_UNSIGNED_CHAR, up,   0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_image + width,                 width, MPI_UNSIGNED_CHAR, up,   1,
                 h_image + (local_rows+1)*width,   width, MPI_UNSIGNED_CHAR, down, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* ================================================================
     * Allocate GPU buffers and copy host → device
     * ============================================================== */
    unsigned char *d_image, *d_blur, *d_edge;
    CUDA_CHECK(cudaMalloc(&d_image, buf_pixels));
    CUDA_CHECK(cudaMalloc(&d_blur,  buf_pixels));
    CUDA_CHECK(cudaMalloc(&d_edge,  buf_pixels));
    CUDA_CHECK(cudaMemset(d_blur, 0, buf_pixels));
    CUDA_CHECK(cudaMemset(d_edge, 0, buf_pixels));

    /* ================================================================
     * Measure H2D transfer time separately
     * ============================================================== */
    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventCreate(&ev3));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_image, h_image, buf_pixels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));

    /* ================================================================
     * MPI barrier before GPU timing
     * ============================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ================================================================
     * Launch CUDA kernels
     *
     * Grid dimensions: ceil(width/BLOCK_W) × ceil(buf_rows/BLOCK_H)
     * so every pixel in the padded slab has exactly one thread.
     * ============================================================== */
    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid(
        (width    + BLOCK_W - 1) / BLOCK_W,
        (buf_rows + BLOCK_H - 1) / BLOCK_H
    );

    /* Row range: skip global image border rows to match serial output (RMSE = 0) */
    int row_first = (rank == 0)        ? 2             : 1;
    int row_last  = (rank == size - 1) ? local_rows - 1 : local_rows;

    gaussianBlurKernel<<<grid, block>>>(d_image, d_blur, width, buf_rows, row_first, row_last);
    CUDA_CHECK(cudaGetLastError());

    /*
     * cudaDeviceSynchronize() ensures all blur pixels are written before
     * we copy boundary rows back to the host for the ghost exchange.
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ================================================================
     * Ghost-row exchange on the BLUR buffer
     *
     * The Sobel kernel reads a 3×3 neighbourhood from d_blur, so each
     * rank needs the adjacent rank's boundary blur rows — exactly like
     * the MPI-only version (lines 277-290 of mpi_sobel.c).
     *
     * Strategy:  D2H (boundary rows) → MPI exchange → H2D (ghost rows)
     * ============================================================== */
    unsigned char *h_blur = (unsigned char *)calloc(buf_pixels, 1);

    /* Copy only the two boundary real rows from d_blur → h_blur:
     *   - Row 1             (first real row)   → for sending UP
     *   - Row local_rows    (last  real row)    → for sending DOWN
     * We copy the full buffer for simplicity; for large images you
     * could copy only the two rows.                                   */
    CUDA_CHECK(cudaMemcpy(h_blur, d_blur, buf_pixels, cudaMemcpyDeviceToHost));

    /* Send my last blur row → down;  recv top ghost ← up */
    MPI_Sendrecv(
        h_blur + local_rows * width,          /* send: last real blur row  */
        width, MPI_UNSIGNED_CHAR, down, 2,
        h_blur,                                /* recv: top ghost           */
        width, MPI_UNSIGNED_CHAR, up,   2,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    /* Send my first blur row → up;  recv bottom ghost ← down */
    MPI_Sendrecv(
        h_blur + width,                        /* send: first real blur row */
        width, MPI_UNSIGNED_CHAR, up,   3,
        h_blur + (local_rows + 1) * width,    /* recv: bottom ghost        */
        width, MPI_UNSIGNED_CHAR, down, 3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    /* Copy the updated ghost rows back to the device */
    CUDA_CHECK(cudaMemcpy(d_blur, h_blur, buf_pixels, cudaMemcpyHostToDevice));
    free(h_blur);

    sobelKernel<<<grid, block>>>(d_blur, d_edge, width, buf_rows, row_first, row_last);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_end       = MPI_Wtime();
    double kernel_time = t_end - t_start;

    /* ================================================================
     * Copy results device → host
     * ============================================================== */
    CUDA_CHECK(cudaEventRecord(ev2));
    CUDA_CHECK(cudaMemcpy(h_edge, d_edge, buf_pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev3));
    CUDA_CHECK(cudaEventSynchronize(ev3));

    float t_h2d_ms = 0.0f, t_d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d_ms, ev0, ev1));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h_ms, ev2, ev3));

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaEventDestroy(ev2); cudaEventDestroy(ev3);

    /* ================================================================
     * Reduce to find max kernel time across all ranks
     * ============================================================== */
    double max_kernel_time;
    MPI_Reduce(&kernel_time, &max_kernel_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("GPU kernel time (max) : %.6f seconds  (%d ranks, %d GPU(s))\n",
               max_kernel_time, size, device_count);
        printf("H2D transfer (rank 0) : %.3f ms\n", t_h2d_ms);
        printf("D2H transfer (rank 0) : %.3f ms\n", t_d2h_ms);
    }

    /* ================================================================
     * Gather edge results back to rank 0
     * ============================================================== */
    MPI_Gatherv(
        h_edge + width,      /* skip ghost row 0 */
        local_pixels, MPI_UNSIGNED_CHAR,
        full_edge, sendcounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD
    );

    /* ================================================================
     * Rank 0: Write output PGM
     * ============================================================== */
    if (rank == 0)
    {
        FILE *out = fopen(out_path, "wb");
        if (!out)
        {
            fprintf(stderr, "ERROR: Cannot write '%s'\n", out_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(out, "P5\n%d %d\n255\n", width, height);
        fwrite(full_edge, 1, width * height, out);
        fclose(out);
        printf("Output saved to       : %s\n", out_path);
        printf("Hybrid edge detection finished.\n");
        free(full_edge);
    }

    /* ---- Cleanup ---- */
    cudaFree(d_image); cudaFree(d_blur); cudaFree(d_edge);
    free(h_image); free(h_edge);
    free(sendcounts); free(displs); free(row_counts);

    MPI_Finalize();
    return 0;
}
