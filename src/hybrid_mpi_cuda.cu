#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

/* =========================================================
                    DEVICE SOBEL FUNCTION
   ========================================================= */

__device__ int sobel(int gx, int gy)
{
    /*
        Calculate gradient magnitude
    */

    int val =
    (int)sqrtf((float)(gx * gx + gy * gy));

    /*
        Clamp value to 0-255
    */

    if(val > 255)
        val = 255;

    if(val < 0)
        val = 0;

    return val;
}

/* =========================================================
                    CUDA GAUSSIAN BLUR
   ========================================================= */

__global__ void gaussianBlur(
    unsigned char *input,
    unsigned char *blur,
    int width,
    int ghost_rows,
    int rank,
    int size)
{
    /*
        Thread coordinates
    */

    int j =
    blockIdx.x * blockDim.x + threadIdx.x;

    int i =
    blockIdx.y * blockDim.y + threadIdx.y;

    /*
        Valid pixels only
        Skip:
        - ghost rows
        - image borders
    */

    if(i >= 1 &&
       i < ghost_rows - 1 &&
       j >= 1 &&
       j < width - 1)
    {
        /*
            Match serial code behavior

            Serial code skips:
            i = 0
            i = height - 1

            Therefore:
            - rank 0 skips first real row
            - last rank skips last real row
        */

        if((rank == 0 && i == 1) ||
           (rank == size - 1 &&
            i == ghost_rows - 2))
        {
            return;
        }

        /*
            Gaussian Kernel
        */

        int G[3][3] =
        {
            {1,2,1},
            {2,4,2},
            {1,2,1}
        };

        int sum = 0;

        /*
            3x3 convolution
        */

        for(int x = -1; x <= 1; x++)
        {
            for(int y = -1; y <= 1; y++)
            {
                sum +=
                input[(i + x) * width + (j + y)]
                *
                G[x + 1][y + 1];
            }
        }

        /*
            Normalize blur value
        */

        blur[i * width + j] =
        sum / 16;
    }
}

/* =========================================================
                        CUDA SOBEL
   ========================================================= */

__global__ void sobelKernel(
    unsigned char *blur,
    unsigned char *edge,
    int width,
    int ghost_rows,
    int rank,
    int size)
{
    /*
        Thread coordinates
    */

    int j =
    blockIdx.x * blockDim.x + threadIdx.x;

    int i =
    blockIdx.y * blockDim.y + threadIdx.y;

    /*
        Valid region only
    */

    if(i >= 1 &&
       i < ghost_rows - 1 &&
       j >= 1 &&
       j < width - 1)
    {
        /*
            Match serial boundary behavior
        */

        if((rank == 0 && i == 1) ||
           (rank == size - 1 &&
            i == ghost_rows - 2))
        {
            return;
        }

        /*
            Sobel Kernels
        */

        int Gx[3][3] =
        {
            {-1,0,1},
            {-2,0,2},
            {-1,0,1}
        };

        int Gy[3][3] =
        {
            {-1,-2,-1},
            {0,0,0},
            {1,2,1}
        };

        int gx = 0;
        int gy = 0;

        /*
            3x3 Sobel convolution
        */

        for(int x = -1; x <= 1; x++)
        {
            for(int y = -1; y <= 1; y++)
            {
                int pixel =
                blur[(i + x) * width + (j + y)];

                gx +=
                pixel * Gx[x + 1][y + 1];

                gy +=
                pixel * Gy[x + 1][y + 1];
            }
        }

        /*
            Remove top ghost offset

            Ghost layout:
            row 0          -> top ghost
            row 1..rows    -> real rows
            row rows+1     -> bottom ghost
        */

        int out_row = i - 1;

        /*
            Store edge result
        */

        edge[out_row * width + j] =
        sobel(gx, gy);
    }
}

/* =========================================================
                            MAIN
   ========================================================= */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &size
    );

    int width, height, maxval;

    unsigned char *image = NULL;
    unsigned char *final_edge = NULL;

    /* =====================================================
                    ROOT LOADS IMAGE
       ===================================================== */

    if(rank == 0)
    {
        FILE *fp =
        fopen("input.pgm", "rb");

        if(fp == NULL)
        {
            printf("Cannot open image\n");

            MPI_Abort(
                MPI_COMM_WORLD,
                1
            );
        }

        char format[3];

        fscanf(fp,
               "%s",
               format);

        fscanf(fp,
               "%d %d",
               &width,
               &height);

        fscanf(fp,
               "%d",
               &maxval);

        /*
            Skip newline after header
        */

        fgetc(fp);

        /*
            Allocate full image
        */

        image =
        (unsigned char*)
        calloc(width * height,
               sizeof(unsigned char));

        /*
            Read image pixels
        */

        fread(image,
              1,
              width * height,
              fp);

        fclose(fp);

        printf(
            "Image Loaded: %dx%d\n",
            width,
            height
        );
    }

    /* =====================================================
                    BROADCAST IMAGE SIZE
       ===================================================== */

    MPI_Bcast(
        &width,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        &height,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /*
        Rows per process
    */

    int rows =
    height / size;

    /*
        Real image size per process
    */

    int local_size =
    rows * width;

    /*
        +2 ghost rows
    */

    int ghost_rows =
    rows + 2;

    int ghost_size =
    ghost_rows * width;

    /* =====================================================
                    LOCAL MEMORY
       ===================================================== */

    unsigned char *local_input =
    (unsigned char*)
    calloc(ghost_size,
           sizeof(unsigned char));

    unsigned char *local_blur =
    (unsigned char*)
    calloc(ghost_size,
           sizeof(unsigned char));

    unsigned char *local_edge =
    (unsigned char*)
    calloc(local_size,
           sizeof(unsigned char));

    /* =====================================================
                    DISTRIBUTE IMAGE
       ===================================================== */

    MPI_Scatter(
        image,
        local_size,
        MPI_UNSIGNED_CHAR,

        &local_input[width],

        local_size,
        MPI_UNSIGNED_CHAR,

        0,
        MPI_COMM_WORLD
    );

    /* =====================================================
                EXCHANGE INPUT GHOST ROWS
       ===================================================== */

    /*
        Send top real row upward
        Receive top ghost row
    */

    if(rank > 0)
    {
        MPI_Sendrecv(
            &local_input[width],
            width,
            MPI_UNSIGNED_CHAR,

            rank - 1,
            0,

            &local_input[0],
            width,
            MPI_UNSIGNED_CHAR,

            rank - 1,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /*
        Send bottom real row downward
        Receive bottom ghost row
    */

    if(rank < size - 1)
    {
        MPI_Sendrecv(
            &local_input[rows * width],
            width,
            MPI_UNSIGNED_CHAR,

            rank + 1,
            0,

            &local_input[(rows + 1) * width],
            width,
            MPI_UNSIGNED_CHAR,

            rank + 1,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /* =====================================================
                        GPU MEMORY
       ===================================================== */

    unsigned char *d_input;
    unsigned char *d_blur;
    unsigned char *d_edge;

    cudaMalloc(
        (void**)&d_input,
        ghost_size
    );

    cudaMalloc(
        (void**)&d_blur,
        ghost_size
    );

    cudaMalloc(
        (void**)&d_edge,
        local_size
    );

    /*
        Initialize GPU memory
    */

    cudaMemset(
        d_blur,
        0,
        ghost_size
    );

    cudaMemset(
        d_edge,
        0,
        local_size
    );

    /* =====================================================
                    COPY INPUT TO GPU
       ===================================================== */

    cudaMemcpy(
        d_input,
        local_input,
        ghost_size,
        cudaMemcpyHostToDevice
    );

    /* =====================================================
                        CUDA CONFIG
       ===================================================== */

    dim3 threads(16,16);

    dim3 blocks(
        (width + 15) / 16,
        (ghost_rows + 15) / 16
    );

    double start =
    MPI_Wtime();

    /* =====================================================
                    GAUSSIAN BLUR
       ===================================================== */

    gaussianBlur<<<blocks, threads>>>(
        d_input,
        d_blur,
        width,
        ghost_rows,
        rank,
        size
    );

    /*
        Wait for kernel completion
    */

    cudaDeviceSynchronize();

    /*
        Check CUDA errors
    */

    cudaError_t err =
    cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf(
            "Gaussian CUDA Error: %s\n",
            cudaGetErrorString(err)
        );
    }

    /* =====================================================
                COPY BLUR BACK TO CPU
       ===================================================== */

    cudaMemcpy(
        local_blur,
        d_blur,
        ghost_size,
        cudaMemcpyDeviceToHost
    );

    /* =====================================================
                EXCHANGE BLUR GHOST ROWS
       ===================================================== */

    if(rank > 0)
    {
        MPI_Sendrecv(
            &local_blur[width],
            width,
            MPI_UNSIGNED_CHAR,

            rank - 1,
            1,

            &local_blur[0],
            width,
            MPI_UNSIGNED_CHAR,

            rank - 1,
            1,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if(rank < size - 1)
    {
        MPI_Sendrecv(
            &local_blur[rows * width],
            width,
            MPI_UNSIGNED_CHAR,

            rank + 1,
            1,

            &local_blur[(rows + 1) * width],
            width,
            MPI_UNSIGNED_CHAR,

            rank + 1,
            1,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /* =====================================================
                COPY UPDATED BLUR TO GPU
       ===================================================== */

    cudaMemcpy(
        d_blur,
        local_blur,
        ghost_size,
        cudaMemcpyHostToDevice
    );

    /* =====================================================
                        SOBEL
       ===================================================== */

    sobelKernel<<<blocks, threads>>>(
        d_blur,
        d_edge,
        width,
        ghost_rows,
        rank,
        size
    );

    cudaDeviceSynchronize();

    /*
        Check CUDA errors
    */

    err =
    cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf(
            "Sobel CUDA Error: %s\n",
            cudaGetErrorString(err)
        );
    }

    double end =
    MPI_Wtime();

    printf(
        "Process %d GPU Time: %f seconds\n",
        rank,
        end - start
    );

    /* =====================================================
                    COPY EDGE BACK
       ===================================================== */

    cudaMemcpy(
        local_edge,
        d_edge,
        local_size,
        cudaMemcpyDeviceToHost
    );

    /* =====================================================
                ROOT ALLOCATES OUTPUT
       ===================================================== */

    if(rank == 0)
    {
        final_edge =
        (unsigned char*)
        calloc(width * height,
               sizeof(unsigned char));
    }

    /* =====================================================
                    GATHER FINAL IMAGE
       ===================================================== */

    MPI_Gather(
        local_edge,
        local_size,
        MPI_UNSIGNED_CHAR,

        final_edge,

        local_size,
        MPI_UNSIGNED_CHAR,

        0,
        MPI_COMM_WORLD
    );

    /* =====================================================
                    SAVE OUTPUT IMAGE
       ===================================================== */

    if(rank == 0)
    {
        FILE *out =
        fopen("hybrid_output.pgm",
              "wb");

        fprintf(out,
                "P5\n%d %d\n255\n",
                width,
                height);

        fwrite(
            final_edge,
            1,
            width * height,
            out
        );

        fclose(out);

        printf(
            "Hybrid MPI + CUDA Finished\n"
        );

        free(image);
        free(final_edge);
    }

    /* =====================================================
                        CLEANUP
       ===================================================== */

    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_edge);

    free(local_input);
    free(local_blur);
    free(local_edge);

    MPI_Finalize();

    return 0;
}