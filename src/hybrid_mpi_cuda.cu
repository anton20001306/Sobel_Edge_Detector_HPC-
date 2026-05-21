#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

__device__ int sobel(int gx, int gy)
{
    int val =
    (int)sqrtf((float)(gx * gx + gy * gy));

    if(val > 255)
        val = 255;

    if(val < 0)
        val = 0;

    return val;
}

__global__ void sobelKernel(
    unsigned char *input,
    unsigned char *output,
    int width,
    int rows)
{
    int j =
    blockIdx.x * blockDim.x + threadIdx.x;

    int i =
    blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && i < rows - 1 &&
       j > 0 && j < width - 1)
    {
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

        for(int x = -1; x <= 1; x++)
        {
            for(int y = -1; y <= 1; y++)
            {
                int pixel =
                input[(i + x) * width + (j + y)];

                gx += pixel *
                      Gx[x + 1][y + 1];

                gy += pixel *
                      Gy[x + 1][y + 1];
            }
        }

        output[i * width + j] =
        sobel(gx, gy);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD,
                  &rank);

    MPI_Comm_size(MPI_COMM_WORLD,
                  &size);

    int width, height, maxval;

    unsigned char *image = NULL;
    unsigned char *edge = NULL;

    /* Root process loads image */

    if(rank == 0)
    {
        FILE *fp =
        fopen("input.pgm", "rb");

        if(fp == NULL)
        {
            printf("Cannot open image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char format[3];

        fscanf(fp, "%s", format);
        fscanf(fp, "%d %d",
               &width,
               &height);

        fscanf(fp, "%d",
               &maxval);

        fgetc(fp);

        image =
        (unsigned char*)
        calloc(width * height,
               sizeof(unsigned char));

        fread(image,
              1,
              width * height,
              fp);

        fclose(fp);

        printf("Image Loaded: %dx%d\n",
                width,
                height);
    }

    /* Broadcast image size */

    MPI_Bcast(&width,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    MPI_Bcast(&height,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    int rows = height / size;

    int local_size =
    rows * width;

    /* Local buffers */

    unsigned char *local_image =
    (unsigned char*)
    calloc(local_size,
           sizeof(unsigned char));

    unsigned char *local_edge =
    (unsigned char*)
    calloc(local_size,
           sizeof(unsigned char));

    /* Scatter image */

    MPI_Scatter(image,
                local_size,
                MPI_UNSIGNED_CHAR,
                local_image,
                local_size,
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);

    /* GPU memory */

    unsigned char *d_input;
    unsigned char *d_output;

    cudaMalloc((void**)&d_input,
               local_size);

    cudaMalloc((void**)&d_output,
               local_size);

    /* Copy local chunk to GPU */

    cudaMemcpy(d_input,
               local_image,
               local_size,
               cudaMemcpyHostToDevice);

    /* CUDA configuration */

    dim3 threads(16,16);

    dim3 blocks(
        (width + threads.x - 1)
        / threads.x,

        (rows + threads.y - 1)
        / threads.y
    );

    double start = MPI_Wtime();

    /* Launch CUDA kernel */

    sobelKernel<<<blocks, threads>>>(
        d_input,
        d_output,
        width,
        rows
    );

    cudaDeviceSynchronize();

    double end = MPI_Wtime();

    printf("Process %d GPU Time: %f seconds\n",
            rank,
            end - start);

    /* Copy result back */

    cudaMemcpy(local_edge,
               d_output,
               local_size,
               cudaMemcpyDeviceToHost);

    /* Root allocates output */

    if(rank == 0)
    {
        edge =
        (unsigned char*)
        calloc(width * height,
               sizeof(unsigned char));
    }

    /* Gather results */

    MPI_Gather(local_edge,
               local_size,
               MPI_UNSIGNED_CHAR,
               edge,
               local_size,
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);

    /* Save output */

    if(rank == 0)
    {
        FILE *out =
        fopen("hybrid_output.pgm",
              "wb");

        fprintf(out,
                "P5\n%d %d\n255\n",
                width,
                height);

        fwrite(edge,
               1,
               width * height,
               out);

        fclose(out);

        printf("Hybrid MPI + CUDA Finished\n");

        free(image);
        free(edge);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    free(local_image);
    free(local_edge);

    MPI_Finalize();

    return 0;
}