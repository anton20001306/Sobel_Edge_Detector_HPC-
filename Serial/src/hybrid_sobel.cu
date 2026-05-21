#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// CUDA KERNEL: Executed on the GPU
// ---------------------------------------------------------
__global__ void sobel_kernel(unsigned char *d_in, unsigned char *d_out, int width, int rows, int rank, int size) 
{
    // Calculate global thread positions (x = column, y = row)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are inside the image boundaries (excluding the 1-pixel outer edge)
    if (x > 0 && x < width - 1 && y > 0 && y < rows - 1) 
    {
        // Skip the absolute top and bottom image borders for Rank 0 and the last Rank
        if ((rank == 0 && y == 1) || (rank == size - 1 && y == rows - 2)) return;

        int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

        int gx = 0;
        int gy = 0;

        // Apply 3x3 convolution
        for (int i = -1; i <= 1; i++) 
        {
            for (int j = -1; j <= 1; j++) 
            {
                int pixel = d_in[(y + i) * width + (x + j)];
                gx += pixel * Gx[i + 1][j + 1];
                gy += pixel * Gy[i + 1][j + 1];
            }
        }

        // Calculate magnitude and clamp to 0-255
        int val = (int)sqrtf((float)(gx * gx + gy * gy));
        if (val > 255) val = 255;
        if (val < 0) val = 0;

        d_out[y * width + x] = val;
    }
}

// ---------------------------------------------------------
// HOST CODE: Executed on the CPU (MPI)
// ---------------------------------------------------------
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, maxval;
    unsigned char *image = NULL;
    unsigned char *edge = NULL;

    /* 1. Process 0 reads image */
    if(rank == 0)
    {
        FILE *fp = fopen("input.pgm", "rb");
        if(fp == NULL) {
            printf("Cannot open image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char format[3];
        fscanf(fp, "%s", format);
        fscanf(fp, "%d %d", &width, &height);
        fscanf(fp, "%d", &maxval);
        fgetc(fp);

        image = (unsigned char*)malloc(width * height);
        fread(image, 1, width * height, fp);
        fclose(fp);

        printf("Image Loaded: %dx%d\n", width, height);
    }

    /* 2. Broadcast dimensions */
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows = height / size;
    int total_chunk_size = (rows + 2) * width; // Includes ghost rows

    /* 3. Allocate CPU memory (with halos) */
    unsigned char *local_image = (unsigned char*)calloc(total_chunk_size, sizeof(unsigned char));
    unsigned char *local_edge  = (unsigned char*)calloc(total_chunk_size, sizeof(unsigned char));

    /* 4. Scatter image chunks */
    MPI_Scatter(image, rows * width, MPI_UNSIGNED_CHAR,
                local_image + width, rows * width, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    /* 5. CPU Halo Exchange */
    // Identify neighbors. MPI_PROC_NULL elegantly handles edge cases.
    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Send my top row to top neighbor, receive top halo from top neighbor
    // Tag is set to 0
    MPI_Sendrecv(local_image + width, width, MPI_UNSIGNED_CHAR, top_neighbor, 0,
                 local_image, width, MPI_UNSIGNED_CHAR, top_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send my bottom row to bottom neighbor, receive bottom halo from bottom neighbor
    // Tag is ALSO set to 0
    MPI_Sendrecv(local_image + rows * width, width, MPI_UNSIGNED_CHAR, bottom_neighbor, 0,
                 local_image + (rows + 1) * width, width, MPI_UNSIGNED_CHAR, bottom_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* 6. CUDA GPU Execution */
    double start = MPI_Wtime();

    unsigned char *d_in, *d_out;
    
    // Allocate GPU memory
    cudaMalloc((void**)&d_in, total_chunk_size);
    cudaMalloc((void**)&d_out, total_chunk_size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_in, local_image, total_chunk_size, cudaMemcpyHostToDevice);

    // Define CUDA Grid/Block configuration
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, ((rows + 2) + 15) / 16);

    // Launch Kernel
    sobel_kernel<<<blocks, threads>>>(d_in, d_out, width, rows + 2, rank, size);
    cudaDeviceSynchronize();

    // Copy results back from GPU to CPU
    cudaMemcpy(local_edge, d_out, total_chunk_size, cudaMemcpyDeviceToHost);

    double end = MPI_Wtime();
    printf("Process %d finished GPU computation in %f seconds\n", rank, end - start);

    /* 7. Clean up GPU memory */
    cudaFree(d_in);
    cudaFree(d_out);

    /* 8. Gather results back to Root */
    if(rank == 0) {
        edge = (unsigned char*)malloc(width * height);
    }

    MPI_Gather(local_edge + width, rows * width, MPI_UNSIGNED_CHAR,
               edge, rows * width, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    /* 9. Save final image */
    if(rank == 0)
    {
        FILE *out = fopen("hybrid_output.pgm", "wb");
        fprintf(out, "P5\n%d %d\n255\n", width, height);
        fwrite(edge, 1, width * height, out);
        fclose(out);

        printf("MPI+CUDA Hybrid Edge Detection Finished\n");
        free(image);
        free(edge);
    }

    free(local_image);
    free(local_edge);

    MPI_Finalize();
    return 0;
}