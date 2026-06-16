#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ int sobel(int gx, int gy)
{
    int val = (int)sqrtf((float)(gx * gx + gy * gy));

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
    int height)
{
    int j =
    blockIdx.x * blockDim.x + threadIdx.x;

    int i =
    blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && i < height - 1 &&
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

int main()
{
    FILE *fp =
    fopen("input.pgm", "rb");

    if(fp == NULL)
    {
        printf("Cannot open image\n");
        return 1;
    }

    char format[3];

    int width, height, maxval;

    fscanf(fp, "%s", format);
    fscanf(fp, "%d %d", &width, &height);
    fscanf(fp, "%d", &maxval);

    fgetc(fp);

    int size = width * height;

    unsigned char *image =
    (unsigned char*)calloc(size,
                           sizeof(unsigned char));

    unsigned char *edge =
    (unsigned char*)calloc(size,
                           sizeof(unsigned char));

    fread(image,
          1,
          size,
          fp);

    fclose(fp);

    /* GPU memory */

    unsigned char *d_input;
    unsigned char *d_output;

    cudaMalloc((void**)&d_input,
               size);

    cudaMalloc((void**)&d_output,
               size);

    /* Copy image to GPU */

    cudaMemcpy(d_input,
               image,
               size,
               cudaMemcpyHostToDevice);

    /* CUDA configuration */

    dim3 threads(16,16);

    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );

    /* Launch kernel */

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobelKernel<<<blocks, threads>>>(
        d_input,
        d_output,
        width,
        height
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        printf("CUDA Error: %s\n",
            cudaGetErrorString(err));
    }
    else
    {
        printf("CUDA Kernel Executed Successfully\n");
    }



    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    printf("CUDA Execution Time: %f ms\n",
            milliseconds);

    /* Copy result back */

    cudaMemcpy(edge,
               d_output,
               size,
               cudaMemcpyDeviceToHost);

    /* Save image */

    FILE *out =
    fopen("cuda_output.pgm", "wb");

    fprintf(out,
            "P5\n%d %d\n255\n",
            width,
            height);

    fwrite(edge,
           1,
           size,
           out);

    fclose(out);

    /* Cleanup */

    cudaFree(d_input);
    cudaFree(d_output);

    free(image);
    free(edge);

    printf("CUDA Sobel Finished\n");

    return 0;
}