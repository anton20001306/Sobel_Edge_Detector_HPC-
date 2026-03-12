#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



int sobel(int gx, int gy)
{
    int val = (int)sqrt(gx*gx + gy*gy);

    if(val > 255) val = 255;
    if(val < 0) val = 0;

    return val;
}


int main()
{
    int width, height, channels;

    /* Load image (force grayscale) */
    unsigned char *image = stbi_load("input.jpg",
                                     &width,
                                     &height,
                                     &channels,
                                     1);

    if(image == NULL)
    {
        printf("Cannot load image\n");
        return 1;
    }

    printf("Image loaded: %d x %d\n", width, height);

    unsigned char *blur = malloc(width * height);
    unsigned char *edge = malloc(width * height);

    struct timespec start, end;

    /* Start timing */
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Gaussian Kernel */
    int G[3][3] =
    {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };

    for(int i=1;i<height-1;i++)
    {
        for(int j=1;j<width-1;j++)
        {
            int sum = 0;

            for(int x=-1;x<=1;x++)
            {
                for(int y=-1;y<=1;y++)
                {
                    sum += image[(i+x)*width+(j+y)] * G[x+1][y+1];
                }
            }

            blur[i*width+j] = sum / 16;
        }
    }

    /* Sobel Kernels */

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

    for(int i=1;i<height-1;i++)
    {
        for(int j=1;j<width-1;j++)
        {
            int gx = 0;
            int gy = 0;

            for(int x=-1;x<=1;x++)
            {
                for(int y=-1;y<=1;y++)
                {
                    int pixel = blur[(i+x)*width+(j+y)];

                    gx += pixel * Gx[x+1][y+1];
                    gy += pixel * Gy[x+1][y+1];
                }
            }

            edge[i*width+j] = sobel(gx,gy);
        }
    }

    /* Stop timing */
    clock_gettime(CLOCK_MONOTONIC, &end);

    double execution_time =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Execution Time: %.6f seconds\n", execution_time);

    /* Performance statistics */
    double pixels = width * height;
    double throughput = pixels / execution_time;

    printf("Pixels processed: %.0f\n", pixels);
    printf("Processing rate: %.2f pixels/sec\n", throughput);

    /* Save output image */
    stbi_write_png("output.png",
                   width,
                   height,
                   1,
                   edge,
                   width);

    printf("Edge detection finished\n");

    stbi_image_free(image);

    free(blur);
    free(edge);

    return 0;
}