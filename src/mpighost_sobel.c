#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int sobel(int gx, int gy)
{
    int val = (int)sqrt(gx * gx + gy * gy);

    if(val > 255)
        val = 255;

    if(val < 0)
        val = 0;

    return val;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, maxval;

    unsigned char *image = NULL;
    unsigned char *edge = NULL;

    /* Root process reads image */

    if(rank == 0)
    {
        FILE *fp = fopen("input.pgm", "rb");

        if(fp == NULL)
        {
            printf("Cannot open image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char format[3];

        fscanf(fp, "%s", format);
        fscanf(fp, "%d %d", &width, &height);
        fscanf(fp, "%d", &maxval);

        fgetc(fp);

        image = calloc(width * height,
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

    /* Broadcast image dimensions */

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

    /* Buffers */

    unsigned char *local_image =
    calloc((rows + 2) * width,
           sizeof(unsigned char));

    unsigned char *local_blur =
    calloc((rows + 2) * width,
           sizeof(unsigned char));

    unsigned char *local_edge =
    calloc(rows * width,
           sizeof(unsigned char));

    /* Scatter image chunks */

    MPI_Scatter(image,
                rows * width,
                MPI_UNSIGNED_CHAR,
                &local_image[width],
                rows * width,
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);

    /* Halo Exchange for Original Image */

    if(rank > 0)
    {
        MPI_Sendrecv(
            &local_image[1 * width],
            width,
            MPI_UNSIGNED_CHAR,
            rank - 1,
            0,

            &local_image[0],
            width,
            MPI_UNSIGNED_CHAR,
            rank - 1,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if(rank < size - 1)
    {
        MPI_Sendrecv(
            &local_image[rows * width],
            width,
            MPI_UNSIGNED_CHAR,
            rank + 1,
            0,

            &local_image[(rows + 1) * width],
            width,
            MPI_UNSIGNED_CHAR,
            rank + 1,
            0,

            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    /* Gaussian Kernel */

    int G[3][3] =
    {
        {1,2,1},
        {2,4,2},
        {1,2,1}
    };

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

    double start = MPI_Wtime();

    /* Gaussian Blur */

    for(int i = 1; i <= rows; i++)
    {
        int global_i =
        rank * rows + (i - 1);

        /* Match serial version */

        if(global_i == 0 ||
           global_i == height - 1)
        {
            continue;
        }

        for(int j = 1; j < width - 1; j++)
        {
            int sum = 0;

            for(int x = -1; x <= 1; x++)
            {
                for(int y = -1; y <= 1; y++)
                {
                    int pixel =
                    local_image[(i + x) * width + (j + y)];

                    sum += pixel *
                           G[x + 1][y + 1];
                }
            }

            local_blur[i * width + j] =
            sum / 16;
        }
    }

    /* Halo Exchange for Blurred Image */

    if(rank > 0)
    {
        MPI_Sendrecv(
            &local_blur[1 * width],
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

    /* Sobel Computation */

    for(int i = 1; i <= rows; i++)
    {
        int global_i =
        rank * rows + (i - 1);

        /* Match serial version */

        if(global_i == 0 ||
           global_i == height - 1)
        {
            continue;
        }

        for(int j = 1; j < width - 1; j++)
        {
            int gx = 0;
            int gy = 0;

            for(int x = -1; x <= 1; x++)
            {
                for(int y = -1; y <= 1; y++)
                {
                    int pixel =
                    local_blur[(i + x) * width + (j + y)];

                    gx += pixel *
                          Gx[x + 1][y + 1];

                    gy += pixel *
                          Gy[x + 1][y + 1];
                }
            }

            int out_row = i - 1;

            local_edge[out_row * width + j] =
            sobel(gx, gy);
        }
    }

    double end = MPI_Wtime();

    printf("Process %d finished in %f seconds\n",
            rank,
            end - start);

    /* Root allocates final image */

    if(rank == 0)
    {
        edge = calloc(width * height,
                      sizeof(unsigned char));
    }

    /* Gather final output */

    MPI_Gather(local_edge,
               rows * width,
               MPI_UNSIGNED_CHAR,
               edge,
               rows * width,
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);

    /* Save output */

    if(rank == 0)
    {
        FILE *out =
        fopen("mpighost_output.pgm", "wb");

        fprintf(out,
                "P5\n%d %d\n255\n",
                width,
                height);

        fwrite(edge,
               1,
               width * height,
               out);

        fclose(out);

        printf("MPI Ghost Sobel Finished\n");

        free(image);
        free(edge);
    }

    free(local_image);
    free(local_blur);
    free(local_edge);

    MPI_Finalize();

    return 0;
}