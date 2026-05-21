#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int sobel(int gx, int gy)
{
    int val = (int)sqrt(gx * gx + gy * gy);

    if(val > 255) val = 255;
    if(val < 0) val = 0;

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

    /* Process 0 reads image */
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

        image = malloc(width * height);
        fread(image, 1, width * height, fp);
        fclose(fp);

        printf("Image Loaded: %dx%d\n", width, height);
    }

    /* Broadcast image dimensions */
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Rows per process */
    int rows = height / size;

    /* * Allocate local memory with 2 extra "ghost" rows (halos) 
     * Using calloc to ensure top/bottom image borders default to black (0)
     */
    unsigned char *local_image = calloc((rows + 2) * width, sizeof(unsigned char));
    unsigned char *local_edge  = calloc((rows + 2) * width, sizeof(unsigned char));

    /* * Scatter image chunks. 
     * We offset the receive buffer by 'width' so the scattered data 
     * lands in the middle, leaving the first row open for the top halo.
     */
    MPI_Scatter(image,
                rows * width,
                MPI_UNSIGNED_CHAR,
                local_image + width, 
                rows * width,
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);

    /* --- HALO EXCHANGE --- */
    
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

    /* Sobel Kernels */
    int Gx[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,1} };
    int Gy[3][3] = { {-1,-2,-1}, {0,0,0}, {1,2,1} };

    double start = MPI_Wtime();

    /* * Local Sobel Computation
     * We iterate from 1 to 'rows' (the actual data we own).
     * Row 0 is the top halo, Row 'rows + 1' is the bottom halo.
     */
    for(int i = 1; i <= rows; i++)
    {
        // Skip the absolute top and bottom image borders 
        if (rank == 0 && i == 1) continue;
        if (rank == size - 1 && i == rows) continue;

        for(int j = 1; j < width - 1; j++)
        {
            int gx = 0;
            int gy = 0;

            // Neighborhood convolution
            for(int x = -1; x <= 1; x++)
            {
                for(int y = -1; y <= 1; y++)
                {
                    int pixel = local_image[(i + x) * width + (j + y)];
                    gx += pixel * Gx[x + 1][y + 1];
                    gy += pixel * Gy[x + 1][y + 1];
                }
            }

            local_edge[i * width + j] = sobel(gx, gy);
        }
    }

    double end = MPI_Wtime();

    printf("Process %d finished in %f seconds\n", rank, end - start);

    /* Root gathers results */
    if(rank == 0)
    {
        edge = malloc(width * height);
    }

    /* * Gather the computed data.
     * We send from 'local_edge + width' to skip the empty ghost row at the top.
     */
    MPI_Gather(local_edge + width,
               rows * width,
               MPI_UNSIGNED_CHAR,
               edge,
               rows * width,
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);

    /* Save final image */
    if(rank == 0)
    {
        FILE *out = fopen("mpi_output.pgm", "wb");
        fprintf(out, "P5\n%d %d\n255\n", width, height);
        fwrite(edge, 1, width * height, out);
        fclose(out);

        printf("MPI Edge Detection Finished\n");
        free(image);
        free(edge);
    }

    free(local_image);
    free(local_edge);

    MPI_Finalize();

    return 0;
}