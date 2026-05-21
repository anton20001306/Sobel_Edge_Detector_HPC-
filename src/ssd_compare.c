#include <stdio.h>
#include <stdlib.h>

long long calculate_ssd(unsigned char *img1,
                        unsigned char *img2,
                        int size)
{
    long long ssd = 0;

    for(int i = 0; i < size; i++)
    {
        int diff = img1[i] - img2[i];

        ssd += diff * diff;
    }

    return ssd;
}

void read_pgm(const char *filename,
              unsigned char **image,
              int *width,
              int *height)
{
    FILE *fp = fopen(filename, "rb");

    if(fp == NULL)
    {
        printf("Cannot open %s\n", filename);
        exit(1);
    }

    char format[3];
    int maxval;

    fscanf(fp, "%s", format);
    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);

    *image = malloc((*width) * (*height));

    fread(*image, 1, (*width) * (*height), fp);

    fclose(fp);
}

int main()
{
    unsigned char *serial_img;
    unsigned char *openmp_img;
    unsigned char *mpi_img;
    unsigned char *mpighost_img;

    int width, height;

    /* Read Images */

    read_pgm("serial_output.pgm",
             &serial_img,
             &width,
             &height);

    read_pgm("omp_output.pgm",
             &openmp_img,
             &width,
             &height);

    read_pgm("mpi_output.pgm",
             &mpi_img,
             &width,
             &height);

    read_pgm("mpighost_output.pgm",
             &mpighost_img,
             &width,
             &height);

    int size = width * height;

    /* Calculate SSD */

    long long ssd_openmp =
    calculate_ssd(serial_img,
                  openmp_img,
                  size);

    long long ssd_mpi =
    calculate_ssd(serial_img,
                  mpi_img,
                  size);

    long long ssd_mpighost =
    calculate_ssd(serial_img,
                  mpighost_img,
                  size);

    printf("\n========== SSD RESULTS ==========\n");

    printf("SSD (Serial vs OpenMP): %lld\n",
            ssd_openmp);

    printf("SSD (Serial vs MPI): %lld\n",
            ssd_mpi);

    printf("SSD (Serial vs MPI Ghost): %lld\n",
            ssd_mpighost);

    printf("\n========== ANALYSIS ==========\n");

    if(ssd_openmp == 0)
        printf("OpenMP output is IDENTICAL to Serial\n");
    else
        printf("OpenMP output has differences\n");

    if(ssd_mpi == 0)
        printf("MPI output is IDENTICAL to Serial\n");
    else
        printf("MPI output has differences\n");

    if(ssd_mpighost == 0)
        printf("MPI Ghost output is IDENTICAL to Serial\n");
    else
        printf("MPI Ghost output has differences\n");

    free(serial_img);
    free(openmp_img);
    free(mpi_img);
    free(mpighost_img);

    return 0;
}