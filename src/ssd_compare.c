#include <stdio.h>
#include <stdlib.h>

long long calculateSSD(
    const char *file1,
    const char *file2)
{
    FILE *fp1 = fopen(file1, "rb");
    FILE *fp2 = fopen(file2, "rb");

    if(fp1 == NULL || fp2 == NULL)
    {
        printf("Error opening files\n");
        return -1;
    }

    char format1[3];
    char format2[3];

    int width1, height1, max1;
    int width2, height2, max2;

    fscanf(fp1, "%s", format1);
    fscanf(fp1, "%d %d", &width1, &height1);
    fscanf(fp1, "%d", &max1);
    fgetc(fp1);

    fscanf(fp2, "%s", format2);
    fscanf(fp2, "%d %d", &width2, &height2);
    fscanf(fp2, "%d", &max2);
    fgetc(fp2);

    if(width1 != width2 ||
       height1 != height2)
    {
        printf("Image sizes differ\n");

        fclose(fp1);
        fclose(fp2);

        return -1;
    }

    int size = width1 * height1;

    unsigned char *img1 =
    malloc(size);

    unsigned char *img2 =
    malloc(size);

    fread(img1, 1, size, fp1);
    fread(img2, 1, size, fp2);

    fclose(fp1);
    fclose(fp2);

    long long ssd = 0;

    for(int i = 0; i < size; i++)
    {
        int diff =
        img1[i] - img2[i];

        ssd += diff * diff;
    }

    free(img1);
    free(img2);

    return ssd;
}

int main()
{
    printf("\n========== SSD RESULTS ==========\n");

    long long omp_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/omp_output.pgm"
    );

    long long mpi_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/mpi_output.pgm"
    );

    long long mpighost_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/mpighost_output.pgm"
    );

    long long cuda_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/cuda_output.pgm"
    );

    long long hybrid_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/hybrid_output.pgm"
    );

    printf("SSD (Serial vs OpenMP): %lld\n",
            omp_ssd);

    printf("SSD (Serial vs MPI): %lld\n",
            mpi_ssd);

    printf("SSD (Serial vs MPI Ghost): %lld\n",
            mpighost_ssd);

    printf("SSD (Serial vs CUDA): %lld\n",
            cuda_ssd);

    printf("SSD (Serial vs Hybrid): %lld\n",
            hybrid_ssd);

    printf("\n========== ANALYSIS ==========\n");

    if(omp_ssd == 0)
        printf("OpenMP output is IDENTICAL\n");

    if(mpighost_ssd == 0)
        printf("MPI Ghost output is IDENTICAL\n");

    if(cuda_ssd == 0)
        printf("CUDA output is IDENTICAL\n");

    if(hybrid_ssd == 0)
        printf("Hybrid output is IDENTICAL\n");

    return 0;
}