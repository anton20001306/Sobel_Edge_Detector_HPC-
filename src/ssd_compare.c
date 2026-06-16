#include <stdio.h>
#include <stdlib.h>
#include <math.h>

long long calculateSSD(
    const char *file1,
    const char *file2,
    double *rmse)
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

    *rmse = sqrt((double)ssd / size);

    free(img1);
    free(img2);

    return ssd;
}

int main()
{
    printf("\n========== SSD / RMSE RESULTS ==========\n");

    double omp_rmse;
    //double mpi_rmse;
    double mpighost_rmse;

    long long omp_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/openmp_output.pgm",
        &omp_rmse
    );

    // long long mpi_ssd =
    // calculateSSD(
    //     "images/serial_output.pgm",
    //     "images/mpi_output.pgm",
    //     &mpi_rmse
    // );

    long long mpighost_ssd =
    calculateSSD(
        "images/serial_output.pgm",
        "images/mpighost_output.pgm",
        &mpighost_rmse
    );

    printf("\nOpenMP Results\n");
    printf("--------------------------\n");
    printf("SSD  : %lld\n", omp_ssd);
    printf("RMSE : %f\n", omp_rmse);

    // printf("\nMPI Results\n");
    // printf("--------------------------\n");
    // printf("SSD  : %lld\n", mpi_ssd);
    // printf("RMSE : %f\n", mpi_rmse);

    printf("\nMPI Ghost Results\n");
    printf("--------------------------\n");
    printf("SSD  : %lld\n", mpighost_ssd);
    printf("RMSE : %f\n", mpighost_rmse);

    printf("\n========== ANALYSIS ==========\n");

    if(omp_ssd == 0)
        printf("OpenMP output is IDENTICAL\n");

    if(mpighost_ssd == 0)
        printf("MPI Ghost output is IDENTICAL\n");

    return 0;
}