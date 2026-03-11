#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int sobel(int gx, int gy)
{
    int val = (int)sqrt(gx*gx + gy*gy);
    if(val > 255) val = 255;
    if(val < 0) val = 0;
    return val;
}

int main()
{
    FILE *fp = fopen("input.pgm","rb");

    if(fp == NULL)
    {
        printf("Cannot open image\n");
        return 1;
    }

    char format[3];
    int width, height, maxval;

    fscanf(fp,"%s",format);
    fscanf(fp,"%d %d",&width,&height);
    fscanf(fp,"%d",&maxval);
    fgetc(fp);

    unsigned char *image = malloc(width*height);
    unsigned char *blur = malloc(width*height);
    unsigned char *edge = malloc(width*height);

    fread(image,1,width*height,fp);
    fclose(fp);

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

            blur[i*width+j] = sum/16;
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

    FILE *out = fopen("output.pgm","wb");

    fprintf(out,"P5\n%d %d\n255\n",width,height);

    fwrite(edge,1,width*height,out);

    fclose(out);

    free(image);
    free(blur);
    free(edge);

    printf("Edge detection finished\n");

    return 0;
}