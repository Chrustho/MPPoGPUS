#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define BLUR_SIZE 6
#define PASS 100

__global__ void blurKernel(unsigned char *out, unsigned char *in, int w, int h)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h)
    {
        int pixR = 0;
        int pixG = 0;
        int pixB = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++)
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++)
            {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {
                    pixR += in[3 * (curRow * w + curCol)];
                    pixG += in[3 * (curRow * w + curCol) + 1];
                    pixB += in[3 * (curRow * w + curCol) + 2];
                    pixels++;
                }
            }
        out[3 * (Row * w + Col)] = (unsigned char)(pixR / pixels);
        out[3 * (Row * w + Col) + 1] = (unsigned char)(pixG / pixels);
        out[3 * (Row * w + Col) + 2] = (unsigned char)(pixB / pixels);
    }
}

static void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    const char *inFilename = (argc > 1) ? argv[1] : "maialino_triste.png";
    const char *fname = inFilename;
    const char *s1 = strrchr(inFilename, '/');
    const char *s2 = strrchr(inFilename, '\\');
    if (s1 || s2)
        fname = (s1 && s2) ? ((s1 > s2) ? s1 + 1 : s2 + 1)
                           : (s1 ? s1 + 1 : s2 + 1);
    const char *last_dot = strrchr(fname, '.');
    size_t namelen = (last_dot && last_dot != fname) ? (size_t)(last_dot - fname) : strlen(fname);

    char base[256];
    if (namelen >= sizeof(base))
        namelen = sizeof(base) - 1;
    snprintf(base, sizeof(base), "%.*s", (int)namelen, fname);

    char nome[512];
    snprintf(nome, sizeof(nome),
             "blur_%s_mail_NP:%d_DM:%d.png",
             base, PASS, BLUR_SIZE);

    const char *outFilename = (argc > 2) ? argv[2] : nome;

    int w, h, channels;
    unsigned char *h_in = stbi_load(inFilename, &w, &h, &channels, 3);
    if (!h_in)
    {
        fprintf(stderr, "Errore: impossibile aprire immagine '%s'\n", inFilename);
        return EXIT_FAILURE;
    }
    channels = 3;
    size_t imgSize = (size_t)w * h * channels;

    printf("Caricata %s: %d x %d, %d canali\n", inFilename, w, h, channels);
    printf("Output predefinito: %s\n", nome);

    unsigned char *h_out = (unsigned char *)malloc(imgSize);
    if (!h_out)
    {
        fprintf(stderr, "Errore malloc output\n");
        stbi_image_free(h_in);
        return EXIT_FAILURE;
    }

    unsigned char *d_in = NULL, *d_out = NULL;
    checkCuda(cudaMalloc((void **)&d_in, imgSize), "cudaMalloc d_in");
    checkCuda(cudaMalloc((void **)&d_out, imgSize), "cudaMalloc d_out");

    checkCuda(cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D d_in");

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start),"");
    checkCuda(cudaEventCreate(&stop),"");
    checkCuda(cudaEventRecord(start, 0),"");


    for (int i = 0; i < PASS; ++i)
    {
        blurKernel<<<grid, block>>>(d_out, d_in, w, h);
        checkCuda(cudaGetLastError(), "lancio kernel");


        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned char *tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }
    checkCuda(cudaEventRecord(stop, 0),"");
    checkCuda(cudaEventSynchronize(stop),"");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop),"");
    printf("Elapsed time %f\n",ms);

    checkCuda(cudaMemcpy(h_out, d_in, imgSize, cudaMemcpyDeviceToHost), "cudaMemcpy h_out d_in");

    if (!stbi_write_png(outFilename, w, h, channels, h_out, w * channels))
    {
        fprintf(stderr, "Errore: impossibile salvare immagine '%s'\n", outFilename);
    }
    else
    {
        printf("Output salvato in '%s'\n", outFilename);
    }

    checkCuda(cudaFree(d_in), "cudaFree d_in");
    checkCuda(cudaFree(d_out), "cudaFree d_out");
    stbi_image_free(h_in);
    free(h_out);

    return 0;
}