#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void sepiaKernel(const unsigned char *Pin, unsigned char *Pout, int width, int height, int channels)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col >= width || Row >= height)
        return;

    int pixelIndex = (Row * width + Col) * channels;

    unsigned char r = Pin[pixelIndex + 0];
    unsigned char g = Pin[pixelIndex + 1];
    unsigned char b = Pin[pixelIndex + 2];

    float newR = 0.393f * r + 0.769f * g + 0.189f * b;
    float newG = 0.349f * r + 0.686f * g + 0.168f * b;
    float newB = 0.272f * r + 0.534f * g + 0.131f * b;

    unsigned char outR = (unsigned char)(fminf(newR, 255.0f));
    unsigned char outG = (unsigned char)(fminf(newG, 255.0f));
    unsigned char outB = (unsigned char)(fminf(newB, 255.0f));

    Pout[pixelIndex + 0] = outR;
    Pout[pixelIndex + 1] = outG;
    Pout[pixelIndex + 2] = outB;

    if (channels == 4)
    {
        Pout[pixelIndex + 3] = Pin[pixelIndex + 3];
    }
}

int main()
{
    int width, height, channels;
    unsigned char *data = stbi_load("porky.png", &width, &height, &channels, 0);
    if (!data)
    {
        fprintf(stderr, "Impossibile caricare immagine\n");
        return 1;
    }
    printf("Image: %d x %d, channels=%d\n", width, height, channels);

    if (channels < 3)
    {
        fprintf(stderr, "Numero di canali non supportato: %d\n", channels);
        stbi_image_free(data);
        return 1;
    }

    size_t n_pixels = (size_t)width * height;
    size_t in_bytes = n_pixels * channels;
    size_t out_bytes = in_bytes;

    unsigned char *d_In = nullptr, *d_Out = nullptr;
    unsigned char *out = (unsigned char *)malloc(out_bytes);
    if (!out)
    {
        fprintf(stderr, "malloc failed\n");
        stbi_image_free(data);
        return 1;
    }

    cudaMalloc((void **)&d_In, in_bytes);
    cudaMalloc((void **)&d_Out, out_bytes);

    cudaMemcpy(d_In, data, in_bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sepiaKernel<<<grid, block>>>(d_In, d_Out, width, height, channels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Errore nel kernel CUDA: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_Out, out_bytes, cudaMemcpyDeviceToHost);

    if (!stbi_write_png("porky_seppia.png", width, height, channels, out, width * channels))
    {
        fprintf(stderr, "Scrittura immagine fallita\n");
    }
    else
    {
        printf("Creata imm_seppia.png\n");
    }

    stbi_image_free(data);
    free(out);
    cudaFree(d_In);
    cudaFree(d_Out);
    return 0;
}
