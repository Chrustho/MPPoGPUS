#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstdio>
#include <cstdlib>
#include "stb_image.h"
#include "stb_image_write.h"

#define CHANNELS 3
#define COEFF_R 0.5
#define COEFF_G 0.5
#define COEFF_B 0.5

__global__ void colorToGrey(unsigned char *Pin, unsigned char *Pout, int width, int height)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height)
    {
        int greyoffset = Row * width + Col;

        int rgbOffset = greyoffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyoffset] = COEFF_R * r + COEFF_G * g + COEFF_B * b;
    }
}

int main()
{
    int width, height, channels;
    unsigned char *data = stbi_load("maialino.png", &width, &height, &channels, 0);
    if (!data)
    {
        fprintf(stderr, "Fallimento nel caricamento\n");
        return 1;
    }
    printf("Image: %d x %d \n channels=%d\n", width, height, channels);

    size_t in_bytes = (size_t)width * height * channels; // con rgb è fondamentale conoscere il numero di canali
    size_t out_bytes = (size_t)width * height;           // poichè è in scala di grigi ha un solo canale

    unsigned char *d_In = nullptr, *d_Out = nullptr;
    unsigned char *out = (unsigned char *)malloc(out_bytes);

    cudaMalloc((void **)&d_In, in_bytes);
    cudaMalloc((void **)&d_Out, out_bytes);

    cudaMemcpy(d_In, data, in_bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    colorToGrey<<<grid, block>>>(d_In, d_Out, width, height);

    cudaMemcpy(out, d_Out, out_bytes, cudaMemcpyDeviceToHost);

    char filename[256];
    snprintf(filename, sizeof(filename),
             "imm_grigia_%.2f_%.2f_%.2f.png",
             (double)COEFF_R, (double)COEFF_G, (double)COEFF_B);

    if (!stbi_write_png(filename, width, height, 1, out, width))
    {
        fprintf(stderr, "stbi_write_png fallito\n");
    }
    else
    {
        printf("Creato %s\n", filename);
    }

    stbi_image_free(data);
    free(out);
    cudaFree(d_In);
    cudaFree(d_Out);
    return 0;
}
