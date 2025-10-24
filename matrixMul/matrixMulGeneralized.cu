#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 32

__global__ void matrixMulKernel(float *P, const float *M, const float *N, int Width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < Width && Col < Width)
    {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k)
        {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

static void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Errore CUDA (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    printf("DAMMI WIDTH> ");
    int width;
    scanf("%d", &width);
    printf("DAMMI HEIGHT> ");
    int height;
    scanf("%d", &height);
    const size_t n = width * height;
    const size_t bytes = n * sizeof(float);

    float *h_M = (float *)malloc(bytes);
    float *h_N = (float *)malloc(bytes);
    float *h_P = (float *)malloc(bytes);
    if (!h_M || !h_N || !h_P)
    {
        fprintf(stderr, "malloc host fallita\n");
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        h_M[i] = i * 1.0f;
        h_N[i] = i * 1.0f;
    }

    float *d_M = NULL, *d_N = NULL, *d_P = NULL;
    checkCuda(cudaMalloc((void **)&d_M, bytes), "cudaMalloc d_M");
    checkCuda(cudaMalloc((void **)&d_N, bytes), "cudaMalloc d_N");
    checkCuda(cudaMalloc((void **)&d_P, bytes), "cudaMalloc d_P");

    checkCuda(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_M->d_M");
    checkCuda(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_N->d_N");

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    matrixMulKernel<<<gridDim, blockDim>>>(d_P, d_M, d_N, width);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel execution");

    checkCuda(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_P->h_P");

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            printf("%6.1f\t ", h_P[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
