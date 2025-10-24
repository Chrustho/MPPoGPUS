#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 32

__global__ void MatrixMulKernel(float *P, float *M, float *N, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ph++)
    {
        Mds[tr][tc] = M[Row * Width + (ph * TILE_WIDTH + tc)];
        Nds[tr][tc] = N[(ph * TILE_WIDTH + tr) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[tr][k] * Nds[k][tc];
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;

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
