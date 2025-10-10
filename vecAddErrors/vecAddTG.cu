#include <stdio.h>
#include <stdlib.h>
#include <ctime>

/*
 *  VecAdd with Thread Granularity implmentation
 */

__global__ void vecAddKernel(int *C, int *A, int *B, int n, int tgn)
{
    int num = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += tgn)
    {
        C[i] = A[i] + B[i];
        num++;
    }
    printf("Thread %d ha processato %d celle\n", blockIdx.x * blockDim.x + threadIdx.x, num);
}

void vecAddTG(int *C, int *A, int *B, int n, int tgn)
{
    int *d_C, *d_B, *d_A;
    int size = n * sizeof(int);
    int block_size = 32,
        number_of_blocks = ceil((int)n / block_size);

    printf("numero blocchi: %d", number_of_blocks);
    cudaMalloc((void **)&d_A, size); //<-- Async
    cudaMalloc((void **)&d_B, size); //<-- Async
    cudaMalloc((void **)&d_C, size); //<-- Async

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<number_of_blocks, block_size>>>(d_C, d_A, d_B, n, tgn); //<-- Async

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); //<-- Async
}

int main()
{
    int N = 0;
    printf("dammi dimensione array: \n ");
    scanf("%d", &N);
    int tgn = 0;
    printf("dammi thread granularity:\n");
    scanf("%d", &tgn);
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    int *h_C = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }

    vecAddTG(h_C, h_A, h_B, N, tgn);

    for (size_t i = 0; i < N; i++)
    {
        printf("%d\t", h_C[i]);
    }
    printf("\n");
}