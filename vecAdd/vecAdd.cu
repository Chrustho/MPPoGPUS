#include <stdio.h>

__global__ void vecAddKernel(float *C, float *A, float *B, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecAdd(float *C, float *A, float *B, int n)
{
    float *d_C, *d_B, *d_A;
    int size = n * sizeof(float);
    int block_size = 32, number_of_blocks = ceil((float)n / block_size);

    cudaMalloc((void **)&d_A, size); //<-- Async
    cudaMalloc((void **)&d_B, size); //<-- Async
    cudaMalloc((void **)&d_C, size); //<-- Async

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<number_of_blocks, block_size>>>(d_C, d_A, d_B, n); //<-- Async

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); //<-- Async
}

int main()
{
    int N = 4;
    float h_A[] = {16.0f, 17.0f, 18.0f, 19.0f};
    float h_B[] = {256.0f, 256.0f, 256.0f, 256.0f};
    float *h_C = (float *)malloc(N * sizeof(float));
    vecAdd(h_C, h_A, h_B, N);

    for (size_t i = 0; i < N; i++)
    {
        printf("%f\t", h_C[i]);
    }
    printf("\n");
}