#include <cuda.h>
#include <stdio.h>

__global__ void vecAddKernel(float *C, float *A, float *B, int n)
{
    // i is a private variable
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) // to be sure to "intercept" data
        // C[i] = A[i] + B[i+100000]; the B pointer gets "nullified"
        C[i] = A[i] + B[i];
}

void vecAdd(float *C, float *A, float *B, int n)
{
    float *d_C, *d_B, *d_A;
    int size = n * sizeof(float);
    // we alter the block size=1025 to be much bigger, so now the sum adds up to 0.
    int block_size = 32, number_of_blocks = ceil((float)n / block_size);
    cudaMalloc((void **)&d_A, size); //<-- Async
    cudaMalloc((void **)&d_B, size); //<-- Async
    cudaMalloc((void **)&d_C, size); //<-- Async
    // cudaFree (d_A); //here we free before using the d_A pointer, so it becames invalid
    // It's worth to mention that doing so the code still works (probably the kernel treats the invalid elements as 0)
    // for (int i=0; i<3; i++){
    //     printf("%.6f \n", d_A[i]);
    //  }
    // cudaMemcpy(d_A, A, 1100000000, cudaMemcpyHostToDevice); TOO much space
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    // d_A = nullptr; doing this you get an illegal memory access
    vecAddKernel<<<number_of_blocks, block_size>>>(d_C, d_A, d_B, n); //<-- Async
    // This will be the default code to get and print errors.
    // Move it below the line that triggers the error to highlight
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C); //<-- Async
}

int main()
{
    int N = 3;
    float *h_C = (float *)malloc(N * sizeof(float));
    float h_A[] = {12, 16, 18};
    // float * h_A= (float*)malloc(1100000000*sizeof(float));
    /*for (int i=0; i<1100000000; i++){
        h_A[i]=0.07;
    }*/
    float h_B[] = {1, 1, 1};
    vecAdd(h_C, h_A, h_B, N);
    for (int i = 0; i < 3; i++)
    {
        printf("%.6f \n", h_C[i]);
    }
    free(h_C);
}