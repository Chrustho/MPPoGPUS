#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cuda_runtime.h>

#define MAX_TILE 32

#define ROWS_A (1 << 11) /* 2048 */
#define COLS_A (1 << 10) /* 1024 */
#define ROWS_B (1 << 10) /* 1024 */
#define COLS_B (1 << 9)  /* 512 */

#define ROWS_C ROWS_A
#define COLS_C COLS_B

const float VAL_A = 0.5f * (1.0f / 1024.0f); /* 0.5 * 2^-10 */
const float VAL_B = 2.0f;
const float VAL_C = 0.0f;
const float EXPECTED_C = 1.0f;

#define CUDA_CHECK(call)                                                                          \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = (call);                                                                   \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

__global__ void matrixInit(float *A, int rowsA, int colsA,
                           float *B, int rowsB, int colsB,
                           float *C, int rowsC, int colsC,
                           float valA, float valB, float valC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int totalA = rowsA * colsA;
    int totalB = rowsB * colsB;
    int totalC = rowsC * colsC;
    int maxTotal = totalA;
    if (totalB > maxTotal)
        maxTotal = totalB;
    if (totalC > maxTotal)
        maxTotal = totalC;

    for (int i = idx; i < maxTotal; i += stride)
    {
        if (i < totalA)
            A[i] = valA;
        if (i < totalB)
            B[i] = valB;
        if (i < totalC)
            C[i] = valC;
    }
}

__global__ void MatrixMulTiled(float *d_P, float *d_M, float *d_N, int j, int k, int l)
{
    __shared__ float Mds[MAX_TILE][MAX_TILE];
    __shared__ float Nds[MAX_TILE][MAX_TILE];

    int tc = threadIdx.x;
    int tr = threadIdx.y;
    int Row = blockIdx.y * blockDim.y + tr;
    int Col = blockIdx.x * blockDim.x + tc;

    int TILE = blockDim.x;

    float Pvalue = 0.0f;

    int numPhases = (k + TILE - 1) / TILE;

    for (int ph = 0; ph < numPhases; ++ph)
    {
        int colIdxM = ph * TILE + tc;
        int rowIdxN = ph * TILE + tr;

        if ((Row < j) && (colIdxM < k))
        {
            Mds[tr][tc] = d_M[(size_t)Row * k + colIdxM];
        }
        else
        {
            Mds[tr][tc] = 0.0f;
        }

        if ((rowIdxN < k) && (Col < l))
        {
            Nds[tr][tc] = d_N[(size_t)rowIdxN * l + Col];
        }
        else
        {
            Nds[tr][tc] = 0.0f;
        }

        __syncthreads();

        for (int t = 0; t < TILE; ++t)
        {
            Pvalue += Mds[tr][t] * Nds[t][tc];
        }

        __syncthreads();
    }

    if ((Row < j) && (Col < l))
    {
        d_P[(size_t)Row * l + Col] = Pvalue;
    }
}

int checkCorrectness(const float *hostC, size_t rows, size_t cols)
{
    double tol = 1e-3;
    size_t r, c;
    for (r = 0; r < rows; ++r)
    {
        for (c = 0; c < cols; ++c)
        {
            float v = hostC[r * cols + c];
            if (fabs((double)v - (double)EXPECTED_C) > tol)
            {
                fprintf(stderr, "Mismatch at (%zu,%zu): got %f expected %f\n", r, c, v, EXPECTED_C);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    printf("tiled_matmul: A=(%d x %d), B=(%d x %d), C=(%d x %d)\n",
           ROWS_A, COLS_A, ROWS_B, COLS_B, ROWS_C, COLS_C);

    size_t sizeA = (size_t)ROWS_A * COLS_A * sizeof(float);
    size_t sizeB = (size_t)ROWS_B * COLS_B * sizeof(float);
    size_t sizeC = (size_t)ROWS_C * COLS_C * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));

    int threadsInit = 256;
    int blocksInit = (int)(((sizeA / sizeof(float)) + threadsInit - 1) / threadsInit);
    if (blocksInit < 128)
        blocksInit = 128;

    matrixInit<<<blocksInit, threadsInit>>>(d_A, ROWS_A, COLS_A, d_B, ROWS_B, COLS_B, d_C, ROWS_C, COLS_C, VAL_A, VAL_B, VAL_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int blockSizes[3] = {8, 16, 32};

    float *h_C = (float *)malloc(sizeC);
    if (!h_C)
    {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    FILE *fout = fopen("tiled_timings.csv", "w");
    if (!fout)
    {
        fprintf(stderr, "Cannot open tiled_timings.csv for writing\n");
        return 1;
    }
    fprintf(fout, "kernel,block,ms,gflops,correct\n");

    double totalFlops = 2.0 * (double)ROWS_A * (double)COLS_B * (double)COLS_A;

    /* array per memorizzare i gflops misurati per ogni block-size */
    double gflops_arr[3] = {0.0, 0.0, 0.0};

    int i;
    for (i = 0; i < 3; ++i)
    {
        int blk = blockSizes[i];
        dim3 block(blk, blk);
        dim3 grid((COLS_C + block.x - 1) / block.x, (ROWS_C + block.y - 1) / block.y);

        /* re-inizializza C su GPU a zero prima di ogni misura */
        matrixInit<<<256, 256>>>(d_A, ROWS_A, COLS_A, d_B, ROWS_B, COLS_B, d_C, ROWS_C, COLS_C, VAL_A, VAL_B, 0.0f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        MatrixMulTiled<<<grid, block>>>(d_C, d_A, d_B, ROWS_A, COLS_A, COLS_B);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double gflops = 0.0;
        if (ms > 0.0f) /* protezione divisione per zero */
        {
            gflops = (totalFlops / (ms / 1000.0)) / 1e9;
        }
        gflops_arr[i] = gflops;

        CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        int ok = checkCorrectness(h_C, ROWS_C, COLS_C);

        printf("Tiled block %dx%d: %.3f ms, %.3f GFLOPS, correct=%s\n", blk, blk, ms, gflops, ok ? "YES" : "NO");
        fprintf(fout, "Tiled,%d,%.6f,%.6f,%d\n", blk, ms, gflops, ok);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    fclose(fout);

    /* scrivo file roofline con intensity teorica e GFLOPS misurati */
    FILE *fr = fopen("tiled_roofline.dat", "w");
    if (fr)
    {
        double minimalBytes = ((double)ROWS_A * COLS_A + (double)COLS_A * COLS_B + (double)ROWS_A * COLS_B) * sizeof(float);
        double intensity = totalFlops / minimalBytes;
        fprintf(fr, "#kernel intensity gflops\n");
        for (i = 0; i < 3; ++i)
        {
            fprintf(fr, "Tiled_%d %.6f %.6f\n", blockSizes[i], intensity, gflops_arr[i]);
        }
        fclose(fr);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_C);

    printf("tiled_matmul done. Files: tiled_timings.csv, tiled_roofline.dat\n");
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
