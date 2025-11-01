#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#define MAX_TILE 32
#define MAX_GRAN 8
#define TILE 32

#define ROWS_A (1 << 11)
#define COLS_A (1 << 10)
#define ROWS_B (1 << 10)
#define COLS_B (1 << 9)

#define ROWS_C ROWS_A
#define COLS_C COLS_B

const float VAL_A = 0.5f * (1.0f / 1024.0f);
const float VAL_B = 2.0f;
const float VAL_C = 0.0f;
const float EXPECTED_C = 1.0f;

#define CUDA_CHECK(call)                                      \
    do {                                                     \
        cudaError_t e = (call);                              \
        if (e != cudaSuccess) {                              \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
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
    if (totalB > maxTotal) maxTotal = totalB;
    if (totalC > maxTotal) maxTotal = totalC;

    for (int i = idx; i < maxTotal; i += stride) {
        if (i < totalA) A[i] = valA;
        if (i < totalB) B[i] = valB;
        if (i < totalC) C[i] = valC;
    }
}

__global__ void MatrixMulTiled_GranularTG(float *d_P, const float *d_M, const float *d_N,
                                          int j, int k, int l, int gran)
{
    extern __shared__ float sdata[]; 
    float *Mds = sdata; 
    float *Nds = sdata + (TILE * TILE); 

    int tc = threadIdx.x; 
    int tr = threadIdx.y; 
    int Row = blockIdx.y * TILE + tr;
    int ColBase = blockIdx.x * (TILE * gran);

    float Pvalue[MAX_GRAN];
    for (int g = 0; g < gran; ++g) Pvalue[g] = 0.0f;

    int numPhases = (k + TILE - 1) / TILE;

    for (int ph = 0; ph < numPhases; ++ph) {
        int colIdxM = ph * TILE + tc;
        int rowIdxN = ph * TILE + tr;

        if ((Row < j) && (colIdxM < k)) {
            Mds[tr * TILE + tc] = d_M[(size_t)Row * k + colIdxM];
        }

        int pitchN = TILE * gran;
        for (int g = 0; g < gran; ++g) {
            int globalCol = ColBase + g * TILE + tc;
            if ((rowIdxN < k) && (globalCol < l)) {
                Nds[tr * pitchN + g * TILE + tc] = d_N[(size_t)rowIdxN * l + globalCol];
            }
        }

        __syncthreads();

        for (int t = 0; t < TILE; ++t) {
            float m = Mds[tr * TILE + t];
            for (int g = 0; g < gran; ++g) {
                Pvalue[g] += m * Nds[t * pitchN + g * TILE + tc];
            }
        }

        __syncthreads();
    }

    for (int g = 0; g < gran; ++g) {
        int outCol = ColBase + g * TILE + tc;
        if ((Row < j) && (outCol < l)) {
            d_P[(size_t)Row * l + outCol] = Pvalue[g];
        }
    }
}


int checkCorrectness(const float *hostC, size_t rows, size_t cols)
{
    double tol = 1e-3;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float v = hostC[r * cols + c];
            if (fabs((double)v - (double)EXPECTED_C) > tol) {
                fprintf(stderr, "Mismatch at (%zu,%zu): got %f expected %f\n", r, c, v, EXPECTED_C);
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    printf("matmul_granularity: A=(%d x %d), B=(%d x %d), C=(%d x %d)\n",
           ROWS_A, COLS_A, ROWS_B, COLS_B, ROWS_C, COLS_C);

    size_t sizeA = (size_t)ROWS_A * COLS_A * sizeof(float);
    size_t sizeB = (size_t)ROWS_B * COLS_B * sizeof(float);
    size_t sizeC = (size_t)ROWS_C * COLS_C * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));

    int threadsInit = 1024;
    int blocksInit = (int)(((sizeA / sizeof(float)) + threadsInit - 1) / threadsInit);
    matrixInit<<<blocksInit, threadsInit>>>(d_A, ROWS_A, COLS_A, d_B, ROWS_B, COLS_B, d_C, ROWS_C, COLS_C, VAL_A, VAL_B, VAL_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_C = (float *)malloc(sizeC);
    if (!h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    FILE *fout = fopen("granularity_timings.csv", "w");
    if (!fout) {
        fprintf(stderr, "Unable to open output CSV\n");
        return 1;
    }
    fprintf(fout, "granularity,block,ms,gflops,intensity,correct\n");

    double totalFlops = (double)ROWS_A * (double)COLS_B * (2.0 * (double)COLS_A - 1.0);
    double minimalBytes = ((double)ROWS_A * COLS_A + (double)COLS_A * COLS_B + (double)ROWS_A * COLS_B) * sizeof(float);
    double intensity = totalFlops / minimalBytes;
 

    for (int gran = 1; gran <= MAX_GRAN; ++gran) {
        if (gran < 1) gran = 1;
        if (gran > MAX_GRAN) {
            fprintf(stderr, "Requested gran > MAX_GRAN (%d) - skipping\n", MAX_GRAN);
            break;
        }

        dim3 block(TILE, TILE);
        dim3 grid((COLS_C + (TILE * gran) - 1) / (TILE * gran), (ROWS_C + TILE - 1) / TILE);

        matrixInit<<<blocksInit, threadsInit>>>(d_A, ROWS_A, COLS_A, d_B, ROWS_B, COLS_B, d_C, ROWS_C, COLS_C, VAL_A, VAL_B, 0.0f);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        MatrixMulTiled_GranularTG<<<grid, block>>>(d_C, d_A, d_B, ROWS_A, COLS_A, COLS_B, gran);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double gflops = 0.0;
        if (ms > 0.0f) gflops = (totalFlops / (ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        int ok = checkCorrectness(h_C, ROWS_C, COLS_C);

        printf("gran=%d tile=%d: %.3f ms, %.3f GFLOPS, intensity=%.6f, correct=%s\n",
               gran, TILE, ms, gflops, intensity, ok ? "YES" : "NO");
        fprintf(fout, "%d,%d,%.6f,%.6f,%.6f,%d\n", gran, TILE, ms, gflops, intensity, ok);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    fclose(fout);

    FILE *fr = fopen("granularity_roofline.dat", "w");
    if (fr) {
        fprintf(fr, "#granularity gflops intensity\n");
        for (int gran = 1; gran <= MAX_GRAN; ++gran) {
            fprintf(fr, "%d 0.0 %.6f\n", gran, intensity);
        }
        fclose(fr);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_C);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
