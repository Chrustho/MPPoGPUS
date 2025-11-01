NVCC = nvcc
CFLAGS = -O3 -arch=sm_52

all: basic_matmul tiled_matmul

basic_matmul: basic_matmul.cu
	$(NVCC) $(CFLAGS) -o basic_matmul basic_matmul.cu

tiled_matmul: tiled_matmul.cu
	$(NVCC) $(CFLAGS) -o tiled_matmul tiled_matmul.cu

clean:
	rm -f basic_matmul tiled_matmul *.csv *.dat nvprof_*.txt
