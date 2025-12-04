/*
Develop a CUDA blur kernel program that implements 
the naïve/tiled algorithms 
described in the convolution lecture, namely:
- Basic: see page 131 in mppogpus_2025-11-07.pdf
- Tiled with halos: page 141 in mppogpus_2025-11-07.pdf
- Tiled without halos: page 146 in mppogpus_2025-11-07.pdf
- Tiled with halos and larger blocks: page 149 in mppogpus_2025-11-07.pdf
 
Evaluate the performance of each kernel on the Nvidia GTX 980 GPU using 
blockSize = 8x8, 16x16, 32x32 via nvprof.
Use the profiled data to evaluate the operational intensity 
and performance of the kernels and report the result in a Roofline graph.
Also, plot the recorded elapsed times in a single bar graph.
 
For the graphs, I suggest using Gnluplot.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define CHANNELS 3

#define BLUR_RADIUS 3
#define TILE_SIZE 32
#define SM_WIDTH (TILE_SIZE+2*BLUR_RADIUS)


__global__ 
void conv_blur_halos(float *P, float *M, int Mask_Width, int Width, int Height){

    __shared__ unsigned char I_sm[SM_WIDTH][SM_WIDTH][CHANNELS];

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int row= blockIdx.y*blockDim.y+ty;
    int col= blockIdx.x*blockDim.x+tx;


}
