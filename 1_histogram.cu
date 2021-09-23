
#include "1_histogram.h"

#define INTENSITIES 256

__global__ void hist(int* d_blue, int* d_green, int* d_red, int* d_hist_blue, int* d_hist_green, int* d_hist_red, int len) {

    //for (int i = 0; i < 256; i++)
    //    d_hist_blue[i] = 100;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // count 
    if (tid < len) {
        
        atomicAdd(&(d_hist_blue[d_blue[tid]]), 1);
        atomicAdd(&(d_hist_green[d_green[tid]]), 1);
        atomicAdd(&(d_hist_red[d_red[tid]]), 1);
    }
}

void histogram_cuda(int* blue, int* green, int* red, int rows, int cols, int* output_blue, int* output_green, int* output_red) {
    int* d_blue, * d_green, * d_red, * d_hist_blue, * d_hist_green, * d_hist_red;
    int* zeros = new int[INTENSITIES];
    int length = rows * cols;
    int size = sizeof(int) * length;

    for (int i = 0; i < INTENSITIES; i++)
        zeros[i] = 0;

    // Allocate space for device 
    cudaMalloc((void**)&d_blue, size);
    cudaMalloc((void**)&d_green, size);
    cudaMalloc((void**)&d_red, size);
    
    cudaMalloc((void**)&d_hist_blue, sizeof(int) * INTENSITIES);
    cudaMalloc((void**)&d_hist_green, sizeof(int) * INTENSITIES);
    cudaMalloc((void**)&d_hist_red, sizeof(int) * INTENSITIES);
       

    // Copy inputs to device
    cudaMemcpy(d_blue, blue, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_red, red, size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_hist_blue, zeros, INTENSITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist_green, zeros, INTENSITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist_red, zeros, INTENSITIES, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);   

    hist <<< grid_size, block_size >>> (d_blue, d_green, d_red, d_hist_blue, d_hist_green, d_hist_red, length);

    cudaMemcpy(output_blue, d_hist_blue, sizeof(int) * INTENSITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_green, d_hist_green, sizeof(int) * INTENSITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_red, d_hist_red, sizeof(int) * INTENSITIES, cudaMemcpyDeviceToHost);       

    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_red);
    cudaFree(d_hist_blue);
    cudaFree(d_hist_green);
    cudaFree(d_hist_red);
    
}



