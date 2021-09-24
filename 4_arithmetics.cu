
#include "2_equalize.h"

#define INTENSITIES 256

__global__ void arithmetics(int* d_img_1, int* d_img_2, 
                            int* d_output_1,  int* d_output_2, int* d_output_3, int* d_output_4,
                            int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        d_output_1[tid] = int(0.5*d_img_1[tid] + 0.5*d_img_2[tid]);
        d_output_2[tid] = d_img_1[tid] - d_img_2[tid];
        d_output_3[tid] = d_img_1[tid] * d_img_2[tid];
        d_output_4[tid] = d_img_1[tid] / d_img_2[tid];
    }
}

void arithmetics_cuda(  int* img_1, int* img_2, int rows, int cols, 
                        int* output_1, int* output_2, int* output_3, int* output_4) {
    int* d_img_1, * d_img_2;
    int* d_output_1, * d_output_2, * d_output_3, * d_output_4;
    int length = rows * cols;
    int size = sizeof(int) * length * 3;

    cudaMalloc((void**)&d_img_1, size);
    cudaMalloc((void**)&d_img_2, size);    
    cudaMalloc((void**)&d_output_1, size);
    cudaMalloc((void**)&d_output_2, size);
    cudaMalloc((void**)&d_output_3, size);
    cudaMalloc((void**)&d_output_4, size);

    cudaMemcpy(d_img_1, img_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_2, img_2, size, cudaMemcpyHostToDevice);

    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);
    arithmetics << < grid_size, block_size >> > (   d_img_1, d_img_2, d_output_1, 
                                                    d_output_2, d_output_3, d_output_4, length);

    cudaMemcpy(output_1,    d_output_1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_2,    d_output_2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_3,    d_output_3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_4,    d_output_4, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img_1);
    cudaFree(d_img_2);
    cudaFree(d_output_1);
    cudaFree(d_output_2);
    cudaFree(d_output_3);
    cudaFree(d_output_4);
}
