
#include "2_equalize.h"

#define INTENSITIES 256

__global__ void arithmetics(int* d_blue, int* d_green, int* d_red,
    int** d_output_1, int** d_output_2, int** d_output_3,
    int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // count 
    if (tid < len) {
        float A = 1.5;
        float B = 30;
        float res_b = (float)d_blue[tid] * A + B;
        float res_g = (float)d_green[tid] * A + B;
        float res_r = (float)d_red[tid] * A + B;
        if (res_b > 255) res_b = 255;
        if (res_g > 255) res_g = 255;
        if (res_r > 255) res_r = 255;

        d_out_blue[tid] = (int)res_b;
        d_out_green[tid] = (int)res_g;
        d_out_red[tid] = (int)res_r;
    }
}

/*
void global_func_cuda(int*** img, int rows, int cols, int channels, int*** o_output1) {
    int* d_img;
    int* d_output1;
    int* zeros = new int[INTENSITIES];
    int length = rows * cols;
    int size = sizeof(int) * length * channels;

    for (int i = 0; i < INTENSITIES; i++)
        zeros[i] = 0.0;

    cudaMalloc((void**)&d_img, size);
    cudaMalloc((void**)&d_output1, size);

    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);
    apply_function << < grid_size, block_size >> > (d_img, d_output1, length);

    cudaMemcpy(o_output1, d_output1, sizeof(int) * INTENSITIES, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_output1);
}
*/



void arithmetics_cuda(int* blue, int* green, int* red, int rows, int cols, int* o_blue, int* o_green, int* o_red) {
    int* d_blue, * d_green, * d_red;
    int** d_out_blue, ** d_out_green, ** d_out_red;
    int length = rows * cols;
    int size = sizeof(int) * length;

    cudaMalloc((void**)&d_blue, size);
    cudaMalloc((void**)&d_green, size);
    cudaMalloc((void**)&d_red, size);
    cudaMalloc((void**)&d_out_blue, size * 3);
    cudaMalloc((void**)&d_out_green, size * 3);
    cudaMalloc((void**)&d_out_red, size * 3);

    cudaMemcpy(d_blue, blue, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_red, red, size, cudaMemcpyHostToDevice);

    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);
    arithmetics << < grid_size, block_size >> > (d_blue, d_green, d_red, d_out_blue, d_out_green, d_out_red, length);

    cudaMemcpy(o_blue, d_out_blue, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_green, d_out_green, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(o_red, d_out_red, size, cudaMemcpyDeviceToHost);

    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_red);
    cudaFree(d_out_blue);
    cudaFree(d_out_green);
    cudaFree(d_out_red);
}
