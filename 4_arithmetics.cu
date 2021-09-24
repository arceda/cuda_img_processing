
#include "4_arithmetics.h"

#define INTENSITIES 256

__global__ void arithmetics(int* d_img_1, int* d_img_2, 
                            int* d_output_1,  int* d_output_2, int* d_output_3, int* d_output_4,
                            int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        // sum
        int sum_b = (int)(0.5 * d_img_1[tid] + 0.5 * d_img_2[tid]);
        int sum_g = (int)(0.5 * d_img_1[tid + len] + 0.5 * d_img_2[tid + len]);
        int sum_r = (int)(0.5 * d_img_1[tid + len*2] + 0.5 * d_img_2[tid + len*2]);
        if (sum_b > 255) sum_b = 255;
        if (sum_g > 255) sum_g = 255;
        if (sum_r > 255) sum_r = 255;
        
        d_output_1[tid] = sum_b;
        d_output_1[tid + len] = sum_g;
        d_output_1[tid + len*2] = sum_r;

        // mul
        int mul_b = (int)(d_img_1[tid] * d_img_2[tid])/255;
        int mul_g = (int)(d_img_1[tid + len] * d_img_2[tid + len]) / 255;
        int mul_r = (int)(d_img_1[tid + len * 2] * d_img_2[tid + len * 2]) / 255;
        if (mul_b > 255) mul_b = 255;
        if (mul_g > 255) mul_g = 255;
        if (mul_r > 255) mul_r = 255;

        d_output_2[tid] = mul_b;
        d_output_2[tid + len] = mul_g;
        d_output_2[tid + len * 2] = mul_r;

        // sub
        int sub_b = (int)(d_img_1[tid] - d_img_2[tid]) + 100;
        int sub_g = (int)(d_img_1[tid + len] - d_img_2[tid + len]) + 100;
        int sub_r = (int)(d_img_1[tid + len * 2] - d_img_2[tid + len * 2]) + 100;
        if (sub_b < 0) sub_b = 0; if (sub_b > 255) sub_b = 255;
        if (sub_g < 0) sub_g = 0; if (sub_g > 255) sub_g = 255;
        if (sub_r < 0) sub_r = 0; if (sub_r > 255) sub_r = 255;

        d_output_3[tid] = sub_b;
        d_output_3[tid + len] = sub_g;
        d_output_3[tid + len * 2] = sub_r;

        // div
        int div_b = (int)(d_img_1[tid] / d_img_2[tid]) * 255;
        int div_g = (int)(d_img_1[tid + len] / d_img_2[tid + len]) * 255;
        int div_r = (int)(d_img_1[tid + len * 2] / d_img_2[tid + len * 2]) * 255;
        if (div_b < 0) div_b = 0; if (div_b > 255) div_b = 255;
        if (div_g < 0) div_g = 0; if (div_g > 255) div_g = 255;
        if (div_r < 0) div_r = 0; if (div_r > 255) div_r = 255;

        d_output_4[tid] = div_b;
        d_output_4[tid + len] = div_g;
        d_output_4[tid + len * 2] = div_r;
                
       
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
