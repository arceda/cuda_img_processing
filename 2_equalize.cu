
#include "2_equalize.h"

#define INTENSITIES 256

__global__ void hist_frecuencies(int* d_data, int* d_hist, float* d_hist_p, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // count 
    if (tid < len) {
        atomicAdd(&(d_hist_p[d_data[tid]]), 1.0/len);
        atomicAdd(&(d_hist[d_data[tid]]), 1);
    }
}

__global__ void rebuild_img(float* d_CH, int* d_new_img, int* old_img, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // count 
    if (tid < len) {
        d_new_img[tid] = d_CH[old_img[tid]];
    }
}

// esto no se puede paralelizar
void sum_probabilities(float * in, float *out, int len){
    float acc = 0.0;
    for (int i = 0; i < len; i++) {
        acc += in[i];
        out[i] = floor( (INTENSITIES - 1) * acc);
    }
}

void equalize_hist_cuda(int* data, int rows, int cols, int* o_hist, int* new_img) {
    int* d_data;
    int* d_hist;
    float* d_hist_p;
    int* zeros = new int[INTENSITIES];
    int length = rows * cols;
    int size = sizeof(int) * length;

    for (int i = 0; i < INTENSITIES; i++)
        zeros[i] = 0.0;

    // Allocate space for device 
    cudaMalloc((void**)&d_data, size);
    cudaMalloc((void**)&d_hist, sizeof(int) * INTENSITIES);
    cudaMalloc((void**)&d_hist_p, sizeof(float) * INTENSITIES);

    // Copy inputs to device
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, zeros, INTENSITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist_p, zeros, INTENSITIES, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);

    hist_frecuencies << < grid_size, block_size >> > (d_data, d_hist, d_hist_p, length);

    float* hist_p = new float[INTENSITIES];
    float* CH = new float[INTENSITIES];

    cudaMemcpy(o_hist, d_hist, sizeof(int) * INTENSITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hist_p, d_hist_p, sizeof(float) * INTENSITIES, cudaMemcpyDeviceToHost);
    
    sum_probabilities(hist_p, CH, INTENSITIES);
    /*
    float acc = 0.0;
    for (int i = 0; i < 256; i++) {
        cout << CH[i] << ", ";
        acc += hist_p[i];
    }   
    cout << "acc: " << acc << endl << endl;
    */

    /// ////////////////////////////////////////////////////////////////
    /// rebuilt image
    /// ////////////////////////////////////////////////////////////////
    int* d_old_image, *d_new_image;
    float* d_CH;
    cudaMalloc((void**)&d_old_image, sizeof(int) * length);
    cudaMalloc((void**)&d_new_image, sizeof(int) * length);
    cudaMalloc((void**)&d_CH, sizeof(float) * INTENSITIES);

    cudaMemcpy(d_old_image, data, sizeof(int) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CH, CH, sizeof(float) * INTENSITIES, cudaMemcpyHostToDevice);

    rebuild_img << < grid_size, block_size >> > (d_CH, d_new_image, d_old_image, length);    
    cudaMemcpy(new_img, d_new_image, sizeof(int) * length, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < length; i++) {
    //    cout << new_img[i] << ", ";
    //}


    cudaFree(d_data);
    cudaFree(hist_p);
    cudaFree(d_old_image);
    cudaFree(d_new_image);
    cudaFree(d_CH);
}




