
#include "5_conv.h"
/*
#define INTENSITIES 256

__global__ void conv(int* d_img, int* d_kernel, int* d_output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        // sum
        int sum_b = (int)(0.5 * d_img[tid] + 0.5 * d_img[tid]);
        int sum_g = (int)(0.5 * d_img[tid + len] + 0.5 * d_img[tid + len]);
        int sum_r = (int)(0.5 * d_img[tid + len * 2] + 0.5 * d_img[tid + len * 2]);
        if (sum_b > 255) sum_b = 255;
        if (sum_g > 255) sum_g = 255;
        if (sum_r > 255) sum_r = 255;

        d_output[tid] = sum_b;
        d_output[tid + len] = sum_g;
        d_output[tid + len * 2] = sum_r;
    }
}

void conv_cuda(int* img, int *kernel, int rows, int cols, int* output) {
    int* d_img;
    int* d_kernel;
    int* d_output;
    int length = rows * cols;
    int size = sizeof(int) * length * 3;

    cudaMalloc((void**)&d_img, size);    
    cudaMalloc((void**)&d_kernel, sizeof(int)*9);
    cudaMalloc((void**)&d_output, size);
    
    cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, size, cudaMemcpyHostToDevice);
    
    int block_size = rows;
    int grid_size = ((length + block_size) / block_size);
    conv << < grid_size, block_size >> > (d_img, d_kernel, d_output, length);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);    

    cudaFree(d_img);    
    cudaFree(d_output);
    
}
*/


#include "5_conv.h"

#define maskCols 5
#define maskRows 5

// mask in constant memory
__constant__ float deviceMaskData[maskRows * maskCols];

__global__ void constantKernelConvolution(int* d_img, const float* __restrict__ kernel,
	int* d_output, int channels, int width, int height) {


	float accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;	//col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;	//row index
	int maskRowsRadius = maskRows / 2;
	int maskColsRadius = maskCols / 2;


	for (int k = 0; k < channels; k++) {    //cycle on channels
		if (row < height && col < width) {
			accum = 0;
			int startRow = row - maskRowsRadius;    //row index shifted by mask radius
			int startCol = col - maskColsRadius;	//col index shifted by mask radius

			for (int i = 0; i < maskRows; i++) {	//cycle on mask rows

				for (int j = 0; j < maskCols; j++) {	//cycle on mask cols

					int currentRow = startRow + i;	//row index to fetch data from input image
					int currentCol = startCol + j;	//col index to fetch data from input image

					if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {

						accum += d_img[(currentRow * width + currentCol) * channels + k] *
							deviceMaskData[i * maskRows + j];
					}
					else accum = 0;
				}

			}
			d_output[(row * width + col) * channels + k] = (int)accum;
		}

	}

}



void conv_cuda(int* img, int* kernel, int rows, int cols, int channels, int* output) {

	
	float hostMaskData[maskRows * maskCols] = {
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04
	};

	
	int* d_img;
	int* d_kernel;
	int* d_output;
	int length = rows * cols;
	int size = sizeof(int) * length * 3;

	cudaMalloc((void**)&d_img, size);
	//cudaMalloc((void**)&d_kernel, sizeof(int) * 9);
	cudaMalloc((void**)&d_output, size);

	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_kernel, kernel, size, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(deviceMaskData, hostMaskData, maskRows * maskCols * sizeof(float));

	dim3 dimGrid(ceil((float)cols / 16),
		ceil((float)rows / 16));
	dim3 dimBlock(16, 16, 1);
		
	constantKernelConvolution << <dimGrid, dimBlock >> > (d_img, deviceMaskData, d_output,
		channels, cols, rows);
	

	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	
	cudaFree(d_img);
	cudaFree(d_output);
	cudaFree(deviceMaskData);
;

}