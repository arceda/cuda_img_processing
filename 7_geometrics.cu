
#include "7_geometrics.h"

__global__ void geometrics(int* d_img, int* d_output, float* d_M, int channels, int cols, int rows) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int length = cols * rows;
	
	float a = d_M[0];
	float b = d_M[1];
	float c = d_M[2];
	float d = d_M[3];
	float e = d_M[4];
	float f = d_M[5];
	
	/*float a = 1;
	float b = 0;
	float c = 0;
	float d = 1;
	float e = 50;
	float f = 50;*/

	for (int k = 0; k < channels; k++) {
		// compute A*X + B 
		int new_col, new_row;
		new_col = a * col + b * row + e;
		new_row = a * row + b * col + e;
		//d_output[(row * cols + col) + length * k] = d_img[(row * cols + col) + length * k];
		if (new_col > 0 && new_col < cols && new_row > 0 && new_row < rows)
			d_output[(new_row * cols + new_col) + length * k] = d_img[(row * cols + col) + length * k];
	}
	//d_output[row * cols + col] = 100;	
}


void geometrics_cuda(int* img, float* M, int rows, int cols, int channels, int* output) {
	int* d_img;
	float* d_M;
	int* d_output;
	int length = rows * cols;
	int size = sizeof(int) * length * channels;
	//int size = sizeof(int) * length;


	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&d_M, sizeof(float) * 6);
	cudaMalloc((void**)&d_output, size);

	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, sizeof(float) * 6, cudaMemcpyHostToDevice);

	dim3 num_blocks(ceil((float)cols / 16), ceil((float)rows / 16));
	dim3 threads_per_block(16, 16, 1);
	geometrics << <num_blocks, threads_per_block >> > (d_img, d_output, d_M, channels, cols, rows);
	
	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_M);
	cudaFree(d_output);
}

