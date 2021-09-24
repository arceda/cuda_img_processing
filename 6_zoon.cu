
#include "6_zoon.h"

__global__ void zoon(int* d_img, int* d_output, int channels, int cols, int rows) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
		
	int index = (row * cols/2 + col);

	row *= 2;
	col *= 2;
	d_output[row * cols + col] = d_img[index];
	__syncthreads();
}

__global__ void zoon_corner(int* d_img, int* d_output, int channels, int cols, int rows) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
		
	int index = row * cols / 2 + col;
	row *= 2;
	col *= 2;
	//d_output[(row + 1) * cols + col + 1] = (d_output[(row + 2) * cols + col] + d_output[(row * cols + col + 2)] + d_output[(row + 2) * cols + col + 2] + d_output[(row * cols + col)]) / 4;
	__syncthreads();
}

__global__ void zoon_v_h(int* d_img, int* d_output, int channels, int cols, int rows) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int index = row * cols / 2 + col;
	row *= 2;
	col *= 2;

	//d_output[row * cols + col + 1] = (d_output[row * cols + col] + d_output[(row - 1) * cols + col + 1] + d_output[row * cols + col + 2] + d_output[(row + 1) * cols + col + 1]) / 4;
	//d_output[(row + 1) * cols + col] = (d_output[row * cols + col] + d_output[(row + 1) * cols + col - 1] + d_output[(row + 1) * cols + col + 1] + d_output[(row + 2) * cols + col]) / 4;
	d_output[row * cols + col + 1] = (d_output[row * cols + col] + d_output[(row - 1) * cols + col + 1] + d_output[row * cols + col + 2] + d_output[(row + 1) * cols + col + 1]) / 2;
	d_output[(row + 1) * cols + col] = (d_output[row * cols + col] + d_output[(row + 1) * cols + col - 1] + d_output[(row + 1) * cols + col + 1] + d_output[(row + 2) * cols + col]) / 2;
	__syncthreads();
}

void zoon_cuda(int* img, int rows, int cols, int channels, int* output) {
	int* d_img;
	int* d_output;
	int length = rows * cols;
	//int size = sizeof(int) * length * channels;
	int size = sizeof(int) * length;
	

	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&d_output, size*4);

	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

	dim3 block_dim(16, 16);
	//dim3 block_dim(1, 1);
	dim3 grid_dim(cols /block_dim.x, rows/block_dim.y);
	zoon << <block_dim, grid_dim >> > (d_img, d_output, channels, cols * 2, rows * 2);
	zoon_corner << <block_dim, grid_dim >> > (d_img, d_output, channels, cols * 2, rows * 2);
	zoon_v_h << <block_dim, grid_dim >> > (d_img, d_output, channels, cols * 2, rows * 2);

	//dim3 num_blocks(ceil((float)cols*2 / 16), ceil((float)rows*2 / 16));
	//dim3 threads_per_block(16, 16, 1);
	//zoon << <num_blocks, threads_per_block >> > (d_img, d_output, channels, cols * 2, rows * 2);

	cudaMemcpy(output, d_output, size*4, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_output);
}
