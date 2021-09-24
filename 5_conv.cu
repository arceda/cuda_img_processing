
#include "5_conv.h"

__global__ void conv(int* d_img, float* d_kernel,
	int* d_output, int channels, int width, int height, int kernel_rows, int kernel_cols) {

	float acc;
	int col = threadIdx.x + blockIdx.x * blockDim.x;	//col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;	//row index
	int radius_y = kernel_rows / 2;
	int radius_x = kernel_cols / 2;

	for (int k = 0; k < channels; k++) {    
		if (row < height && col < width) {
			acc = 0;
			int start_row = row - radius_y; 
			int start_col = col - radius_x;	

			for (int i = 0; i < kernel_rows; i++) {	//cycle on mask rows
				for (int j = 0; j < kernel_cols; j++) {	//cycle on mask cols
					int current_row = start_row + i;	//row index to fetch data from input image
					int current_col = start_col + j;	//col index to fetch data from input image

					if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
						acc += d_img[(current_row * width + current_col) * channels + k] *
							d_kernel[i * kernel_rows + j];
					}
					else acc = 0;
				}
			}
			d_output[(row * width + col) * channels + k] = (int)acc;
		}
	}
}

void conv_cuda(	int* img, float* kernel, int rows, int cols, int channels, 
				int kernel_rows, int kernel_cols, int* output) {
	
	int* d_img;
	float* d_kernel;
	int* d_output;
	int length = rows * cols;
	int size = sizeof(int) * length * channels;

	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&d_kernel, sizeof(float) * kernel_rows* kernel_cols);
	cudaMalloc((void**)&d_output, size);

	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, sizeof(float) * kernel_rows * kernel_cols, cudaMemcpyHostToDevice);
		
	dim3 num_blocks(ceil((float)cols / 16), ceil((float)rows / 16));
	dim3 threads_per_block(16, 16, 1);
	//int num_blocks = rows;
	//int threads_per_block = ((length + num_blocks) / num_blocks);
	//cout << num_blocks << " " << threads_per_block << endl;
	//cout << cols << " " << rows << " " << ceil((float)cols / 16) << " " << ceil((float)rows / 16) << endl;
	conv << <num_blocks, threads_per_block >> > (d_img, d_kernel, d_output,
		channels, cols, rows, kernel_rows, kernel_cols);
	
	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_output);
	cudaFree(d_kernel);
;

}