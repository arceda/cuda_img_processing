
#include "8_search.h"

__global__ void search(	int* d_img, int* d_output, int* d_pattern, int channels, 
						int cols, int rows,
						int cols_pattern, int rows_pattern, int* d_num_boxes) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int length = cols * rows;
	int length_pattern = cols_pattern * rows_pattern;

	int radius_y = rows_pattern / 2;
	int radius_x = cols_pattern / 2;
	float acc;


	//acc = 0;
	for (int k = 0; k < channels; k++) {
		// mal, creamos un histograma del patron
		int hist_window[255];
		for (int index = 0; index < 256; index++) hist_window[index] = 0;		

		// creamos un histograma para cada ventana
		int hist_pattern[256];
		for (int index = 0; index < 256; index++) hist_pattern[index] = 0;

		// al principio la imagen de salida es igual a la imagend de entrada
		//d_output[(row * cols + col) + length * k] = d_img[(row * cols + col) + length * k]; 
		d_output[(row * cols + col) + length * k] = 0;

		if (row < rows && col < cols) {
			acc = 0;
			int start_row = row - radius_y;
			int start_col = col - radius_x;

			for (int i = 0; i < rows_pattern; i++) {	
				for (int j = 0; j < cols_pattern; j++) {
					int current_row = start_row + i;	
					int current_col = start_col + j;	

					hist_pattern[d_pattern[(i * cols_pattern + j) + length_pattern * k]] += 1;

					if (current_row >= 0 && current_row < rows && current_col >= 0 && current_col < cols) {						
						hist_window[d_img[(current_row * cols + current_col) + length * k]] += 1;
					}
					//else acc = 0;
				}
			}

			// comparamos los histogramas
			//acc = 0;
			for (int i = 0; i < 256; i++)
				acc += (hist_pattern[i] - hist_window[i]) * (hist_pattern[i] - hist_window[i]);

			float error = acc / (rows_pattern * cols_pattern);
			if (error < 20) {				
				d_output[(row * cols + col) + length * k] = 255;
				//d_num_boxes += 1;
				atomicAdd(&(d_num_boxes[0]), 1);
			}
			
		}

				
	}	

	/*float error = acc / (rows_pattern * cols_pattern * channels);
	if (error < 40) {
		d_output[(row * cols + col) + length * 2] = 255;
		//d_num_boxes += 1;
		atomicAdd(&(d_num_boxes[0]), 1);
	}*/
}


void search_pattern_cuda(	int* img, int* pattern,
					int rows, int cols, 
					int rows_pattern, int cols_pattern, 
					int channels, int* output, int* num_boxes) {
	int* d_img;
	int* d_pattern;
	int* d_output;
	int* d_num_boxes;
	int length = rows * cols;
	int size = sizeof(int) * length * channels;
	int size_pattern = sizeof(int) * rows_pattern * cols_pattern * channels;
	
	cudaMalloc((void**)&d_img, size);
	cudaMalloc((void**)&d_pattern, size_pattern);
	cudaMalloc((void**)&d_output, size);
	cudaMalloc((void**)&d_num_boxes, sizeof(int));

	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_boxes, 0, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pattern, pattern, size_pattern, cudaMemcpyHostToDevice);

	dim3 num_blocks(ceil((float)cols / 16), ceil((float)rows / 16));
	dim3 threads_per_block(16, 16, 1);
	search << <num_blocks, threads_per_block >> > (d_img, d_output, d_pattern, channels, cols, rows, 
													cols_pattern, rows_pattern, d_num_boxes);

	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_boxes, d_num_boxes, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_pattern);
	cudaFree(d_output);
}


