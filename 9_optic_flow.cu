
#include "9_optic_flow.h"

#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters:
			&a GPU device pointer to a m X n matrix (A)
			&b GPU device pointer to a n X k matrix (B)
			&c GPU device output purpose pointer to a m X k matrix (C)
			to store the result
Note:
	grid and block should be configured as:
		dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(float* a, float* b, float* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters:
			&a GPU device pointer to a n X n matrix (A)
			&b GPU device pointer to a n X n matrix (B)
			&c GPU device output purpose pointer to a n X n matrix (C)
			to store the result
Note:
	grid and block should be configured as:
		dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
		dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(float* d_a, float* d_b, float* d_result, int n)
{
	__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int tmp = 0;
	int idx;

	for (int sub = 0; sub < gridDim.x; ++sub)
	{
		idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
		if (idx >= n * n)
		{
			// n may not divisible by BLOCK_SIZE
			tile_a[threadIdx.y][threadIdx.x] = 0;
		}
		else
		{
			tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
		}

		idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
		if (idx >= n * n)
		{
			tile_b[threadIdx.y][threadIdx.x] = 0;
		}
		else
		{
			tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
		}
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < n && col < n)
	{
		d_result[row * n + col] = tmp;
	}
}


__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < cols && idy < rows)
	{
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}




__global__ void optic_flow(int* d_img_1, int* img_2, int* d_output, int channels, int cols, int rows) {
	int	col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int length = cols * rows;
	

	for (int k = 0; k < channels; k++) {
		
		d_output[(row * cols + col) + length * k] = d_img_1[(row * cols + col) + length * k];
		
	}
}

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (ROW < N && COL < N) {
		// each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
	}
	C[ROW * N + COL] = tmpSum;
}


void matrixMultiplication(float* A, float* B, float* C, int N) {

	int* d_img_1;
	int* d_img_2;
	int* d_output;

	int length = N * N;
	int size = sizeof(float) * length;

	cudaMalloc((void**)&d_img_1, size);
	cudaMalloc((void**)&d_img_2, size);
	cudaMalloc((void**)&d_output, size);

	cudaMemcpy(d_img_1, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_2, B, size, cudaMemcpyHostToDevice);
	
	// declare the number of blocks per grid and the number of threads per block
	// use 1 to 512 threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N * N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, N);
	
	cudaMemcpy(C, d_output, size, cudaMemcpyDeviceToHost);
	cudaFree(d_img_1);
	cudaFree(d_img_2);
	cudaFree(d_output);
}

// solo para matrices cuadradas
Mat my_cv_matrix_multiplication(Mat  A, Mat B) {
	int N = A.rows;
	float* A_vec = new float[N * N];
	float* B_vec = new float[N * N];
	float* C_vec = new float[N * N];
	int index = 0;
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			A_vec[index] = (float)(A.at<double>(i, j));
			B_vec[index] = (float)(B.at<double>(i, j));

			index += 1;

			/*if (index < 10) {
				cout << (float)(A.at<double>(i, j)) << " ";
				cout << (float)(B.at<double>(i, j)) << " ";

				cout << "a_vec:"<< A_vec[index - 1] << " ";
				cout << "b_vec:" << B_vec[index - 1] << " ";
			}*/
				
		}
	}
	//cout << "\nse convirtio los mat a vectores" << endl;

	matrixMultiplication(A_vec, B_vec, C_vec, N);
	//cout << "\nse multiplico" << endl;

	//for (int i = 0; i < 10; i++) {
	//	cout << C_vec[i] << " ";
	//}

	Mat C(A.rows, A.cols, CV_64FC1, cv::Scalar(0));

	//cout << "\n\nse convittio el vector a mat. Mat C:" << endl;
	index = 0;
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			C.at<double>(i, j) = C_vec[index];
			index += 1;

			//if (index < 10) {
			//	cout << (float)(C.at<double>(i, j)) << " ";
			//}
		}
	}
	
	return C;
}



void optic_flow_cuda(int* img_1, int* img_2,
	int rows, int cols, int channels, int* output) {
	int* d_img_1;
	int* d_img_2;
	int* d_output;
	
	int length = rows * cols;
	int size = sizeof(int) * length * channels;
	int size_output = sizeof(int) * rows * cols * 2;

	cudaMalloc((void**)&d_img_1, size);
	cudaMalloc((void**)&d_img_2, size);
	cudaMalloc((void**)&d_output, size_output);

	cudaMemcpy(d_img_1, img_1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_2, img_2, size, cudaMemcpyHostToDevice);		

	dim3 num_blocks(ceil((float)cols / 16), ceil((float)rows / 16));
	dim3 threads_per_block(16, 16, 1);
	optic_flow << <num_blocks, threads_per_block >> > (d_img_1, d_img_2, d_output, channels, cols, rows);

	cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);	

	cudaFree(d_img_1);
	cudaFree(d_img_2);	
	cudaFree(d_output);
}



Mat make_color_wheel_cuda() {

	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int num_cols = RY + YG + GC + CB + BM + MR;
	int col = 0;

	Mat color_wheel = Mat::zeros(num_cols, 3, CV_64FC1);

	//RY calculation
	for (int i = 0; i < RY; i++) {
		color_wheel.at<double>(i, 0) = 255;
		color_wheel.at<double>(i, 1) = floor(255 * i / RY);
	}
	col += RY;

	//YG calculation
	for (int i = 0; i < YG; i++) {
		color_wheel.at<double>(col + i, 0) = 255 - floor(255 * i / YG);
		color_wheel.at<double>(col + i, 1) = 255;
	}
	col += YG;

	//GC calculation
	for (int i = 0; i < GC; i++) {
		color_wheel.at<double>(col + i, 1) = 255;
		color_wheel.at<double>(col + i, 2) = floor(255 * i / GC);
	}
	col += GC;

	//CB calculation
	for (int i = 0; i < CB; i++) {
		color_wheel.at<double>(col + i, 1) = 255 - floor(255 * i / CB);
		color_wheel.at<double>(col + i, 2) = 255;
	}
	col += CB;

	//BM calculation
	for (int i = 0; i < BM; i++) {
		color_wheel.at<double>(col + i, 2) = 255;
		color_wheel.at<double>(col + i, 0) = floor(255 * i / BM);
	}
	col += BM;

	//MR calculation
	for (int i = 0; i < MR; i++) {
		color_wheel.at<double>(col + i, 2) = 255 - floor(255 * i / MR);
		color_wheel.at<double>(col + i, 0) = 255;
	}

	return color_wheel;
}

//indirect translation to C++ from MATLAB source file: cgm.technion.ac.il/people/Viki/figure1&5_cluster/computeColor.m
Mat compute_color_cuda(Mat U, Mat V) {

	Mat img;
	Mat color_wheel = make_color_wheel_cuda();
	int num_cols = color_wheel.rows;

	Mat U_squared, V_squared, rad;
	cv::pow(U, 2, U_squared);
	cv::pow(V, 2, V_squared);
	cv::sqrt(U_squared + V_squared, rad);

	Mat a = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			double v_element = V.at<double>(i, j);
			double u_element = U.at<double>(i, j);
			a.at<double>(i, j) = atan2(-v_element, -u_element) / M_PI;
		}
	}

	Mat fk = (a + 1) / 2 * (num_cols - 1); //remove +1 for C++ indexing
	Mat k0 = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < k0.rows; i++) {
		for (int j = 0; j < k0.cols; j++) {
			k0.at<double>(i, j) = floor(fk.at<double>(i, j));
		}
	}

	Mat k1 = k0 + 1;

	for (int i = 0; i < k1.rows; i++) {
		for (int j = 0; j < k1.cols; j++) {
			if (k1.at<double>(i, j) == num_cols) //adjust for overflow in indexing
				k1.at<double>(i, j) = 0;
		}
	}

	Mat f = fk - k0;
	Mat f_prime = 1 - f;

	vector<cv::Mat> channels; //to store the RGB channels

	for (int i = 0; i < color_wheel.cols; i++) {
		Mat col0 = Mat::zeros(k0.rows, k0.cols, CV_64FC1);
		Mat col1 = Mat::zeros(k1.rows, k1.cols, CV_64FC1);

		for (int j = 0; j < k0.rows; j++) {
			for (int k = 0; k < k0.cols; k++) {

				double col0_index = k0.at<double>(j, k);
				col0.at<double>(j, k) = color_wheel.at<double>(col0_index, i) / 255.0;

				double col1_index = k1.at<double>(j, k);
				col1.at<double>(j, k) = color_wheel.at<double>(col1_index, i) / 255.0;
			}
		}

		Mat col_first, col_second, col;

		multiply(f_prime, col0, col_first);
		multiply(f, col1, col_second);

		col = col_first + col_second;

		for (int l = 0; l < col.rows; l++) {
			for (int m = 0; m < col.cols; m++) {
				if (rad.at<double>(l, m) <= 1) {
					double col_val = 1 - rad.at<double>(l, m) * (1 - col.at<double>(l, m));
					col.at<double>(l, m) = col_val;
				}
				else
					col.at<double>(l, m) *= 0.75;
			}
		}
		channels.push_back(col);
	}
	reverse(channels.begin(), channels.end());
	cv::merge(channels, img);

	return img;
}


void optical_flow_analysis_cuda(Mat mat1, Mat mat2, int iterations, int avg_window, double alpha) {

	//C++ time elapsed: stackoverflow.com/questions/2808398/easily-measure-elapsed-time
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	Mat img_gray, img2_gray;

	//convert to grayscale; ints from 0 to 255
	cvtColor(mat1, img_gray, COLOR_BGR2GRAY);
	cvtColor(mat2, img2_gray, COLOR_BGR2GRAY);

	//convert to double precision floats
	Mat img_gray_db, img2_gray_db;

	//CV_64FC1 for single channel
	img_gray.convertTo(img_gray_db, CV_64FC1, 1.0 / 255.0);
	img2_gray.convertTo(img2_gray_db, CV_64FC1, 1.0 / 255.0);

	//calculate directional gradients of first image
	Mat I_t = img2_gray_db - img_gray_db;

	/*cout << img_gray_db.at<double>(0, 0) << endl;
	cout << img2_gray_db.at<double>(0, 0) << endl;
	cout << I_t.at<double>(0, 0) << endl;
	cout << "Rows in img_gray_db is " << img_gray_db.rows << " and cols is " << img_gray_db.cols << endl;
	cout << "Rows in img2_gray_db is " << img2_gray_db.rows << " and cols is " << img2_gray_db.cols << endl;
	cout << "Rows in I_t is " << I_t.rows << " and cols is " << I_t.cols << endl;*/

	//calculate equivalent of imgradientxy for img_gray_db

	Mat I_x, I_y;
	int ddepth = -1; //outputs same depth as input

	Sobel(img_gray_db, I_x, ddepth, 1, 0, 3); //X gradient
	Sobel(img_gray_db, I_y, ddepth, 0, 1, 3); //Y gradient

	//initialize zero-filled matrices
	Mat U = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
	Mat V = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);

	//initialize kernel
	Mat kernel = Mat::ones(avg_window, avg_window, CV_64FC1) / pow(avg_window, 2);

	//run multiple iterations to get horizontal and vertical flow

	for (int i = 0; i < iterations; i++) {

		Mat U_avg, V_avg;

		// from stackoverflow.com/questions/10309561/is-there-any-function-in-opencv-which-is-equivalent-to-matlab-conv2 :
		//perform 2D convolutions equivalent to "same" argument in MATLAB
		Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);

		//no need to flip kernel because it's symmetric
		filter2D(U, U_avg, U.depth(), kernel, anchor, 0, BORDER_CONSTANT);
		filter2D(V, V_avg, V.depth(), kernel, anchor, 0, BORDER_CONSTANT);

		//update U and V
		Mat C_prod1, C_prod2, I_x_squared, I_y_squared, I_x_C, I_y_C, C;

		//////////////////////////////////////////////////////////////////////
		C_prod1 = my_cv_matrix_multiplication(I_x, U_avg);
		C_prod2 = my_cv_matrix_multiplication(I_y, V_avg);
		I_x_squared = my_cv_matrix_multiplication(I_x, I_x);
		I_y_squared = my_cv_matrix_multiplication(I_y, I_y);
		/*cout << "\n\n RESULT MY MUL" << endl;
		for (int i = 0; i <5; i++) {
			for (int j = 0; j < 5; j++) {
				cout<<(int)(C_prod1.at<double>(i, j)) << " ";
			}
		}*/
		//////////////////////////////////////////////////////////////////////

		//multiply(I_x, U_avg, C_prod1);

		/*cout << endl << "\n RESULT CV MUL" << endl;
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				cout << (int)(C_prod1.at<double>(i, j)) << " ";
			}
		}*/

		//multiply(I_x, U_avg, C_prod1);
		//multiply(I_y, V_avg, C_prod2);
		//multiply(I_x, I_x, I_x_squared);
		//multiply(I_y, I_y, I_y_squared);

		Mat C_num = C_prod1 + C_prod2 + I_t;
		Mat C_den = pow(alpha, 2) + I_x_squared + I_y_squared;

		divide(C_num, C_den, C);

		multiply(I_x, C, I_x_C);
		multiply(I_y, C, I_y_C);
		//I_x_C = my_cv_matrix_multiplication(I_x, C);
		//I_y_C = my_cv_matrix_multiplication(I_y, C);

		U = U_avg - I_x_C;
		V = V_avg - I_y_C;
	}

	//compute color equivalence

	Mat img = compute_color_cuda(U, V);


	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "The time elapsed is " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << endl;

	string window1 = "First image";
	string window2 = "Second image";
	string window3 = "Third image";

	//namedWindow(window1, WINDOW_AUTOSIZE);
	//imshow(window1, U);

	//namedWindow(window2, WINDOW_AUTOSIZE);
	//imshow(window2, V);

	namedWindow(window3, WINDOW_AUTOSIZE);
	imshow(window3, img);

	waitKey(0);
}


