
#include "1_histogram.h"
#include "2_equalize.h"
#include "3_global.h"
#include "4_arithmetics.h"
#include "5_conv.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;
using namespace std;

void get_histogram(Mat img) {
    imshow("Image", img);
    waitKey(0);
    int vector_size = img.rows * img.cols;
    //cout << "rows: " << img.rows << " cols: " << img.cols << endl;

    int* blue = new int[vector_size];
    int* green = new int[vector_size];
    int* red = new int[vector_size];

    int* output_blue = new int[256];
    int* output_green = new int[256];
    int* output_red = new int[256];

    mat_to_vec(img, blue, green, red);
    histogram_cuda(blue, green, red, img.rows, img.cols, output_blue, output_green, output_red);
    histDisplay(output_blue, "hist blue");
    histDisplay(output_green, "hist green");
    histDisplay(output_red, "hist red");

    /*show_vec(output[0], 256);
    show_vec(output[1], 256);
    show_vec(output[2], 256);

    int acc = 0;
    for (int i = 0; i < 256; i++)
        acc += output[0][i];
    cout << "total: " << acc;
    */
}

void equalize_hist(Mat img) {
    Mat grey;
    cvtColor(img, grey, COLOR_BGR2GRAY);
    int vector_size = grey.rows * grey.cols;
    int* data = new int[vector_size];
    //float* data_out = new float[256];
    int* hist = new int[256];
    int* new_img_vec = new int[vector_size];

       
    imshow("Original image", grey);
    waitKey(0);

    mat_to_vec(grey, data);
    equalize_hist_cuda(data, grey.rows, grey.cols, hist, new_img_vec);
    
    //histDisplay(hist, "hist");
    Mat new_img = vec_to_mat(new_img_vec, grey.rows, grey.cols);
    imshow("Image equalized", new_img);
    waitKey(0);
 
    //show_vec(new_img_vec, 256);
}

void aply_function(Mat img) {
    //int*** data = create_tensor_int(img.rows, img.cols, img.channels());
    //mat_to_tensor(img, data);
    //show_tensor(data, img.rows, img.cols, img.channels());

    imshow("Image", img);
    waitKey(0);
    int vector_size = img.rows * img.cols;
    //cout << "rows: " << img.rows << " cols: " << img.cols << endl;

    int* blue = new int[vector_size];
    int* green = new int[vector_size];
    int* red = new int[vector_size];

    int* output_blue = new int[vector_size];
    int* output_green = new int[vector_size];
    int* output_red = new int[vector_size];

    mat_to_vec(img, blue, green, red);
    global_func_cuda(blue, green, red, img.rows, img.cols, output_blue, output_green, output_red);
    show_vec(output_blue, 100);

    Mat new_img = vec_to_mat(output_blue, output_green, output_red, img.rows, img.cols);
    imshow("Image global", new_img);
    waitKey(0);
}

void arithmetics(Mat img_1, Mat img_2) {      
    int vector_size = img_1.rows * img_1.cols;
    //cout << "rows: " << img.rows << " cols: " << img.cols << endl;

    int* img_vec_1 = new int[vector_size * 3];
    int* img_vec_2 = new int[vector_size * 3];
    int* output_1 = new int[vector_size * 3];
    int* output_2 = new int[vector_size * 3];
    int* output_3 = new int[vector_size * 3];
    int* output_4 = new int[vector_size * 3];

    mat_to_vec_1d(img_1, img_vec_1);
    mat_to_vec_1d(img_2, img_vec_2);
    //show_vec(img_vec_1, 100);      
    
    arithmetics_cuda(img_vec_1, img_vec_2, img_1.rows, img_1.cols, output_1, output_2, output_3, output_4);
    //show_vec(output_1, 100);
    Mat img_sum = vec_1d_to_mat(output_1, img_1.rows, img_1.cols);
    Mat img_mul = vec_1d_to_mat(output_2, img_1.rows, img_1.cols);
    Mat img_sub = vec_1d_to_mat(output_3, img_1.rows, img_1.cols);
    Mat img_div = vec_1d_to_mat(output_4, img_1.rows, img_1.cols);

    imshow("Image 1", img_1);    waitKey(0);
    imshow("Image 2", img_2);    waitKey(0);    
    imshow("sum", img_sum);    waitKey(0);      
    imshow("mul", img_mul);    waitKey(0);
    imshow("sub", img_sub);    waitKey(0);
    imshow("div", img_div);    waitKey(0);
}

void convolutions(Mat img, float *kernel, int kernel_rows, int kernel_cols) {
    int vector_size = img.rows * img.cols;
    int* img_vec = new int[vector_size * 3];    
    int* output = new int[vector_size * 3];
    
    mat_to_vec_1d(img, img_vec);         
    conv_cuda(img_vec, kernel, img.rows, img.cols, img.channels(), kernel_rows, kernel_cols, output);
    Mat img_result = vec_1d_to_mat(output, img.rows, img.cols);    

    imshow("Image", img);               waitKey(0);
    imshow("New Image", img_result);    waitKey(0);
}

int main() {    
    // PREGUNTA 1
    Mat img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    //get_histogram(img);    

    // PREGUNTA 2
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena2.jpg");
    //equalize_hist(img);

    // PREGUNTA 3
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    //aply_function(img); 

    // PREGUNTA 4
    Mat img_1 = imread("D:\\CUDA\\HelloCUDAopenCV\\leon.jpg");
    Mat img_2 = imread("D:\\CUDA\\HelloCUDAopenCV\\aqp.jpg");
    Mat img_3 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_10.jpg");
    Mat img_4 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_11.jpg");
    //arithmetics(img_1, img_2);
    //arithmetics(img_3, img_4);

    // PREGUNTA 5
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    float kernel[] = {
            0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04
    }; // mean
    convolutions(img, kernel, 5, 5);

    img = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_10.jpg");
    float kernel_sobel[] = {
            2, 1, 0, -1, -2,
            2, 1, 0, -1, -2,
            4, 2, 0, -2, -4,
            2, 1, 0, -1, -2,
            2, 1, 0, -1, -2,
    };
    convolutions(img, kernel_sobel, 5, 5);
    return 0;

}


