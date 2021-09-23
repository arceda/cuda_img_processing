﻿
#include "1_histogram.h"
#include "2_equalize.h"
#include "3_global.h"
#include "4_arithmetics.h"
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
    //int*** data = create_tensor_int(img.rows, img.cols, img.channels());
    //mat_to_tensor(img, data);
    //show_tensor(data, img.rows, img.cols, img.channels());

    imshow("Image 1", img_1);
    imshow("Image 2", img_2);
    waitKey(0);
    int vector_size = img_1.rows * img_1.cols;
    //cout << "rows: " << img.rows << " cols: " << img.cols << endl;

    int* blue_1 = new int[vector_size];
    int* green_1 = new int[vector_size];
    int* red_1 = new int[vector_size];
    int* blue_2 = new int[vector_size];
    int* green_2 = new int[vector_size];
    int* red_2 = new int[vector_size];

    int* output_blue_1 = new int[vector_size];
    int* output_green_1 = new int[vector_size];
    int* output_red_1 = new int[vector_size];
    int* output_blue_2 = new int[vector_size];
    int* output_green_2 = new int[vector_size];
    int* output_red_2 = new int[vector_size];

    mat_to_vec(img_1, blue_1, green_1, red_1);
    mat_to_vec(img_2, blue_2, green_2, red_2);
    arithmetics_cuda(blue_1, green_1, red_1, img_1.rows, img_1.cols, output_blue_1, output_green_1, output_red_1);
    show_vec(output_blue_1, 100);

    Mat new_img = vec_to_mat(output_blue_1, output_green_1, output_red_1, img_1.rows, img_1.cols);
    imshow("sum", new_img);
    waitKey(0);
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
    arithmetics(img_1, img_2);


    return 0;

}


