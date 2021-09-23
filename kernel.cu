
#include "1_histogram.h"
#include "2_equalize.h"
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

int main() {    
    // PREGUNTA 1
    Mat img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    //get_histogram(img);    

    // PREGUNTA 2
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena2.jpg");
    //equalize_hist(img);

    
    return 0;

}


