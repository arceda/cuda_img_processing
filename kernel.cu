
#include "1_histogram.h"
#include "2_equalize.h"
#include "3_global.h"
#include "4_arithmetics.h"
#include "5_conv.h"
#include "6_zoon.h"
#include "7_geometrics.h"
#include "8_search.h"
#include "9_optic_flow.h"

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
    imshow("1-Image", img);
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
    histDisplay(output_blue, "1-hist blue");
    histDisplay(output_green, "1-hist green");
    histDisplay(output_red, "1-hist red");

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

       
    imshow("2-Original image", grey);
    waitKey(0);

    mat_to_vec(grey, data);
    equalize_hist_cuda(data, grey.rows, grey.cols, hist, new_img_vec);
    
    //histDisplay(hist, "hist");
    Mat new_img = vec_to_mat(new_img_vec, grey.rows, grey.cols);
    imshow("2-Image equalized", new_img);
    waitKey(0);
 
    //show_vec(new_img_vec, 256);
}

void aply_function(Mat img) {
    //int*** data = create_tensor_int(img.rows, img.cols, img.channels());
    //mat_to_tensor(img, data);
    //show_tensor(data, img.rows, img.cols, img.channels());

    imshow("3-Image", img);
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
    imshow("3-Image global", new_img);
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

    imshow("4-Image 1", img_1);    waitKey(0);
    imshow("4-Image 2", img_2);    waitKey(0);    
    imshow("4-sum", img_sum);    waitKey(0);      
    imshow("4-mul", img_mul);    waitKey(0);
    imshow("4-sub", img_sub);    waitKey(0);
    imshow("4-div", img_div);    waitKey(0);
}

void convolutions(Mat img, float *kernel, int kernel_rows, int kernel_cols) {
    int vector_size = img.rows * img.cols;
    int* img_vec = new int[vector_size * 3];    
    int* output = new int[vector_size * 3];
    
    mat_to_vec_1d(img, img_vec);         
    conv_cuda(img_vec, kernel, img.rows, img.cols, img.channels(), kernel_rows, kernel_cols, output);
    Mat img_result = vec_1d_to_mat(output, img.rows, img.cols);    

    imshow("5-Image original", img);           waitKey(0);
    imshow("5-After convolution", img_result);    waitKey(0);
}

void zoon(Mat img) {
    Mat grey;
    cvtColor(img, grey, COLOR_BGR2GRAY);
    int vector_size = img.rows * img.cols;
    //int* img_vec = new int[vector_size * img.channels()];
    //int* output = new int[vector_size * 2 * img.channels()];
    //mat_to_vec_1d(img, img_vec);
    //zoon_cuda(img_vec, img.rows, img.cols, img.channels(),  output);
    //Mat img_result = vec_1d_to_mat(output, img.rows, img.cols);


    /*int length = 9;
    int rows = 3; int cols = 3;
    int* img_vec = new int[length];
    img_vec[0] = 100; img_vec[1] = 200; img_vec[2] = 300; img_vec[3] = 400; img_vec[4] = 500;
    img_vec[5] = 600; img_vec[6] = 700; img_vec[7] = 800; img_vec[8] = 900;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << img_vec[i * cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    int* output = new int[length * 4];
    zoon_cuda(img_vec, rows, cols, 1, output);
    
    cout << endl << "output\n";
    for (int i = 0; i < rows*2; i++) {
        for (int j = 0; j < cols*2; j++) {
            cout << output[i * cols * 2 + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    */



    int* img_vec = new int[vector_size];
    int* output = new int[vector_size * 4];
    mat_to_vec(grey, img_vec);
    zoon_cuda(img_vec, grey.rows, grey.cols, grey.channels(),  output);
    Mat img_result = vec_to_mat(output, grey.rows*2, grey.cols*2);

    show_vec(img_vec, 20);
    cout << endl;
    show_vec(output, 400);

    imshow("6-Image low resolution", grey);                    waitKey(0);
    imshow("6-Image high resolution", img_result);            waitKey(0);
    
}

void geometrics(Mat img, int* x_1, int* y_1, int* x_2, int* y_2) {
    float* M = new float[6]; // coeficient of M matrix for affine transformation
    AffineSolver(M, x_1, y_1, x_2, y_2);
    PrintMatrix(M);    

    Mat img_input = img;
    //cvtColor(img, img_input, COLOR_BGR2GRAY);
    int vector_size = img_input.rows * img_input.cols * img_input.channels();

    int* img_vec = new int[ vector_size ];
    int* output = new int[ vector_size ];
    mat_to_vec_1d(img_input, img_vec);
    geometrics_cuda(img_vec, M, img_input.rows, img_input.cols, img_input.channels(), output);
    Mat img_result = vec_1d_to_mat(output, img_input.rows, img_input.cols);

    imshow("7-Original", img_input);    waitKey(0);
    imshow("7-Affine", img_result);     waitKey(0);
}

void search_pattern(Mat img, Mat img_pattern) {    
    Mat img_input = img;
    //cvtColor(img, img_input, COLOR_BGR2GRAY);
    int vector_size = img_input.rows * img_input.cols * img_input.channels();
    int vector_pattern_size = img_pattern.rows * img_pattern.cols * img_pattern.channels();

    int* img_vec = new int[vector_size];
    int* img_pattern_vec = new int[vector_pattern_size];
    int* output = new int[vector_size];
    int num_boxes;
    mat_to_vec_1d(img_input, img_vec);
    mat_to_vec_1d(img_pattern, img_pattern_vec);
    search_pattern_cuda(    img_vec, img_pattern_vec, 
                            img_input.rows, img_input.cols, 
                            img_pattern.rows, img_pattern.cols,
                            img_input.channels(), output, &num_boxes);
    Mat img_result = vec_1d_to_mat(output, img_input.rows, img_input.cols);

    // find contours ////////////////////////////////////////////////////////////////////////////////////////
    Mat new_output_img = img_input.clone();
    Mat gray_result, binary_result;
    cvtColor(img_result, gray_result, COLOR_BGR2GRAY);
    threshold(gray_result, binary_result, 100, 255, THRESH_BINARY);

    RNG rng(12345);
    vector<vector<Point> > contours;
    findContours(binary_result, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f>centers(contours.size());
    vector<float>radius(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
        minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
    }
    Mat drawing = Mat::zeros(binary_result.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        //drawContours(drawing, contours_poly, (int)i, color);
        //rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
        //circle(drawing, centers[i], (int)radius[i], color, 2);
        int width = boundRect[i].br().x - boundRect[i].tl().x;
        int height = boundRect[i].br().y - boundRect[i].tl().y;        
        int area = width * height;
        cout << area <<endl;

        int center_x = boundRect[i].tl().x + width / 2;
        int center_y = boundRect[i].tl().y + height / 2;
        Point new_potin_tl(center_x - img_pattern.cols/2, center_y - img_pattern.rows/2);
        Point new_potin_br(center_x + img_pattern.cols / 2, center_y + img_pattern.rows / 2);

        if (area > 150) {
            //rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
            //rectangle(new_output_img, boundRect[i].tl(), boundRect[i].br(), color, 2);
            rectangle(new_output_img, new_potin_tl, new_potin_br, cv::Scalar(0, 0, 255), 2);
        }
        
        
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

   

    imshow("8-Image original", img_input);    waitKey(0);
    imshow("8-Pattern", img_pattern);         waitKey(0);
    imshow("8-Patterns finded", new_output_img);  waitKey(0);
}

void optic_flow(Mat img_1, Mat img_2) {

    imshow("9-Frame 1", img_1);    waitKey(0);
    imshow("9-Frame 2", img_2);    waitKey(0);

    //optical_flow_analysis(img_1, img_2, 20);
    optical_flow_analysis_cuda(img_1, img_2, 20);


    /*Mat gray_1, gray_2;
    cvtColor(img_1, gray_1, COLOR_BGR2GRAY);
    cvtColor(img_2, gray_2, COLOR_BGR2GRAY);
    int vector_size = gray_1.rows * gray_1.cols * gray_1.channels();

    int* img_vec_1 = new int[vector_size];
    int* img_vec_2 = new int[vector_size];    
    int* output = new int[gray_1.rows * gray_1.cols * gray_1.channels() * 2];
    
    mat_to_vec(gray_1, img_vec_1);
    mat_to_vec(gray_2, img_vec_2);
    optic_flow_cuda(img_vec_1, img_vec_2,
        gray_1.rows, gray_1.cols, gray_1.channels(), output);
    Mat img_result = vec_to_mat(output, gray_1.rows, gray_1.cols);

   
    imshow("Frame 1", img_1);    waitKey(0);
    imshow("Frame 2", img_2);    waitKey(0);    
    imshow("Optic flow", img_result);  waitKey(0);*/
}

int main() {    
    // PREGUNTA 1 /////////////////////////////////////////////////////////////////////////////
    Mat img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    get_histogram(img);    

    // PREGUNTA 2 /////////////////////////////////////////////////////////////////////////////
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena2.jpg");
    equalize_hist(img);

    // PREGUNTA 3 /////////////////////////////////////////////////////////////////////////////
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\lena.jpg");
    aply_function(img); 

    // PREGUNTA 4
    Mat img_1 = imread("D:\\CUDA\\HelloCUDAopenCV\\leon.jpg");
    Mat img_2 = imread("D:\\CUDA\\HelloCUDAopenCV\\aqp.jpg");
    Mat img_3 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_10.jpg");
    Mat img_4 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_11.jpg");
    arithmetics(img_1, img_2);
    arithmetics(img_3, img_4);

    // PREGUNTA 5 /////////////////////////////////////////////////////////////////////////////
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
    

    // PREGUNTA 6 /////////////////////////////////////////////////////////////////////////////
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\orange.jpg");
    zoon(img);

    // PREGUNTA 7 /////////////////////////////////////////////////////////////////////////////
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\orange.jpg");
    int x_1[] = { 0, 0, 10 };   int y_1[] = { 10, 0, 0 };    int x_2[] = { 0, 0, 5 };    int y_2[] = { 5, 0, 0 };
    int x_3[] = { 50, 200, 50 };   int y_3[] = { 50, 50, 200 };    int x_4[] = { 10, 200, 100 };    int y_4[] = { 100, 50, 250 };    
    geometrics(img, x_1, y_1, x_2, y_2);
    geometrics(img, x_3, y_3, x_4, y_4);

    // PREGUNTA 8 /////////////////////////////////////////////////////////////////////////////
    img = imread("D:\\CUDA\\HelloCUDAopenCV\\clouds.jpg");
    Mat img_pattern = imread("D:\\CUDA\\HelloCUDAopenCV\\cloud.jpg");
    search_pattern(img, img_pattern);

    // PREGUNTA 9 /////////////////////////////////////////////////////////////////////////////
    img_1 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_0.jpg");
    img_2 = imread("D:\\CUDA\\HelloCUDAopenCV\\sub_1.jpg");
    optic_flow(img_1, img_2);

    return 0;

}


