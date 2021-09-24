#ifndef __UTILS_H__
#define __UTILS_H__


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


void show_vec(int* arr, int len) {
    for (int i = 0; i < len; i++)
        cout << arr[i] << ", ";
    cout << endl;
}

void show_vec(float* arr, int len) {
    for (int i = 0; i < len; i++)
        cout << arr[i] << ", ";
    cout << endl;
}

void show_tensor(int*** data, int rows, int cols, int channels) {
    for (int i = 0; i < rows; i++) {  
        for (int j = 0; j < cols; j++) {
            cout << "[";
            for (int k = 0; k < channels; k++) {
                cout<<data[i][j][k]<<" ";
            }
            cout << "] ; ";
        }
        cout << endl;
    }
}

int*** create_tensor_int(int rows, int cols, int channels) {
    int*** data = new int**[rows];
    for (int i = 0; i < rows; i++) {
        int** temp = new int*[cols];
        data[i] = temp;
           
        for (int j = 0; j < channels; j++) {
            int* temp2 = new int[channels];
            data[i][j] = temp2;
        }
    }

    return data;
}

void mat_to_tensor(Mat img, int*** data) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            data[i][j][0] = (int)(img.at<Vec3b>(i, j)[0]);
            data[i][j][1] = (int)(img.at<Vec3b>(i, j)[1]);
            data[i][j][2] = (int)(img.at<Vec3b>(i, j)[2]);
        }
    }
}

void mat_to_vec(Mat img, int* c_data) {    
    int index = 0;
    /*for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            c_data[index] = (int)(img.at<uchar>(i, j));
            index += 1;
        }
    }*/
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            c_data[index] = (int)(img.at<uchar>(i, j));
            index += 1;
        }
    }
}

Mat vec_to_mat(int* c_data, int rows, int cols) {
    Mat image(rows, cols, CV_8UC1, cv::Scalar(0));
    
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image.at<uchar>(i,j) = c_data[index];
            //image.at<uchar>(i, j) = 255;
            //cout << image.at<uchar>(i, j);
            //cout << 0;
            index += 1;
        }
    }
    return image;
}

Mat vec_to_mat(int* blue, int* green, int* red, int rows, int cols) {
    Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image.at<Vec3b>(i, j)[0] = blue[index];
            image.at<Vec3b>(i, j)[1] = green[index];
            image.at<Vec3b>(i, j)[2] = red[index];
            index += 1;
        }
    }
    return image;
}

void mat_to_vec(Mat img, int* c_blue, int* c_green, int* c_red) {
    int vector_size = img.rows * img.cols;
    /*Mat img_vec = img.reshape(3, vector_size);

    for (int i = 0; i < img_vec.rows; i++) {
        c_blue[i] = (int)(img.at<Vec3b>(i)[0]);
        c_green[i] = (int)(img.at<Vec3b>(i)[1]);
        c_red[i] = (int)(img.at<Vec3b>(i)[2]);
        //cout << blue << " " << green << " " << red << " ";      
    }*/

    //cout << "channels " << img.channels();
    int index = 0;
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            c_blue[index] = (int)(img.at<Vec3b>(i, j)[0]);
            c_green[index] = (int)(img.at<Vec3b>(i, j)[1]);
            c_red[index] = (int)(img.at<Vec3b>(i, j)[2]);
            //cout << blue << " " << green << " " << red << " ";

            index += 1;
        }
    }
}

// convierte un vector de 1d (rows*cols*channles) a un Mat
Mat vec_1d_to_mat(int* data, int rows, int cols) {
    Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    int size = rows * cols;
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image.at<Vec3b>(i, j)[0] = data[index];
            image.at<Vec3b>(i, j)[1] = data[index + size];
            image.at<Vec3b>(i, j)[2] = data[index + size*2];
            index += 1;
        }
    }
    return image;
}

// convierte un Mat en un vector de enteros, este vector contine tambien todos los canales
void mat_to_vec_1d(Mat img, int* data) {
    int vector_size = img.rows * img.cols;    
    //cout << "rows:" << img.rows << "cols:" << img.cols << endl;

    //cout << "channels " << img.channels();
    int index = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            //cout << i << " " << j << " " << end;
            data[index] =                   (int)(img.at<Vec3b>(i, j)[0]);
            data[index + vector_size] =     (int)(img.at<Vec3b>(i, j)[1]);
            data[index + vector_size*2] =   (int)(img.at<Vec3b>(i, j)[2]);
            index += 1;
        }
    }
}

void histDisplay(int histogram[], const char* name)
{
    int hist[256];
    for (int i = 0; i < 256; i++)
    {
        hist[i] = histogram[i];
    }
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    int max = hist[0];
    for (int i = 1; i < 256; i++) {
        if (max < hist[i]) {
            max = hist[i];
        }
    }

    // normalize the histogram between 0 and histImage.rows
    for (int i = 0; i < 256; i++)
    {
        hist[i] = ((double)hist[i] / max) * histImage.rows;
    }


    // draw the intensity line for histogram
    for (int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - hist[i]), Scalar(0, 0, 0), 1, 8, 0);
    }

    // display histogram
    namedWindow(name);
    imshow(name, histImage);
    waitKey(0);
}

#endif