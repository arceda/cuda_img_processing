#pragma once
#ifndef __OPTIC_FLOW_H__
#define __OPTIC_FLOW_H__

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "math.h"

using namespace std;
using namespace std::chrono;

using namespace cv;
using namespace std;

//void global_func_cuda(int**, int, int, int**);
void optic_flow_cuda(int* img, int*, int rows, int cols, int channels, int*);
void optical_flow_analysis_cuda(Mat mat1, Mat mat2, int iterations = 100, int avg_window = 5, double alpha = 1); // 100 iter
Mat compute_color_cuda(Mat U, Mat V);
Mat make_color_wheel_cuda();

#endif


