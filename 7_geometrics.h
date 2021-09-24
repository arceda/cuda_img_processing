#pragma once
#ifndef __GEOMETRICS_H__
#define __GEOMETRICS_H__

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
void geometrics_cuda(int* img, float* M, int rows, int cols, int channels, int* output);

#endif


