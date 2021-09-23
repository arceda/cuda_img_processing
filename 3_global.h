#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "math.h"



using namespace cv;
using namespace std;

//void global_func_cuda(int**, int, int, int**);
void global_func_cuda(int*, int*, int*, int, int, int*, int*, int*);

#endif

