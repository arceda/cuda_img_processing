#ifndef __EQUALIZE_H__
#define __EQUALIZE_H__

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

void sum_probabilities(float*, float*, int);
void equalize_hist_cuda(int*, int, int, int*, int*);


#endif

