#ifndef __HIST_H__
#define __HIST_H__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>



using namespace cv;
using namespace std;

void histogram_cuda(int*, int*, int*, int, int, int*, int*, int*);


#endif

