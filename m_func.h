#pragma once
#include <cuda.h>
#include <vector_functions.h>
#include <curand.h>
__global__ void seq_minmaxKernel(float* max, float* min, const float* __restrict__ a, const unsigned int n);
__global__ void seq_finalminmaxKernel(float* max, float* min, float* __restrict__ mimmax, const unsigned int offset, const unsigned int n);