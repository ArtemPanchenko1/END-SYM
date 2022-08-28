#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>
#include <float.h>
#include "m_func.h"

__global__ void seq_minmaxKernel(float* max, float* min, const float* __restrict__ a, const unsigned int n) {
	__shared__ float maxtile[SMEMDIM];
	__shared__ float mintile[SMEMDIM];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		maxtile[tid] = a[i];
		mintile[tid] = a[i];
	}
	else
	{
		maxtile[tid] = -FLT_MAX;
		mintile[tid] =  FLT_MAX;
	}
		__syncthreads();

		//sequential addressing by reverse loop and thread-id based indexing
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				//printf("FMM %u %u\n", tid, s);
				if (maxtile[tid + s] > maxtile[tid])
					maxtile[tid] = maxtile[tid + s];
				if (mintile[tid + s] < mintile[tid])
					mintile[tid] = mintile[tid + s];
			}
			__syncthreads();
		}
	

	if (tid == 0) {
		//printf("FMaxMin %u | %e %e\n", blockIdx.x, mintile[0], maxtile[0]);
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
		
}

__global__ void seq_finalminmaxKernel(float* max, float* min, float* __restrict__ mimmax, const unsigned int offset, const unsigned int n) {
	__shared__ float maxtile[SMEMDIM];
	__shared__ float mintile[SMEMDIM];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		maxtile[tid] = max[i];
		mintile[tid] = min[i];
	}
	else
	{
		maxtile[tid] = -FLT_MAX;
		mintile[tid] =  FLT_MAX;
	}
	
	__syncthreads();

	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		//printf("FMaxMin %u | %e %e\n", blockIdx.x, mintile[0], maxtile[0]);
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
		mimmax[offset] = mintile[0];
		mimmax[offset+1] = maxtile[0];
	}
}