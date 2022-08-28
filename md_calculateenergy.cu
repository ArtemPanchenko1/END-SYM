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

//#include <cudpp.h>
//#include <cudpp_plan.h>


__global__ void d_calculateKineticEnergy(	
	const float* __restrict__ VU, const float* __restrict__ VV,
	float* __restrict__ aEk, unsigned int offset,
	unsigned int n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < n)
	{		
		aEk[offset +         idx] += VU[idx] *     VU[idx];
		aEk[offset +     n + idx] += VU[idx + n] * VU[idx + n];
		aEk[offset + 2 * n + idx] += VV[idx] *     VV[idx];
		aEk[offset + 3 * n + idx] += VV[idx + n] * VV[idx + n];
	}	
}

__global__ void d_calculateKineticEnergy_precision(
	const float* __restrict__ VU, const float* __restrict__ VV,
	const float* __restrict__ FU, const float* __restrict__ FV,
	float* __restrict__ aEk, unsigned int offset,
	unsigned int n, const float P_dtm)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n)
	{
		float uvx = VU[idx] + 0.5 * FU[idx] * P_dtm;
		float uvy = VU[idx + n] + 0.5 * FU[idx + n] * P_dtm;
		float vvx = VV[idx] + 0.5 * FV[idx] * P_dtm;
		float vvy = VV[idx + n] + 0.5 * FV[idx + n] * P_dtm;
		aEk[offset + idx] += uvx * uvx;
		aEk[offset + n + idx] += uvy * uvy;
		aEk[offset + 2 * n + idx] += vvx * vvx;
		aEk[offset + 3 * n + idx] += vvy * vvy;
	}
}

