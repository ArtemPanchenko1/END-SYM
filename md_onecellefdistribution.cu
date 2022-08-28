#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
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

__global__ void d_distrubuteBoundToCircle(const int* __restrict__ In, const float* __restrict__ U, const float* __restrict__ RU0, const float* __restrict__ EFbound, const float* __restrict__ EFbound0, float* CEF, const unsigned int n, const unsigned int ni, const unsigned int nT, const double X0, const double Y0, const double _1d_DR)
{
	__shared__ float s_mem[2*NumberCircles];

	int i, j, ir;
	float rcx, rcy, r, EF, uu;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	while (idx < ni)
	{
		//if (idx == 0)printf("F0I %i %i %i\n", idx, ni, n);
		
		if (threadIdx.x < 2*NumberCircles)
			s_mem[threadIdx.x] = 0;

		i = In[idx];
		j = In[idx + ni];
#ifndef pre_OneCellMaterialdistribution		
		rcx = 0.5f * (RU0[i] + RU0[j]) - X0;
		rcy = 0.5f * (RU0[i + n] + RU0[j + n]) - Y0;
#endif // !pre_OneCellMaterialdistribution
#ifdef pre_OneCellMaterialdistribution
		rcx = 0.5f * (RU0[i] + U[i] + RU0[j] + U[j]) - X0;
		rcy = 0.5f * (RU0[i + n] + U[i + n] + RU0[j + n] + U[j + n]) - Y0;
#endif // pre_OneCellMaterialdistribution
		r = __fsqrt_rn(rcx * rcx + rcy * rcy);
		ir = roundf(r * _1d_DR);

		//if (threadIdx.x == 0)
		//	printf("F1I %i %f\n", idx, EFbound[idx]);
		//if(threadIdx.x==0)
		//	printf("F1I %i %i %f | %f %f\n", idx, ir, r, rcx, rcy);
		//if(ir> NumberCircles-1)printf("T %i %f\n", ir, r);

		__syncthreads();
		//if (idx == 0)printf("F0I %i %i %i\n", idx, ni, nT);
		//if (fabsf(ir-10)<0.1)
		{
			/*rcx = 0.5f * (U[i] + U[j]);
			rcy = 0.5f * (U[i + n] + U[j + n]);
			uu = __fsqrt_rn(rcx * rcx + rcy * rcy);
			atomicAdd(s_mem + ir, uu);/**/
			atomicAdd(s_mem + ir, fabsf(EFbound[idx] - EFbound0[idx]));
			//atomicAdd(s_mem + ir, r);
			atomicAdd(s_mem + ir + NumberCircles, 1.0f);
		}
		
		__syncthreads();
		//if (idx == 0 && nT>100)printf("F0I %i %i %i\n", idx, ni, n);
		if (threadIdx.x < NumberCircles-1)
		{
			/*if (s_mem[threadIdx.x + NumberCircles] > 1e-1)
				EF = __fdiv_rn(s_mem[threadIdx.x], s_mem[threadIdx.x + NumberCircles]);
			else
				EF = 0;/**/
			//if(threadIdx.x ==20)
			//	printf("T %i %i %i %i | %e %e\n", idx, 2 * NumberCircles, threadIdx.x, nT, s_mem[threadIdx.x], s_mem[threadIdx.x + NumberCircles]);
			atomicAdd(CEF + 2 * NumberCircles * nT + threadIdx.x, s_mem[threadIdx.x]);
			atomicAdd(CEF + 2 * NumberCircles * nT + NumberCircles + threadIdx.x, s_mem[threadIdx.x + NumberCircles]);
		}
			
		
		__syncthreads();
		//if (idx == 0)
		//	printf("F2I %f\n", CEF[NumberCircles * nT + 10]);/**/
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_calculateForcesI_EF(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ F, float* __restrict__ EFbound, const unsigned int n, const unsigned int ni, const float P_c)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drx, dry, drm, rmada, _1d_drm, _1d_a, c, fsumx, fsumy;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsumx = 0;
		fsumy = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			drx = Ir0[k] + U[j] - U[i];
			dry = Ir0[k + ni] + U[j + n] - U[i + n];
			_1d_a = Ir0[k + 2 * ni];
			c = P_c;

#ifdef pre_nonlinearC
			drm = __fsqrt_rn(drx * drx + dry * dry);
			_1d_drm = __frcp_rn(drm);
			rmada = drm * _1d_a - 1.0f;
			c = dd_nonlinearC(rmada, P_c);
#endif
#ifndef pre_nonlinearC
			_1d_drm = __frsqrt_rn(drx * drx + dry * dry);
#endif // !pre_nonlinearC		

			fsumx += c * drx * (_1d_a - _1d_drm);
			fsumy += c * dry * (_1d_a - _1d_drm);

#ifdef pre_OneCellEdistribution
			drm = __fsqrt_rn(drx * drx + dry * dry);
			rmada = drm - __frcp_rn(_1d_a);
			EFbound[k] = c * _1d_a * rmada * rmada;
#endif // pre_OneCellEdistribution
#ifdef pre_OneCellFdistribution
			drm = __fsqrt_rn(drx * drx + dry * dry);			
			EFbound[k] = c * drm * (_1d_a - _1d_drm);
#endif // pre_OneCellFdistribution
			//if(fsumx* fsumx+ fsumy* fsumy>1e-5)
			//if(idx==4822)
			//	printf("F %i %i %i | %e %e %e | %e %e | %e %e\n", idx, i, j, fsumx, fsumy, _1d_a, U[i], U[i+n], U[j], U[j + n]);
			//printf("FF %e %e %e \n", drx, (1.0 - P_a * _1d_drm), fx[1]);
			//printf("F %u %u | %e %e %e | %e %e | %e %e\n",  j1, idx, fx[1], fy[1], drm, U[j1], U[idx], U[j1 + n], U[idx + n]);

		}
		//if (fsumx * fsumx + fsumy * fsumy > 1e-5)

		//if(idx==4818)
		//	printf("F %u | %e %e %e\n", idx, fsumx, fsumy, fsumx * fsumx + fsumy * fsumy);

		F[idx] = fsumx;
		F[idx + n] = fsumy;
		idx += blockDim.x * gridDim.x;
	}
}