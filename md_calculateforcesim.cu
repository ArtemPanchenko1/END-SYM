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

//__constant__ char IM[2 * IMatrixSize];

__device__ float dd_nonlinearC(const double rmada, const double c)
{
	//if (rmada < -1.0 + 1e-7)std::cerr << "Error!\n";
	if (rmada < -0.9)return c * (0.01 - 1000.1 * (rmada + 0.9));
	if (rmada < -0.1)return c * 0.01;
	if (rmada < 0.0)return c * (0.01 + 9.9 * (rmada + 0.1));
	return c * __expf(2.3025851 * rmada);
}

__global__ void d_calculateForcesI(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float P_c)
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
		//if (idx == 4818)	printf("F %u | %e %e\n", idx, F[idx], F[idx + n]);
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_calculateForcesIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Fbound, const unsigned int n, const unsigned int ni, const float P_c)
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
			drm = __fsqrt_rn(drx * drx + dry * dry);
			_1d_drm = __frcp_rn(drm);

			c = P_c;

#ifdef pre_nonlinearC
			rmada = drm * _1d_a - 1.0f;
			c = dd_nonlinearC(rmada, P_c);
#endif


			//fsumx = c * drx * (_1d_a - _1d_drm);
			//fsumy = c * dry * (_1d_a - _1d_drm);
			//Fbound[k] = __fsqrt_rn(fsumx * fsumx + fsumy * fsumy);
			Fbound[k] = c * (drm * _1d_a - 1.0f);
			//if(fsumx* fsumx+ fsumy* fsumy>1e-10)
			//printf("F %i %i %i | %e %e | %e %e | %e %e\n", idx, i, j, c * drx * (_1d_a - _1d_drm), c * dry * (_1d_a - _1d_drm), U[i], U[i+n], U[j], U[j + n]);
			//printf("FF %e %e %e \n", drx, (1.0 - P_a * _1d_drm), fx[1]);
			//printf("F %u %u | %e %e %e | %e %e | %e %e\n",  j1, idx, fx[1], fy[1], drm, U[j1], U[idx], U[j1 + n], U[idx + n]);

		}
		//F[idx] = fsumx;
		//F[idx + n] = fsumy;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_calculateEnergyIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Ebound, const unsigned int n, const unsigned int ni, const float P_c)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drx, dry, drm, rmada, _1d_a, c, drmma;//, _1d_drm
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		//fsumx = 0;
		//fsumy = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			drx = Ir0[k] + U[j] - U[i];
			dry = Ir0[k + ni] + U[j + n] - U[i + n];
			_1d_a = Ir0[k + 2 * ni];
			drm = __fsqrt_rn(drx * drx + dry * dry);
			//_1d_drm = __frcp_rn(drm);

			c = P_c;

#ifdef pre_nonlinearC
			rmada = drm * _1d_a - 1.0f;
			c = dd_nonlinearC(rmada, P_c);
#endif

			rmada = drm - __frcp_rn(_1d_a);
			//fsumx = c * drx * (_1d_a - _1d_drm);
			//fsumy = c * dry * (_1d_a - _1d_drm);
			//Fbound[k] = __fsqrt_rn(fsumx * fsumx + fsumy * fsumy);
			Ebound[k] = c * _1d_a * rmada * rmada;
			//if(fsumx* fsumx+ fsumy* fsumy>1e-10)
			//printf("F %i %i %i | %e %e | %e %e | %e %e\n", idx, i, j, c * drx * (_1d_a - _1d_drm), c * dry * (_1d_a - _1d_drm), U[i], U[i+n], U[j], U[j + n]);
			//printf("FF %e %e %e \n", drx, (1.0 - P_a * _1d_drm), fx[1]);
			//printf("F %u %u | %e %e %e | %e %e | %e %e\n",  j1, idx, fx[1], fy[1], drm, U[j1], U[idx], U[j1 + n], U[idx + n]);

		}
		//F[idx] = fsumx;
		//F[idx + n] = fsumy;
		idx += blockDim.x * gridDim.x;
	}
}


__global__ void d_calculateUIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Ubound, const unsigned int n, const unsigned int ni, const float P_c)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float dr;//, _1d_drm
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		//fsumx = 0;
		//fsumy = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];
#ifdef pre_SaveLammpsUx
			dr = 0.5f * (U[j] + U[i]);
			Ubound[k] = dr;
#endif // pre_SaveLammpsUx
		}		
		idx += blockDim.x * gridDim.x;
	}
}