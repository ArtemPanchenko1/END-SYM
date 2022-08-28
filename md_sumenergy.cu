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

__device__ float d_calculatePEnergyIk(const int* __restrict__ In, const float* __restrict__ Ir0, const float* __restrict__ U, const unsigned int n, const unsigned int ni, const int ks, const int kmax, const float P_c)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k;
	float drx, dry, drm, rmada, _1d_a, c, energy;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	energy = 0;
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
		energy += c * _1d_a * rmada * rmada;
		//if(fsumx* fsumx+ fsumy* fsumy>1e-5)
		//if(idx==4818)
		//	printf("F %i %i %i | %e %e %e | %e %e | %e %e\n", idx, i, j, (_1d_a - _1d_drm),(_1d_a - _1d_drm), _1d_a, U[i], U[i+n], U[j], U[j + n]);
		//printf("FF %e %e %e \n", drx, (1.0 - P_a * _1d_drm), fx[1]);
		//printf("F %u %u | %e %e %e | %e %e | %e %e\n",  j1, idx, fx[1], fy[1], drm, U[j1], U[idx], U[j1 + n], U[idx + n]);

	}
	return energy;
}

__device__ float d_calculateKEnergyi(const float* __restrict__ V, const unsigned int i, const unsigned int n, const float m)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	//int i, j, k;
	//float drx, dry, drm, rmada, _1d_a, c, energy;
	//unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	float energy;
	energy = 0.5 * m * (V[i] * V[i] + V[i + n] * V[i + n]);
	return energy;
}

__global__ void d_getEnergyEntire(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ _1d_Mass, const float* __restrict__ U, const float* __restrict__ V, float* Esum, const unsigned int n, const unsigned int ni, const float P_c)
{
	// static shared memory
	__shared__ float s_mem[2 * SMEMDIM];

	// set thread ID
	// global index, 4 blocks of input data processed at a time
	unsigned int tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 4 + threadIdx.x, i;
	int ks, kmax;
	// unrolling 4 blocks
	float ek = 0, ep = 0, m;

	// boundary check
	if (idx + 3 * blockDim.x < n)
	{
		float t_ek0 = 0, t_ek1 = 0, t_ek2 = 0, t_ek3 = 0;
		float t_ep0 = 0, t_ep1 = 0, t_ep2 = 0, t_ep3 = 0;

		i = idx;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + 2 * blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + 3 * blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy	
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);		
		ek = t_ek0 + t_ek1 + t_ek2 + t_ek3;
		ep = t_ep0 + t_ep1 + t_ep2 + t_ep3;
	}
	else if (idx + 2 * blockDim.x < n)
	{
		float t_ek0 = 0, t_ek1 = 0, t_ek2 = 0;
		float t_ep0 = 0, t_ep1 = 0, t_ep2 = 0;

		i = idx;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + 2 * blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);		
		ek = t_ek0 + t_ek1 + t_ek2;
		ep = t_ep0 + t_ep1 + t_ep2;
	}
	else if (idx + blockDim.x < n)
	{
		float t_ek0 = 0, t_ek1 = 0;
		float t_ep0 = 0, t_ep1 = 0;

		i = idx;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		i = idx + blockDim.x;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);		
		ek = t_ek0 + t_ek1;
		ep = t_ep0 + t_ep1;
	}
	else if (idx < n)
	{
		float t_ek0 = 0;
		float t_ep0 = 0;

		i = idx;
		ks = ShIn[i];
		kmax = ks + ShIn[i + n];
		t_ep0 = d_calculatePEnergyIk(In, Ir0, U, n, ni, ks, kmax, P_c);
#ifdef pre_CalcFullKEnergy
		m = __frcp_rn(_1d_Mass[i]);
		t_ek0 = d_calculateKEnergyi(V, i, n, m);
#endif // pre_CalcFullKEnergy
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);		
		ek = t_ek0;
		ep = t_ep0;
	}/**/

	//if(idx + 5 * blockDim.x >4619700)
	//   printf("TT %i %i %i %f %i\n", tid, idx, blockIdx.x, n);
	//if (ns>1e-3f)
	//   printf("TT %i %i %f\n", tid, idx, ns);
	s_mem[tid] = ep;
	s_mem[tid + SMEMDIM] = ek;
	__syncthreads();

	//if(idx==0)
	//	printf("TT %i %f %f %i %i\n", tid, s_ek, e_ek, s_n, e_n);

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 64];
	}

	__syncthreads();
	/*if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		for (int i = 0; i < SMEMDIM; ++i)
			printf("GM %i %e\n", i, smem[i + 3 * SMEMDIM]);
	}/**/

	// unrolling warp
	if (tid < 32)
	{
		volatile float* vsmem = s_mem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 2];
		vsmem[tid] += vsmem[tid + 1];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		Esum[blockIdx.x] = s_mem[0];
		Esum[blockIdx.x + gridDim.x] = s_mem[SMEMDIM];
		//if (smem[tid + 3 * SMEMDIM] > 1e-3f)
		//	printf("TT %i %i %f\n", tid, idx, smem[tid + 3 * SMEMDIM]);
		//if (smem[3 * SMEMDIM] > 1e-3f)
		//printf("T %i %f\n", blockIdx.x, gridDim.x, smem[3 * SMEMDIM]);
	}/**/
}