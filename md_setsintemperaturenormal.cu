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

__global__ void d_getSinVelocityEntire(
	const float* __restrict__ i_r,
	const float* __restrict__ i_v,
	float* __restrict__ o_D,	
	unsigned int n)
{
	// static shared memory
	__shared__ float s_mem[4 * SMEMDIM];

	// set thread ID
	unsigned int tid = threadIdx.x;

	// global index, 4 blocks of input data processed at a time
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	// unrolling 4 blocks
	float vx = 0, vy = 0, ek = 0, _n = 0;

	// boundary check
	if (idx + 3 * blockDim.x < n)
	{
		float vx0 = 0, vy0 = 0;
		float vx1 = 0, vy1 = 0;
		float vx2 = 0, vy2 = 0;
		float vx3 = 0, vy3 = 0;		
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);

		vx0 = i_v[idx];					 vy0 = i_v[idx + n];				 
		vx1 = i_v[idx + blockDim.x];	 vy1 = i_v[idx + blockDim.x + n];	 
		vx2 = i_v[idx + 2 * blockDim.x]; vy2 = i_v[idx + 2 * blockDim.x + n];
		vx3 = i_v[idx + 3 * blockDim.x]; vy3 = i_v[idx + 3 * blockDim.x + n];
				
		vx = vx0 + vx1 + vx2 + vx3;
		vy = vy0 + vy1 + vy2 + vy3;
		ek = vx0 * vx0 + vx1 * vx1 + vx2 * vx2 + vx3 * vx3 +
			 vy0 * vy0 + vy1 * vy1 + vy2 * vy2 + vy3 * vy3;
		_n = 4;
		
	}
	else if (idx + 2 * blockDim.x < n)
	{
		float vx0 = 0, vy0 = 0;
		float vx1 = 0, vy1 = 0;
		float vx2 = 0, vy2 = 0;
		
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);
		vx0 = i_v[idx];					 vy0 = i_v[idx + n];				 
		vx1 = i_v[idx + blockDim.x];	 vy1 = i_v[idx + blockDim.x + n];	  
		vx2 = i_v[idx + 2 * blockDim.x]; vy2 = i_v[idx + 2 * blockDim.x + n]; 
		
		vx = vx0 + vx1 + vx2;
		vy = vy0 + vy1 + vy2;
		ek = vx0 * vx0 + vx1 * vx1 + vx2 * vx2 +
			vy0 * vy0 + vy1 * vy1 + vy2 * vy2;
		_n = 3;		
	}
	else if (idx + blockDim.x < n)
	{
		float vx0 = 0, vy0 = 0, n0 = 0;
		float vx1 = 0, vy1 = 0, n1 = 0;		
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);
		vx0 = i_v[idx];					 vy0 = i_v[idx + n];			 
		vx1 = i_v[idx + blockDim.x];	 vy1 = i_v[idx + blockDim.x + n];
				
		vx = vx0 + vx1;
		vy = vy0 + vy1;
		ek = vx0 * vx0 + vx1 * vx1 +
			vy0 * vy0 + vy1 * vy1;
		_n = 2;		
	}
	else if (idx < n)
	{
		float vx0 = 0, vy0 = 0;		
	
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);
		vx0 = i_v[idx];					 vy0 = i_v[idx + n];	

		vx = vx0;
		vy = vy0;
		ek = vx0 * vx0 + vy0 * vy0;
		_n = 1;		
	}/**/
	//if(idx + 5 * blockDim.x >4619700)
	//   printf("TT %i %i %i %f %i\n", tid, idx, blockIdx.x, n);
	//if (ns>1e-3f)
	//   printf("TT %i %i %f\n", tid, idx, ns);
	s_mem[tid              ] = vx;
	s_mem[tid +     SMEMDIM] = vy;
	s_mem[tid + 2 * SMEMDIM] = ek;
	s_mem[tid + 3 * SMEMDIM] = _n;
	__syncthreads();

	//if(idx==0)
	//	printf("TT %i %f %f %i %i\n", tid, s_ek, e_ek, s_n, e_n);

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid              ] += s_mem[tid +               512];
		s_mem[tid +     SMEMDIM] += s_mem[tid +     SMEMDIM + 512];
		s_mem[tid + 2 * SMEMDIM] += s_mem[tid + 2 * SMEMDIM + 512];
		s_mem[tid + 3 * SMEMDIM] += s_mem[tid + 3 * SMEMDIM + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 256];
		s_mem[tid + 2 * SMEMDIM] += s_mem[tid + 2 * SMEMDIM + 256];
		s_mem[tid + 3 * SMEMDIM] += s_mem[tid + 3 * SMEMDIM + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{

		s_mem[tid] += s_mem[tid + 128];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 128];
		s_mem[tid + 2 * SMEMDIM] += s_mem[tid + 2 * SMEMDIM + 128];
		s_mem[tid + 3 * SMEMDIM] += s_mem[tid + 3 * SMEMDIM + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
		s_mem[tid + SMEMDIM] += s_mem[tid + SMEMDIM + 64];
		s_mem[tid + 2 * SMEMDIM] += s_mem[tid + 2 * SMEMDIM + 64];
		s_mem[tid + 3 * SMEMDIM] += s_mem[tid + 3 * SMEMDIM + 64];
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
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 32];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 16];
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 16];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 8];
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 8];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 4];
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 4];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 2];
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 2];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 2];
		vsmem[tid] += vsmem[tid + 1];
		vsmem[tid + SMEMDIM] += vsmem[tid + SMEMDIM + 1];
		vsmem[tid + 2 * SMEMDIM] += vsmem[tid + 2 * SMEMDIM + 1];
		vsmem[tid + 3 * SMEMDIM] += vsmem[tid + 3 * SMEMDIM + 1];
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		o_D[blockIdx.x] = s_mem[0];
		o_D[blockIdx.x + gridDim.x] = s_mem[SMEMDIM];
		o_D[blockIdx.x + 2 * gridDim.x] = s_mem[2 * SMEMDIM];
		o_D[blockIdx.x + 3 * gridDim.x] = s_mem[3 * SMEMDIM];
		//if (smem[tid + 3 * SMEMDIM] > 1e-3f)
		//	printf("TT %i %i %f\n", tid, idx, smem[tid + 3 * SMEMDIM]);
		//if (smem[3 * SMEMDIM] > 1e-3f)
		//printf("T %i %f\n", blockIdx.x, gridDim.x, smem[3 * SMEMDIM]);
	}/**/
}

__global__ void d_setSinVelocityEntire(
	const float* __restrict__ i_r, 
	const float* __restrict__ i_v,
	float* __restrict__ o_v,
	unsigned int n, float x0,  
	float A_vx, float A_vy, float vc, float vt, float vr)

	
{
	// static shared memory
	//__shared__ float smem[4 * SMEMDIM];

	// set thread ID
	unsigned int tid = threadIdx.x;

	// global index, 4 blocks of input data processed at a time
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// unrolling 4 blocks
	float dx, vcc;

	// boundary check
	if (idx < n)
	{
		//printf("T %i %i %i\n", tid, idx, blockIdx.x);
		//float a1 = g_idata[idx];		
		//printf("T3 %i %i %i\n", tid, idx, blockIdx.x);
		dx = i_r[idx] - x0;
		//dy = i_r[idx + n] - yc;		
		//r = sqrtf(dx * dx + dy * dy);
		vcc = vc * sqrtf(1.0f + vt * sinf(vr * dx));			
		o_v[idx] = vcc * (i_v[idx] - A_vx);
		o_v[idx + n] = vcc * (i_v[idx + n] - A_vy);		
	}
}


//void template <256> __global__ reduceCompleteUnroll(float* g_idata, float* g_odata, unsigned int n);

void setSinTemperatureNormal(p_data& P, p0_data& P0, param_data& Pr, potential_data& Po, pAdd_data& Padd, l_data &L)
{
	cudaEvent_t start, stop;
	float gpuTime;
	double gpuTimeAver = 0;
	double average_vx=0, average_vy=0, average_ek=0, vcoeff=0, tcoeff, rcoeff;
	double n=0, x0;

	
	//thrust::device_ptr<float> d_VUXptr = thrust::device_pointer_cast(P.d_VUX);
	//curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);	
	//curandGenerateNormal(Padd.gen, Padd.d_SR_V, 2 * P.N, 0.0f, sqrtf(1.0f * Po._1d_m * Pr.EkSpot));
	curandGenerateNormal(Padd.gen, Padd.d_ER_V, 2 * P.N, 0.0f, sqrtf(1.0f * Po._1d_m * Pr.Ek));
	
	//std::cerr << "Pr " << Pr.xCenter << " " << Pr.yCenter << " " << Pr.rSpot << "\n";

	d_getSinVelocityEntire << <Padd.bloks4, SMEMDIM >> > (P0.d_RU0, Padd.d_ER_V, Padd.d_bD4, P.N);	
	
	HANDLE_ERROR(cudaMemcpy(Padd.h_bD4, Padd.d_bD4, 4 * Padd.bloks4 * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < Padd.bloks4; ++i)
	{
		average_vx += Padd.h_bD4[i];
		average_vy += Padd.h_bD4[i + Padd.bloks4];
		average_ek += Padd.h_bD4[i + 2 * Padd.bloks4];
		n += Padd.h_bD4[i + 3 * Padd.bloks4];		
		//fprintf(stderr, "n %i %i %e\n", i, bloks, vsh[4][i + bloks]);
	}
	average_vx /= n;
	average_vy /= n;
	average_ek *= 0.5 * Po.m / n;
	average_ek -= 0.5f * Po.m * (average_vx * average_vx + average_vy * average_vy);
	vcoeff = sqrt(Pr.Ek / average_ek);
	tcoeff = Pr.EkSpot / Pr.Ek;
	rcoeff = 2.0 * MC_pi / L.PS[0].x;
	x0 = P0.h_RU0[0];
	//fprintf(stderr, "N %e %e %i | %e %e %e %e | %e %e | %e %e\n", _n, environment_n, P.N, 
	//	average_vx, average_vy, environmentaverage_vx, environmentaverage_vy, 
	//	average_ek, Pr.Ek, environmentaverage_ek, Pr.Ek);
	//fprintf(stderr, "NN %e %e \n", _vcoeff, environment_vcoeff);
	
	d_setSinVelocityEntire << < Padd.bloks, SMEMDIM >> > (P0.d_RV0, Padd.d_ER_V, P.d_VU, P.N, x0, 
		float(average_vx), float(average_vy), float(vcoeff), float(tcoeff), float(rcoeff));

	curandGenerateNormal(Padd.gen, Padd.d_ER_V, 2 * P.N, 0.0f, sqrtf(1.0f * Po._1d_m * Pr.Ek));
	   
	d_getSinVelocityEntire <<<Padd.bloks4, SMEMDIM >> > (P0.d_RV0, Padd.d_ER_V,Padd.d_bD4, P.N);

	HANDLE_ERROR(cudaMemcpy(Padd.h_bD4, Padd.d_bD4, 4 * Padd.bloks4 * sizeof(float), cudaMemcpyDeviceToHost));
	average_vx = 0;
	average_vy = 0;
	average_ek = 0;
	n = 0;	
	for (int i = 0; i < Padd.bloks4; ++i)
	{
		average_vx += Padd.h_bD4[i];
		average_vy += Padd.h_bD4[i + Padd.bloks4];
		average_ek += Padd.h_bD4[i + 2 * Padd.bloks4];
		n += Padd.h_bD4[i + 3 * Padd.bloks4];		
		//fprintf(stderr, "n %i %i %e\n", i, bloks, vsh[4][i + bloks]);
	}
	average_vx /= n;
	average_vy /= n;
	average_ek *= 0.5 * Po.m / n;
	average_ek -= 0.5f * Po.m * (average_vx * average_vx + average_vy * average_vy);
	vcoeff = sqrt(Pr.Ek / average_ek);
	tcoeff = Pr.EkSpot / Pr.Ek;
	rcoeff = 2.0 * MC_pi / L.PS[0].x;
	
	//fprintf(stderr, "N %e %e %i | %e %e %e %e | %e %e | %e %e\n", _n, environment_n, P.N,
	//	average_vx, average_vy, environmentaverage_vx, environmentaverage_vy,
	//	average_ek, Pr.Ek, environmentaverage_ek, Pr.Ek);
	//fprintf(stderr, "NN %e %e \n", _vcoeff, environment_vcoeff);
	
	d_setSinVelocityEntire <<< Padd.bloks, SMEMDIM >> > (P0.d_RV0, Padd.d_ER_V,	P.d_VV, P.N, x0,
		float(average_vx), float(average_vy), float(vcoeff), float(tcoeff), float(rcoeff));
}

