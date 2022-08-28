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


__global__ void d_calculateIncrements(const float* __restrict__ _1d_Mass, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vis)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	if (idx < n)
	{
		/*if (idx == 700)
			{
				//Fetx[nt] = F[700];
				//Fety[nt] = F[700+n];

				//printf("F00 %e %e\n", F[500], F[500 + n]);
				//printf("F0 %e %e\n", F[990], F[990+n]);
				//printf("F1 %e %e\n", F[995], F[995 + n]);
				//printf("F2 %e %e\n", F[999], F[999 + n]);
			}/**/
		//if(F[idx] > 0)printf("F %i %e %e\n", idx, F[idx], F[idx + n]);
		//if(V[idx]>0)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);

		//if(idx>76648 && idx<76771)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);

		//V[idx] += (F[idx] * _1d_Mass[idx] - P_vis * V[idx]) * P_dt;
		//V[idx + n] += (F[idx + n] * _1d_Mass[idx] - P_vis * V[idx + n]) * P_dt;
		V[idx] += F[idx] * _1d_Mass[idx] * P_dt;
		V[idx + n] += F[idx + n] * _1d_Mass[idx] * P_dt;

		//if (V[idx]* V[idx] + V[idx + n] * V[idx + n] > 1e-5)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);

		U[idx] += V[idx] * P_dt;
		U[idx + n] += V[idx + n] * P_dt;
	}	
}

__global__ void d_calculateIncrementsVis(const float* __restrict__ _1d_Mass, const float* __restrict__ VisR, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vis)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float visc;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	if (idx < n)
	{
		/*if (idx == 700)
			{
				//Fetx[nt] = F[700];
				//Fety[nt] = F[700+n];

				//printf("F00 %e %e\n", F[500], F[500 + n]);
				//printf("F0 %e %e\n", F[990], F[990+n]);
				//printf("F1 %e %e\n", F[995], F[995 + n]);
				//printf("F2 %e %e\n", F[999], F[999 + n]);
			}/**/
			//if(F[idx] > 0)printf("F %i %e %e\n", idx, F[idx], F[idx + n]);
			//if(V[idx]>0)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);

			//if(idx>76648 && idx<76771)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);
		visc = 6.0f * MCf_pi * P_vis * VisR[idx];
		V[idx] += (F[idx] - visc * V[idx]) * _1d_Mass[idx] * P_dt;
		V[idx + n] += (F[idx + n] - visc * V[idx + n]) * _1d_Mass[idx] * P_dt;

		//if (V[idx]* V[idx] + V[idx + n] * V[idx + n] > 1e-5)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);

		U[idx] += V[idx] * P_dt;
		U[idx + n] += V[idx + n] * P_dt;
	}
}