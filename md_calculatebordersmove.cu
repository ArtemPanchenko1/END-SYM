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


__global__ void d_calculateBordersMove(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, const float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1)
{

	__shared__ float FA[2*SMEMDIM];
	__shared__ float FB[2*SMEMDIM];
#ifdef pre_RotateCell
	__shared__ float MA[2*SMEMDIM];
	__shared__ float MB[SMEMDIM];
#endif // pre_RotateCell

	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, j, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rux, ruy, rum, m, ax, ay, vtx, vty, vr;
	if (idx < nbp)
	{
		i = BP[idx];
		iA = n - 2;
		iB = n - 1;
		if (idx < nbpt1)
		{			
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			FA[idx] = F[i];
			FA[idx + nbpt1] = F[i + n];
#ifdef pre_RotateCell
			MA[idx] = rx * F[i + n] - ry * F[i];
			MA[idx + nbpt1] = __fsqrt_rn(rx * rx + ry * ry);
			//printf("A0 %u %u | %e\n", idx, Step, MA[idx + nbpt1]);
#endif // pre_RotateCell
			//printf("FA %i | %e %e | %i | %e %e\n", idx, F[i], F[i + n], ResultFRNum * Step * (nbpt1 + nbpt2), FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)]);
			//printf("B0 %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, _1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx < nbpt1 + nbpt2)
		{			
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			//printf("FAB  %i %i | %i %i | %e %e\n", idx, ResultFRNum, Step, nbpt1 + nbpt2, F[i], F[i + n]);			
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			FB[idx - nbpt1] = F[i];
			FB[idx - nbpt1 + nbpt2] = F[i + n];
#ifdef pre_RotateCell
			MB[idx - nbpt1] = rx * F[i + n] - ry * F[i];
#endif // pre_RotateCell
		}
		else
		{
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
		}
		if (idx == 0)
		{
			
			for (j = 1; j < nbpt1-1; ++j)
			{
				FA[0] += FA[j];
				FA[0 + nbpt1] += FA[j + nbpt1];
#ifdef pre_RotateCell
				MA[0] += MA[j];
				MA[nbpt1] += MA[j + nbpt1];
#endif // pre_RotateCell
			}
			//FA[0] = 0; FA[0 + nbpt1] = 0; V[iA] = 0; V[iA + n] = 0;
			F[iA] = FA[0];
			F[iA + n] = FA[0 + nbpt1];
			FA[0] *= _1d_Mass[iA];
			FA[0 + nbpt1] *= _1d_Mass[iA];
#ifdef pre_RotateCell			
			MA[nbpt1] = __fdiv_rn(float(nbpt1), MA[nbpt1]);
			MA[0] *= 2.0f * _1d_Mass[iA] * MA[nbpt1] * MA[nbpt1];	
			//printf("AA %u %u | %e %e\n", idx, Step, MA[nbpt1], MA[0]);
#endif // pre_RotateCell
			//printf("AA %i %i\n", idx, Step);
			
//F[iA] = 0;	F[iA + n] = 0;	V[iA] = 0;	V[iA + n] = 0;
			//printf("FAB  %e %e | %e %e\n", FA[0], FB[0], FA[0 + nbpt1], FB[0 + nbpt2]);
		}
		else if (idx == 1)
		{
			for (j = 1; j < nbpt2 - 1; ++j)
			{
				FB[0] += FB[j];
				FB[0 + nbpt2] += FB[j + nbpt2];
#ifdef pre_RotateCell
				MB[0] += MB[j];
				//printf("BB %i | %e %e\n", j, MB[0], MB[j]);
#endif // pre_RotateCell
			}
			//FB[0] = 0; FB[0 + nbpt2] = 0; V[iB] = 0; V[iB + n] = 0;
			F[iB] = FB[0];
			F[iB + n] = FB[0 + nbpt2];
			FB[0] *= _1d_Mass[iB];
			FB[0 + nbpt2] *= _1d_Mass[iB];
#ifdef pre_RotateCell
			MB[0] *= 2.0f * _1d_Mass[iB] * 628.93252341f;
			//printf("BB %u %u | %e\n", idx, Step, MB[0]);
#endif // pre_RotateCell
//F[iB] = 0;	F[iB + n] = 0;	V[iB] = 0;	V[iB + n] = 0;
		}
		__syncthreads();
		if (idx < nbpt1-1)
		{

			
			//if(Step%10000==0)printf("A %i %i | %e %e | \%e %e\n", idx, Step, FR[idx + (ResultFRNum * Step + 0) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)]);
			//printf("A %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			vtx = V[i] - V[iA];
			vty = V[i + n] - V[iA + n];
			vr = (vtx * rx + vty * ry) * _1d_rm * _1d_rm;
			vtx -= vr * rx;
			vty -= vr * ry;
			//printf("A %i | %e %e | %e %e | %e\n", idx, rx, ry, vtx, vty, rx*vtx+ry*vty);
			V[i] = V[iA] + V1 * rx * _1d_rm + vtx;
			V[i + n] = V[iA + n] + V1 * ry * _1d_rm + vty;
			m = __frcp_rn(_1d_Mass[i]);
			ax = FA[0];
			ay = FA[0 + nbpt1];
#ifdef pre_RotateCell
			ax -= MA[0] * ry;
			ay += MA[0] * rx;
#endif // pre_RotateCell
#ifdef pre_Relaxation
			ax += P_vis * vr * rx;
			ay += P_vis * vr * ry;
#endif // pre_Relaxation
#ifdef pre_Viscocity
			ax += P_vis * V1 * rx * _1d_rm;
			ay += P_vis * V1 * ry * _1d_rm;
#endif // pre_Relaxation
			F[i] = ax * m;
			F[i + n] = ay * m;
			//F[i] = 0;
			//F[i + n] = 0;
//F[i] = 0;	F[i + n] = 0;
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, _1d_rm, BPR[idx], BPR[idx + nbp]);
		} else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{
			
			//if (Step % 10000 == 0)printf("B %i %i | %e %e | \%e %e\n", idx, Step, FR[idx + (ResultFRNum * Step + 0) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)]);
			_1d_rm = __frcp_rn(rx * rx + ry * ry);
			vtx = V[i] - V[iB];
			vty = V[i + n] - V[iB + n];
			vr = (vtx * rx + vty * ry) * _1d_rm;
			vtx -= vr * rx;
			vty -= vr * ry;
			V[i] = V[iB] + vtx;
			V[i + n] = V[iB + n] + vty;
			//printf("B %i | %e %e | %e %e | %e\n", idx, rx, ry, vtx, vty, rx* vtx + ry * vty);
			//V[i] = V[iB] + vtx;
			//V[i + n] = V[iB + n] + vty;
			m = __frcp_rn(_1d_Mass[i]);
			ax = FB[0];
			ay = FB[0 + nbpt2];
#ifdef pre_RotateCell
			ax -= MB[0] * ry;
			ay += MB[0] * rx;
#endif // pre_RotateCell
			F[i] = ax * m;
			F[i + n] = ay * m;
//F[i] = 0;	F[i + n] = 0;	V[i] = 0;	V[i + n] = 0;
		}
	}	
}

__global__ void d_calculateBordersMove(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, const float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1, const int ibp)
{
	__shared__ float FA[2 * SMEMDIM];
	__shared__ float FB[2 * SMEMDIM];
#ifdef pre_RotateCell
	__shared__ float MA[2 * SMEMDIM];
	__shared__ float MB[SMEMDIM];
#endif // pre_RotateCell
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, j, iA, iB, iibp;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rux, ruy, rum, m, ax, ay, vtx, vty, vr;
	if (idx < nbp)
	{
		i = BP[idx];
		iibp = BP[ibp];
		iA = n - 2;
		iB = n - 1;
		if (idx < nbpt1)
		{
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			FA[idx] = F[i];
			FA[idx + nbpt1] = F[i + n];
#ifdef pre_RotateCell
			MA[idx] = rx * F[i + n] - ry * F[i];
			MA[idx + nbpt1] = __fsqrt_rn(rx * rx + ry * ry);
			//printf("A0 %u %u | %e\n", idx, Step, MA[idx + nbpt1]);
#endif // pre_RotateCell
			//printf("FA %i | %e %e | \%e %e\n", idx, F[i], F[i + n]);
			//printf("B0 %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, _1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx < nbpt1 + nbpt2)
		{
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			//printf("FAB  %i %i | %i %i | %e %e\n", idx, ResultFRNum, Step, nbpt1 + nbpt2, F[i], F[i + n]);
			FB[idx - nbpt1] = F[i];
			FB[idx - nbpt1 + nbpt2] = F[i + n];
#ifdef pre_RotateCell
			MB[idx - nbpt1] = rx * F[i + n] - ry * F[i];
#endif // pre_RotateCell
		}
		else
		{
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
		}
		if (idx == 0)
		{

			for (j = 1; j < nbpt1 - 1; ++j)
			{
				FA[0] += FA[j];
				FA[0 + nbpt1] += FA[j + nbpt1];
#ifdef pre_RotateCell
				MA[0] += MA[j];
				MA[nbpt1] += MA[j + nbpt1];
#endif // pre_RotateCell
			}
			//FA[0] = 0; FA[0 + nbpt1] = 0; V[iA] = 0; V[iA + n] = 0;
			F[iA] = FA[0];
			F[iA + n] = FA[0 + nbpt1];
			FA[0] *= _1d_Mass[iA];
			FA[0 + nbpt1] *= _1d_Mass[iA];
#ifdef pre_RotateCell			
			MA[nbpt1] = __fdiv_rn(float(nbpt1), MA[nbpt1]);
			MA[0] *= 2.0f * _1d_Mass[iA] * MA[nbpt1] * MA[nbpt1];
			//printf("AA %u %u | %e %e\n", idx, Step, MA[nbpt1], MA[0]);
#endif // pre_RotateCell
			//F[iA] = 0;	F[iA + n] = 0;	V[iA] = 0;	V[iA + n] = 0;
						//printf("FAB  %e %e | %e %e\n", FA[0], FB[0], FA[0 + nbpt1], FB[0 + nbpt2]);
		}
		else if (idx == 1)
		{
			for (j = 1; j < nbpt2 - 1; ++j)
			{
				FB[0] += FB[j];
				FB[0 + nbpt2] += FB[j + nbpt2];
#ifdef pre_RotateCell
				MB[0] += MB[j];
				//printf("BB %i | %e %e\n", j, MB[0], MB[j]);
#endif // pre_RotateCell
			}
			//FB[0] = 0; FB[0 + nbpt2] = 0; V[iB] = 0; V[iB + n] = 0;
			F[iB] = FB[0];
			F[iB + n] = FB[0 + nbpt2];
			FB[0] *= _1d_Mass[iB];
			FB[0 + nbpt2] *= _1d_Mass[iB];
#ifdef pre_RotateCell
			MB[0] *= 2.0f * _1d_Mass[iB] * 628.93252341f;
			//printf("BB %u %u | %e\n", idx, Step, MB[0]);
#endif // pre_RotateCell
			//F[iB] = 0;	F[iB + n] = 0;	V[iB] = 0;	V[iB + n] = 0;
		}		
		__syncthreads();
		if (idx < nbpt1 - 1)
		{

			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			vtx = V[i] - V[iA];
			vty = V[i + n] - V[iA + n];
			vr = (vtx * rx + vty * ry) * _1d_rm * _1d_rm;
			vtx -= vr * rx;
			vty -= vr * ry;
			
			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			ax = FA[0];
			ay = FA[0 + nbpt1];
			if (idx == ibp)
			{				
				V[i] = V[iA] + V1 * rx * _1d_rm + vtx;
				V[i + n] = V[iA + n] + V1 * ry * _1d_rm + vty;
#ifdef pre_Relaxation
				ax += P_vis * V1 * rx * _1d_rm;
				ay += P_vis * V1 * ry * _1d_rm;
#endif // pre_Relaxation
#ifdef pre_Viscocity
				ax += P_vis * vr * rx;
				ay += P_vis * vr * ry;
#endif // pre_Relaxation
			}
			else
			{
				V[i] = V[iA] + vtx;
				V[i + n] = V[iA + n] + vty;
			}
			m = __frcp_rn(_1d_Mass[i]);
#ifdef pre_RotateCell
			ax -= MA[0] * ry;
			ay += MA[0] * rx;
#endif // pre_RotateCell
			F[i] = ax * m;
			F[i + n] = ay * m;
//F[i] = 0;	F[i + n] = 0;
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, _1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{
			_1d_rm = __frcp_rn(rx * rx + ry * ry);
			vtx = V[i] - V[iB];
			vty = V[i + n] - V[iB + n];
			vr = (vtx * rx + vty * ry) * _1d_rm;
			vtx -= vr * rx;
			vty -= vr * ry;
			V[i] = V[iB] + vtx;
			V[i + n] = V[iB + n] + vty;
			//printf("B %i | %e %e | %e %e | %e\n", idx, rx, ry, vtx, vty, rx* vtx + ry * vty);
			//V[i] = V[iB] + vtx;
			//V[i + n] = V[iB + n] + vty;
			m = __frcp_rn(_1d_Mass[i]);
			ax = FB[0];
			ay = FB[0 + nbpt2];
#ifdef pre_RotateCell
			ax -= MB[0] * ry;
			ay += MB[0] * rx;
#endif // pre_RotateCell
			F[i] = ax * m;
			F[i + n] = ay * m;
			//F[i] = 0;	F[i + n] = 0;	V[i] = 0;	V[i + n] = 0;
		}
	}
}

__global__ void d_PrintFR(const float* __restrict__ FR, const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step)
{	
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, j, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rux, ruy, rum, m;
	//printf("FR %i %i\n", idx, Step);
	if (idx == 0)
	{
		//printf("FR %i %i\n", idx, Step);
		//for (j = 0; j < Step; ++j)
		j = Step;
		{
			printf("FR %i | %e %e | %e %e\n", j, ResultFRNum, FR[idx + ResultFRNum * j * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * j + 1) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * j + 2) * (nbpt1 + nbpt2)], FR[idx + (ResultFRNum * j + 3) * (nbpt1 + nbpt2)]);
		}

		
	}		
}