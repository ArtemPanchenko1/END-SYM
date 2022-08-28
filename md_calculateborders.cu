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

__global__ void d_calculateBorders(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rrm, rr0m, m;
	if (idx < nbp)
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
			//printf("F %i | %e %e\n", idx, F[idx], F[idx + n]);
		i = BP[idx];
		iA = n - 2;
		iB = n - 1;

		if (idx < nbpt1 - 1)
		{

			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];
			//rux = rx - U[i];
			//ruy = ry - U[i + n];
			///rum = rux * rux + ruy * ruy - 0.000397498921890625f; //0.001287896506925625f;(0.9)
			//rum = rux * rux + ruy * ruy - 0.0143620218536932355625f
			//F[i] = -1.0f;
			//F[i + n] = -2.0f;			
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // !pre_FreeCell	
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			//FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			//FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // pre_FreeCell
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;

			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
#ifndef pre_FreeCellHalf
			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			V[i] = V1 * rx * _1d_rm;
			V[i + n] = V1 * ry * _1d_rm;
			//m = __frcp_rn(_1d_Mass[i]);
			F[i] = 0.0;// 6.0f * MCf_pi * P_vis * VisR[i] * V[i] * m;
			F[i + n] = 0.0;// 6.0f * MCf_pi * P_vis * VisR[i] * V[i + n] * m;
#endif // !pre_FreeCellHalf
#ifdef pre_FreeCellHalf
			if (V1 > 1e-12)
			{
				_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
				V[i] = V1 * rx * _1d_rm;
				V[i + n] = V1 * ry * _1d_rm;
				m = 1.0f / _1d_Mass[i];
				F[i] = P_vis * V[i] * m;
				F[i + n] = P_vis * V[i + n] * m;
			}
			else
			{
				rrm = rx * rx + ry * ry;
				rr0m = (BPR[nbpt1 - 1] - BPR[idx]) * (BPR[nbpt1 - 1] - BPR[idx]) + (BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp]) * (BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp]);
				//printf("BP %i %i | %e |%e %e | %e\n", idx, i, V1, rrm, rr0m, rrm-rr0m);
				if (rrm > rr0m)
				{
					//_1d_rm = __frsqrt_rn(rrm);
					V[i] = 0;
					V[i + n] = 0;
					F[i] = 0;
					F[i + n] = 0;
				}
			}
#endif // pre_FreeCellHalf



			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, 1.0f/_1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;/**/
#endif // !pre_FreeCell			
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			//if(i==4822)
			//	printf("B %i %i | %e %e %e | %e %e | %e %e\n", idx, i, F[i], F[i+n], _1d_Mass[i], V[i], V[i+n], U[i], U[i+n]);			
#endif // pre_FreeCell
			/*rx = 0.121052175f - BPR[idx];
			ry = -BPR[idx + nbp];
			FR[idx + 2 * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (2 * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			V[i] = V1 * rx * _1d_rm;
			V[i + n] = V1 * ry * _1d_rm;
			m = 1.0f / _1d_Mass[i];
			F[i] = P_vis * V[i] * m;
			F[i + n] = P_vis * V[i + n] * m;/**/
		}/**/
		else
		{
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
		}
		/*if (idx == nbpt1-1)
		{
			F[iA] = 0;	F[iA + n] = 0;	V[iA] = 0;	V[iA + n] = 0;
		}
		else if (idx == nbpt1 + nbpt2 - 1)
		{
			F[iB] = 0;	F[iB + n] = 0;	V[iB] = 0;	V[iB + n] = 0;
		}/**/
	}
}

__global__ void d_calculateBordersVis(const float* __restrict__ _1d_Mass, const float* __restrict__ VisR, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rrm, rr0m, m;
	if (idx < nbp)
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
		//printf("F %i | %e %e\n", idx, F[idx], F[idx + n]);
		i = BP[idx];
		iA = n - 2;
		iB = n - 1;
		
		if (idx < nbpt1 - 1)
		{
			
			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];
			//rux = rx - U[i];
			//ruy = ry - U[i + n];
			///rum = rux * rux + ruy * ruy - 0.000397498921890625f; //0.001287896506925625f;(0.9)
			//rum = rux * rux + ruy * ruy - 0.0143620218536932355625f
			//F[i] = -1.0f;
			//F[i + n] = -2.0f;			
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // !pre_FreeCell	
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			//FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			//FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // pre_FreeCell
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;

			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
#ifndef pre_FreeCellHalf
			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			V[i] = V1 * rx * _1d_rm;
			V[i + n] = V1 * ry * _1d_rm;
			//m = __frcp_rn(_1d_Mass[i]);
			F[i] = 6.0f * MCf_pi * P_vis * VisR[i] * V[i];
			F[i + n] = 6.0f * MCf_pi * P_vis * VisR[i] * V[i + n];
#endif // !pre_FreeCellHalf
#ifdef pre_FreeCellHalf
			if (V1 > 1e-12)
			{
				_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
				V[i] = V1 * rx * _1d_rm;
				V[i + n] = V1 * ry * _1d_rm;
				m = 1.0f / _1d_Mass[i];
				F[i] = P_vis * V[i] * m;
				F[i + n] = P_vis * V[i + n] * m;
			}
			else
			{
				rrm = rx * rx + ry * ry;
				rr0m = (BPR[nbpt1 - 1] - BPR[idx]) * (BPR[nbpt1 - 1] - BPR[idx]) + (BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp]) * (BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp]);
				//printf("BP %i %i | %e |%e %e | %e\n", idx, i, V1, rrm, rr0m, rrm-rr0m);
				if (rrm > rr0m)
				{
					//_1d_rm = __frsqrt_rn(rrm);
					V[i] = 0;
					V[i + n] = 0;
					F[i] = 0;
					F[i + n] = 0;
				}
			}
#endif // pre_FreeCellHalf

			
			
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, 1.0f/_1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{		
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;/**/
#endif // !pre_FreeCell			
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			//if(i==4822)
			//	printf("B %i %i | %e %e %e | %e %e | %e %e\n", idx, i, F[i], F[i+n], _1d_Mass[i], V[i], V[i+n], U[i], U[i+n]);			
#endif // pre_FreeCell
			/*rx = 0.121052175f - BPR[idx];
			ry = -BPR[idx + nbp];			
			FR[idx + 2 * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (2 * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
			V[i] = V1 * rx * _1d_rm;
			V[i + n] = V1 * ry * _1d_rm;
			m = 1.0f / _1d_Mass[i];
			F[i] = P_vis * V[i] * m;
			F[i + n] = P_vis * V[i + n] * m;/**/
		}/**/
		else
		{
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
		}		
		/*if (idx == nbpt1-1)
		{
			F[iA] = 0;	F[iA + n] = 0;	V[iA] = 0;	V[iA + n] = 0;
		}
		else if (idx == nbpt1 + nbpt2 - 1)
		{
			F[iB] = 0;	F[iB + n] = 0;	V[iB] = 0;	V[iB + n] = 0;
		}/**/
	}	
}

__global__ void d_calculateBorders(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1, const int ibp)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rux, ruy, rum, m;
	if (idx < nbp)
	{		
		i = BP[idx];
		iA = n - 2;
		iB = n - 1;

		if (idx < nbpt1 - 1)
		{



			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);
			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];			
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // !pre_FreeCell	
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			//FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			//FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // pre_FreeCell
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, 1.0f / _1d_rm, BPR[idx], BPR[idx + nbp]);
			if (idx == ibp)
			{
				//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, 1.0f / _1d_rm, BPR[idx], BPR[idx + nbp]);
				_1d_rm = __frsqrt_rn(rx * rx + ry * ry);
				V[i] = V1 * rx * _1d_rm;
				V[i + n] = V1 * ry * _1d_rm;
				m = 1.0f / _1d_Mass[i];
				F[i] = P_vis * V[i] * m;
				F[i + n] = P_vis * V[i + n] * m;
				//F[iA] = 0;	F[iA + n] = 0;
				//F[iB] = 0;	F[iB + n] = 0;
			} else
			{
				F[i] = 0;
				F[i + n] = 0;
				V[i] = 0;
				V[i + n] = 0;
			}
			//printf("B %i | %e %e | \%e %e\n", idx, rum, rux * rux + ruy * ruy, rx, ry);			
		}
		else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
#endif // !pre_FreeCell			
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			//printf("B %i %i | %e %e %e | %e %e | %e %e\n", idx, i, F[i], F[i+n], _1d_Mass[i], V[i], V[i+n], U[i], U[i+n]);			
#endif // pre_FreeCell
		}
		else
		{
			F[i] = 0;
			F[i + n] = 0;
			V[i] = 0;
			V[i + n] = 0;
		}		
	}
}

__global__ void d_calculateBordersFix(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, i, iA, iB;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//printf("B %i %i\n", idx, nbp);
	float rx, ry, _1d_rm, rrm, rr0m, m;
	if (idx < nbp)
	{
		
		i = BP[idx];
		iA = n - 2;
		iB = n - 1;

		if (idx < nbpt1 - 1)
		{

			rx = BPR[nbpt1 - 1] - BPR[idx] + U[iA] - U[i];
			ry = BPR[nbpt1 - 1 + nbp] - BPR[idx + nbp] + U[iA + n] - U[i + n];
				
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
#endif // !pre_FreeCell	
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
#endif // pre_FreeCell
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;

			
			//printf("F %i %i | %e %e | %e %e | %e %e %e | %e %e\n", idx, i, F[i], F[i + n], V[i], V[i + n], rx, ry, 1.0f/_1d_rm, BPR[idx], BPR[idx + nbp]);
		}
		else if (idx >= nbpt1 && idx < nbpt1 + nbpt2 - 1)
		{
#ifndef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = F[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = F[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;			
#endif // !pre_FreeCell			
#ifdef pre_FreeCell
			FR[idx + ResultFRNum * Step * (nbpt1 + nbpt2)] = U[i];
			FR[idx + (ResultFRNum * Step + 1) * (nbpt1 + nbpt2)] = U[i + n];
			rx = BPR[nbpt1 + nbpt2 - 1] - BPR[idx] + U[iB] - U[i];
			ry = BPR[nbpt1 + nbpt2 - 1 + nbp] - BPR[idx + nbp] + U[iB + n] - U[i + n];
			FR[idx + (ResultFRNum * Step + 2) * (nbpt1 + nbpt2)] = rx;
			FR[idx + (ResultFRNum * Step + 3) * (nbpt1 + nbpt2)] = ry;
			//printf("B %i %i | %e %e %e | %e %e | %e %e\n", idx, i, F[i], F[i+n], _1d_Mass[i], V[i], V[i+n], U[i], U[i+n]);
#endif // pre_FreeCell			
		}		
		V[i] = 0;
		V[i + n] = 0;
		F[i] = 0;
		F[i + n] = 0;
	}
}