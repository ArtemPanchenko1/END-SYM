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
__global__ void d_calculateIncrementsFIRE(const float* __restrict__ _1d_Mass, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vis, const float F_alpha)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float vx, vy, v, f, v_d_f;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	if (idx < n)
	{
		//printf("In %e %e %e %e %e %e\n", F[idx], F[idx + n], V[idx], V[idx + n], _1d_Mass[idx], P_dt);
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
	//Leapfrog
#ifndef pre_Viscocity
		vx = V[idx] + F[idx] * _1d_Mass[idx] * P_dt;
		vy = V[idx + n] + F[idx + n] * _1d_Mass[idx] * P_dt;
#endif // !pre_Viscocity		
#ifdef pre_Viscocity
		vx = (F[idx] * _1d_Mass[idx] - P_vis * V[idx]) * P_dt;
		vy = (F[idx + n] * _1d_Mass[idx] - P_vis * V[idx + n]) * P_dt;
#endif // pre_Viscocity

		v = vx * vx + vy * vy;
		f = F[idx] * F[idx] + F[idx + n] * F[idx + n];
		if (f > 1e-12)
		{
			v_d_f = __fsqrt_rn(v * __frcp_rn(f));
		}
		else
		{
			v_d_f = __fsqrt_rn(v)*1e6;
		}
			vx = (1.0f - F_alpha) * vx + F_alpha * v_d_f * F[idx];
			vy = (1.0f - F_alpha) * vy + F_alpha * v_d_f * F[idx + n];
		
		//if (idx == 700)printf("F %i %e %e %e | %e %e\n", idx, v, f, v_d_f, vx, vy);
		

		V[idx] = vx;
		V[idx + n] = vy;
		//V[idx] += (F[idx] * _1d_Mass[idx] - P_vis * V[idx]) * P_dt;
		//V[idx + n] += (F[idx + n] * _1d_Mass[idx] - P_vis * V[idx + n]) * P_dt;
		//if (V[idx]* V[idx] + V[idx + n] * V[idx + n] > 1e-5)printf("V %i %e %e\n", idx, V[idx], V[idx + n]);
		U[idx] += vx * P_dt;
		U[idx + n] += vy * P_dt;
	}
}

__global__ void d_calculateDecrementsHalfStepFIRE(const float* __restrict__ V, float* __restrict__ U, const unsigned int n, const float P_dt)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (idx < n)
	{		
	//Leapfrog
		U[idx] -= V[idx] * P_dt * 0.5;
		U[idx + n] -= V[idx + n] * P_dt * 0.5;
	}
}

void calculateGPUStepsContractRelaxFIRE(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet, firerelax_data &Fire)
{
	//std::cerr << "W1\n";
	/*cudaEvent_t start, stop;
	float gpuTime;
	double gpuTimeAver = 0;/**/
	//float dV, maxShift = 10.0, Vl = sqrt(Po.a * Po.a * Po.c * Po._1d_m), Vd = 1.01;
	///Padd.V = 2e-2;
	char filename[256] = "";
	unsigned int time, bloks, estep, Estep, esize = Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]), i, j, steps;// , timestart, t1, t2;
	double d = 0, povis0 = Po.vis, timereal=0;
	float v = Padd.V, fa_max, fb_min, Ep0 = 2.21821;
	bool contraction = true;


	time = Padd.time;
	//timestart = 1.01*(2.0 * Padd.MaxShift / Padd.V) / Po.dt + 1;

	//float v, v2;
	//time = 1500000;
	
	Fire.bloks4 = P.N / (4 * SMEMDIM) + 1;
	Fire.h_FdotV = (float*)malloc(Fire.bloks4 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Fire.d_FdotV, Fire.bloks4 * sizeof(float)));
	Padd.bloks = P.N / (SMEMDIM)+2;
	Padd.bloksb = P.NBP / (SMEMDIM)+2;

	std::cerr << "Bloks " << Padd.bloks << " " << Padd.bloksb << "\n";
#ifdef pre_OneNodeContractRelax
	double rMoveOne = 1.0 * 0.01 * 0.03987475 * ReadCoordinatesCoefficient, v1;
	unsigned int onecontracttime = unsigned int(rMoveOne / (v * Po.dt));
	v1 = rMoveOne / (onecontracttime * Po.dt);
	std::cerr << "Time_OCC " << onecontracttime << " " << rMoveOne << " " << v1 << "\n";
	for (steps = 0; steps < onecontracttime; ++steps)
	{
		estep = 0;
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
		//std::cerr << "Q2\n";
#ifdef pre_OneNodeContract
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v1, P.iBP[0]);
#endif // pre_OneNodeContract
		d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);

		//if (steps % 100000 == 0)std::cerr << "steps " << steps << "\n";
		/*if (steps % 10000 == 0)
		{
			end = std::chrono::high_resolution_clock::now();
			dr = end - begin;
			std::cerr << "Fin Step" << steps << " " << v << " " << d << " | " << std::chrono::duration_cast<std::chrono::milliseconds>(dr).count() <<" ms " << "\n";
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);
			begin = std::chrono::high_resolution_clock::now();
			std::cin.get();
			//sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", steps);
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, filename);
		}/**/

}
#ifdef pre_SaveLammps
	//if (steps % 100000 == 0)
	{
		std::cerr << "Fin Step" << steps << " " << v << " " << d << "\n";
		d_calculateForcesIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Fbound, P.N, P.NI, Po.c);


		cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(Padd.h_Fbound, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
		//sprintf(filename, "./result/steps/CP_%li.txt", steps);
		//SaveTXTParticles(P, P0, Po, Pnet, filename);
		sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", steps);
		SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
	}
#endif // pre_SaveLammps
#endif // pre_OneNodeContractRelax

#ifdef pre_CalcFullEnergy
	sprintf(filename, "./result/CP_Energy_FIRE.txt");
	std::ofstream file_energy_txt;
	file_energy_txt.open(filename, std::ios::out);
	file_energy_txt << "step time Ep Ek Efull\n";
	file_energy_txt.precision(10);
#endif // pre_CalcFullEnergy
	/*cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	sprintf(filename, "./result/steps/CP_%li.txt", 0);
	SaveTXTParticles(P, P0, Po, Pnet, filename);/**/

	unsigned int contracttime = unsigned int(Padd.RMove / (Padd.V * Po.dt));
	v = Padd.RMove / (contracttime * Po.dt);
	std::cerr << "Time_OCC " << contracttime << " " << Padd.RMove << " " << v << " " << Padd.V << " " << Po.dt << "\n";
	timereal = 0;
	for (steps = 0; steps < contracttime; ++steps)
	{
		estep = 0;
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
#ifndef pre_MoveCell
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_MoveCell
#ifdef pre_MoveCell
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // pre_MoveCell	
		d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);
		timereal += Po.dt;
		/*if (steps % 1000 == 0)
		{			
			std::cerr << "Fin Step" << steps << " " << v << "\n";
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);
			//std::cin.get();
		}/**/
	}

	/*cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	sprintf(filename, "./result/steps/CP_%li.txt", 1);
	SaveTXTParticles(P, P0, Po, Pnet, filename);/**/

	std::cerr << "FIRE Start\n";
	Fire.NPpositive = 0;
	Fire.NPnegative = 0;
	Fire.dt = Fire.dt0;
	Po.vis = 0;
	Po.vism = Po.vis * Po.m;
	v = 0;
	timereal = 0;
	for (steps = 0; steps < Padd.RelaxationTime; ++steps)
	{
		estep = 0;
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
#ifndef pre_MoveCell
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_MoveCell
#ifdef pre_MoveCell
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // pre_MoveCell	
		//d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);
		//std::cerr << "B " << Fire.bloks4 << "\n";
		d_FdotVEntire << < Fire.bloks4, SMEMDIM >> > (P.d_V, P.d_F, Fire.d_FdotV, P.N);
		cudaMemcpy(Fire.h_FdotV, Fire.d_FdotV, Fire.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
		Fire.FdotV = 0;
		for (i = 0; i < Fire.bloks4; ++i)
		{
			Fire.FdotV += Fire.h_FdotV[i];
		}


		if (Fire.FdotV > 0)
		{
			++Fire.NPpositive;
			Fire.NPnegative = 0;
			if (Fire.NPpositive > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtgrow < Fire.dtmax) ? Fire.dt * Fire.dtgrow : Fire.dtmax;
				Fire.alpha *= Fire.alphashrink;
			}
			//std::cerr << "FdV POS " << steps <<" "<<Fire.dt << " " << Fire.alpha << " " << Fire.FdotV << "\n";
			//std::cin.get();
		}
		else
		{
			Fire.NPpositive = 0;
			++Fire.NPnegative;
			if (Fire.NPnegative > Fire.NPnegativeMax)
				break;
			if (steps > Fire.Ndelay)
			{
				Fire.dt = (Fire.dt * Fire.dtshrink > Fire.dtmin) ? Fire.dt * Fire.dtshrink : Fire.dtmin;
				Fire.alpha = Fire.alpha0;
			}
			d_calculateDecrementsHalfStepFIRE << < Padd.bloks, SMEMDIM >> > (P.d_V, P.d_U, P.N, Fire.dt);
			cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));
			//std::cerr << "FdV NEG " << steps << " " << Fire.dt << " " << Fire.alpha << " " << Fire.FdotV << "\n";
			//std::cin.get();
		}

#ifdef pre_CalcFullEnergy
		if (steps % 1000 == 0)
		{
			d_getEnergyEntire << < Padd.bloks4, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_1d_IM, P.d_U, P.d_V, Padd.d_Esum, P.N, P.NI, Po.c);
			cudaMemcpy(Padd.h_Esum, Padd.d_Esum, 2 * Padd.bloks4 * sizeof(float), cudaMemcpyDeviceToHost);
			Padd.Esum[0] = 0;
			Padd.Esum[1] = 0;
			for (i = 0; i < Padd.bloks4; ++i)
			{
				Padd.Esum[0] += Padd.h_Esum[i];
				Padd.Esum[1] += Padd.h_Esum[i + Padd.bloks4];
			}
			file_energy_txt << steps << " " << timereal << " " << Padd.Esum[0] - Ep0 << " " << Padd.Esum[1] << " " << Padd.Esum[0] - Ep0 + Padd.Esum[1] << "\n";
			if (steps % 10000 == 0)
				std::cerr << "E " << Padd.Esum[0] - Ep0 << " " << Padd.Esum[1] << " | " << Fire.FdotV << " | " << Fire.NPpositive << " " << Fire.NPnegative << " | " << Fire.dt << "\n";
			//if (steps == 0)Ep0 = Padd.Esum[0];
			
			
			if (steps % 1000000 == 0)
			{
				cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
				sprintf(filename, "./result/steps/CP_%li.txt", steps);
				SaveTXTParticles(P, P0, Po, Pnet, filename);
			}/**/
			
			//if (steps % 2000000 == 0)
			//	std::cin.get();
		}/**/
#endif // pre_CalcFullEnergy
		d_calculateIncrementsFIRE << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Fire.dt, Po.vis, Fire.alpha);
		
		/*if (steps % 1000 == 0)
		{
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);
			//std::cin.get();
		}/**/
		if (steps % 1000000 == 0)
			std::cerr << "Fire relaxation " << steps << "\n";
		timereal += Fire.dt;
		//std::cin.get();
	}	
	std::cerr << "FIN FIRE! " << steps << "\n"; //std::cin.get();
	Po.vis = 0;
	Po.vism = Po.vis * Po.m;
	cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));
	v = 0;
	for (unsigned int steps = 0; steps < Padd.time; ++steps)
	{

		estep = steps % Padd.ElementSteps;
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
#ifndef pre_MoveCell
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_MoveCell
#ifdef pre_MoveCell
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // pre_MoveCell	
		d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);
		/*if (estep == 0 && steps > 0)
		{
			cudaMemcpy(Padd.h_FResult + Estep * esize, Padd.d_FResult, esize * sizeof(float), cudaMemcpyDeviceToHost);
			std::cerr << "AAA!\n";
			++Estep;
		}/**/
		/*if (steps % 10 == 0)
		{
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);
			std::cin.get();
		}/**/
		//std::cin.get();
	}
	/*cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	sprintf(filename, "./result/steps/CP_%li.txt", steps);
	SaveTXTParticles(P, P0, Po, Pnet, filename);/**/
	//std::cerr << "AA! " << Padd.h_FResult << " " << Padd.d_FResult << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	cudaMemcpy(Padd.h_FResult, Padd.d_FResult, Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cerr << "AA! " << Padd.h_FResult << " " << Padd.d_FResult << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	//std::cerr << "AA! " << Padd.h_FResult[0] << " " << Padd.h_FResult[100] << " " << Padd.h_FResult[1000] << "\n";
	//std::cerr << "AAA!\n";
	calculate_Faver2(Padd, P, P0, Po);
	//calculate_Fminmax2(Padd, P, P0, Po);
	cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));


	//cudaMemcpy(Padd.h_Fstx, Padd.d_Fstx, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fsty, Padd.d_Fsty, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fetx, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fety, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
	//SaveTXTGraphsFR(Padd, P, P0, Po);

	//std::cin.get();/**/

	//HANDLE_ERROR(cudaMemcpy(Padd.h_Ek, Padd.d_Ek, (NGPUEk * 4 * P0.N * sizeof(float)), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemset((void*)Padd.d_Ek, 0, (NGPUEk * 4 * P0.N * sizeof(float))));
#ifdef pre_CalcFullEnergy	
	file_energy_txt.close();
#endif // pre_CalcFullEnergy

	free(Fire.h_FdotV);
	Fire.h_FdotV = nullptr;
	cudaFree(Fire.d_FdotV);
	Fire.d_FdotV = nullptr;
}


__global__ void d_FdotVEntire(const float* __restrict__ V, const float* __restrict__ F, float* FdotV, const unsigned int n)
{
	// static shared memory
	__shared__ float s_mem[SMEMDIM];

	// set thread ID
	// global index, 4 blocks of input data processed at a time
	unsigned int tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 4 + threadIdx.x, i;	
	// unrolling 4 blocks
	float fdv = 0;

	// boundary check
	if (idx + 3 * blockDim.x < n)
	{
		float t_fdv0 = 0, t_fdv1 = 0, t_fdv2 = 0, t_fdv3 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + 2 * blockDim.x;
		t_fdv2 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + 3 * blockDim.x;
		t_fdv3 = F[i] * V[i] + F[i + n] * V[i + n];
		fdv = t_fdv0 + t_fdv1 + t_fdv2 + t_fdv3;
	}
	else if (idx + 2 * blockDim.x < n)
	{
		float t_fdv0 = 0, t_fdv1 = 0, t_fdv2 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + 2 * blockDim.x;
		t_fdv2 = F[i] * V[i] + F[i + n] * V[i + n];		
		fdv = t_fdv0 + t_fdv1 + t_fdv2;
	}
	else if (idx + blockDim.x < n)
	{
		float t_fdv0 = 0, t_fdv1 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + n] * V[i + n];
		i = idx + blockDim.x;
		t_fdv1 = F[i] * V[i] + F[i + n] * V[i + n];		
		fdv = t_fdv0 + t_fdv1;
	}
	else if (idx < n)
	{
		float t_fdv0 = 0;
		i = idx;
		t_fdv0 = F[i] * V[i] + F[i + n] * V[i + n];		
		fdv = t_fdv0;
	}/**/

	//if(idx + 5 * blockDim.x >4619700)
	//   printf("TT %i %i %i %f %i\n", tid, idx, blockIdx.x, n);
	//if (ns>1e-3f)
	//   printf("TT %i %i %f\n", tid, idx, ns);
	s_mem[tid] = fdv;
	__syncthreads();

	//if(idx==0)
	//	printf("TT %i %f %f %i %i\n", tid, s_ek, e_ek, s_n, e_n);

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512)
	{
		s_mem[tid] += s_mem[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
	{
		//printf("Blok!\n");
		s_mem[tid] += s_mem[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
	{
		s_mem[tid] += s_mem[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
	{
		s_mem[tid] += s_mem[tid + 64];
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
		vsmem[tid] += vsmem[tid + 16];		
		vsmem[tid] += vsmem[tid + 8];		
		vsmem[tid] += vsmem[tid + 4];		
		vsmem[tid] += vsmem[tid + 2];		
		vsmem[tid] += vsmem[tid + 1];		
	}/**/

	// write result for this block to global mem
	if (tid == 0)
	{
		//printf("TT %i %i %i %f\n", tid, idx, blockIdx.x, 0);
		FdotV[blockIdx.x] = s_mem[0];
		//printf("TTT %i %i %i %f\n", tid, idx, blockIdx.x, FdotV[blockIdx.x]);
		//if (smem[tid + 3 * SMEMDIM] > 1e-3f)
		//	printf("TT %i %i %f\n", tid, idx, smem[tid + 3 * SMEMDIM]);
		//if (smem[3 * SMEMDIM] > 1e-3f)
		//printf("T %i %f\n", blockIdx.x, gridDim.x, smem[3 * SMEMDIM]);
	}/**/
}