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
#include <chrono>
#include "m_func.h"

//#include <cudpp.h>
//#include <cudpp_plan.h>


void calculateGPUSteps(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data &Pnet)
{
	//std::cerr << "W1\n";
	/*cudaEvent_t start, stop;
	float gpuTime;
	double gpuTimeAver = 0;/**/
	//float dV, maxShift = 10.0, Vl = sqrt(Po.a * Po.a * Po.c * Po._1d_m), Vd = 1.01;
	//Padd.V = 0.013;
	char filename[256] = "";
	size_t time, bloks, estep, esteps = ResultFRSave, Estep, esize = Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]), i, j,
		EFsize = 2 * Padd.ElementSteps * Padd.EF.NR;// , timestart, t1, t2;
	//std::cerr << "AAA! " << EFsize << " " << Padd.ElementSteps << " " << Padd.EF.NR << " " << 2 * Padd.ElementSteps << "\n";
	double d = 0;
	float v = Padd.V, fa_max, fb_min;
	bool contraction = true;
#ifdef pre_SineImpuls
	double omega = MC_pi * Padd.V / Padd.RMove, t1 = Padd.RMove / (Padd.V * Po.dt);
#endif // pre_SineImpuls

	time = Padd.time;
	//timestart = 1.01*(2.0 * Padd.MaxShift / Padd.V) / Po.dt + 1;
	//time = 10;
	//float v, v2;
	//time = 1500000;
	
	Padd.bloks = P.N / (SMEMDIM) + 2;
	Padd.bloksb = P.NBP / (SMEMDIM) + 2;
	std::cerr << "Bloks " << Padd.bloks << " " << Padd.bloksb << "\n";
	//Padd.blokst = time / (SMEMDIM);
	//t1 = Padd.MaxShift / (Padd.V * Po.dt);
	//v2 = (double(Padd.MaxShift) - t1 * Padd.V * Po.dt) / Po.dt;
	//t2 = (fabs(v2) < 1e-8) ? 0 : 1;
	//printf("cGS %u %u | %e %e %e %e | %e %e\n", time, Padd.blokst, Padd.MaxShift, 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vl, Padd.Vl, Po.dt, Padd.V, Po.vism);

	std::cerr << "0 Step" << " " << v << " " << d << " " << time << " | " << unsigned int(0.016235485 / Po.dt) << " " << Padd.ImpulsSteps + unsigned int(0.016235485 / Po.dt) << " " << "\n";
#ifdef pre_SaveLammps
	d_calculateForcesIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Fbound, P.N, P.NI, Po.c);
	cudaMemcpy(Padd.h_Fbound0, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	d_calculateUIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ubound, P.N, P.NI, Po.c);
	cudaMemcpy(Padd.h_Ubound0, Padd.d_Ubound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	unsigned int stepssavelamms = unsigned int(0.005 * 0.016235485 / Po.dt);
	stepssavelamms = unsigned int(Padd.time / 500);
	stepssavelamms = 5000;
	
	std::cerr << "SSLammps " << stepssavelamms << " " << Padd.ImpulsSteps + stepssavelamms * 100 << " | " << (Padd.ImpulsSteps + stepssavelamms / 0.005) / stepssavelamms << "\n"; //std::cin.get();
#ifdef pre_SaveLammpsEnergy
	d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
	cudaMemcpy(Padd.h_Ebound0, Padd.d_Ebound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	for (i = 0; i < P.N; ++i)
		Padd.h_Ek0[i] = P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N]);
#endif // pre_SaveLammpsEnergy	
#endif // pre_SaveLammps
#ifdef pre_SaveEnergyData
	//std::cerr << "A1\n"; std::cin.get();
	unsigned int stepssaveenergy = unsigned int(0.01 * 0.016235485 / Po.dt);
	stepssaveenergy = 18144;//0.001
	stepssaveenergy = 22889;//0.005
	stepssaveenergy = 30316;//0.01
	//stepssaveenergy = 23959;//0.05
	//stepssaveenergy = 46282;//0.1
	//stepssaveenergy = 45266;//0.5
	
	std::cerr << "SSEnergy " << stepssaveenergy <<" "<< stepssaveenergy* Po.dt << "\n";
	stepssaveenergy *= 0.01;
	//std::cin.get();
	d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
	//std::cerr << "A2\n"; std::cin.get();
	sprintf(filename, "./result/CP_Energy.txt");
	SaveEnergyDataStart(P, P0, Pnet, Padd, Po, filename, 0);
	//std::cerr << "A3\n"; std::cin.get();
#endif // pre_SaveEnergyData
#ifdef pre_OneCellEFdistribution
	d_calculateForcesI_EF << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, Padd.EF.d_EFb0, P.N, P.NI, Po.c);
#endif // pre_OneCellEFdistribution
	
	//std::cerr << "T1 " << time * Po.dt << " " << t1 << " " << Po.dt << "\n";
	Estep = 0;
	//esteps = 0;
	Po.CShfreefiber = 1.0 / (double(0.5 * Po.hfreefiber) * log(0.5 * double(Po.hfreefiber) / double(Po.rfiber)));
#ifdef pre_CylinderViscocityShapovalov
	std::cerr << "Po.CShfreefiber " << Po.CShfreefiber << " " << Po.hfreefiber << " " << Po.rfiber << "\n";
#endif // pre_CylinderViscocityShapovalov
	
	for (unsigned long int steps = 0; steps < time; ++steps)
	{
		if (steps % 100000 == 0)std::cerr << "Steps " << steps << "\n";
		//if (steps % 10 == 0)++esteps;
		estep = (steps/ esteps) % Padd.ElementSteps;
		if (estep >= Padd.ElementSteps)std::cerr << "AAAA! " << estep << " " << esize << "\n";
		//d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_U, P.d_F, P.N, Po.c, Po.a, Po._1d_a);
		//d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (const int* __restrict__ In, const float* __restrict__ Ir0, const float* __restrict__ U, float* __restrict__ IF, const unsigned int n, const unsigned int ni, const float P_c, const float P_a, const float P_1d_a);
		//d_sumForcesI << < Padd.bloks, SMEMDIM >> > (const int* __restrict__ In, const float* __restrict__ IF, float* F, const unsigned int n, const unsigned int ni)
		//std::cerr << "Q1\n";
#ifndef pre_OneCellEFdistribution
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
#ifdef pre_CylinderViscocity
#ifdef pre_CylinderViscocity1
		d_calculateVIscosForces << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_V, P.d_F, P.N, P.NI, Po.vis, 1.0 / Po.vis, float(Po.rfiber), float(Po.roliquid));
#endif // pre_CylinderViscocity1
#ifdef pre_CylinderViscocity2
		d_calculateVIscosForces2 << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_V, P.d_F, P.N, P.NI, Po.vis, 1.0 / Po.vis, float(Po.rfiber), float(Po.roliquid), float(2.0 / Po.hfreefiber));
#endif // pre_CylinderViscocity2
#ifdef pre_CylinderViscocity3
		d_calculateVIscosForces3 << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_V, P.d_F, P.N, P.NI, Po.vis, 1.0 / Po.vis, float(Po.rfiber), float(Po.roliquid), float(2.0 / Po.hfreefiber));
#endif // pre_CylinderViscocity3
#ifdef pre_CylinderViscocityShapovalov
		d_calculateVIscosForcesShapovalov << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_V, P.d_F, P.N, P.NI, Po.vis, 1.0 / Po.vis, float(Po.rfiber), float(Po.roliquid), Po.CShfreefiber);
#endif // pre_CylinderViscocityShapovalov
#ifdef pre_CylinderViscocityLindstrom
		d_calculateVIscosForcesLindstrom << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_V, P.d_F, P.N, P.NI, Po.vis, float(Po.rfiber), float(1.0 / Po.rfiber), float(Po.roliquid));
		//std::cin.get();
#endif // pre_CylinderViscocityLindstrom
#endif // pre_CylinderViscocity

		
		

#endif // !pre_OneCellEFdistribution
		
#ifdef pre_OneCellEFdistribution
		//std::cerr << "start distr "<<estep<<"\n";
		d_calculateForcesI_EF << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, Padd.EF.d_EFb, P.N, P.NI, Po.c);
		d_distrubuteBoundToCircle << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_U, P0.d_RU0, Padd.EF.d_EFb, Padd.EF.d_EFb0, Padd.EF.d_CEF, P.N, P.NI, estep, Padd.EF.P00[0], Padd.EF.P00[1], Padd.EF._1d_DR);
		//std::cerr << "E " << estep << " " << 2 * NumberCircles * estep << "\n";
		//std::cin.get();
#endif // pre_OneCellEFdistribution


		//std::cerr << "Q2\n";/
#ifndef pre_MoveCell
#ifndef pre_OneNodeContract
		d_calculateBordersVis << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_VisR, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, 0.0f, v);//Po.vis=0
#endif // !pre_OneNodeContract
#ifdef pre_OneNodeContract
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v, P.iBP[0]);
#endif // pre_OneNodeContract

#endif // !pre_MoveCell
#ifdef pre_MoveCell
#ifndef pre_OneNodeContract
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_OneNodeContract
#ifdef pre_OneNodeContract
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v, P.iBP[0]);
#endif // pre_OneNodeContract
#endif // pre_MoveCell	
		//std::cerr << "Q3\n";
		d_calculateIncrementsVis << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_VisR, P.d_F, P.d_V, P.d_U, P.N, Po.dt, 0.0f);//Po.vis=0
		//d_calculateIncrementsMove << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, Padd.d_FResult, P.d_V, P.d_U, P.N, Po.dt, Po.vis, estep);
		//std::cerr << "Q4\n";
		//std::cin.get();
		//if (steps % 10000 == 0)	std::cerr << "steps " << steps << "\n";
		/*if (steps % 1000 == 0)
		{
			std::cerr << steps << " " << v << " | " << d << " " << Padd.RMove << "\n";
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaMemcpy(Padd.h_Fbound, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);
		}/**/
#ifdef pre_SaveLammps
#ifdef pre_SaveLammpsPoint
		//if (steps % stepssavelamms == 0 && abs(int(steps) - int(Padd.LammpsPointSaveTime)) - 10 < stepssavelamms)
		if (steps == Padd.LammpsPointSaveTime)
#endif // pre_SaveLammpsPoint
#ifndef pre_SaveLammpsPoint
		if (steps % stepssavelamms == 0)
#endif // !pre_SaveLammpsPoint		
		{
			std::cerr << "Fin Step" << steps << " " << v << " " << d << "\n";
			d_calculateForcesIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Fbound, P.N, P.NI, Po.c);
			//std::cin.get();

			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(Padd.h_Fbound, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
			//std::cin.get();
#ifdef pre_SaveLammpsEnergy
			d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
			//std::cin.get();
			cudaMemcpy(Padd.h_Ebound, Padd.d_Ebound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);

#endif // pre_SaveLammpsEnergy			
#ifdef pre_SaveLammpsU
			d_calculateUIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ubound, P.N, P.NI, Po.c);
			//std::cin.get();
			cudaMemcpy(Padd.h_Ubound, Padd.d_Ubound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
#endif // pre_SaveLammpsU	
			//sprintf(filename, "./result/steps/CP_%li.txt", steps);
			//SaveTXTParticles(P, P0, Po, Pnet, filename);
			sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", steps);
			SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
			//std::cin.get();
		}
#endif // pre_SaveLammps	
#ifdef pre_SaveEnergyData
		if (steps % stepssaveenergy == 0)
		{
			std::cerr << "Fin Step" << steps << " " << v << " " << d << "\n";
			d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);

			//cudaMemcpy(Padd.h_Fbound, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
			//sprintf(filename, "./result/steps/CP_%li.txt", steps);
			//SaveTXTParticles(P, P0, Po, Pnet, filename);
			sprintf(filename, "./result/CP_Energy.txt");
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
			SaveEnergyDataStep(P, P0, Pnet, Padd, Po, filename, steps);
			//std::cin.get();
		}
#endif // pre_SaveEnergyData
		//std::cerr << "Q5\n";
		if (estep + 1 == Padd.ElementSteps && steps > 0 && steps% esteps ==0)
		//if (estep + 1 == Padd.ElementSteps && steps > 0)
		{
			//std::cerr << "AAA! "<< steps <<" "<< Estep << " " << esize << "\n";
			cudaMemcpy(Padd.h_FResult + Estep * esize, Padd.d_FResult, esize * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef pre_OneCellEFdistribution
			//if(esteps%10==0)
			cudaMemcpy(Padd.EF.h_CEF + Estep * EFsize, Padd.EF.d_CEF, EFsize * sizeof(float), cudaMemcpyDeviceToHost);
			HANDLE_ERROR(cudaMemset((void*)Padd.EF.d_CEF, 0, Padd.ElementSteps * 2 * Padd.EF.NR * sizeof(float)));
#endif // pre_OneCellEFdistribution			
			++Estep;
		}/**/
		//std::cerr << "Q6\n";
		/**if (steps % StepsToGPU == 0)
		{
			ai = steps / StepsToGPU;
			//fprintf(stderr, "calculateGPUSteps %i %i %i\n", steps, steps % StepsToGPU, ai);
			d_calculateKineticEnergy <<< Padd.bloks, SMEMDIM >>> (P.d_VU, P.d_VV, Padd.d_Ek, (4 * P.N * ai), P.N);
			//d_calculateKineticEnergy_precision <<< Padd.bloks, SMEMDIM >>> (P.d_VU, P.d_VV, P.d_FU, P.d_FV, Padd.d_Ek, (4 * P.N * ai), P.N, Po.dtm);
		}/**/
		//if (steps % 100 == 0)std::cin.get();
		//std::cin.get();
		/*if (fabs(steps - t1) < 1 || steps==0)
		{
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			double rrr = sqrt((P0.h_RU0[P.h_BP[10]] + P.h_U[P.h_BP[10]] + 0.121052175) * (P0.h_RU0[P.h_BP[10]] + P.h_U[P.h_BP[10]] + 0.121052175) + (P0.h_RU0[P.h_BP[10] + P0.N] + P.h_U[P.h_BP[10] + P.N]) * (P0.h_RU0[P.h_BP[10] + P0.N] + P.h_U[P.h_BP[10] + P.N]));
			std::cerr << "Check " << steps << " " << steps * Po.dt << " " << t1 << " " << Po.dt << " | " << P.h_BP[10] << " " << P0.h_RU0[P.h_BP[10]] << " " << P0.h_RU0[P.h_BP[10] + P0.N]
				<< " " << P.h_U[P.h_BP[10]] << " " << P.h_U[P.h_BP[10] + P.N]
				<< " | " << rrr << " " << rrr+Padd.RMove << " " << rrr - Padd.RMove - 0.03987475 <<" " << Padd.RMove << "\n";
			std::cin.get();
		}/**/
#ifdef pre_SineImpuls
		
#ifndef pre_ReleaseHalfSine
		if (steps < 2 * t1) v = 0.5 * Padd.RMove * omega * sin(omega * Po.dt * steps);
		else v = 0;/**/
#endif // !pre_ReleaseHalfSine
#ifdef pre_ReleaseHalfSine
		if (steps < t1) v = -0.5 * Padd.RMove * omega * sin(omega * Po.dt * steps);
		else v = 0;/**/
#endif // pre_ReleaseHalfSine
		//v = 0;
#endif // pre_SineImpuls
#ifdef pre_SquareImpuls
		if (contraction)
		{
			d += v * Po.dt;
			if (d + v * Po.dt > Padd.RMove)
			{
				v = (Padd.RMove - d) / Po.dt;
				//std::cerr << "VV " << v << " " << d << "\n";
				///d = Padd.RMove;
				contraction = false;
			}
		}
		else
		{
#ifdef TwoDirectionMove
			d += v * Po.dt;
			if (v > 0)
			{
				v = -Padd.V;
			}
			else if (d < 1e-12)
			{
				v = 0;
				d = 0;
			}
			else if (d + v * Po.dt < 0)
			{
				v = -d / Po.dt;
			}
#endif // TwoDirectionMove
#ifndef TwoDirectionMove
			v = 0;
#endif // !TwoDirectionMove		
		}		
#endif // pre_SquareImpuls
		//if(estep%10000==0)
		//if(v<1e-12)
		//std::cin.get();
		//if (steps % 10 == 0)std::cerr << steps << "\n";
	}
	//std::cerr << "AA! "<< Estep * esize << " " << Estep << " " << esize << " " << (estep + 1) * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << " " << ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	cudaMemcpy(Padd.h_FResult + Estep * esize, Padd.d_FResult, (estep + 1)* esize* size_t(sizeof(float)), cudaMemcpyDeviceToHost);
	//std::cerr << "AAA! " << Estep << " " << EFsize << " " << Padd.ElementSteps << " " << Padd.EF.NR << " " << estep << "\n";
#ifdef pre_OneCellEFdistribution
	cudaMemcpy(Padd.EF.h_CEF + Estep * EFsize, Padd.EF.d_CEF, (estep + 1) * 2 * Padd.EF.NR * sizeof(float), cudaMemcpyDeviceToHost);
	EFMinMaxData(P, P0, Pnet, Padd, Po);
#endif // pre_OneCellEFdistribution		
	/*for (j = 0; j < Padd.time; j += 500)
	{
		std::cerr << "RE";
		for (i = 0; i < Padd.EF.NR; ++i)
		{
			std::cerr << " " << Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i];
		}
		std::cerr << "\n";
		std::cin.get();
	}/**/
	//std::cin.get();


	calculate_Fminmax2(Padd, P, P0, Po);

	//cudaMemcpy(Padd.h_Fstx, Padd.d_Fstx, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fsty, Padd.d_Fsty, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fetx, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fety, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef pre_SaveFR
	SaveTXTGraphsFR(Padd, P, P0, Po,10);
#endif // pre_SaveFR



	//std::cin.get();/**/

	//HANDLE_ERROR(cudaMemcpy(Padd.h_Ek, Padd.d_Ek, (NGPUEk * 4 * P0.N * sizeof(float)), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemset((void*)Padd.d_Ek, 0, (NGPUEk * 4 * P0.N * sizeof(float))));	
}

void calculateGPUStepsContractRelax(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet)
{
	//std::cerr << "W1\n";
	/*cudaEvent_t start, stop;
	float gpuTime;
	double gpuTimeAver = 0;/**/
	//float dV, maxShift = 10.0, Vl = sqrt(Po.a * Po.a * Po.c * Po._1d_m), Vd = 1.01;
	///Padd.V = 2e-2;
	char filename[256] = "";
	unsigned int time, bloks, estep, Estep, esize = Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]), i, j, steps, stepStop = Padd.RelaxationTime;// , timestart, t1, t2;
	double d = 0, povis0 = Po.vis;
	float v = Padd.V, fa_max, fb_min, Ep0 = 2.21821;
	bool contraction = true;


	time = Padd.time;
	//timestart = 1.01*(2.0 * Padd.MaxShift / Padd.V) / Po.dt + 1;

	//float v, v2;
	//time = 1500000;

	Padd.bloks = P.N / (SMEMDIM)+2;
	Padd.bloksb = P.NBP / (SMEMDIM)+2;

	std::cerr << "Bloks " << Padd.bloks << " " << Padd.bloksb << "\n";
	//Padd.blokst = time / (SMEMDIM);
	//t1 = Padd.MaxShift / (Padd.V * Po.dt);
	//v2 = (double(Padd.MaxShift) - t1 * Padd.V * Po.dt) / Po.dt;
	//t2 = (fabs(v2) < 1e-8) ? 0 : 1;
	//printf("cGS %u %u | %e %e %e %e | %e %e\n", time, Padd.blokst, Padd.MaxShift, 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vl, Padd.Vl, Po.dt, Padd.V, Po.vism);
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
	std::chrono::steady_clock::time_point end;
	std::chrono::nanoseconds dr;
	Estep = 0;
#ifdef pre_SaveLammps
	d_calculateForcesIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Fbound, P.N, P.NI, Po.c);
	cudaMemcpy(Padd.h_Fbound0, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
#endif // pre_SaveLammps
	
#ifdef pre_SaveEnergyDataRelax
	//std::cerr << "A1\n"; std::cin.get();
	d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
	//std::cerr << "A2\n"; std::cin.get();
	sprintf(filename, "./result/CP_Energy.txt");
	SaveEnergyDataStart(P, P0, Pnet, Padd, Po, filename, 0);
	//std::cerr << "A3\n"; std::cin.get();
#endif // pre_SaveEnergyData
#ifdef pre_CalcFullEnergy
	sprintf(filename, "./result/CP_Energy.txt");
	std::ofstream file_energy_txt;
	file_energy_txt.open(filename, std::ios::out);
	file_energy_txt << "step time Ep Ek Efull\n";
	file_energy_txt.precision(10);
#endif // pre_CalcFullEnergy

	

	
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

	//std::cerr << "AAAAA!!!\n";
	//std::cin.get();
	if (d + v * Po.dt > Padd.RMove)
	{
		v = 0;
		contraction = false;
	}
	//for (unsigned int steps = 0; steps < 4000001; ++steps)
	for (steps = 0; steps < Padd.RelaxationTime; ++steps)
	{
		estep = 0;
		if (contraction)
		{			
			if (d + v * Po.dt > Padd.RMove)
			{
				v = (Padd.RMove - d) / Po.dt;
				//std::cerr << "VV " << v << " " << d << "\n";
				///d = Padd.RMove;
				contraction = false;
				stepStop = steps;
			}
			d += v * Po.dt;
		}
		else					
			v = 0;
		
		//d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_U, P.d_F, P.N, Po.c, Po.a, Po._1d_a);
		//d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (const int* __restrict__ In, const float* __restrict__ Ir0, const float* __restrict__ U, float* __restrict__ IF, const unsigned int n, const unsigned int ni, const float P_c, const float P_a, const float P_1d_a);
		//d_sumForcesI << < Padd.bloks, SMEMDIM >> > (const int* __restrict__ In, const float* __restrict__ IF, float* F, const unsigned int n, const unsigned int ni)
		//std::cerr << "Q1\n";
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
		//std::cerr << "Q2\n";
#ifndef pre_MoveCell
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_MoveCell
#ifdef pre_MoveCell
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // pre_MoveCell	
		//std::cerr << "Q3\n";
		d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);
		//std::cerr << "Q4\n";
		//std::cin.get();
#ifdef pre_CalcFullEnergy
		if ((steps - stepStop) % 1000 == 0 && steps >= stepStop)
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
			
			file_energy_txt << steps - stepStop << " " << (steps - stepStop) * Po.dt << " " << Padd.Esum[0] - Ep0 << " " << Padd.Esum[1] << " " << Padd.Esum[0] - Ep0 + Padd.Esum[1] << "\n";
			if (steps % 10000 == 0)
				std::cerr << "E " << Padd.Esum[0] << " " << Padd.Esum[1] << " | " << contraction << " " << v << "\n";
			//if (steps == stepStop)Ep0 = Padd.Esum[0];
			
			//if (steps % 2000000 == 0)
			//	std::cin.get();
		}/**/
#endif // pre_CalcFullEnergy

		/*if (steps == 500000)
		{
			Po.vis = 0;
			Po.vism = Po.vis * Po.m;
		}
		else if (steps == 1000000)
		{
			Po.vis = povis0;
			Po.vism = Po.vis * Po.m;
		}
		else if (steps > 1000000 && steps % 50000 == 0)
		{
			Po.vis *= 1.01;
			Po.vism = Po.vis * Po.m;
		}/**/
		if (steps > 1000000 && steps % 200000 == 0)
		{
			cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));			
		}/**/
		/*if (steps == 4000000)
		{
			Po.vis = 0;
			Po.vism = Po.vis * Po.m;
		}/**/
		if (steps > 499999 && steps % 10000 == 0)
		{
			Po.vis = povis0 * (0.5 - 0.5 * sin((steps - 500000) * 1e-5 * MC_pi + 1.5 * MC_pi));
			//std::cerr << "VIs " << Po.vis << " " << ((steps - 500000) * 1e-5 * MC_pi + 1.5 * MC_pi) / MC_pi << "\n";
			Po.vism = Po.vis * Po.m;
			//std::cin.get();
		}
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
#ifdef pre_SaveLammps
		if (steps % 100000 == 0)
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
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
		}
#endif // pre_SaveLammps

#ifdef pre_SaveEnergyDataRelax
		if (steps % 10000 == 0)
		{
			std::cerr << "Fin Step" << steps << " " << v << " " << d << "\n";
			d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
			

			//cudaMemcpy(Padd.h_Fbound, Padd.d_Fbound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
			//sprintf(filename, "./result/steps/CP_%li.txt", steps);
			//SaveTXTParticles(P, P0, Po, Pnet, filename);
			sprintf(filename, "./result/CP_Energy.txt");
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
			SaveEnergyDataStep(P, P0, Pnet, Padd, Po, filename, steps);
			std::cin.get();
		}		
#endif // pre_SaveEnergyData
		//std::cerr << "Q5\n";
		/*if (estep == 0 && steps > 0)
		{
			cudaMemcpy(Padd.h_FResult + Estep * esize, Padd.d_FResult, esize * sizeof(float), cudaMemcpyDeviceToHost);
			std::cerr << "AAA!\n";
			++Estep;
		}/**/
		//std::cerr << "Q6\n";
		/**if (steps % StepsToGPU == 0)
		{
			ai = steps / StepsToGPU;
			//fprintf(stderr, "calculateGPUSteps %i %i %i\n", steps, steps % StepsToGPU, ai);
			d_calculateKineticEnergy <<< Padd.bloks, SMEMDIM >>> (P.d_VU, P.d_VV, Padd.d_Ek, (4 * P.N * ai), P.N);
			//d_calculateKineticEnergy_precision <<< Padd.bloks, SMEMDIM >>> (P.d_VU, P.d_VV, P.d_FU, P.d_FV, Padd.d_Ek, (4 * P.N * ai), P.N, Po.dtm);
		}/**/
		//if (steps % 1000 == 0)std::cin.get();
		//std::cin.get();
		
		//std::cin.get();
	}
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
		//std::cin.get();
	}
	
	//std::cerr << "AA! " << Padd.h_FResult << " " << Padd.d_FResult << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	cudaMemcpy(Padd.h_FResult, Padd.d_FResult, Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cerr << "AA! " << Padd.h_FResult << " " << Padd.d_FResult << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	//std::cerr << "AA! " << Padd.h_FResult[0] << " " << Padd.h_FResult[100] << " " << Padd.h_FResult[1000] << "\n";
	//std::cerr << "AAA!\n";
	calculate_Faver2(Padd, P, P0, Po);
	//calculate_Fminmax2(Padd, P, P0, Po);
	cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));

#ifdef pre_CalcFullEnergy	
	file_energy_txt.close();
#endif // pre_CalcFullEnergy
	//cudaMemcpy(Padd.h_Fstx, Padd.d_Fstx, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fsty, Padd.d_Fsty, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fetx, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Padd.h_Fety, Padd.d_Fety, time * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef pre_SaveFR
	SaveTXTGraphsFR(Padd, P, P0, Po);
#endif // pre_SaveFR

	

	//std::cin.get();/**/

	//HANDLE_ERROR(cudaMemcpy(Padd.h_Ek, Padd.d_Ek, (NGPUEk * 4 * P0.N * sizeof(float)), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemset((void*)Padd.d_Ek, 0, (NGPUEk * 4 * P0.N * sizeof(float))));	
}

void calculateGPUStepsAverage(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet)
{	
	char filename[256] = "";
	unsigned int time, bloks, estep, Estep, esize = Padd.ElementSteps * 2 * (P.NBPT[0] + P.NBPT[1]), i, j;// , timestart, t1, t2;
	double d = 0;
	float v = Padd.V, fa_max, fb_min;
	bool contraction = true;


	time = Padd.time;
	//timestart = 1.01*(2.0 * Padd.MaxShift / Padd.V) / Po.dt + 1;

	//float v, v2;
	//time = 1500000;

	Padd.bloks = P.N / (SMEMDIM)+2;
	Padd.bloksb = P.NBP / (SMEMDIM)+2;
	std::cerr << "Bloks " << Padd.bloks << " " << Padd.bloksb << "\n";
	//Padd.blokst = time / (SMEMDIM);
	//t1 = Padd.MaxShift / (Padd.V * Po.dt);
	//v2 = (double(Padd.MaxShift) - t1 * Padd.V * Po.dt) / Po.dt;
	//t2 = (fabs(v2) < 1e-8) ? 0 : 1;
	//printf("cGS %u %u | %e %e %e %e | %e %e\n", time, Padd.blokst, Padd.MaxShift, 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vl, Padd.Vl, Po.dt, Padd.V, Po.vism);
	Estep = 0;
	v = 0;
#ifdef pre_OneCellEFdistribution	
	HANDLE_ERROR(cudaMemset((void*)Padd.EF.d_CEF, 0, Padd.ElementSteps * 2 * Padd.EF.NR * sizeof(float)));
#endif // pre_OneCellEFdistribution
	//std::cin.get();

	/*double maxE;
	int nimax;
	d_calculateEnergyIBound << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, Padd.d_Ebound, P.N, P.NI, Po.c);
	cudaMemcpy(Padd.h_Ebound0, Padd.d_Ebound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	maxE = Padd.h_Ebound0[0]; nimax = 0;
	for (i = 1; i < P.NI; ++i)
	{
		//std::cerr << i << " " << Padd.h_Ebound0[i] << "\n";
		if (fabs(maxE) < fabs(Padd.h_Ebound0[i]))
		{
			maxE = Padd.h_Ebound0[i];
			nimax = i;
		}
	}
	std::cerr.precision(10);
	std::cerr << "MaxE " << maxE << " " << nimax << "\n"; std::cin.get();/**/


	for (unsigned int steps = 0; steps < Padd.time; ++steps)
	{

		estep = steps % Padd.ElementSteps;
#ifndef pre_OneCellEFdistribution
		d_calculateForcesI << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, P.N, P.NI, Po.c);
#endif // !pre_OneCellEFdistribution

#ifdef pre_OneCellEFdistribution
		//std::cerr << "start distr "<<estep<<"\n";
		d_calculateForcesI_EF << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_Ir0, P.d_ShIn, P.d_U, P.d_F, Padd.EF.d_EFb, P.N, P.NI, Po.c);
		d_distrubuteBoundToCircle << < Padd.bloks, SMEMDIM >> > (P.d_In, P.d_U, P0.d_RU0, Padd.EF.d_EFb, Padd.EF.d_EFb0, Padd.EF.d_CEF, P.N, P.NI, estep, Padd.EF.P00[0], Padd.EF.P00[1], Padd.EF._1d_DR);
		//std::cerr << "E " << estep << " " << 2 * NumberCircles * estep << "\n";
		//std::cin.get();
#endif // pre_OneCellEFdistribution
#ifndef pre_MoveCell
		//d_calculateBordersFix << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#ifndef pre_OneNodeContract
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // !pre_OneNodeContract
#ifdef pre_OneNodeContract
		d_calculateBorders << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v, P.iBP[0]);
#endif // pre_OneNodeContract
#endif // !pre_MoveCell
#ifdef pre_MoveCell
		d_calculateBordersMove << < Padd.bloksb, SMEMDIM >> > (P.d_1d_IM, P.d_BP, P.d_BPR, P.d_F, P.d_V, P.d_U, Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep, Po.vis, v);
#endif // pre_MoveCell			
		d_calculateIncrements << < Padd.bloks, SMEMDIM >> > (P.d_1d_IM, P.d_F, P.d_V, P.d_U, P.N, Po.dt, Po.vis);

		/*if (steps % 10000 == 0)
		{			
			cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "./result/steps/CP_%li.txt", steps);
			SaveTXTParticles(P, P0, Po, Pnet, filename);			
			//sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", steps);
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, filename);
		}/**/
		/*if (estep == 0 && steps > 0)
		{
			cudaMemcpy(Padd.h_FResult + Estep * esize, Padd.d_FResult, esize * sizeof(float), cudaMemcpyDeviceToHost);
			std::cerr << "AAA!\n";
			++Estep;
		}/**/
#ifdef pre_SaveLammps
		if (steps % 40000 == 0)
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
			//SaveLammpsDATAParticles(P, P0, Po, Pnet, Padd, filename);
		}/**/
#endif // pre_SaveLammps
		//if(steps%10000==0)
		//	d_PrintFR << < Padd.bloksb, SMEMDIM >> > (Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep);
		//std::cin.get();
	}
#ifdef pre_FreeCellHalf
	unsigned int ii;
	cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	for (i = 0; i < P.NBPT[0]-1; ++i)
	{
		ii = P.h_BP[i];
		//std::cerr << "PBR " << P.h_BPR[i] << " " << P.h_BPR[i + P.NBP] << " | " << sqrt((P.h_BPR[i] + 0.121052175) * (P.h_BPR[i] + 0.121052175) + P.h_BPR[i + P.NBP] * P.h_BPR[i + P.NBP]) << " | ";
		P.h_BPR[i] = P0.h_RU0[ii] + P.h_U[ii];
		P.h_BPR[i + P.NBP] = P0.h_RU0[ii + P.N] + P.h_U[ii + P.N];
		//std::cerr << P.h_BPR[i] << " " << P.h_BPR[i + P.N] << " | " << P.h_BPR[i] + 0.121052175 << " | " << sqrt((P.h_BPR[i] + 0.121052175) * (P.h_BPR[i] + 0.121052175) + P.h_BPR[i + P.NBP] * P.h_BPR[i + P.NBP]) << "\n";
	}
	cudaMemcpy(P.d_BPR, P.h_BPR, 2 * P.NBP * sizeof(float), cudaMemcpyHostToDevice);
#endif // pre_FreeCellHalf

	
	//std::cerr << "FR " << Padd.d_FResult << " " << P.N << " " << P.NBP << " " << estep <<" | "<< Padd.bloksb << " "<< SMEMDIM << "\n";
	//std::cin.get();
	
	//for (unsigned int steps = 0; steps < 10000; ++steps)
	//	d_PrintFR << < Padd.bloksb, SMEMDIM >> > (Padd.d_FResult, P.N, P.NBP, P.NBPT[0], P.NBPT[1], P.NBPT[2], estep);
	
	//std::cin.get();
	//std::cerr << "AA!\n";
	//std::cerr << "FR " << Padd.time <<" "<< ResultFRNum <<" "<< (P.NBPT[0] + P.NBPT[1]) <<" "<< Padd.h_FResult << "\n";
	cudaMemcpy(Padd.h_FResult, Padd.d_FResult, Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef pre_OneCellEFdistribution
	//if(esteps%10==0)
	cudaMemcpy(Padd.EF.h_CEF, Padd.EF.d_CEF, Padd.time * 2 * Padd.EF.NR * sizeof(float), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(cudaMemset((void*)Padd.EF.d_CEF, 0, Padd.ElementSteps * 2 * Padd.EF.NR * sizeof(float)));
	EFAverageData(P, P0, Pnet, Padd, Po);
#endif // pre_OneCellEFdistribution	
	cudaMemset(P.d_V, 0, 2 * P.N * sizeof(float));
	//std::cerr << "AAA!\n";
	

	//std::cerr << "FR " << Padd.h_FResult[0] << " " << Padd.h_FResult[1] << " " << Padd.h_FResult[2] << "\n";
	calculate_Faver2(Padd, P, P0, Po);
	//SaveTXTGraphsFR(Padd, P, P0, Po);
	//std::cin.get();
}