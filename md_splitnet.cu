#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math_functions.h>
#include <iostream>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <vector>
//#include "lattice_constans.h"



void SplitNet(p_data& P, p0_data& P0, l_data& L, pAdd_data& Padd, potential_data& Po, pNet_data& Pnet)
{
	//fprintf(stderr, "Start createLattice\n");
	P.h_F = nullptr;
	P.h_V = nullptr;
	P.h_U = nullptr;

	P.d_F = nullptr;
	P.d_V = nullptr;
	P.d_U = nullptr;


	P0.h_RU0 = nullptr;
	//P0.h_Ri = nullptr;

	P0.d_RU0 = nullptr;
	P0.d_U0 = nullptr;
	//P0.d_Ri = nullptr;


	Padd.h_Fmm = (float*)malloc(ResultFmmNum * (Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float));
	Padd.h_V = (float*)malloc((Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float));

	//Padd.d_Fmm = nullptr;
	//Padd.d_Fstx = nullptr;
	//Padd.d_Fsty = nullptr;
	//Padd.d_Fetx = nullptr;
	//Padd.d_Fety = nullptr;
	//Padd.d_cmax = nullptr;
	//Padd.d_cmin = nullptr;

	//Pnet.ScShift = int(Pnet.L/Pnet.a_aver);
#ifndef pre_OnlyOneCell
	P.N = Pnet.SN + 2;
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	P.N = Pnet.SN + 1;
#endif // pre_OnlyOneCell


	P.h_U = (float*)malloc(2 * P.N * sizeof(float));
	P.h_V = (float*)malloc(2 * P.N * sizeof(float));
	P.h_F = (float*)malloc(2 * P.N * sizeof(float));
	P0.N = P.N;
	P0._1d_N = P._1d_N;
	P0.h_RU0 = (float*)malloc(2 * P.N * sizeof(float));

	memset(P.h_U, 0, 2 * P.N * sizeof(float));
	memset(P.h_V, 0, 2 * P.N * sizeof(float));
	memset(P.h_F, 0, 2 * P.N * sizeof(float));	
	int_fast32_t i, j, k, kk, m, mm[3], mmm;

#ifndef pre_OnlyOneCell
	double r[3][2] = { { -Pnet.CellDistance * 0.5 * ReadCoordinatesCoefficient,0 }, { Pnet.CellDistance * 0.5 * ReadCoordinatesCoefficient,0 }, { 0,0 } },//0.121052175
		rr[3] = { 0.03987475 * 0.03987475 * RCC2, 0.03987475 * 0.03987475 * RCC2, Pnet.Rcut * Pnet.Rcut * RCC2 },
		drr[3] = { 1e-4 * RCC2, 1e-4 * RCC2, 2e-2 * RCC2 },
		drx, dry, drm, _1d_rm;
	P.NBP = 2; P.NBPT[0] = 1; P.NBPT[1] = 1; P.NBPT[2] = 0;
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	double r[3][2] = { { 0,0 }, { 1e30,1e30 }, { 0,0 } },
		rr[3] = { 0.03987475 * 0.03987475, 0.03987475 * 0.03987475, Pnet.Rcut * Pnet.Rcut },
		drr[3] = { 1e-4, 1e-4, 2e-2 },
		drx, dry, drm, _1d_rm;
	P.NBP = 1; P.NBPT[0] = 1; P.NBPT[1] = 0; P.NBPT[2] = 0;
#endif // pre_OnlyOneCell	
#ifdef pre_AlignCell
	double rn, an, nx, ny;
#endif // pre_AlignCell
	for (i = 0; i < Pnet.SN; ++i)
	{
		P0.h_RU0[i] = Pnet.h_S[i];
		P0.h_RU0[i + P.N] = Pnet.h_S[i + Pnet.SN];
		for (m = 0; m < 3; ++m)
		{
			drx = P0.h_RU0[i] - r[m][0];
			dry = P0.h_RU0[i + P.N] - r[m][1];
			drm = drx * drx + dry * dry;
			if (fabs(drm - rr[m]) < drr[m])
			{
#ifdef pre_AlignCell				
				//rn = sqrt(drm);
				if (m < 2)
				{
					//std::cerr << "R " << drx << " " << dry << " " << sqrt(drm) - 0.03987475 << " | ";
					an = -1 + sqrt(rr[m] / drm);
					P0.h_RU0[i] = r[m][0] + drx + an * drx;
					P0.h_RU0[i + P.N] = r[m][1] + dry + an * dry;
					drx = P0.h_RU0[i] - r[m][0];
					dry = P0.h_RU0[i + P.N] - r[m][1];
					drm = drx * drx + dry * dry;
					//std::cerr << drx << " " << dry << " " << sqrt(drm) - 0.03987475 << "\n";
				}				
#endif // pre_AlignCell
				++P.NBP;
				++P.NBPT[m];
			}
		}
		//std::cerr<<"P "<< P0.h_RU0[i + Pnet.ScShift]<< " "
	}
	//std::cin.get();
#ifndef pre_OnlyOneCell
	P0.h_RU0[P.N - 2] = r[0][0];
	P0.h_RU0[2 * P.N - 2] = r[0][1];
	P0.h_RU0[P.N - 1] = r[1][0];
	P0.h_RU0[2 * P.N - 1] = r[1][1];
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	P0.h_RU0[P.N - 1] = r[0][0];
	P0.h_RU0[2 * P.N - 1] = r[0][1];
#endif // pre_OnlyOneCell

	P.h_BP = (int*)malloc(P.NBP * sizeof(int));
	P.h_BPR = (float*)malloc(2 * P.NBP * sizeof(float));
	std::cerr << "BP " << P.NBP << " " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << " | " << r[0][0] << " " << r[1][0] << "\n";
	double mmax=0, mmin=1e39, minPdt=1e99, varPdt, maxC = 0, varC;
	k = 0;
	mm[0] = 0; mm[1] = 0; mm[2] = 0;
	
	/**/
#ifndef pre_OnlyOneCell
	P.h_BP[P.NBPT[0] - 1] = P.N - 2;
	P.h_BPR[P.NBPT[0] - 1] = P0.h_RU0[P.N - 2];
	P.h_BPR[P.NBPT[0] - 1 + P.NBP] = P0.h_RU0[2 * P.N - 2];
	P.h_BP[P.NBPT[0] + P.NBPT[1] - 1] = P.N - 1;
	P.h_BPR[P.NBPT[0] + P.NBPT[1] - 1] = P0.h_RU0[P.N - 1];
	P.h_BPR[P.NBPT[0] + P.NBPT[1] - 1 + P.NBP] = P0.h_RU0[2 * P.N - 1];
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	P.h_BP[P.NBPT[0] - 1] = P.N - 1;
	P.h_BPR[P.NBPT[0] - 1] = P0.h_RU0[P.N - 1];
	P.h_BPR[P.NBPT[0] - 1 + P.NBP] = P0.h_RU0[2 * P.N - 1];
#endif // pre_OnlyOneCell
#ifdef pre_ConnectCellSurface
	//unsigned int panN = (P.NBPT[0] > P.NBPT[1]) ? P.NBPT[0] : P.NBPT[1];
	double *pangle = (double*)malloc((P.NBPT[0] + P.NBPT[1]) * sizeof(double));
#endif // pre_ConnectCellSurface
	for (i = 0; i < P.N; ++i)
	{
		mmm = 0;
		for (m = 0; m < 3; ++m)
		{
			drx = P0.h_RU0[i] - r[m][0];
			dry = P0.h_RU0[i + P.N] - r[m][1];
			drm = drx * drx + dry * dry;
			if (fabs(drm - rr[m]) < drr[m])
			{
				//if (m == 0)
				//	std::cerr << "BP " << i << " " << m << " " << mm[m] << " " << mmm << " | " << P0.h_RU0[i] << " " << P0.h_RU0[i + P.N] << "\n";
				P.h_BP[mm[m] + mmm] = i;
				P.h_BPR[mm[m] + mmm] = P0.h_RU0[i];
				P.h_BPR[mm[m] + mmm + P.NBP] = P0.h_RU0[i + P.N];
#ifdef pre_ConnectCellSurface
				if (m < 2 && drm>1e-10)
				{
					_1d_rm = 1.0 / sqrt(drm);
					if (dry * _1d_rm > 0)
						pangle[mm[m] + mmm] = acos(drx * _1d_rm);
					else
						pangle[mm[m] + mmm] = 2.0 * MC_pi - acos(drx * _1d_rm);
					//std::cerr << "BP " << m << " " << mm[m] << " | " << 180.0 * (pangle[mm[m] + mmm] * MC_1d_pi) << " | " << drx << " " << dry << " " << drm << "\n";
				}				
#endif // pre_ConnectCellSurface
				++mm[m];
				//goto L__EndCicle;
			}
			mmm += P.NBPT[m];
			//std::cin.get();
		}
	}
	/*std::cin.get();
	for (i = 0; i < P.NBPT[0] - 1; ++i)
		std::cerr << "PBR0 " << P.h_BPR[i] << " " << P.h_BPR[i + P.N] << " | " << sqrt((P.h_BPR[i] + 0.121052175) * (P.h_BPR[i] + 0.121052175) + P.h_BPR[i + P.N] * P.h_BPR[i + P.N]) << "\n";
	std::cin.get();/**/
#ifdef pre_ConnectCellSurface
#ifndef pre_OnlyOneCell
	unsigned int ACSc = P.NBPT[0] - 1 + P.NBPT[1] - 1, iACSc;
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	unsigned int ACSc = P.NBPT[0] - 1, iACSc;
#endif // DEBUG	
	unsigned int* h_Sc_new = (unsigned int*)malloc(2 * (Pnet.ScN + ACSc) * sizeof(unsigned int));
	memcpy(h_Sc_new, Pnet.h_Sc, Pnet.ScN * sizeof(unsigned int));
	memcpy(h_Sc_new + Pnet.ScN + ACSc, Pnet.h_Sc + Pnet.ScN, Pnet.ScN * sizeof(unsigned int));
	unsigned int* h_Sc_t = Pnet.h_Sc;
	Pnet.h_Sc = h_Sc_new;
	free(h_Sc_t); h_Sc_t = nullptr; h_Sc_new = nullptr;
	P.h_BPDfi = (float*)malloc((P.NBPT[0] + P.NBPT[1]) * sizeof(float));
	memset(P.h_BPDfi, 0, (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
	double dpa, dpamin=1e+30;
	unsigned int ibpn;
	mmm = 0;
	iACSc = 0;
	for (m = 0; m < 2; ++m)
	{		
		for (i = 0; i < P.NBPT[m]-1; ++i)
		{
			dpamin = 2.0 * MC_pi;
			for (j = 0; j < P.NBPT[m]-1; ++j)
			{
				if (i == j)continue;
				dpa = pangle[j + mmm] - pangle[i + mmm];
				if (dpa < 0) dpa += 2.0 * MC_pi;
				if (dpamin > dpa)
				{
					dpamin = dpa;
					ibpn = j;
				}
			}
			Pnet.h_Sc[Pnet.ScN + iACSc] = P.h_BP[i + mmm];
			Pnet.h_Sc[Pnet.ScN + iACSc + ACSc + Pnet.ScN] = P.h_BP[ibpn + mmm];
			dpa = pangle[ibpn + mmm] - pangle[i + mmm];
			if (dpa < 0) dpa += 2.0 * MC_pi;
			P.h_BPDfi[i + mmm] += 0.5 * dpa;
			P.h_BPDfi[ibpn + mmm] += 0.5 * dpa;
			++iACSc;

			//if(i + mmm> P.NBPT[0] + P.NBPT[1] || ibpn + mmm > P.NBPT[0] + P.NBPT[1])std::cerr<<"ERR!!!\n";
			//std::cerr << "BPC " << i << " " << ibpn << " | " << pangle[i + mmm] << " " << pangle[ibpn + mmm] << " | " << P.h_BP[i + mmm] << " " << P.h_BP[ibpn + mmm] 
			//	<<" | "<<sqrt((P0.h_RU0[P.h_BP[i + mmm]]- P0.h_RU0[P.h_BP[ibpn + mmm]])* (P0.h_RU0[P.h_BP[i + mmm]] - P0.h_RU0[P.h_BP[ibpn + mmm]]))<< "\n";
			//std::cin.get();
		}
		mmm += P.NBPT[m];
	}
	Pnet.ScN += ACSc;
	std::cerr << "BPSC " << iACSc << " " << P.NBPT[0] - 1 + P.NBPT[1] - 1 << "\n";
	/*dpa = 0;
	for (i = 0; i < P.NBPT[0] + P.NBPT[1]; ++i)
	{
		dpa += P.h_BPDfi[i];
		std::cerr << "Angle " << i << " " << P.h_BPDfi[i] << " " << dpa << "\n";
		if (i == P.NBPT[0])dpa = 0;
	}/**/
	free(pangle); pangle = nullptr;
#endif // pre_ConnectCellSurface
	//std::cin.get();

	P.NI = 2 * Pnet.ScN;
	P.h_In = (int*)malloc(2 * P.NI * sizeof(int));
	P.h_ShIn = (int*)malloc(2 * P.N * sizeof(int));
	P.h_Ir0 = (float*)malloc(3 * P.NI * sizeof(float));
	P.h_IM = (float*)malloc(P.N * sizeof(float));
	P.h_1d_IM = (float*)malloc(P.N * sizeof(float));
	P.h_VisR = (float*)malloc(P.N * sizeof(float));

	for (i = 0; i < P.N; ++i)
	{	
		P.h_ShIn[i] = k;		
		P.h_IM[i] = 0;
		P.h_VisR[i] = 0;
		kk = 0;
		maxC = 0;
		for (j = 0; j < Pnet.ScN; ++j)
		{
			if (Pnet.h_Sc[j] == i)
			{
				P.h_In[k] = i;// +Pnet.ScShift;
				P.h_In[k + P.NI] = Pnet.h_Sc[j + Pnet.ScN];// +Pnet.ScShift;
				drx = P0.h_RU0[P.h_In[k + P.NI]] - P0.h_RU0[i];
				dry = P0.h_RU0[P.h_In[k + P.NI] + P.N] - P0.h_RU0[i + P.N];
				P.h_Ir0[k] = drx;
				P.h_Ir0[k + P.NI] = dry;
				drm = sqrt(drx * drx + dry * dry);
				/*if (drm > 1e-1)
				{
					std::cerr << "ERR1 " << drm << " " << P.h_In[k] << " " << P.h_In[k + P.NI] << "\n";
					std::cin.get();
				}/**/
				P.h_Ir0[k + 2 * P.NI] = float(1.0 / ((1.0 - Pnet.InitialDeformation) * drm));
				P.h_IM[i] += 0.5 * Po.m * drm;
				P.h_VisR[i] += 0.5 * Po.Sfiber * drm;
				varC = Po.c / drm;
				if (varC > maxC) maxC = varC;
				//std::cerr << "A " << P.In[k] << " " << P.In[k + P.NI] << "\n";
				++k;
				++kk;
			}
			else if (Pnet.h_Sc[j + Pnet.ScN] == i)
			{
				P.h_In[k] = i;// +Pnet.ScShift;
				P.h_In[k + P.NI] = Pnet.h_Sc[j];// +Pnet.ScShift;
				drx = P0.h_RU0[P.h_In[k + P.NI]] - P0.h_RU0[i];
				dry = P0.h_RU0[P.h_In[k + P.NI] + P.N] - P0.h_RU0[i + P.N];
				P.h_Ir0[k] = drx;
				P.h_Ir0[k + P.NI] = dry;
				drm = sqrt(drx * drx + dry * dry);
				/*if (drm > 1e-1)
				{
					std::cerr << "ERR2 " << drm << " " << P.h_In[k] << " " << P.h_In[k + P.NI] << "\n";
					std::cin.get();
				}/**/
				P.h_Ir0[k + 2 * P.NI] = float(1.0 / ((1.0 - Pnet.InitialDeformation) * drm));
				P.h_IM[i] += 0.5 * Po.m * drm;
				//std::cerr << "B " << P.In[k] << " " << P.In[k + P.NI] << "\n";
				P.h_VisR[i] += 0.5 * Po.Sfiber * drm;
				varC = Po.c / drm;
				if (varC > maxC) maxC = varC;
				++k;
				++kk;
			}
		}/**/
		P.h_1d_IM[i] = 1.0f / P.h_IM[i];
		P.h_VisR[i] = pow(0.75 * P.h_VisR[i] * MC_1d_pi, MC_1d3);
		P.h_ShIn[i + P.N] = kk;
		if (mmax < P.h_IM[i]) mmax = P.h_IM[i];
		//if (mmin > P.h_IM[i]) std::cerr << "Mmin " << i << " " << P.h_IM[i] << "\n";
		if (mmin > P.h_IM[i] && P.h_IM[i]>1e-12) mmin = P.h_IM[i];
		varPdt = 0.01 * MC_pi * sqrt(P.h_IM[i] / maxC);
		//std::cerr << "maxC " << maxC << " " << varPdt<< " " << P.h_IM[i] << "\n";
		//std::cin.get();
		if (minPdt > varPdt) minPdt = varPdt;
		//std::cerr << "M " << i << " " << P.h_IM[i] << "\n";
		//std::cin.get();
//L__EndCicle:
		if(i%10000==0)std::cerr << "i "<< i <<" "<< P.NI << " " << k << " "<< P.N << "\n";
		//std::cin.get();
	}
#ifndef pre_Relaxation
	P.h_1d_IM[P.N - 2] = 0.1 / mmax;
	P.h_1d_IM[P.N - 1] = 0.1 / mmax;
#endif // !pre_Relaxation	
#ifdef pre_Relaxation
	P.h_1d_IM[P.N - 2] = 0.1 / mmax;
	P.h_1d_IM[P.N - 1] = 0.1 / mmax;
#endif // pre_Relaxation
	//std::cin.get();
	
	std::cerr << "BP LAST A " << P.h_BP[P.NBPT[0] - 1] << " " << P.h_BPR[P.NBPT[0] - 1] << " " << P.h_BPR[P.NBPT[0] - 1 + P.NBP] << "\n";
	std::cerr << "BP LAST B " << P.h_BP[P.NBPT[0] + P.NBPT[1] - 1] << " " << P.h_BPR[P.NBPT[0] + P.NBPT[1] - 1] << " " << P.h_BPR[P.NBPT[0] + P.NBPT[1] - 1 + P.NBP] << "\n";
	std::cerr << "I " << P.NI << " " << k << "\n";
	std::cerr << "Mass " << mmin << " " << mmax << " | " << Po.m << "\n";
	std::cerr << "dT " << Po.dt << " " << minPdt << " " << maxC << "\n";
	Po.dt = minPdt;
	//std::cin.get();
	P.NI = k;

	//std::cerr << "BB " << P0.h_RU0[0] << " " << P0.h_RU0[P.N] << "\n";
	std::cerr << "Data " << (6 * P.N * sizeof(float) + 2 * P.N * sizeof(int) + 5 * P.NI * sizeof(float) + 2 * P.NI * sizeof(int))/(1024*1024) << "Mb\n";
	//std::cin.get();
	//for (m = 0; m < P.NBPT[0]; ++m)	
	//	std::cerr << "BP " << m << " " << P.h_BPR[m] << " " << P.h_BPR[m + P.NBP] << "\n";		
	//std::cin.get();
	mmax = -1e39;
	for (i = 0; i < P.NBPT[0]; ++i)
	{
		if (P.h_BPR[i] > mmax)
		{
			k = i;
			mmax = P.h_BPR[i];
		}

	}
	P.iBP[0] = k;
	mmax = 1e39;
	for (i = P.NBPT[0]; i < P.NBPT[0] + P.NBPT[1]; ++i)
	{
		if (P.h_BPR[i] < mmax)
		{
			k = i;
			mmax = P.h_BPR[i];
		}
	}
	P.iBP[1] = k;
	std::cerr << "BP " << P.iBP[0] << " " << P.h_BP[P.iBP[0]] << " " << P0.h_RU0[P.h_BP[P.iBP[0]]] << "\n";
	std::cerr << "BP " << P.iBP[1] << " " << P.h_BP[P.iBP[1]] << " " << P0.h_RU0[P.h_BP[P.iBP[1]]] << "\n";
	//std::cerr << "PN " << P.N << "\n";
	//std::cin.get();
	HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_In, 2 * P.NI * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_ShIn, 2 * P.N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_Ir0, 3 * P.NI * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_1d_IM, P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_BP, P.NBP * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_BPR, 2 * P.NBP * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_VisR, P.N * sizeof(float)));

	std::cerr << "Fin create\n";
	//std::cin.get();
	//std::cin.get();
	HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(P.d_In, P.h_In, 2 * P.NI * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_ShIn, P.h_ShIn, 2 * P.N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_Ir0, P.h_Ir0, 3 * P.NI * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_1d_IM, P.h_1d_IM, P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_BP, P.h_BP, P.NBP * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_BPR, P.h_BPR, 2 * P.NBP * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_VisR, P.h_VisR, P.N * sizeof(float), cudaMemcpyHostToDevice));

	//std::cerr << "FR " << Padd.d_FResult << " " << Padd.h_FResult << " " << Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << "\n";
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_FResult, Padd.ElementSteps * ResultFRNum* (P.NBPT[0] + P.NBPT[1]) * sizeof(float)));
	//std::cerr<<"Data2 "<< (Padd.ElementSteps * 2 * (P.NBPT[0] + P.NBPT[1]) * sizeof(float)) / (1024.0 * 1024.0) << "Mb "<< P.NBPT[0] + P.NBPT[1] << "\n";
	Padd.h_FResult = (float*)malloc(Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
	//std::cerr << "FR " << Padd.d_FResult << " " << Padd.h_FResult << "\n";
#ifdef pre_CalcFullEnergy
	Padd.bloks4 = P.N / (4 * SMEMDIM) + 1;
	Padd.h_Esum = (float*)malloc(2 * Padd.bloks4 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Esum, 2 * Padd.bloks4 * sizeof(float)));
#endif // pre_CalcFullEnergy


#ifdef pre_SaveLammps
	Padd.h_Fbound0 = (float*)malloc(P.NI * 1 * sizeof(float));
	Padd.h_Fbound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fbound, P.NI * 1 * sizeof(float)));
	Padd.h_LammpsAddParticles = (float*)malloc((P.NI / 2) * 4 * sizeof(float));
	Padd.h_LammpsSumF = (float*)malloc(P.N * 1 * sizeof(float));
#endif // pre_SaveLammps
#ifdef pre_SaveEnergyData
	//Padd.h_Ebound0 = (float*)malloc(P.NI * 1 * sizeof(float));
	Padd.h_Ebound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Ebound, P.NI * 1 * sizeof(float)));
#endif // pre_SaveEnergyData

	
	fprintf(stderr, "Finish createLattice %i\n", P.N);
	//std::cin.get();
}