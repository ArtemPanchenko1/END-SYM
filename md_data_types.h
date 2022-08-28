#pragma once
#include <cuda.h>
#include <vector_functions.h>
#include <curand.h>
#include <iostream>
#include <fstream>
#define ResultFmmNum 66
#define ResultFRNum 4
#define NumberCircles 61
#define ResultEFNum 4
extern "C"
{
	struct EFDist_data
	{
		double P00[2], DR, _1d_DR, FCAverage;
		double dR, Dfi, dfi;
		unsigned int NR, iStep, NSteptoH, NStepDef;
		unsigned int Nfi, NRfi, NStep, NStepR, NStepRfi;
		unsigned int* h_BtoC, *h_CtoB;
		unsigned int* d_BtoC, * d_CtoB;
		float* h_EFb, * h_EFb0, *h_CEF;
		float* d_EFb, * d_EFb0, *d_CEF;
		double* EFMinMax, * CEFAverage, *EFcell0, * EFcell;
	};

	struct p_data
	{
		float* h_F, *h_V, *h_U, *h_Ir0, *h_BPR, *h_IM, * h_1d_IM, * h_BPDfi, *h_VisR;
		float* d_F, *d_V, *d_U, *d_Ir0, *d_1d_IM, *d_BPR, * d_VisR;
		int* h_In, *h_BP, * h_ShIn;
		int* d_In, *d_BP, *d_ShIn;
		//float* d_VU1, *d_U1, *d_VV1, *d_V1;
		//float* h_EkU, *h_EkV, *d_EkU, *d_EkV;
		int N, NI, NBP, NBPT[3], iBP[2];
		float _1d_N;
	};
	struct p0_data
	{
		float *h_RU0;
		float *d_RU0, *d_U0;
		//int* h_Ri;
		//int* d_Ri;
		int N;
		float _1d_N;
	};
	struct pAdd_data
	{
		float* d_Fmm, /*d_cmax, *d_cmin, *d_Fstx, * d_Fsty, * d_Fetx, * d_Fety,*/* d_FResult, * h_V, * h_Fmm,
			/* h_Fstx, * h_Fsty, * h_Fetx, * h_Fety,*/* h_FResult, * d_Fbound, * h_Fbound, * h_LammpsAddParticles, * h_LammpsSumF,
			* h_Fbound0, * d_Ebound, * h_Ebound, * h_Ebound0, * h_Ek0, * d_Esum, * h_Esum, * d_Ubound, * h_Ubound, * h_Ubound0;
		unsigned int bloks, bloksb, blokst, bloks4, StepVN[3], StepV, MaxTimeStep, ElementSteps, ImpulsSteps, RelaxationTime, LammpsPointSaveTime;
		size_t time;
		float V, Eps0, Eps, V0[3], dV[3], Fmm, MaxShift, Vl, Vt, RMove;
		double FSAB_aver[34], FABi_aver[8], Esum[2], LammpsPointSavetime, logvcoeff, mintauILT, maxtauILT;
		curandGenerator_t gen;
		EFDist_data EF;
	}; 
	struct pNet_data
	{
		double* h_S;
		unsigned int* h_Sc;
		unsigned int SN, ScN, AdN;
		double L, minrm, maxrm, a_aver, Rcut, InitialDeformation;
		double Connectivity, CellDistance;
		int Nnodes;
	};

	struct h_data
	{
		
	};
	struct EnergyDist_data
	{
		double P00[2], DR, dR, Dfi, dfi;
		unsigned int NR, Nfi, NRfi, iStep, NStep, NStepR, NStepRfi;
		unsigned int* h_Step;
		double* h_E, *h_sE, * h_Ebound0, * h_Ek0;
		
	};
	

	//struct p_data2
	//{
	//	float *Cx, *Cy, * Cz, *Vx, * Vy, * Vz, /* W,*/* Fx, * Fy, * Fz;
	//	float* devCx, * devCy, * devCz, * devVx, * devVy, * devVz, /* W,*/* devFx, * devFy, * devFz;
	//	int N;
	//	float _1d_N;
	//};

	struct b_data
	{
		
		
	};
	struct l_data
	{
		float2 PS, LV, rCenter;
		int2 n;
		float V, _1d_V;
		int iCenter;
	};
	struct param_data
	{
		//float Ek, EkSpot, rSpot, xCenter, yCenter;
		//int NSpot;
		//unsigned int StepV1, StepV2;
		//float V, Eps0, V01, V02, dV1, dV2;
	};
	/*struct Lindstromparam_data
	{
		double l, b, a, e, L, XA, YA, XC, YC, YH;
		void calcparam(double lvar, double rvar, double muvar)
		{
			double c = MC_pi * muvar;
			l = lvar;
			a = 0.5 * lvar;
			b = 1.0 / 1.24 * rvar * sqrt(log(0.5 * lvar / rvar));
			e = sqrt(1.0 - b * b / (a * a));
			L = log((1.0 + e) / (1.0 - e));
			XA = c * 8.0 * MC_1d3 * e * e * e / (-2.0 * e + (1.0 + e * e) * L);
			YA = c * 16.0 * MC_1d3 * e * e * e / (2.0 * e + (3.0 * e * e - 1.0) * L);
			XC = c * 4.0 * MC_1d3 * e * e * e * (1.0 - e * e) / (2.0 * e - (1.0 - e * e) * L);
			YC = c * 4.0 * MC_1d3 * e * e * e * (2.0 - e * e) / (-2.0 * e + (1.0 + e * e) * L);
			YH = c * 4.0 * MC_1d3 * e * e * e * e * e / (-2.0 * e + (1.0 + e * e) * L);
			std::cerr << "Par " << l << " " << a << " " << b << " " << e << " " << L << " " << XA << " " << YA << " " << XC << " " << YC << " " << YH << "\n";
		}
	};/**/
	struct potential_data
	{
		float a, _1d_a, c, m, _1d_m, dt, dtm, k, vis, vism;	
		double Efiber, rfiber, rofiber, Sfiber, roliquid, hfreefiber, CShfreefiber;
		//Lindstromparam_data Lp;
	};
	struct firerelax_data
	{
		unsigned int bloks4, NPpositiveMax, NPnegativeMax, NPpositive, NPnegative, Ndelay;
		float dtmax, dtmin, dt0, dt, alpha0, alpha, dtgrow, dtshrink, alphashrink, FdotV;
		float* h_FdotV, *d_FdotV;
	};
}

void write_p_data(std::ofstream& file, p_data& P);
void read_p_data(std::ifstream& file, p_data& P);
void write_p0_data(std::ofstream& file, p0_data& P0);
void read_p0_data(std::ifstream& file, p0_data& P0);
void write_pNet_data(std::ofstream& file, pNet_data& Pnet);
void read_pNet_data(std::ifstream& file, pNet_data& Pnet);
void write_potential_data(std::ofstream& file, potential_data& Po);
void read_potential_data(std::ifstream& file, potential_data& Po);