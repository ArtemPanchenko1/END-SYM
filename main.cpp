#include <stdio.h>
#include <iostream>
#include "md.h"
#include "md_data_types.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <time.h>
#include <iomanip>
#include <chrono>

int main()
{	
	cudaError_t cudaStatus;
	h_data Host;
	p_data P;	
	p0_data P0;
	l_data L;
	param_data Pr;
	potential_data Po;
	EnergyDist_data ED;
	Po.Efiber = 100e6 / pressure_const; //50e6 / pressure_const; //14.5e6 / pressure_const; //1e9 / pressure_const;// 11.5e3;
	Po.rfiber = 0.1e-6 / length_const; //0.03e-6 / length_const; //274e-9 / length_const; //1e-7 / length_const;
	Po.rofiber = 1350 / density_const; //1270 / density_const;
	Po.Sfiber = Po.rfiber * Po.rfiber * MC_pi;
	Po.roliquid = 1e+3 / density_const;
	Po.hfreefiber = 1.04198e-06 / length_const;

	Po.k = 1.0;
	Po.m = Po.rofiber * Po.Sfiber / Po.k;
	//Po.m = 1.0 / Po.k;
	Po._1d_m = 1.0 / Po.m;
	Po.a = 1.0 / Po.k;
	Po._1d_a = 1.0 / Po.a;
	//Po.c = Po.k * 100.0;
	
	Po.c = Po.k * Po.Efiber * Po.Sfiber;
	//Po.dt = 0.01 * MC_s2d2 * MC_pi * sqrt(Po.m / Po.c);
	Po.dt = 0.002 * MC_pi * sqrt(Po.rofiber * Po.Sfiber * def_a_aver * def_a_aver / Po.c);
	Po.dtm = Po.dt * Po._1d_m;
	Po.vis = 0.18;
	Po.vism = Po.vis * Po.m;
	fprintf(stderr, "CONST m=%e kg; a=%e m; c=%e N; dt=%e s\n", Po.m*mass_const, Po.a*length_const, Po.c*force_const, Po.dt*time_const);
	fprintf(stderr, "HELP omega=%e 1/c; v=%e m/s; d=%e a; dt*m=%e %e | Time %e\n", 1e-12*sqrt((Po.c * force_const)/(Po.m * mass_const)), 
		9.0*sqrt((Po.c * force_const) / (Po.m * mass_const))* (Po.a * length_const), 
		9.0 * sqrt((Po.c * force_const) / (Po.m * mass_const)) * Po.dt*time_const*1500.0,
		Po.dtm, Po.dtm*time_const/mass_const, Po.dt*9000/(2.0*MC_pi * sqrt(Po.m / Po.c)));
	//std::cerr << "P " << Po.Efiber << " " << Po.Sfiber << "\n";
	//std::cin.get();
	pAdd_data Padd;
	P.N = Po.k * 999 + 1;
	P._1d_N = 1.0 / P.N;

	Padd.StepVN[0] = 199;
	Padd.StepVN[1] = 300;
	Padd.StepVN[2] = 150;
	//Padd.StepV = Padd.StepV1 + Padd.StepV2;
	Padd.V0[0] = 0.00005;
	Padd.dV[0] = 0.00005;
	Padd.V0[1] = 0.01;
	Padd.dV[1] = 0.005;
	Padd.V0[2] = 0.16;
	Padd.dV[2] = 0.01;//0.2692

#ifdef pre_logveloscale
	Padd.StepVN[0] = 1;
	Padd.StepVN[1] = 1;
	Padd.StepVN[2] = 98;
#endif // pre_logveloscale

	

	/*Padd.V0[0] = 0.0002;
	Padd.dV[0] = 0.0002;
	Padd.V0[1] = 0.010;
	Padd.dV[1] = 0.002;
	Padd.V0[2] = 0.1;
	Padd.dV[2] = 0.05;/**/
	
	Padd.MaxShift = 10.0;
	Padd.Vl = sqrt(Po.a * Po.a * Po.c * Po._1d_m);
	Padd.Vt = 1.01;
	Padd.V = Padd.V0[2];
	std::cerr << "Vl " << Padd.Vl * velocity_const << " " << sqrt(Po.Efiber / Po.rofiber) << " " << 1004e-6 / (pressure_const * time_const)
		<< " | " << (0.08 * RCC / Padd.Vl) * time_const << "\n";
	//std::cin.get();
	pNet_data Pnet;
	char filename1[256] = "", filename2[256] = "", filename3[256] = "";
	

	
	Pnet.a_aver = def_a_aver;
	Pnet.Rcut = 0.5;
	Pnet.InitialDeformation = 0.0;
	Padd.ElementSteps = 51000;
	Padd.time = 49999;
	int def = 1, def0 = 0;
	Padd.RMove = 0.01 * def * 0.03987475 * ReadCoordinatesCoefficient;

	Padd.time = SMEMDIM * (Padd.time / SMEMDIM + 1);

	Padd.StepV = 0;
	Padd.time = 49999;
	int i_def0 = 0, i_RT = 10, i_defi = 0, i_P1 = 0, i_defc=1;
	double i_ID = 0.0, povis = Po.vis;
	unsigned int A_def0[] = { 0,1,5,10,20,40 };
	double param1[17][5] = { {8.0, 50000, 0.24, 25794, 100840}, {6.5, 50000, 0.24, 25812, 79917}, {5.5, 50000, 0.24, 25818, 73217}, {4.5, 50000, 0.24, 25755, 58965}, {3.5, 50000, 0.24, 25810, 51265},
	{8.0, 10000, 0.24, 5158, 20068}, {8.0, 30000, 0.24, 15640, 61074}, {8.0, 70000, 0.24, 36095, 141277}, {8.0, 90000, 0.24, 46469, 181883},
	{8.0, 50000, 0.12, 25813, 100881}, {8.0, 50000, 0.18, 25885, 101203}, {8.0, 50000, 0.30, 25952, 101564}, {8.0, 50000, 0.36, 25788, 100759}, {8.0, 50001, 0.16, 25946, 101588}, {3.0, 50000, 0.16, 25759, 46456}, {8.0, 50000, 0.24, 25794, 100840}, {3.0, 50000, 0.24, 25735, 46438} };
	//double hfreefiberNetworks[17] = { 1.04198e-06 / length_const,1.43288e-06 / length_const, 1.6123e-06 / length_const, 2.11546e-06 / length_const, 2.47456e-06 / length_const,
	//2.47017e-06 / length_const, 1.37622e-06 / length_const, 8.64549e-07 / length_const, 7.48488e-07 / length_const, 1.0364e-06 / length_const, 1.02915e-06 / length_const,
	//	1.04062e-06 / length_const, 1.05214e-06 / length_const, 1.02827e-06 / length_const, 2.78302e-06 / length_const, 1.04198e-06 / length_const, 2.79696e-06 / length_const };
	double hfreefiberNetworks[17] = { 0 / length_const, 0 / length_const, 0 / length_const, 0 / length_const, 0 / length_const,
	0 / length_const, 0 / length_const, 0 / length_const, 0 / length_const, 0 / length_const, 0 / length_const,
		0 / length_const, 0 / length_const, 2.25653e-06 / length_const, 5.76604e-06 / length_const, 2.28397e-06 / length_const, 5.79392e-06 / length_const };

#ifndef pre_OnlyOneCell
	sprintf(filename1, "./data/0.txt", 0);
	sprintf(filename2, "./data/elements.txt", 0);
	Pnet.SN = 76827;
	Pnet.ScN = 301698;
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	sprintf(filename1, "./data/1_0.txt", 0);
	sprintf(filename2, "./data/1_elements.txt", 0);
	Pnet.SN = 76829;
	Pnet.ScN = 301901;
#endif // pre_OnlyOneCell
	initArrays(P, P0, Padd, Pnet, ED);
#ifdef pre_LoadEnergyData
	
#ifndef pre_OnlyOneCell
	ED.P00[0] = -0.121052175;
	ED.P00[1] = 0;
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
	ED.P00[0] = 0;
	ED.P00[1] = 0;
#endif // pre_OnlyOneCell
	ED.DR = 0.03987475 * 0.2;
	ED.Dfi = MC_pi / 180.0;
	sprintf(filename3, "./result/CP_Energy.txt", 0);
	LoadEnergyDataStep(P, P0, Pnet, Padd, Po, ED, filename3);
	return 0;
#endif // pre_LoadEnergyData

	//std::cerr << "C " << Po.c << "\n";
	
#ifdef pre_Relaxation	
	//strcat(filename, filenamelast);
	
	char filenameAverageF[200];
	strcpy(filenameAverageF, "./result/AverageF.dat");
	std::ofstream fileAverageF;
	fileAverageF.open(filenameAverageF, std::ios::out);
	fileAverageF << "Vis Step FxA FyA FxB FyB dFxA dFyA dFxB dFyB\n";
	fileAverageF.close();
	//i_def0 = 50;
	i_RT = 11;
	i_ID = 0*0.15;
	i_defi = 2;
	i_defc = 1;
	i_P1 = 13;
	//for (i_defi = 2; i_defi < 6; i_defi+=3)
	//for (i_def0 = 1; i_def0 < 16; i_def0 += 5)
		//for (i_ID = 0; i_ID < 0.5; i_ID += 0.15)
	//for (i_vis = 0.05; i_vis < 0.31; i_vis += 0.05)
	//	for (i_RT = 3; i_RT < 11; i_RT += 1)
	//for (i_defc = 2; i_defc < 6; i_defc += 1)
	for (i_P1 = 13; i_P1 < 17; i_P1 += 1)
		{			
#ifndef pre_OnlyOneCell
		/*sprintf(filename1, "./data/0.txt", 0);
		sprintf(filename2, "./data/elements.txt", 0);
		Pnet.SN = 76827;
		Pnet.ScN = 301698;/**/


		Pnet.Connectivity = param1[i_P1][0];
		Pnet.Nnodes = param1[i_P1][1];
		Pnet.CellDistance = param1[i_P1][2];
		Pnet.SN = param1[i_P1][3];
		Pnet.ScN = param1[i_P1][4];
		if (Pnet.CellDistance > 0.24)
			Pnet.Rcut = 0.34 + Pnet.CellDistance * 0.5 + 0.04;
		sprintf(filename1, "../data/xyz_network_c%.1f_N%i_RR%.2f.txt", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);
		sprintf(filename2, "../data/couples_network_c%.1f_N%i_RR%.2f.txt", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);/**/
		std::cerr << "Param: " << Pnet.Connectivity << " " << Pnet.Nnodes << " " << Pnet.CellDistance << " " << Pnet.SN << " " << Pnet.ScN
			<< " | " << filename1 << " | " << filename2 << "\n " << (Pnet.CellDistance - 0.08) * ReadCoordinatesCoefficient / Padd.Vl * time_const << "  " << 0.01*0.4/(Pnet.CellDistance - 0.08)  / Padd.Vl * velocity_const << "\n";
#endif // !pre_OnlyOneCell
#ifdef pre_OnlyOneCell
		sprintf(filename1, "./data/1_0.txt", 0);
		sprintf(filename2, "./data/1_elements.txt", 0);
		Pnet.SN = 76829;
		Pnet.ScN = 301901;
#endif // pre_OnlyOneCell
			Pnet.a_aver = def_a_aver;
			//Pnet.Rcut = 0.5;
			Pnet.InitialDeformation = i_ID;
			Padd.ElementSteps = 51000;
			Padd.time = 49999;
			Padd.RelaxationTime = i_RT * 1000000 + 1;
			//Po.vis = povis;
			Po.vis = 0;// i_vis;
			Po.vism = Po.vis * Po.m;
			//def0 = i_def0;
			def0 = A_def0[i_defi];// +i_defc;
			Padd.RMove = def0 * 0.01 * 0.03987475 * ReadCoordinatesCoefficient;
			Padd.time = SMEMDIM * (Padd.time / SMEMDIM + 1);
			ReadLattice(Pnet, Po, filename1, filename2);
			//std::cerr << "A1\n"; 
			//std::cin.get();
			SplitNet(P, P0, L, Padd, Po, Pnet);
			std::cerr << "Time " << Padd.time << " " << 0.023 / Po.dt << " | " << Po.dt << " | " << Padd.d_FResult << " " << Padd.h_FResult << "\n";
			//std::cin.get();
			Padd.V = 100.0*Padd.V0[2] / velocity_const;
			renewLattice(P, P0);
			std::cerr << "StartR " << Padd.RelaxationTime << " " << Po.vis << " | " << 0.0 << " " << i_RT << " | " << def0 << " " << Padd.RMove << "\n";
			//std::cin.get(); continue;
#ifndef pre_FireRelaxation
			calculateGPUStepsContractRelax(P, P0, Po, Padd, L, Pnet);
#endif // !pre_FireRelaxation
#ifdef pre_FireRelaxation
			firerelax_data Fire;			
			Fire.dt0 = Po.dt;
			Fire.dtmax = 100.0 * Fire.dt0;
			Fire.dtmin = 0.02 * Fire.dt0;
			Fire.dtgrow = 1.1;
			Fire.dtshrink = 0.5;
			Fire.alpha0 = 0.25;
			Fire.alphashrink = 0.99;
			Fire.NPnegativeMax = 2000;
			Fire.Ndelay = 20;
			sprintf(filename1, "./result/steps/CP_%i.txt", 1 + 100 * i_P1 + 10 * i_defi);
			SaveTXTParticles(P, P0, Po, Pnet, filename1);/**/
			//std::cin.get();
			calculateGPUStepsContractRelaxFIRE(P, P0, Po, Padd, L, Pnet, Fire);
#endif // pre_FireRelaxation
						
			//std::cerr << "R1\n";
			//std::cin.get();
			//SaveTXTGraphsFmm(Padd, Po, Pnet);
			double m_temp = 4.0 * 2.0 * MC_pi * 0.03987475 * ReadCoordinatesCoefficient;
			//std::cerr << "MCell " << m_temp << "\n";
			P.h_IM[P.N - 2] = m_temp;
			P.h_IM[P.N - 1] = m_temp;
			P.h_1d_IM[P.N - 2] = 1.0 / m_temp;
			P.h_1d_IM[P.N - 1] = 1.0 / m_temp;
			Po.vis = 0;// i_vis;
			//std::cin.get();
			SaveAllData(P, P0, Pnet, Padd, Po, def0);
			sprintf(filename1, "./result/steps/CP_%i.txt", 2 + 100 * i_P1 + 10 * i_defi);
			SaveTXTParticles(P, P0, Po, Pnet, filename1);/**/
			//std::cin.get();
			deleteArrays(P, P0, Padd, Pnet, ED);
			//std::cin.get();
		}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
	//std::cin.get();/**/
#endif // pre_Relaxation	
	//std::cin.get();
	//fprintf(stderr, "createIMatrix\n");
	//std::cin.get();

	double Po_dt, CellImpulsTransportTime, CellDistanceImpulsTransportTime;
	unsigned int i, j, maxtime = 0, i_vis = 3;
	std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
	std::chrono::steady_clock::time_point end;
	std::chrono::nanoseconds dr;
	
	int SVnn, SVn;
	i_def0 = 0;
	i_defi = 2;
	i_ID=0*0.15;
	//i_vis = 0.0;
	double VisCrit = 300.0 * sqrt(Po.c * Po.m), visArray[4] = {694.7e-6 / (pressure_const * time_const) , 900e-6 / (pressure_const * time_const), 3500e-6 / (pressure_const * time_const), 0.0};
	std::cerr << "Vis " << VisCrit << " " << Po.c << " " << Po.m << "\n";
	Po.vis = VisCrit;
	//for (i_defi = 0; i_defi < 6; ++i_defi)
	//for (i_def0 = 0; i_def0 < 51; i_def0 += 5)
	//for (i_ID = 0; i_ID < 0.5; i_ID += 0.15)
	//i_vis = 1004e-6/(pressure_const*time_const);
	//i_vis = 694.7e-6 / (pressure_const * time_const);
	//i_vis = 3500e-6 / (pressure_const * time_const);
	Po_dt = Po.dt;
	i_P1 = 15;
	i_defc = 1;
	Padd.mintauILT = 0.09;
	Padd.maxtauILT = 1.0;
	
	//for (i_vis = 0; i_vis < 4; i_vis+=2)
	//for (i_defc=1; i_defc < 6; i_defc += 1)
	//for (i_defi = 2; i_defi < 6; i_defi+=3)
	//for (i_P1 = 13; i_P1 < 17; i_P1 += 1)
	{
		//def0 = i_def0;
		def0 = A_def0[i_defi];// +1;
#ifdef pre_ReleaseHalfSine
		def0 += 1;
#endif // pre_ReleaseHalfSine
		/*Pnet.SN = 76827;
		Pnet.ScN = 301698;/**/

		Pnet.Connectivity = param1[i_P1][0];
		Pnet.Nnodes = param1[i_P1][1];
		Pnet.CellDistance = param1[i_P1][2];
		Pnet.SN = param1[i_P1][3];
		Pnet.ScN = param1[i_P1][4];
		Po.hfreefiber = hfreefiberNetworks[i_P1];

		Pnet.a_aver = def_a_aver;
		Pnet.Rcut = 0.5;
		Pnet.InitialDeformation = i_ID;
		Padd.ElementSteps = 51000;
		Padd.Eps0 = 0.01 * (def0);
		def = i_defc;
		Padd.Eps = 0.01 * (def + def0);
		Padd.RMove = def * 0.01 * 0.03987475 * ReadCoordinatesCoefficient;
		///Padd.time = (3.0 * Padd.RMove) / (Padd.V0[0] * Po.dt) + 0.023 / Po.dt;
		Padd.time = 1700000;
		Padd.time = SMEMDIM * (Padd.time / SMEMDIM + 1);


		/*std::cerr << "Static 0.00: " << 1.174251592 / 2.558858187 << "\n"
			<< "0.01 : " << (2.382552952 - 1.174251592) / (5.162101901 - 2.558858187) << "\n"
			<< "0.1 : " << (14.32067217 - 12.90772277) / (30.18190246 - 27.27178346) << "\n"
			<< "0.5 : " << (83.5325888 - 81.54184229) / (163.4198859 - 159.7178667) << "\n";/**/
		LoadAllData(P, P0, Pnet, Padd, Po, def0);
		std::cerr << "Time " << Padd.time << " | " << Po.dt << "\n";
#ifdef pre_OneCellEFdistribution
		EFInitData(P, P0, Pnet, Padd, Po);
		Padd.EF.NStepDef = i_defi;
#endif // pre_OneCellEFdistribution
		Padd.V = 1.0;
		//Po.dt *= 2.0;

		Po.vis = 0.00; Po.vism = Po.vis * Po.m;
		Padd.time = 49999;
		//std::cin.get();
		calculateGPUStepsAverage(P, P0, Po, Padd, L, Pnet);
		//std::cin.get();// continue;
		//Po.vis = i_vis; Po.vism = Po.vis * Po.m;
		//std::cerr << "Vis " << VisCrit << " " << Po.vis << " " << i_vis << " " << Po.vism << "\n";
		Padd.time = 49999;
		//continue; std::cin.get();
		Padd.time = SMEMDIM * (Padd.time / SMEMDIM + 1);
		//CellImpulsTransportTime = 0.04 * Pnet.CellDistance / 0.24;
		CellImpulsTransportTime = 1.3 * (Pnet.CellDistance - 0.08) * ReadCoordinatesCoefficient / Padd.Vl;
		CellDistanceImpulsTransportTime = (Pnet.CellDistance - 0.08) * ReadCoordinatesCoefficient / Padd.Vl;
		std::cerr << "Calculation tau^I_N = " << 2.0 * Padd.RMove / (Padd.V0[0] * 0.1 * Padd.Vl * CellDistanceImpulsTransportTime * 0.16 / (Pnet.CellDistance - 0.08)) << " " << Padd.RMove / ((Padd.V0[2] + Padd.StepVN[2] * Padd.dV[2]) * 0.1 * Padd.Vl * CellDistanceImpulsTransportTime * 0.16 / (Pnet.CellDistance - 0.08))
			<< " | " << Pnet.CellDistance << " Vis " << i_vis <<" "<< visArray[i_vis] << " coeff = " << 2.0 * Padd.RMove / (CellDistanceImpulsTransportTime) << "\n";
		Padd.StepV = 0;
		//std::cin.get();
#ifdef pre_logveloscale
		Padd.V = 2.0 * Padd.RMove / (Padd.mintauILT * CellDistanceImpulsTransportTime);
		Padd.logvcoeff = -log10(pow(Padd.maxtauILT / Padd.mintauILT, 1.0 / double(Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2])));
		SVnn = 0;
		for (SVn = 0; SVn < Padd.StepVN[0]+ Padd.StepVN[1]+ Padd.StepVN[2]; ++SVn, ++Padd.StepV)
		{
			Padd.V = Padd.V*pow(10, Padd.logvcoeff);
			std::cerr << "Tau " << Padd.StepV << " " << Padd.V << " " << 2.0 * Padd.RMove / (Padd.V * CellDistanceImpulsTransportTime) <<" "<< 2.0 * Padd.RMove / (CellDistanceImpulsTransportTime) << " " << Padd.logvcoeff << "\n";
			//std::cin.get();

#endif // pre_logveloscale
#ifndef pre_logveloscale
			for (SVnn = 0; SVnn < 3; ++SVnn)
			{
				for (SVn = 0; SVn < Padd.StepVN[SVnn]; ++SVn, ++Padd.StepV)
				{
		
					Padd.V = Padd.V0[SVnn] + SVn * Padd.dV[SVnn];
					Padd.V *= i_defc;
					Padd.V *= 0.1 * Padd.Vl * 0.16 / (Pnet.CellDistance - 0.08);
#endif // !pre_logveloscale	
					

					//Padd.V = 0.1;
					//Padd.V = 0.1;
					//Padd.V = 0.001;
					//Padd.V = 0.04912049/24.5612;
#ifdef pre_SaveFR
					Padd.V = 0.082;
					Padd.V = 0.01;
					Padd.V = 0.2692 * 0.1 * Padd.Vl * i_defc;
					Padd.V = 2.0 * Padd.RMove / (1.0 * CellDistanceImpulsTransportTime);
					std::cerr << "Tau " << 2.0 * Padd.RMove * Padd.Vl / ((Pnet.CellDistance - 0.08) * ReadCoordinatesCoefficient * Padd.V) << " " << Pnet.CellDistance << " " << Padd.RMove << " " << CellImpulsTransportTime << '\n';
#endif // pre_SaveFR
					//Padd.V = (Padd.V0[2] + Padd.StepVN[2] * Padd.dV[2]) * 0.1 * Padd.Vl;
					//Padd.V = 0.068;Padd.LammpsPointSaveTime=0;
					//Padd.V = 0.41;
					//Padd.V = 0.18;
					//Padd.V = 0.72;
					//Padd.V = 0.05;
					//Padd.V = 0.19;
					// 
					//fig3a					
					//Padd.V = 0.0022; Padd.LammpsPointSavetime = 9.821391905; //Padd.LammpsPointSaveTime = 28712;
					//Padd.V = 0.06300000101; Padd.LammpsPointSavetime = 1.574656917; //Padd.LammpsPointSaveTime = 36827;//S2
					//Padd.V = 0.41; Padd.LammpsPointSavetime = 1.16815453; //Padd.LammpsPointSaveTime = 54640;//S1
					//fig3b					
					//Padd.V = 0.0022; Padd.LammpsPointSavetime = 9.821391905; //Padd.LammpsPointSaveTime = 28712;
					//Padd.V = 0.007; Padd.LammpsPointSavetime = 3.695849338; //Padd.LammpsPointSaveTime = 21609;//S2
					//Padd.V = 0.049; Padd.LammpsPointSavetime = 1.673385634; //Padd.LammpsPointSaveTime = 39136;
					//Padd.V = 0.18; Padd.LammpsPointSavetime = 1.317508969; //Padd.LammpsPointSaveTime = 61626;//S1
					//fig4a					
					//Padd.V = 0.12; Padd.LammpsPointSavetime = 1.347119037; //Padd.LammpsPointSaveTime = 63011;
					//Padd.V = 0.72; Padd.LammpsPointSavetime = 1.131403867; //Padd.LammpsPointSaveTime = 52921;//S3
					//fig4b
					//Padd.V = 0.0018; Padd.LammpsPointSavetime = 11.48143722; //Padd.LammpsPointSaveTime = 33565;
					//Padd.V = 0.085; Padd.LammpsPointSavetime = 1.408113652; //Padd.LammpsPointSaveTime = 32932;
					//Padd.V = 0.39; Padd.LammpsPointSavetime = 1.198470168; //Padd.LammpsPointSaveTime = 56058;//S3
					//Padd.V = 0.72; Padd.LammpsPointSavetime = 1.131403867; //Padd.LammpsPointSaveTime = 52921;
					//fig5a
					//Padd.V = 0.29; Padd.LammpsPointSavetime = 26.42882964; //Padd.LammpsPointSaveTime = 1137864;//S4
					//fig5
					//Padd.V = 0.0005; Padd.LammpsPointSavetime = 48.6251951; //Padd.LammpsPointSaveTime = 32711;
					//Padd.V = 0.0036; Padd.LammpsPointSavetime = 8.598153714; //Padd.LammpsPointSaveTime = 46273;
					//Padd.V = 0.025; Padd.LammpsPointSavetime = 2.615791333; //Padd.LammpsPointSaveTime = 56310;
					//Padd.V = 0.29; Padd.LammpsPointSavetime = 26.42882964; Padd.LammpsPointSavetime = 1.380757851;//Padd.LammpsPointSaveTime = 1137864;//S4
					//fig6a					
					//Padd.V = 0.19; Padd.LammpsPointSavetime = 1.488924567; //Padd.LammpsPointSaveTime = 64104;//S5
					//fig6
					//Padd.V = 0.0005; Padd.LammpsPointSavetime = 52.14227807; //Padd.LammpsPointSaveTime = 35077;
					//Padd.V = 0.014; Padd.LammpsPointSavetime = 3.276730331; //Padd.LammpsPointSaveTime = 35269;
					//Padd.V = 0.073; Padd.LammpsPointSavetime = 1.745068656; //Padd.LammpsPointSaveTime = 75132;//S5
					//Padd.V = 0.19; Padd.LammpsPointSavetime = 1.488924567; //Padd.LammpsPointSaveTime = 64104;

					//Padd.V = 0.0007542530657; 0.0053;// 0.0007542530657;

					Po.dt = Po_dt;
					Padd.time = (2.2 * Padd.RMove) / (Padd.V * Po.dt) + CellImpulsTransportTime / Po.dt;
					std::cerr << "Time " << Padd.time * Po.dt << " " << Padd.time << " " << (3.0 * Padd.RMove) / (Padd.V * Po.dt) << " " << CellImpulsTransportTime / Po.dt << "\n";
#ifdef pre_OneCellEFdistribution
					Padd.time = (3.0 * Padd.RMove) / (Padd.V * Po.dt) + 0.05 / Po.dt;
#endif // pre_OneCellEFdistribution
					while (Padd.time < 50000)
					{
						Po.dt *= 0.9;
						Padd.time = (2.2 * Padd.RMove) / (Padd.V * Po.dt) + CellImpulsTransportTime / Po.dt;
#ifdef pre_OneCellEFdistribution
						Padd.time = (3.0 * Padd.RMove) / (Padd.V * Po.dt) + 0.05 / Po.dt;
#endif // pre_OneCellEFdistribution
					}
					//Padd.time *= 2;
					//Padd.time = (0.016235485 * 5.0) / Po.dt;
#ifdef TwoDirectionMove
					Padd.ImpulsSteps = (2.0 * Padd.RMove) / (Padd.V * Po.dt);
					//Padd.ImpulsSteps = Padd.time;
#endif // TwoDirectionMove
#ifndef TwoDirectionMove
					Padd.ImpulsSteps = (1.0 * Padd.RMove) / (Padd.V * Po.dt);
#endif // TwoDirectionMove
#ifdef pre_ReleaseHalfSine
					Padd.ImpulsSteps = (Padd.RMove) / (Padd.V * Po.dt);
#endif // pre_ReleaseHalfSine

					//Padd.time = 1000;
					Padd.LammpsPointSaveTime = unsigned int(Padd.LammpsPointSavetime * 0.016235485 / Po.dt);
					Padd.LammpsPointSaveTime = 5000;
					//std::cerr << "LPST " << Padd.LammpsPointSaveTime; std::cin.get(); exit(0);
#ifdef pre_SaveLammpsPoint
					Padd.time = Padd.LammpsPointSaveTime + 10000;
#endif // pre_SaveLammpsPoint
					//std::cerr << "Time " << Padd.time << " " << (3.0 * Padd.RMove) / (Padd.V * Po.dt) << " " << CellImpulsTransportTime / Po.dt << "\n";

					Po.dtm = Po.dt * Po._1d_m;
					//Padd.time = 1000;
					Padd.time = SMEMDIM * (Padd.time / SMEMDIM + 1);
					if (Padd.time > maxtime)maxtime = Padd.time;
					//Padd.time = 1000;					
					///renewLattice(P, P0);
					reloadLattice(P, P0);
					//Padd.V = 1e-3;
					//std::cerr << "E1 "<< Padd.StepV << "\n";
					Padd.h_V[Padd.StepV] = Padd.V;
					//std::cerr << "E2\n";
					//std::cin.get();
	//if(Padd.StepV%5==0)
					Po.vis = visArray[i_vis]; Po.vism = Po.vis * Po.m;

					std::cerr << "Time " << Padd.V << " " << Padd.time << " " << Padd.ImpulsSteps << " | " << Po.dt << " " << 0.023 / Po.dt << " | " << Padd.V0[SVnn]
						<< " " << SVn << " " << Padd.dV[SVnn] << " | " << Po.vis << " " << Po.hfreefiber << "\n";
					calculateGPUSteps(P, P0, Po, Padd, L, Pnet);

					///SaveAllData(P, P0, Pnet, Padd, Po, 50);
					//std::cin.get();
					//std::cerr << "E3\n";
					if (Padd.StepV % 10 == 0)
					{
						end = std::chrono::high_resolution_clock::now();
						dr = end - begin;
						std::cerr << "Fin element " << Padd.StepV << " " << Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2] << " " << std::chrono::duration_cast<std::chrono::milliseconds>(dr).count() * 0.1 << "ms\n";
						std::cerr << "Remain " << (double(Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2] - Padd.StepV) * std::chrono::duration_cast<std::chrono::milliseconds>(dr).count()) * 0.0001 * MC_1d_60 << "min " << std::chrono::duration_cast<std::chrono::milliseconds>(dr).count() * 0.1 * P._1d_N << " ms/particle" << "\n";
						begin = end;
					}
#ifdef pre_SaveFR
					std::cin.get();
#endif // pre_SaveFR
					//std::cin.get();
					//exit(0); 

					//fprintf(stderr, "Fin element %i %i\n", element, RepeatEnsemble);
#ifndef pre_logveloscale
				}
			}
#endif // !pre_logveloscale	
#ifdef pre_logveloscale
		}
		//std::cin.get();
#endif // pre_logveloscale	
#ifdef pre_OneCellEFdistribution
			SaveTXTGraphsEFmm(Padd, Po, Pnet);
#endif // pre_OneCellEFdistribution
			///cudaMemcpy(Padd.h_Fmm, Padd.d_Fmm, (8 * (Padd.StepV1 + Padd.StepV2) * sizeof(float)), cudaMemcpyDeviceToHost);
			std::cerr << "Maxtime " << maxtime << "\n";
			SaveTXTGraphsFmm(Padd, Po, Pnet);
			fprintf(stderr, "Save TXT\n");
			//std::cin.get();
			deleteArrays(P, P0, Padd, Pnet, ED);

		}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}/**/
	//std::cin.get();
	return 0;
}