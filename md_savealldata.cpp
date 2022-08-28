#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "pcuda_helper.h"

void SaveAllData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, int dn)
{	
	std::ofstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];
	strcpy(filenamelast, "");	
#ifndef pre_nonlinearC
	strcpy(filenamepart[1], "_Cc");
#endif
#ifdef pre_nonlinearC
	strcpy(filenamepart[1], "_Cb");
#endif
	strcat(filenamelast, filenamepart[1]);
	sprintf(filenamepart[2], "_D%i_a%.5f_ID%.3f", dn, Pnet.a_aver, Pnet.InitialDeformation);
	strcat(filenamelast, filenamepart[2]);
#ifdef pre_MoveCell
	strcpy(filenamepart[3], "_MoveCell");
	strcat(filenamelast, filenamepart[3]);
#endif // pre_MoveCell
#ifdef pre_RotateCell
	strcpy(filenamepart[4], "_RotateCell");
	strcat(filenamelast, filenamepart[4]);
#endif // pre_RotateCell
#ifdef pre_FreeCell
	strcpy(filenamepart[5], "_FreeCell");
	strcat(filenamelast, filenamepart[5]);
#endif // pre_FreeCell
#ifdef pre_OnlyOneCell
	strcpy(filenamepart[6], "_OnlyOneCell");
	strcat(filenamelast, filenamepart[6]);
#endif // pre_OnlyOneCell
#ifdef pre_ConnectCellSurface
	strcpy(filenamepart[7], "_CCS");
	strcat(filenamelast, filenamepart[7]);
#endif // pre_ConnectCellSurface
#ifdef OneNodeContractRelax
	strcpy(filenamepart[8], "_ONCR");
	strcat(filenamelast, filenamepart[8]);
#endif // OneNodeContractRelax
	//sprintf(filenamepart[6], "_vis%.3f_SR%im", Po.vis, int(Padd.RelaxationTime/1000000));
	//strcat(filenamelast, filenamepart[6]);
	sprintf(filenamepart[9], "_c%.1f_Nn%i_RR%.2f", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);
	strcat(filenamelast, filenamepart[9]);

	//Po.vis = 0;
	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/DATA");
	strcat(filename, filenamelast);
	
	std::cerr << "Save DATA " << filename << "\n";
	file.open(filename, std::ios::out | std::ios::binary);
	std::cerr << "write_p_data start\n"; //std::cin.get();
	write_p_data(file, P);
	std::cerr << "write_p0_data start\n"; //std::cin.get();
	write_p0_data(file, P0);
	std::cerr << "write_pNet_data start\n"; //std::cin.get();
	write_pNet_data(file, Pnet);
	std::cerr << "write_potential_data start\n"; //std::cin.get();
	write_potential_data(file, Po);	
	file.close();

	/*strcpy(filename, "./result/Average");
	strcat(filename, filenamelast);
	int i;
	file.open(filename, std::ios::out);
	file << "Type Ax Ay Bx By";
	file << "\nF ";
	for (i = 0; i < 4; ++i)
		file << " " << Padd.FSAB_aver[i];
	file << "\nS ";
	for (i = 4; i < 12; ++i)
		file << " " << Padd.FSAB_aver[i];
	file << "\nFi ";
	for (i = 12; i < 16; ++i)
		file << " " << Padd.FSAB_aver[i];
	file.close();/**/
	strcpy(filename, "./result/AverageF.dat");
	//strcat(filename, filenamelast);
	int i;
	file.open(filename, std::ios::out | std::ios::app);
	//file << "Vis Step FxA FyA FxB FyB dFxA dFyA dFxB dFyB\n";
	
	file << "\n" << Po.vis << " " << int(Padd.RelaxationTime / 1000000);
	file.precision(12);
	for (i = 12; i < 16; ++i)
		file << " " << Padd.FSAB_aver[i];
	for (i = 12+18; i < 16+18; ++i)
		file << " " << Padd.FSAB_aver[i];
	file.close();
	fprintf(stderr, "Finish Save Data\n", 0);
	//std::cin.get();
}

void LoadAllData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, int dn)
{
	std::ifstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];
	strcpy(filenamelast, "");
#ifndef pre_nonlinearC
	strcpy(filenamepart[1], "_Cc");
#endif
#ifdef pre_nonlinearC
	strcpy(filenamepart[1], "_Cb");
#endif
	strcat(filenamelast, filenamepart[1]);
	sprintf(filenamepart[2], "_D%i_a%.5f_ID%.3f", dn, Pnet.a_aver, Pnet.InitialDeformation);
	strcat(filenamelast, filenamepart[2]);
#ifdef pre_MoveCell
	strcpy(filenamepart[3], "_MoveCell");
	strcat(filenamelast, filenamepart[3]);
#endif // pre_MoveCell
#ifdef pre_RotateCell
	strcpy(filenamepart[4], "_RotateCell");
	strcat(filenamelast, filenamepart[4]);
#endif // pre_RotateCell
#ifdef pre_FreeCell
	strcpy(filenamepart[5], "_FreeCell");
	strcat(filenamelast, filenamepart[5]);
#endif // pre_FreeCell
#ifdef pre_OnlyOneCell
	strcpy(filenamepart[6], "_OnlyOneCell");
	strcat(filenamelast, filenamepart[6]);
#endif // pre_OnlyOneCell
#ifdef pre_ConnectCellSurface
	strcpy(filenamepart[7], "_CCS");
	strcat(filenamelast, filenamepart[7]);
#endif // pre_ConnectCellSurface
#ifdef OneNodeContractRelax
	strcpy(filenamepart[8], "_ONCR");
	strcat(filenamelast, filenamepart[8]);
#endif // OneNodeContractRelax
	sprintf(filenamepart[9], "_c%.1f_Nn%i_RR%.2f", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);
	strcat(filenamelast, filenamepart[9]);
	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/DATA");
	//strcpy(filename, "../DATA/DATA");
	//strcpy(filename, "../2021_05_23/data/data");
	//strcpy(filename, "../2021_06_06/DATA/Relaxation/DATA");
	//strcpy(filename, "../15/result/DATA");
	//strcpy(filename, "../2021_07_05/CellFixed/8/result/DATA");
	//strcpy(filename, "../2021_10_07/0/result/DATA");
	//strcpy(filename, "../0/result/DATA");
	//strcpy(filename, "../DATA/DATA");
	
	
	strcat(filename, filenamelast);
	std::cerr << "Read DATA " << filename << "\n";
	//std::cin.get();
	file.open(filename, std::ios::in | std::ios::binary);
	std::cerr << "read_p_data start\n";
	read_p_data(file, P);
	HANDLE_ERROR(cudaMemset((void*)P.d_F, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
	//std::cin.get();
	std::cerr << "read_p0_data start\n";
	read_p0_data(file, P0);
	/*for (int i = 0; i < P.NBP; ++i)
		std::cerr << "BP " << i << " " << P.h_BP[i] << " " << P0.h_RU0[P.h_BP[i]] << " " << P0.h_RU0[P.h_BP[i] + P0.N] << "\n";
	std::cin.get();/**/
	//std::cin.get();
	HANDLE_ERROR(cudaMalloc((void**)&P0.d_U0, 2 * P0.N * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(P0.d_U0, P.d_U, 2 * P0.N * sizeof(float), cudaMemcpyDeviceToDevice));
	std::cerr << "set_pNet_data start\n";
	//std::cin.get();
//Padd.time *= 10.0;
	size_t size = ResultFmmNum * (Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float);
	Padd.h_Fmm = (float*)malloc(ResultFmmNum * (Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float));
	size = (Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float);
	Padd.h_V = (float*)malloc((Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(float));
	size = Padd.time * size_t(ResultFRNum) * size_t(P.NBPT[0] + P.NBPT[1]);
	Padd.h_FResult = (float*)malloc(size * sizeof(float));
	//std::cerr << "AAA " << Padd.time << " " << Padd.h_FResult << " " << Padd.time * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << " " << ResultFRNum * (P.NBPT[0] + P.NBPT[1]) << " " << size << "\n";

	//Padd.ElementSteps* ResultFRNum* (P.NBPT[0] + P.NBPT[1])

	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_FResult, Padd.ElementSteps * ResultFRNum * (P.NBPT[0] + P.NBPT[1]) * sizeof(float)));
#ifdef pre_SaveLammps
	Padd.h_Fbound0 = (float*)malloc(P.NI * 1 * sizeof(float));
	Padd.h_Fbound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fbound, P.NI * 1 * sizeof(float)));
	Padd.h_LammpsAddParticles = (float*)malloc((P.NI / 2) * 4 * sizeof(float));
	Padd.h_LammpsSumF = (float*)malloc(P.N * 1 * sizeof(float));
#ifdef  pre_SaveLammpsEnergy
	Padd.h_Ek0 = (float*)malloc(P.N * 1 * sizeof(float));
	Padd.h_Ebound0 = (float*)malloc(P.NI * 1 * sizeof(float));
#endif //  pre_SaveLammpsEnergy
#ifdef pre_SaveLammpsU
	Padd.h_Ubound0 = (float*)malloc(P.NI * 1 * sizeof(float));
	Padd.h_Ubound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Ubound, P.NI * 1 * sizeof(float)));
#endif // pre_SaveLammpsU
#ifndef pre_SaveEnergyData
	Padd.h_Ebound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Ebound, P.NI * 1 * sizeof(float)));
#endif // !pre_SaveEnergyData

#endif // pre_SaveLammps
#ifdef pre_SaveEnergyData
	//Padd.h_Ebound0 = (float*)malloc(P.NI * 1 * sizeof(float));
	Padd.h_Ebound = (float*)malloc(P.NI * 1 * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Ebound, P.NI * 1 * sizeof(float)));
#endif // pre_SaveEnergyData
	std::cerr << "read_pNet_data start\n";
	read_pNet_data(file, Pnet);
	std::cerr << "read_potential_data start\n";
	read_potential_data(file, Po);
//Po.dt *= 0.1;
	file.close();
	fprintf(stderr, "Finish Load Data\n", 0);
}

void reloadLattice(p_data& P, p0_data& P0)
{
	//HANDLE_ERROR(cudaMemset((void*)P.d_U, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(P.d_U, P0.d_U0, 2 * P0.N * sizeof(float), cudaMemcpyDeviceToDevice));
	//HANDLE_ERROR(cudaMemcpy((void*)P.d_U, (void*)P0.d_U0, 2 * P.N * sizeof(float), cudaMemcpyDeviceToDevice));   
	//fprintf(stderr, "Finish renewLattice\n");
}

void write_p_data(std::ofstream& file, p_data& P)
{
	std::cerr << "write_p_data " << P.N << " " << P.NI << " " << P.NBP << " | " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << " " << "\n";
	file.write((char*)(&P.N), sizeof(int));
	file.write((char*)(&P.NI), sizeof(int));
	file.write((char*)(&P.NBP), sizeof(int));	
	file.write((char*)(P.NBPT), 3 * sizeof(int));
	file.write((char*)(P.iBP), 2 * sizeof(int));	
	//std::cerr << "Write0 ConnectCell\n"; std::cin.get();
	//std::cerr << "W1\n";
	//cudaMemset(P.d_F, 0, 2 * P.N * sizeof(float));	
	//std::cerr << "F " << P.h_F <<" "<< P.h_F [100]<< " " << P.d_F << "\n"; std::cin.get();
	HANDLE_ERROR(cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost)); //std::cin.get();
	HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost)); //std::cerr << "V\n"; std::cin.get();
	HANDLE_ERROR(cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost)); //std::cerr << "U\n"; std::cin.get();
	//std::cerr << "W2\n";
	std::cerr << "Write0.1 ConnectCell\n"; //std::cin.get();
	file.write((char*)(P.h_F), 2 * P.N * sizeof(float));
	file.write((char*)(P.h_V), 2 * P.N * sizeof(float));
	file.write((char*)(P.h_U), 2 * P.N * sizeof(float));
	file.write((char*)(P.h_Ir0), 3 * P.NI * sizeof(float));
	file.write((char*)(P.h_BPR), 2 * P.NBP * sizeof(float));
	file.write((char*)(P.h_IM), P.N * sizeof(float));
	file.write((char*)(P.h_1d_IM), P.N * sizeof(float));
	file.write((char*)(P.h_VisR), P.N * sizeof(float));
	std::cerr << "Write1 ConnectCell\n"; //std::cin.get();
#ifdef pre_ConnectCellSurface
	file.write((char*)(P.h_BPDfi), (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
#endif // pre_ConnectCellSurface
	//std::cerr << "Write ConnectCell\n";
	std::cerr << "Write2 ConnectCell\n"; //std::cin.get();
	file.write((char*)(P.h_In), 2 * P.NI * sizeof(int));
	file.write((char*)(P.h_BP), P.NBP * sizeof(int));
	file.write((char*)(P.h_ShIn), 2 * P.N * sizeof(int));
	
	/*for (int i = 0; i < P.NBP; ++i)
		std::cerr << "BP " << i << " " << P.h_BP[i] << "\n";
	std::cin.get();/**/

	std::cerr << "Write p_data\n";
}

void read_p_data(std::ifstream& file, p_data& P)
{
	file.read((char*)(&P.N), sizeof(int));
	file.read((char*)(&P.NI), sizeof(int));
	file.read((char*)(&P.NBP), sizeof(int));
	file.read((char*)(P.NBPT), 3 * sizeof(int));
	file.read((char*)(P.iBP), 2 * sizeof(int));
	P._1d_N = 1.0 / P.N;
	std::cerr << "read_p_data "<< P.N << " " << P.NI << " " << P.NBP << " | " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << " " << "\n";
	//std::cin.get();
	P.h_F = (float*)malloc(2 * P.N * sizeof(float));
	file.read((char*)(P.h_F), 2 * P.N * sizeof(float));	
	P.h_V = (float*)malloc(2 * P.N * sizeof(float));
	file.read((char*)(P.h_V), 2 * P.N * sizeof(float));
	P.h_U = (float*)malloc(2 * P.N * sizeof(float));
	file.read((char*)(P.h_U), 2 * P.N * sizeof(float));
	P.h_Ir0 = (float*)malloc(3 * P.NI * sizeof(float));
	file.read((char*)(P.h_Ir0), 3 * P.NI * sizeof(float));
	P.h_BPR = (float*)malloc(2 * P.NBP * sizeof(float));
	file.read((char*)(P.h_BPR), 2 * P.NBP * sizeof(float));
	P.h_IM = (float*)malloc(P.N * sizeof(float));
	file.read((char*)(P.h_IM), P.N * sizeof(float));
	P.h_1d_IM = (float*)malloc(P.N * sizeof(float));
	file.read((char*)(P.h_1d_IM), P.N * sizeof(float));
	P.h_VisR = (float*)malloc(P.N * sizeof(float));
	file.read((char*)(P.h_VisR), P.N * sizeof(float));
#ifdef pre_ConnectCellSurface
	P.h_BPDfi = (float*)malloc((P.NBPT[0] + P.NBPT[1]) * sizeof(float));
	file.read((char*)(P.h_BPDfi), (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
#endif // pre_ConnectCellSurface	
	//std::cerr << "A2\n";
	std::cerr << "MCell " << 1.0/ P.h_1d_IM[P.N-2] << " " << 1.0 / P.h_1d_IM[P.N - 1] << "\n";
	
	P.h_In = (int*)malloc(2 * P.NI * sizeof(int));
	file.read((char*)(P.h_In), 2 * P.NI * sizeof(int));
	P.h_BP = (int*)malloc(P.NBP * sizeof(int));
	//memset(P.h_BP, 0, P.NBP * sizeof(int));
	file.read((char*)(P.h_BP), P.NBP * sizeof(int));
	P.h_ShIn = (int*)malloc(2 * P.N * sizeof(int));
	file.read((char*)(P.h_ShIn), 2 * P.N * sizeof(int));
	//std::cerr << "A3\n";	
	HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 2 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_Ir0, 3 * P.NI * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_1d_IM, P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_BPR, 2 * P.NBP * sizeof(float)));
	//std::cerr << "A4\n";
	HANDLE_ERROR(cudaMalloc((void**)&P.d_In, 2 * P.NI * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_BP, P.NBP * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_ShIn, 2 * P.N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_VisR, P.N * sizeof(float)));
	//std::cerr << "A5\n";

	/*for (int i = 0; i < P.N; ++i)
		if(fabs(P.h_F[i])+fabs(P.h_F[i+P.N])<1e-15)
			std::cerr << "PF " << i << " " << P.h_F[i] << " " << P.h_F[i+P.N] << "\n";
	std::cin.get();/**/
	//std::cerr << "BP "<<<<" " << "\n";
	/*for (int i = 0; i<P.NBP; ++i)
		std::cerr << "BP " << i << " " << P.h_BP[i]<<" "<< "\n";
	std::cin.get();/**/
	HANDLE_ERROR(cudaMemcpy(P.d_F, P.h_F, 2 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_V, P.h_V, 2 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_U, P.h_U, 2 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_Ir0, P.h_Ir0, 3 * P.NI * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_1d_IM, P.h_1d_IM, P.N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_BPR, P.h_BPR, 2 * P.NBP * sizeof(float), cudaMemcpyHostToDevice));
	//std::cerr << "A6\n";
	HANDLE_ERROR(cudaMemcpy(P.d_In, P.h_In, 2 * P.NI * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_BP, P.h_BP, P.NBP * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_ShIn, P.h_ShIn, 2 * P.N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(P.d_VisR, P.h_VisR, P.N * sizeof(float), cudaMemcpyHostToDevice));
	//std::cerr << "A7\n";
	//std::cin.get();
	/*for (int i = 0; i < P.N; ++i)
	{
		std::cerr << "P " << P.h_F[i] << " " << P.h_F[i + P.N] << " | " << P.h_V[i] << " " << P.h_V[i + P.N] << " | " << P.h_U[i] << " " << P.h_U[i + P.N];
		std::cin.get();
	}/**/

	std::cerr << "Read p_data\n";
}

void write_p0_data(std::ofstream& file, p0_data& P0)
{
	std::cerr << "P0 " << P0.N << "\n";
	file.write((char*)(&P0.N), sizeof(int));	
	file.write((char*)(P0.h_RU0), 2 * P0.N * sizeof(float));	
	std::cerr << "Write p0_data\n";
}

void read_p0_data(std::ifstream& file, p0_data& P0)
{
	file.read((char*)(&P0.N), sizeof(int));	
	P0._1d_N = 1.0 / P0.N;
	P0.h_RU0 = (float*)malloc(2 * P0.N * sizeof(float));
	file.read((char*)(P0.h_RU0), 2 * P0.N * sizeof(float));
	std::cerr << "P0 " << P0.N << "\n";
	/*for (int i = 0; i < P0.N; ++i)
	{
		std::cerr << "P " << P0.h_RU0[i] << " " << P0.h_RU0[i+P0.N] <<"\n";
		//std::cin.get();
	}/**/

	
	P0.d_U0 = nullptr;
	HANDLE_ERROR(cudaMalloc((void**)&P0.d_RU0, 2 * P0.N * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(P0.d_RU0, P0.h_RU0, 2 * P0.N * sizeof(float), cudaMemcpyHostToDevice));	
	std::cerr << "Read p0_data\n";
}

void write_pNet_data(std::ofstream& file, pNet_data& Pnet)
{
	file.write((char*)(&Pnet.SN), sizeof(unsigned int));
	file.write((char*)(&Pnet.ScN), sizeof(unsigned int));
	file.write((char*)(&Pnet.AdN), sizeof(unsigned int));
	file.write((char*)(&Pnet.L), sizeof(double));
	file.write((char*)(&Pnet.minrm), sizeof(double));
	file.write((char*)(&Pnet.maxrm), sizeof(double));
	file.write((char*)(&Pnet.a_aver), sizeof(double));
	file.write((char*)(&Pnet.Rcut), sizeof(double));
	std::cerr << "Pnet " << Pnet.SN << " " << Pnet.AdN << " " << Pnet.Rcut << "\n";
	file.write((char*)(Pnet.h_S), 2 * Pnet.SN * sizeof(double));
	file.write((char*)(Pnet.h_Sc), 2 * Pnet.ScN * sizeof(unsigned int));
	std::cerr << "Write pNet_data\n";
}

void read_pNet_data(std::ifstream& file, pNet_data& Pnet)
{
	//Pnet.SN = 0; Pnet.Rcut = 0; Pnet.AdN = 0;
	file.read((char*)(&Pnet.SN), sizeof(unsigned int));
	file.read((char*)(&Pnet.ScN), sizeof(unsigned int));
	file.read((char*)(&Pnet.AdN), sizeof(unsigned int));
	file.read((char*)(&Pnet.L), sizeof(double));
	file.read((char*)(&Pnet.minrm), sizeof(double));
	file.read((char*)(&Pnet.maxrm), sizeof(double));
	file.read((char*)(&Pnet.a_aver), sizeof(double));
	file.read((char*)(&Pnet.Rcut), sizeof(double));

	Pnet.h_S = (double*)malloc(2 * Pnet.SN * sizeof(double));
	Pnet.h_Sc = (unsigned int*)malloc(2 * Pnet.ScN * sizeof(unsigned int));
	file.read((char*)(Pnet.h_S), 2 * Pnet.SN * sizeof(double));
	file.read((char*)(Pnet.h_Sc), 2 * Pnet.ScN * sizeof(unsigned int));	
	//std::cerr << "Pnet " << Pnet.SN <<" "<< Pnet.AdN << " " << Pnet.Rcut << "\n";
	std::cerr << "Read pNet_data\n";
}

void write_potential_data(std::ofstream& file, potential_data& Po)
{
	file.write((char*)(&Po.a), sizeof(float));
	file.write((char*)(&Po._1d_a), sizeof(float));
	file.write((char*)(&Po.c), sizeof(float));
	//file.write((char*)(&Po.g), sizeof(float));
	file.write((char*)(&Po.m), sizeof(float));
	file.write((char*)(&Po._1d_m), sizeof(float));
	file.write((char*)(&Po.dt), sizeof(float));
	file.write((char*)(&Po.dtm), sizeof(float));
	file.write((char*)(&Po.k), sizeof(float));
	file.write((char*)(&Po.vis), sizeof(float));
	file.write((char*)(&Po.vism), sizeof(float));

	file.write((char*)(&Po.Efiber), sizeof(float));
	file.write((char*)(&Po.rfiber), sizeof(float));
	file.write((char*)(&Po.rofiber), sizeof(float));
	file.write((char*)(&Po.Sfiber), sizeof(float));
	file.write((char*)(&Po.roliquid), sizeof(float));
	file.write((char*)(&Po.hfreefiber), sizeof(float));

	std::cerr << "Write potential_data\n";
}

void read_potential_data(std::ifstream& file, potential_data& Po)
{
	Po.a = 0; Po.vism = 0;
	file.read((char*)(&Po.a), sizeof(float));
	file.read((char*)(&Po._1d_a), sizeof(float));
	file.read((char*)(&Po.c), sizeof(float));
	//file.read((char*)(&Po.g), sizeof(float));
	file.read((char*)(&Po.m), sizeof(float));
	file.read((char*)(&Po._1d_m), sizeof(float));
	file.read((char*)(&Po.dt), sizeof(float));
	file.read((char*)(&Po.dtm), sizeof(float));
	file.read((char*)(&Po.k), sizeof(float));
	file.read((char*)(&Po.vis), sizeof(float));
	file.read((char*)(&Po.vism), sizeof(float));

	file.read((char*)(&Po.Efiber), sizeof(float));
	file.read((char*)(&Po.rfiber), sizeof(float));
	file.read((char*)(&Po.rofiber), sizeof(float));
	file.read((char*)(&Po.Sfiber), sizeof(float));
	file.read((char*)(&Po.roliquid), sizeof(float));
	file.read((char*)(&Po.hfreefiber), sizeof(float));
	std::cerr << "Po " << Po.dt <<" "<< Po.a <<" "<< Po.vism << "\n";
	std::cerr << "Read potential_data\n";
	//std::cin.get();
}