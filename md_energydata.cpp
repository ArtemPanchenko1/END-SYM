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

void SaveEnergyDataStart(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, char* Name, unsigned int step)
{
	cudaMemcpy(Padd.h_Ebound, Padd.d_Ebound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream file;
	file.open(Name, std::ios::out | std::ios::binary);
	/*double maxE = -1e+30;
	for (int i = 0; i < P.NI; ++i)
	{
		if (maxE < Padd.h_Ebound[i])
			maxE = Padd.h_Ebound[i];
	}
	std::cerr << "MaxE " << maxE << "\n";/**/

	std::cerr << "write_p_data " << P.N << " " << P.NI << " " << P.NBP << " | " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << " " << "\n";
	file.write((char*)(&P.N), sizeof(int));
	file.write((char*)(&P.NI), sizeof(int));
	file.write((char*)(&P.NBP), sizeof(int));
	file.write((char*)(P.NBPT), 3 * sizeof(int));
	file.write((char*)(P.iBP), 2 * sizeof(int));
	
	//file.write((char*)(P.h_Ir0), 3 * P.NI * sizeof(float));
	file.write((char*)(P.h_BPR), 2 * P.NBP * sizeof(float));
	file.write((char*)(P.h_IM), P.N * sizeof(float));
	//file.write((char*)(P.h_1d_IM), P.N * sizeof(float));
#ifdef pre_ConnectCellSurface
	file.write((char*)(P.h_BPDfi), (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
#endif // pre_ConnectCellSurface	
	file.write((char*)(P.h_In), 2 * P.NI * sizeof(int));
	file.write((char*)(P.h_BP), P.NBP * sizeof(int));
	//file.write((char*)(P.h_ShIn), 2 * P.N * sizeof(int));

	file.write((char*)(&P0.N), sizeof(int));
	file.write((char*)(P0.h_RU0), 2 * P0.N * sizeof(float));
	write_potential_data(file, Po);
	std::cerr << "PINT " << file.tellp() << "\n";
	//std::cerr << "B0\n"; std::cin.get();
	file.write((char*)(&step), sizeof(unsigned int));
	//std::cerr << "B1\n"; std::cin.get();
	file.write((char*)(P.h_V), 2 * P.N * sizeof(float));
	//std::cerr << "B2\n"; std::cin.get();
	file.write((char*)(P.h_U), 2 * P.N * sizeof(float));
	//std::cerr << "B3 "<< Padd.h_Ebound<<"\n"; std::cin.get();
	file.write((char*)(Padd.h_Ebound), P.NI * sizeof(float));
	//std::cerr << "B4\n"; std::cin.get();
	file.close();
}

void SaveEnergyDataStep(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, char* Name, unsigned int step)
{
	cudaMemcpy(Padd.h_Ebound, Padd.d_Ebound, P.NI * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_U, P.d_U, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_V, P.d_V, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P.h_F, P.d_F, 2 * P.N * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream file;
	file.open(Name, std::ios::out | std::ios::binary | std::ios::app);
	std::cerr << "write_p_data " << "\n";
	//file.write((char*)(P.h_F), 2 * P.N * sizeof(float));
	file.write((char*)(&step), sizeof(unsigned int));
	file.write((char*)(P.h_V), 2 * P.N * sizeof(float));
	file.write((char*)(P.h_U), 2 * P.N * sizeof(float));	
	file.write((char*)(Padd.h_Ebound), P.NI * sizeof(float));
	/*double maxE = -1e+30;
	for (int i = 0; i < P.NI; ++i)
	{
		if (maxE < Padd.h_Ebound[i])
			maxE = Padd.h_Ebound[i];
	}
	std::cerr << "MaxE " << maxE << "\n";/**/
	//write_potential_data(file, Po);
	std::cerr << "PPint " << file.tellp() << "\n";
	file.close();
}

void LoadEnergyDataStep(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED, char* Name)
{
	double P00[2], DR, dR, Dfi, dfi;

	ED.dR = ED.DR;
	ED.dfi = ED.Dfi;
	ED.NR = (unsigned int)(Pnet.Rcut / ED.DR) + 1;
	ED.Nfi = (unsigned int)(2.0 * MC_pi / ED.Dfi + 1e-8);
	ED.NRfi = ED.NR * ED.Nfi;
	ED.NStep = 10000;
	ED.NStepR = ED.NStep * ED.NR;
	ED.NStepRfi = ED.NStep * ED.NR * ED.Nfi;	
	ED.h_Step = (unsigned int*)malloc(ED.NStep * sizeof(unsigned int));
	ED.h_E = (double*)malloc(ED.NStepRfi * sizeof(double));
	ED.h_sE = (double*)malloc(ED.NStepR * sizeof(double));
	std::cerr << "EL " << ED.NR << " " << ED.Nfi << " " << ED.NStep << " " << ED.NStepR << " " << ED.NStepRfi << "\n";

	std::ifstream file;
	file.open(Name, std::ios::in | std::ios::binary);
	file.read((char*)(&P.N), sizeof(int));
	file.read((char*)(&P.NI), sizeof(int));
	file.read((char*)(&P.NBP), sizeof(int));
	file.read((char*)(P.NBPT), 3 * sizeof(int));
	file.read((char*)(P.iBP), 2 * sizeof(int));
	P._1d_N = 1.0 / P.N;
	std::cerr << "read_p_data " << P.N << " " << P.NI << " " << P.NBP << " | " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << " " << "\n";
	
	//std::cin.get();	
	P.h_BPR = (float*)malloc(2 * P.NBP * sizeof(float));
	file.read((char*)(P.h_BPR), 2 * P.NBP * sizeof(float));
	P.h_IM = (float*)malloc(P.N * sizeof(float));
	file.read((char*)(P.h_IM), P.N * sizeof(float));
#ifdef pre_ConnectCellSurface
	P.h_BPDfi = (float*)malloc((P.NBPT[0] + P.NBPT[1]) * sizeof(float));
	file.read((char*)(P.h_BPDfi), (P.NBPT[0] + P.NBPT[1]) * sizeof(float));
#endif // pre_ConnectCellSurface	
	P.h_In = (int*)malloc(2 * P.NI * sizeof(int));
	file.read((char*)(P.h_In), 2 * P.NI * sizeof(int));
	P.h_BP = (int*)malloc(P.NBP * sizeof(int));
	file.read((char*)(P.h_BP), P.NBP * sizeof(int));
	file.read((char*)(&P0.N), sizeof(int));
	P0._1d_N = 1.0 / P0.N;
	P0.h_RU0 = (float*)malloc(2 * P0.N * sizeof(float));
	file.read((char*)(P0.h_RU0), 2 * P0.N * sizeof(float));
	read_potential_data(file, Po);

	P.h_V = (float*)malloc(2 * P.N * sizeof(float));
	P.h_U = (float*)malloc(2 * P.N * sizeof(float));
	Padd.h_Ebound = (float*)malloc(P.NI * sizeof(float));

	ED.h_Ek0 = (double*)malloc(P.N * sizeof(double));
	ED.h_Ebound0 = (double*)malloc(P.NI * sizeof(double));
	std::cerr << "Q0 " << file.eof() << " " << file.tellg() << "\n";
	unsigned int step, i;
	file.read((char*)(&step), sizeof(unsigned int));
	file.read((char*)(P.h_V), 2 * P.N * sizeof(float));
	file.read((char*)(P.h_U), 2 * P.N * sizeof(float));
	file.read((char*)(Padd.h_Ebound), P.NI * sizeof(float));
	for (i = 0; i < P.NI; ++i)
		ED.h_Ebound0[i] = Padd.h_Ebound[i];
	//memcpy((char*)(ED.h_Ebound0), (char*)(Padd.h_Ebound), P.NI * sizeof(float));
	for (i = 0; i < P.N; ++i)	
		ED.h_Ek0[i] = P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N]);
	
	ED.iStep = 0;
	for (;!file.eof();)
	{
		//std::cerr << "Q1 " << file.eof()<<" "<<file.tellg() << "\n";
		file.read((char*)(&step), sizeof(unsigned int));
		file.read((char*)(P.h_V), 2 * P.N * sizeof(float));
		file.read((char*)(P.h_U), 2 * P.N * sizeof(float));
		file.read((char*)(Padd.h_Ebound), P.NI * sizeof(float));
		CalcEnergyDistribution(P, P0, Pnet, Padd, Po, ED, step);
		//std::cin.get();
		file.peek();
	}	
	file.close();
	SaveEnergyRTineDistribution(P, P0, Pnet, Padd, Po, ED);
	SaveEnergyTineDistribution(P, P0, Pnet, Padd, Po, ED);
	//SumEnergyDistribution(P, P0, Pnet, Padd, Po, ED);
}

void CalcEnergyDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED, unsigned int step)
{
	unsigned int i, j, k, ir, ifi, ishift = ED.iStep * ED.NRfi;
	double dx, dy, r, sinfi, cosfi, _1d_r, fi;
	ED.h_Step[ED.iStep] = step;
	memset((ED.h_E + ishift), 0, ED.NRfi * sizeof(double));
	for (i = 0; i < P.N; ++i)
	{
		//std::cerr << "R " << i << " ";
		dx = P0.h_RU0[i] + P.h_U[i] - ED.P00[0];
		dy = P0.h_RU0[i + P.N] + P.h_U[i + P.N] - ED.P00[1];
		r = sqrt(dx * dx + dy * dy);
		_1d_r = 1.0 / r;
		
		cosfi = dx * _1d_r;
		sinfi = dy * _1d_r;
		if (sinfi >= 0)
			fi = acos(cosfi);
		else
			fi = 2.0 * MC_pi - acos(cosfi);
		ir = round(r / ED.DR);
		ifi = round(fi / ED.Dfi);
		//std::cerr << " " << P0.h_RU0[i] << " " << P.h_U[i] << " " << ED.P00[0] << " | " << P0.h_RU0[i + P.N] << " " << P.h_U[i + P.N] << " " << ED.P00[1] << "\n";
		//std::cerr << " " << ir <<" "<<r<<" "<<ED.DR << " | " << ifi << " " << ishift + ir * ED.Nfi + ifi << " " << ED.NStepRfi << "\n";
		ED.h_E[ishift + ir * ED.Nfi + ifi] += P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N])-ED.h_Ek0[i];
	}
	int nn = 0;
	for (k = 0; k < P.NI; ++k)
	{
		i = P.h_In[k];
		j = P.h_In[k + P.NI];
		dx = 0.5 * (P0.h_RU0[i] + P.h_U[i] + P0.h_RU0[j] + P.h_U[j]) - ED.P00[0];
		dy = 0.5 * (P0.h_RU0[i + P.N] + P.h_U[i + P.N] + P0.h_RU0[j + P.N] + P.h_U[j + P.N]) - ED.P00[1];

		r = sqrt(dx * dx + dy * dy);
		_1d_r = 1.0 / r;

		cosfi = dx * _1d_r;
		sinfi = dy * _1d_r;
		if (sinfi >= 0)
			fi = acos(cosfi);
		else
			fi = 2.0 * MC_pi - acos(cosfi);
		ir = round(r / ED.DR);
		ifi = round(fi / ED.Dfi);

		ED.h_E[ishift + ir * ED.Nfi + ifi] += Padd.h_Ebound[k] - ED.h_Ebound0[k];
		/*if (ir == 71 && ifi == 228)
		{
			++nn;
			std::cerr << "dE " << ED.DR << " " << ED.Dfi << " " << r << " " << fi << " " << r / ED.DR << " " << fi / ED.Dfi << "\n";
			//std::cerr << "dE " << k << " " << Padd.h_Ebound[k] << " " << ED.h_Ebound0[k] << " " << Padd.h_Ebound[k] - ED.h_Ebound0[k] << " " << ED.h_E[ishift + ir * ED.Nfi + ifi] << "\n";
			//if (fabs(Padd.h_Ebound[k] - ED.h_Ebound0[k]) > 1e-6)
			std::cin.get();
		}/**/
		
	}
	//std::cerr << "NN " << nn<<" "<< P.NI << "\n";
	//std::cin.get();
	++ED.iStep;
}

void SumEnergyDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED)
{
	unsigned int i, j, k, ir, ifi, ishift = ED.iStep * ED.NRfi;
	double maxj, maxRE;
	memset(ED.h_sE, 0, ED.NStepR * sizeof(double));
	for (k = 0; k < ED.NStep; ++k)
	{
		for (j = 0; j < ED.NR; ++j)
			for (i = 0; i < ED.Nfi; ++i)
			{
				//std::cerr << "W " << i << " " << j << " " << k << " | " << k * ED.NStepR + j << " " << k * ED.NStepRfi + j * ED.Nfi + i << "\n";
				ED.h_sE[k * ED.NR + j] += ED.h_E[k * ED.NRfi + j * ED.Nfi + i];
				/*if (j == 71)
				{
					if(ED.h_E[k * ED.NRfi + j * ED.Nfi + i]>1e-8)
					std::cerr << "Ee " << i << " " << ED.h_E[k * ED.NRfi + j * ED.Nfi + i] << "\n";

				}/**/
			}
		//std::cin.get();
	}
		
	//std::cin.get();
	std::ofstream file1;
	char filename1[256] = "";
	sprintf(filename1, "./result/ER_t.dat");
	file1.open(filename1, std::ios::out);
	file1 << "i time R E\n";
	maxRE = -1e+30;
	for (k = 0; k < ED.NStep; ++k)
	{
		maxRE = -1e+30;
		for (j = 0; j < ED.NR; ++j)
		{
			//std::cerr << "E " << j << " " << ED.h_sE[k * ED.NR + j] << " " << maxj << " " << maxRE << "\n";
			if (maxRE < ED.h_sE[k * ED.NR + j])
			{
				maxRE = ED.h_sE[k * ED.NR + j];
				maxj = j;
			}
		}
		std::cerr << "Emax " << maxj << " " << ED.h_Step[k] * Po.dt << " " << maxj * ED.DR << " " << maxRE << "\n";
		std::cin.get();
		file1 << k << " " << ED.h_Step[k] * Po.dt << " " << maxj * ED.DR << " " << maxRE << "\n";
	}
	file1.close();
}

void SaveEnergyTineDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED)
{
	unsigned int i, j, k, ir, ifi, ishift = ED.iStep * ED.NRfi;
	double maxj, maxRE;
	memset(ED.h_sE, 0, ED.NStepR * sizeof(double));
	std::ofstream file1;
	char filename1[256] = "";
	sprintf(filename1, "./result/E_R_t.dat");
	file1.open(filename1, std::ios::out);
	file1 << "i time";
	for (j = 0; j < ED.NR; ++j)
		file1 << " R"<<j;
	file1 << "\n";
	for (k = 0; k < 150/*ED.NStep*/; ++k)
	{
		file1 << k << " " << ED.h_Step[k] * Po.dt;
		for (j = 0; j < ED.NR; ++j)
		{
			for (i = 0; i < ED.Nfi; ++i)
			{
				//std::cerr << "W " << i << " " << j << " " << k << " | " << k * ED.NStepR + j << " " << k * ED.NStepRfi + j * ED.Nfi + i << "\n";
				ED.h_sE[k * ED.NR + j] += ED.h_E[k * ED.NRfi + j * ED.Nfi + i];
				/*if (j == 71)
				{
					if(ED.h_E[k * ED.NRfi + j * ED.Nfi + i]>1e-8)
					std::cerr << "Ee " << i << " " << ED.h_E[k * ED.NRfi + j * ED.Nfi + i] << "\n";

				}/**/
			}
			file1 << " " << ED.h_sE[k * ED.NR + j];
		}
		file1 << "\n";
		//std::cin.get();
	}
	//std::cin.get();	
	file1.close();
}

void SaveEnergyRTineDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED)
{
	unsigned int i, j, k, ir, ifi, ishift = ED.iStep * ED.NRfi;
	double maxj, maxRE;
	memset(ED.h_sE, 0, ED.NStepR * sizeof(double));
	std::ofstream file1;
	char filename1[256] = "";
	sprintf(filename1, "./result/E_t_R.dat");
	file1.open(filename1, std::ios::out);
	file1 << "i R";
	for (k = 0; k < 2/*ED.NStep*/; ++k)
		file1 << " t"<<ED.h_Step[k] * Po.dt;
	file1 << "\n";
	
	for (j = 0; j < ED.NR; ++j)
	{
		file1 << j << " " << ED.DR*j;
		for (k = 0; k < 101/*ED.NStep*/; k+=100)
		{
			for (i = 0; i < ED.Nfi; ++i)
			{
				//std::cerr << "W " << i << " " << j << " " << k << " | " << k * ED.NStepR + j << " " << k * ED.NStepRfi + j * ED.Nfi + i << "\n";
				ED.h_sE[k * ED.NR + j] += ED.h_E[k * ED.NRfi + j * ED.Nfi + i];
				/*if (j == 71)
				{
					if(ED.h_E[k * ED.NRfi + j * ED.Nfi + i]>1e-8)
					std::cerr << "Ee " << i << " " << ED.h_E[k * ED.NRfi + j * ED.Nfi + i] << "\n";

				}/**/
			}
			file1 << " " << ED.h_sE[k * ED.NR + j];
		}
		file1 << "\n";
		//std::cin.get();
	}
	//std::cin.get();	
	file1.close();
}