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

void EFInitData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po)
{
	unsigned int k, i, j;
	double drx, dry, rcx, rcy;
	Padd.EF.NR = NumberCircles;
	Padd.EF.DR = Pnet.Rcut / double(Padd.EF.NR-1);
	Padd.EF._1d_DR = 1.0 / Padd.EF.DR;

	Padd.EF.P00[0] = 0;
	Padd.EF.P00[1] = 0;
	//Padd.EF.Nfi = (unsigned int)(2.0 * MC_pi / Padd.EF.Dfi + 1e-8);
	//Padd.EF.NRfi = Padd.EF.NR * Padd.EF.Nfi;

	//Padd.EF.h_BtoC = (unsigned int*)malloc(P.NI * sizeof(unsigned int));
	//Padd.EF.h_CtoB = (unsigned int*)malloc(P.NI * sizeof(unsigned int));
	Padd.EF.h_EFb = (float*)malloc(P.NI * sizeof(float));
	Padd.EF.h_EFb0 = (float*)malloc(P.NI * sizeof(float));
	std::cerr << "Time " << Padd.time << "\n";
	Padd.EF.h_CEF = (float*)malloc(2*Padd.time * 2 * Padd.EF.NR * sizeof(float));

	Padd.EF.iStep = 0;
	
	HANDLE_ERROR(cudaMalloc((void**)&Padd.EF.d_EFb, P.NI * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.EF.d_EFb0, P.NI * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&Padd.EF.d_CEF, Padd.ElementSteps * 2 * Padd.EF.NR * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)Padd.EF.d_CEF, 0, Padd.ElementSteps * 2 * Padd.EF.NR * sizeof(float)));
	

	Padd.EF.EFMinMax = (double*)malloc(ResultEFNum * (Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2]) * sizeof(double));
	Padd.EF.CEFAverage = (double*)malloc(Padd.EF.NR * sizeof(double));

	Padd.EF.EFcell0 = (double*)malloc(P.NBPT[0] * sizeof(double));
	Padd.EF.EFcell = (double*)malloc(P.NBPT[0] * sizeof(double));
}

void EFMinMaxData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po)
{
	unsigned int k, i, j;
	double MM[4 * NumberCircles], EFij;/*Min timeMin Max timeMax*/
	for (i = 0; i < NumberCircles; ++i)
	{
		MM[i] = 1e+35;
		MM[i + 2*NumberCircles] = -1e+35;
	}
	for (j = 0; j < Padd.time; j += 1)
	{
		//std::cerr << "RF ";
		for (i = 0; i < Padd.EF.NR; ++i)
		{
			//std::cerr << "(" << Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i] << "," << Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i] << ") ";
			EFij = (Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i]<0.1)? 0 : Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i] / Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i];
			if (MM[i] > EFij)
			{
				MM[i] = EFij;
				MM[i + NumberCircles] = j;
			}

				
			if (MM[i + 2*NumberCircles] < EFij)
			{
				MM[i + 2 * NumberCircles] = EFij;
				MM[i + 3 * NumberCircles] = j;
			}
							
		}
		//std::cerr << "\n";
		//std::cin.get();
	}
	/*std::cerr << "REAA " << Padd.EF.DR / 0.03987475 << "\n";
	for (i = 0; i < NumberCircles; ++i)
	{
		std::cerr << "RE " << i << " " << Padd.EF.DR * i / 0.03987475 << " " << MM[i] << " " << MM[i + NumberCircles]
			<< " | " << MM[i + 2 * NumberCircles] << " " << MM[i + 3 * NumberCircles] << "\n";
	}/**/
	
#ifdef pre_OneCellEdistribution
	k = round(0.03987475 * Padd.EF._1d_DR);
	Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 0] = MM[k + 2 * NumberCircles];
#endif // pre_OneCellEdistribution
#ifdef pre_OneCellFdistribution
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1];
	double fr[4], fs[2], fm[4] = { 1e+35, 0, -1e35, 0 };
	//Padd.ImpulsSteps
	for (j = 0; j < Padd.time / ResultFRSave; ++j)
	{
		fs[0] = 0; fs[1] = 0;
		for (i = 0; i < P.NBPT[0] - 1; ++i)
		{
			k = P.h_BP[i];
			fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + i];
			fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + i];
			fr[2] = Padd.h_FResult[(ResultFRNum * j + 2) * npb2 + i];
			fr[3] = Padd.h_FResult[(ResultFRNum * j + 3) * npb2 + i];

			fs[0] += fabs(sqrt(fr[0] * fr[0] + fr[1] * fr[1]));
		}
		fs[0] /= double(P.NBPT[0]);
		if (fm[0] > fs[0])
		{
			fm[0] = fs[0];
			fm[1] = j;
		}
		if (fm[2] < fs[0])
		{
			fm[2] = fs[0];
			fm[3] = j;
		}
	}

	std::cerr << "FCell " << fm[0] << " " << fm[1] << " " << fm[2] << " " << fm[3] << "\n";
	k = round(0.03987475 * Padd.EF._1d_DR);
	Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 0] = fm[2]- Padd.EF.FCAverage;
Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 0] = Padd.EF.FCAverage;
	
#endif // pre_OneCellFdistribution
	std::cerr << "EFmm " << k << " " << Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 0] << " " << " " << "\n";
	for (i = k; i < Padd.EF.NR; ++i)
	{
		if (MM[i + 2 * NumberCircles] < pre_RatioToCell * Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 0])
		{
			Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 1] = i * Padd.EF.DR;
			Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 2] = MM[i + 2 * NumberCircles];
			break;
		}

		//std::cerr << "RE " << i << " " << Padd.EF.DR * i / 0.03987475 << " " << MM[i] << " " << MM[i + NumberCircles]
		//	<< " | " << MM[i + 2 * NumberCircles] << " " << MM[i + 3 * NumberCircles] << "\n";
	}
	std::cerr << "RE " << Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 1] << " " << Padd.EF.EFMinMax[ResultEFNum * Padd.StepV + 2] << "\n";

	std::ofstream file;
	char filename[200];
	sprintf(filename, "./result/EF.dat");
	file.open(filename, std::ios::out);
	for (j = 0; j < Padd.time; j += 10)
	{
		file << j << " " << j * Po.dt;
		for (i = 0; i < Padd.EF.NR; ++i)
		{
			//EFij = (Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i] < 0.1) ? 0 : Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i] / Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i];
			EFij = Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i];
			file << " " << EFij;

		}
		file << "\n";
		//std::cerr << "\n";
		//std::cin.get();
	}
	file.close();	
}

void SaveTXTGraphsEFmm(pAdd_data& Padd, potential_data& Po, pNet_data& Pnet)
{
	int i, j, is = Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2];
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	std::ofstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];
	strcpy(filenamelast, "");
#ifndef pre_nonlinearC
	strcpy(filenamepart[1], "_Cc");
	//sprintf(filename, "./result/EFmm_Cc_Vx_%.2f-%.2f_Dx%.3f_vis%.2f_a%.4f.dat", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis, Pnet.a_aver);
#endif
#ifdef pre_nonlinearC
	strcpy(filenamepart[1], "_Cb");
	//sprintf(filename, "./result/Fmm_Cb_Vx_%.2f-%.2f_Dx%.3f_vis%.2f_a%.4f.dat", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis, Pnet.a_aver);
#endif
	strcat(filenamelast, filenamepart[1]);
	sprintf(filenamepart[2], "_Vx_%.2f-%.2f_Dx%.3f_vis%.2f_a%.4f_ID%.3f", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis, Pnet.a_aver, Pnet.InitialDeformation);
	strcat(filenamelast, filenamepart[2]);
#ifdef pre_MoveCell
	strcpy(filenamepart[3], "_MoveCell");
	strcat(filenamelast, filenamepart[3]);
#endif // pre_MoveCell
#ifdef TwoDirectionMove
	strcpy(filenamepart[4], "_TwoDirection");
	strcat(filenamelast, filenamepart[4]);
#endif // TwoDirectionMove
#ifdef SineMove
	strcpy(filenamepart[5], "_SineMove");
	strcat(filenamelast, filenamepart[5]);
#endif // TwoDirectionMove
#ifdef pre_OneNodeContract
	strcpy(filenamepart[6], "_ONC");
	strcat(filenamelast, filenamepart[6]);
#endif // pre_OneNodeContract
#ifdef pre_FreeCellHalf
	strcpy(filenamepart[7], "_FCH");
	strcat(filenamelast, filenamepart[7]);
#endif // pre_FreeCellHalf
#ifdef pre_ConnectCellSurface
	strcpy(filenamepart[8], "_CCS");
	strcat(filenamelast, filenamepart[8]);
#endif // pre_ConnectCellSurface
#ifdef pre_OnlyOneCell
	strcpy(filenamepart[9], "_OnlyOneCell");
	strcat(filenamelast, filenamepart[9]);
#endif // pre_OnlyOneCell

	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/EFmm");
	strcat(filename, filenamelast);
	std::cerr << "Start Save EFmm " << is << " | " << filename << "\n";
	file.open(filename, std::ios::out);
	file << is << "\n";
#ifdef pre_OneCellEdistribution
	file << "Step V ECellsurface RE" << pre_RatioToCell << " E" << pre_RatioToCell"\n";
#endif // pre_OneCellEdistribution
#ifdef pre_OneCellFdistribution
	file << "Step V FCellsurface RF"<<pre_RatioToCell<<" F"<<pre_RatioToCell<<"\n";
#endif // pre_OneCellFdistribution	
	file.precision(10);
	for (j = 0; j < is; ++j)
	{
		file << j << " " << Padd.h_V[j];
		for (i = 0; i < ResultEFNum; ++i)
			file << " " << Padd.EF.EFMinMax[ResultEFNum * j + i];
		file << "\n";
	}
	file.close();
	//std::cin.get();
	fprintf(stderr, "Finish Save Fmm\n", 0);
}

void EFAverageData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po)
{
	unsigned int k, i, j;
	double MM[NumberCircles], EFij, _1d_time = 1.0 / double(Padd.time);/*Min timeMin Max timeMax*/
	//double FCAverage0 = 3.82347e-05, CEFAverage[] = {};
	//double CEFAverage[] = {0, 0, 0, 0, 0, 0.102638, 0.0600173, 0.0464955, 0.0425375, 0.0383795, 0.041324, 0.0394512, 0.0342299, 0.0308073, 0.0304403, 0.026243, 0.0310083, 0.0262339, 0.0248811, 0.0243659, 0.0226467, 0.0239804, 0.0216699, 0.0219656, 0.0204865, 0.0201982, 0.0195797, 0.0195828, 0.0203013, 0.018497, 0.0182339, 0.0163506, 0.0175293, 0.0170717, 0.0164732, 0.0171191, 0.0160095, 0.0158247, 0.0155384, 0.0158828, 0.0153249, 0.014938, 0.0150854, 0.0150149, 0.0142931, 0.0145839, 0.0138945, 0.0142389, 0.0142858, 0.0140674, 0.0148281, 0.0142524, 0.0143405, 0.0134903, 0.013117, 0.0133606, 0.0137616, 0.0125681, 0.00191866, 1.42194e-06, 0};
	//double FCAverage0 = 1.28262, CEFAverage[] = {0, 0, 0, 0, 0, 0.498605, 0.291521, 0.235014, 0.208868, 0.198832, 0.209377, 0.196544, 0.174937, 0.149005, 0.15337, 0.130711, 0.15726, 0.130658, 0.125494, 0.12415, 0.114326, 0.119608, 0.108585, 0.110478, 0.102633, 0.101795, 0.0989694, 0.0991925, 0.101542, 0.0934846, 0.0911686, 0.0828877, 0.0878181, 0.0859861, 0.0832549, 0.0862433, 0.0809069, 0.0795374, 0.0785686, 0.0800107, 0.0774534, 0.0754243, 0.0761935, 0.0758118, 0.0719396, 0.0735389, 0.070077, 0.0720623, 0.0720815, 0.0708443, 0.0749137, 0.072117, 0.0722422, 0.0681255, 0.0662652, 0.0674785, 0.0695616, 0.0635031, 0.00969096, 1.42194e-06, 0};
	//double FCAverage0 = 2.58293, CEFAverage[] = {0, 0, 0, 0, 2.1825, 0.711082, 0.566967, 0.485862, 0.425011, 0.400211, 0.412877, 0.388572, 0.341696, 0.306006, 0.310537, 0.264897, 0.317522, 0.258715, 0.248318, 0.253439, 0.228414, 0.240938, 0.220521, 0.220786, 0.207685, 0.206977, 0.199083, 0.199713, 0.204275, 0.188037, 0.184853, 0.165926, 0.177731, 0.173304, 0.168473, 0.175473, 0.161915, 0.161994, 0.158784, 0.161144, 0.156749, 0.152238, 0.154635, 0.152353, 0.145938, 0.148264, 0.141475, 0.145586, 0.146113, 0.14304, 0.151761, 0.145424, 0.146143, 0.137818, 0.133835, 0.136509, 0.140717, 0.128577, 0.0196036, 1.42194e-06, 0};
	//double FCAverage0 = 5.2208, CEFAverage[] = {0, 0, 0, 0, 2.50133, 1.42919, 1.03801, 0.9443, 0.866176, 0.852389, 0.79375, 0.763681, 0.687318, 0.622125, 0.605526, 0.559198, 0.643305, 0.517177, 0.505707, 0.519687, 0.462357, 0.480731, 0.446891, 0.448721, 0.428054, 0.411512, 0.403925, 0.413774, 0.408549, 0.38342, 0.373295, 0.3399, 0.362874, 0.350779, 0.342709, 0.356355, 0.331526, 0.330546, 0.322489, 0.327539, 0.320773, 0.311891, 0.312387, 0.313383, 0.296198, 0.300255, 0.290217, 0.295627, 0.297816, 0.292874, 0.31067, 0.295898, 0.298296, 0.280211, 0.273222, 0.279507, 0.287023, 0.262877, 0.0397558, 1.42194e-06, 0};
	//double FCAverage0 = 10.5026, CEFAverage[] = {0, 0, 0, 5.60169, 3.06956, 2.41194, 2.04705, 1.82837, 1.72878, 1.76541, 1.60823, 1.50459, 1.27529, 1.30718, 1.10377, 1.36899, 1.17791, 1.06886, 1.03934, 0.98688, 1.00005, 0.968739, 0.898243, 0.885247, 0.890206, 0.844036, 0.840555, 0.864706, 0.81081, 0.800933, 0.733221, 0.714475, 0.754135, 0.696777, 0.720476, 0.733595, 0.684725, 0.67267, 0.678056, 0.663732, 0.662792, 0.645173, 0.636977, 0.646612, 0.61756, 0.612253, 0.607134, 0.601795, 0.622902, 0.604853, 0.633216, 0.618102, 0.611405, 0.579425, 0.563677, 0.580063, 0.591589, 0.54443, 0.0815663, 1.42194e-06, 0};
	double CEFAverage[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double FCAverage0[] = { 3.82347e-05, 0.254451, 1.28262, 2.58293, 5.2208, 10.5026 };
	double FCAverage00[] = { 0.254451, 0.255471, 0.258772, 0.261818, 0.26528, 0.260138 };
	
	for (i = 0; i < NumberCircles; ++i)
	{
		MM[i] = 0;
	}
	for (j = 0; j < Padd.time; j += 1)
	{
		//std::cerr << "RF ";
		for (i = 0; i < Padd.EF.NR; ++i)
		{
			//std::cerr << "(" << Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i] << "," << Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i] << ") ";
			EFij = (Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i] < 0.1) ? 0 : Padd.EF.h_CEF[j * 2 * Padd.EF.NR + i] / Padd.EF.h_CEF[j * 2 * Padd.EF.NR + Padd.EF.NR + i];
			MM[i] += EFij;
		}
		//std::cerr << "\n";
		//std::cin.get();
	}
	std::cerr << "CEF ";
	for (i = 0; i < Padd.EF.NR; ++i)
	{
		Padd.EF.CEFAverage[i] = MM[i] * _1d_time- CEFAverage[i];
		//std::cerr << Padd.EF.CEFAverage[i] << ", ";
	}
	std::cerr << "\n";
	/*for (i = 0; i < NumberCircles; ++i)
	{
		std::cerr << "CEFA " << i << " " << Padd.EF.DR * i / 0.03987475 << " " << Padd.EF.CEFAverage[i] << "\n";
	}/**/
	
#ifdef pre_OneCellEdistribution
	k = round(0.03987475 * Padd.EF._1d_DR);
	Padd.EF.FCAverage = MM[k];
#endif // pre_OneCellEdistribution
#ifdef pre_OneCellFdistribution
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1];
	double fr[4], fs[2], fm[2] = { 0, 0 };
	//Padd.ImpulsSteps
	for (j = 0; j < Padd.time / ResultFRSave; ++j)
	{
		fs[0] = 0; fs[1] = 0;
		for (i = 0; i < P.NBPT[0] - 1; ++i)
		{
			k = P.h_BP[i];
			fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + i];
			fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + i];
			fr[2] = Padd.h_FResult[(ResultFRNum * j + 2) * npb2 + i];
			fr[3] = Padd.h_FResult[(ResultFRNum * j + 3) * npb2 + i];

			fs[0] += fabs(sqrt(fr[0] * fr[0] + fr[1] * fr[1]));
		}
		fm[0] += fs[0] / double(P.NBPT[0]);			
	}	
	Padd.EF.FCAverage = fm[0] * _1d_time * double(ResultFRSave) -FCAverage0[Padd.EF.NStepDef];
	Padd.EF.FCAverage = FCAverage00[Padd.EF.NStepDef];
	std::cerr << "FCell " << Padd.EF.FCAverage << "\n"; //std::cin.get();
	k = round(0.03987475 * Padd.EF._1d_DR);
#endif // pre_OneCellFdistribution

	for (i = k; i < Padd.EF.NR; ++i)
	{
		if (Padd.EF.CEFAverage[i] < pre_RatioToCell * Padd.EF.FCAverage)
		{
			k = i;
			std::cerr << "RE " << i << " " << Padd.EF.DR * i / 0.03987475 << " " << Padd.EF.CEFAverage[i] << "\n";
			break;
		}

		//std::cerr << "RE " << i << " " << Padd.EF.DR * i / 0.03987475 << " " << MM[i] << " " << MM[i + NumberCircles]
		//	<< " | " << MM[i + 2 * NumberCircles] << " " << MM[i + 3 * NumberCircles] << "\n";
	}

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
	sprintf(filenamepart[2], "_Dx%.3f_a%.4f_ID%.3f", Padd.Eps, Pnet.a_aver, Pnet.InitialDeformation);
	strcat(filenamelast, filenamepart[2]);
#ifdef pre_MoveCell
	strcpy(filenamepart[3], "_MoveCell");
	strcat(filenamelast, filenamepart[3]);
#endif // pre_MoveCell
#ifdef pre_OneNodeContract
	strcpy(filenamepart[6], "_ONC");
	strcat(filenamelast, filenamepart[6]);
#endif // pre_OneNodeContract
#ifdef pre_FreeCellHalf
	strcpy(filenamepart[7], "_FCH");
	strcat(filenamelast, filenamepart[7]);
#endif // pre_FreeCellHalf
#ifdef pre_ConnectCellSurface
	strcpy(filenamepart[8], "_CCS");
	strcat(filenamelast, filenamepart[8]);
#endif // pre_ConnectCellSurface
#ifdef pre_OnlyOneCell
	strcpy(filenamepart[9], "_OnlyOneCell");
	strcat(filenamelast, filenamepart[9]);
#endif // pre_OnlyOneCell

	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/EFAverage");
	strcat(filename, filenamelast);
	file.open(filename, std::ios::out);
	file << "Cell "<< 0.03987475 << " " << Padd.EF.FCAverage<<"\n";
	file << pre_RatioToCell<<"Cell " << k * Padd.EF.DR << " " << Padd.EF.CEFAverage[k] << "\n";
	for (i = 0; i < Padd.EF.NR; ++i)
	{
		file << i << " " << i * Padd.EF.DR << " " << Padd.EF.CEFAverage[i] << "\n";
	}
	file.close();
}