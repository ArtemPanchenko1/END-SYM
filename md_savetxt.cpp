#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>

void SaveTXTGraphsFmm(pAdd_data& Padd, potential_data& Po, pNet_data& Pnet)
{
	int i, j, is = Padd.StepVN[0] + Padd.StepVN[1] + Padd.StepVN[2];
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	std::ofstream file;
	char filename[200], filenamelast[200], filenamepart[10][100];	
	strcpy(filenamelast, "");
#ifndef pre_nonlinearC
	strcpy(filenamepart[1], "_Cc");
	//sprintf(filename, "./result/Fmm_Cc_Vx_%.2f-%.2f_Dx%.3f_vis%.2f_a%.4f.dat", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis, Pnet.a_aver);
#endif
#ifdef pre_nonlinearC
	strcpy(filenamepart[1], "_Cb");
	//sprintf(filename, "./result/Fmm_Cb_Vx_%.2f-%.2f_Dx%.3f_vis%.2f_a%.4f.dat", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis, Pnet.a_aver);
#endif
	strcat(filenamelast, filenamepart[1]);
	sprintf(filenamepart[2], "_Vx_%.2f-%.2f_Dx%.3f_vis%.4f_a%.4f_ID%.3f", Padd.V0[0], Padd.V0[2] + Padd.dV[2] * Padd.StepVN[2], Padd.Eps, Po.vis*1e3, Pnet.a_aver, Pnet.InitialDeformation);
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
	strcpy(filenamepart[7], "_CCS");
	strcat(filenamelast, filenamepart[7]);
#endif // pre_ConnectCellSurface
	sprintf(filenamepart[9], "_c%.1f_Nn%i_RR%.2f", Pnet.Connectivity, int(Pnet.Nnodes), Pnet.CellDistance);
	strcat(filenamelast, filenamepart[9]);

	
	strcat(filenamelast, ".dat");
	strcpy(filename, "./result/Fmm");
	strcat(filename, filenamelast);
	std::cerr << "Start Save Fmm " << is << " | " << filename << "\n";
	file.open(filename, std::ios::out);
	file << is << "\n";
	file << "Step V A_FXMin A_FYMin B_FXMin B_FYMin A_SXXMin A_SXYMin A_SYXMin A_SYYMin B_SXXMin B_SXYMin B_SYXMin B_SYYMin A_FRXMin A_FRYMin B_FRXMin B_FRYMin A_trSMin B_trSMin A_FXMax A_FYMax B_FXMax B_FYMax A_SXXMax A_SXYMax A_SYXMax A_SYYMax B_SXXMax B_SXYMax B_SYXMax B_SYYMax A_FRXMax A_FRYMax B_FRXMax B_FRYMax A_trSMax B_trSMax\n";
	file.precision(10);
	for (j = 0; j < is; ++j)
	{
		file << j << " " << Padd.h_V[j];
		for (i = 0; i < 36; ++i)
			file << " " << Padd.h_Fmm[ResultFmmNum * j + i];
		file << "\n";
	}
	file.close();

	strcpy(filename, "./result/Imp");
	strcat(filename, filenamelast);
	std::cerr << "Start Save Imp " << is << " | " << filename << "\n";
	file.open(filename, std::ios::out);
	file << is << "\n";
	file << "Step V A_ImpFX A_ImpFY B_ImpFX B_ImpFY A_ImpSXX A_ImpSXY A_ImpSYX A_ImpSYY B_ImpSXX B_ImpSXY B_ImpSYX B_ImpSYY\n";
	file.precision(10);
	for (j = 0; j < is; ++j)
	{
		file << j << " " << Padd.h_V[j];
		for (i = 36; i < 54; ++i)
			file << " " << Padd.h_Fmm[ResultFmmNum * j + i];
		file << "\n";
	}
	file.close();

	strcpy(filename, "./result/FmmIi");
	strcat(filename, filenamelast);
	std::cerr << "Start Save FmmIi " << is << " | " << filename << "\n";
	file.open(filename, std::ios::out);
	file << is << "\n";
	file << "Step V Ai_FXMin Ai_FYMin Bi_FXMin Bi_FYMin Ai_FXMax Ai_FYMax Bi_FXMax Bi_FYMax Ai_ImpFX Ai_ImpFY Bi_ImpFX Bi_ImpFY A_FRXMin A_FRYMin B_FRXMin B_FRYMin A_FRXMax A_FRYMax B_FRXMax B_FRYMax\n";
	file.precision(10);
	for (j = 0; j < is; ++j)
	{
		file << j << " " << Padd.h_V[j];
		for (i = 54; i < 66; ++i)
			file << " " << Padd.h_Fmm[ResultFmmNum * j + i];
		for (i = 12; i < 16; ++i)
			file << " " << Padd.h_Fmm[ResultFmmNum * j + i];
		for (i = 30; i < 34; ++i)
			file << " " << Padd.h_Fmm[ResultFmmNum * j + i];
		file << "\n";
	}
	file.close();


	//std::cin.get();
	fprintf(stderr, "Finish Save Fmm\n", 0);
}

void SaveTXTGraphsFR(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po, int savestep)
{
	int i, j, k, m;
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1];
	const unsigned int varNum = 18;
	double fs[varNum], fr[4], r[2], _1d_V, sumfi[2] = { 0,0 };
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	std::ofstream file;
	char filename[200];
	sprintf(filename, "./result/FR/FR_D%.3f_%.5f.dat", Padd.Eps, Padd.V);
	file.open(filename, std::ios::out);
	file << Padd.time/10 << "\n";
	file << "Step t A_FX A_FY B_FX B_FY A_SXX A_SXY A_SYX A_SYY B_SXX B_SXY B_SYX B_SYY A_FunX A_FunY B_FunX B_FunY A_trS B_trS Ai_FX Ai_FY Bi_FX Bi_FY\n";
	fprintf(stderr, "Start Save FR %.5f\n", Padd.V);
	
	for (i = 0; i < P.NBPT[0] - 1; ++i)
	{
		k = P.h_BP[i];
		if (P0.h_RU0[k] - P0.h_RU0[P.N - 2] > 0)
			sumfi[0] += P.h_BPDfi[i];
	}
	sumfi[0] = 1.0 / sumfi[0];
	for (i = P.NBPT[0]; i < npb2 - 1; ++i)
	{
		k = P.h_BP[i];
		if (P0.h_RU0[k] - P0.h_RU0[P.N - 1] < 0)
			sumfi[1] += P.h_BPDfi[i];
	}
	sumfi[1] = 1.0 / sumfi[1];
	
	for (j = 0; j < Padd.time/ ResultFRSave; j+=savestep)
	{
		memset(fs, 0, varNum * sizeof(double));
		r[0] = 0; r[1] = 0;
		for (i = 0; i < P.NBPT[0] - 1; ++i)
		{
			k = P.h_BP[i];
			fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + i];
			fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + i];
			fr[2] = Padd.h_FResult[(ResultFRNum * j + 2) * npb2 + i];
			fr[3] = Padd.h_FResult[(ResultFRNum * j + 3) * npb2 + i];

			fs[0] += fr[0];
			fs[1] += fr[1];
			fs[4] += fr[0] * fr[2];
			fs[5] += fr[1] * fr[2];
			fs[6] += fr[0] * fr[3];
			fs[7] += fr[1] * fr[3];
			r[0] += sqrt(fr[2] * fr[2] + fr[3] * fr[3]);
#ifdef pre_OneNodeForceA
			if (i == P.iBP[0])
			{
#ifndef pre_FreeCell
				fs[12] = fr[0];
				fs[13] = fr[1];
#endif // !pre_FreeCell

#ifdef pre_FreeCell
				fs[12] = fr[0];
				fs[13] = fr[1];
#endif // pre_FreeCell		
				//std::cerr << "FBA " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 2] << "\n";
			}
#endif // pre_OneNodeForceA
#ifndef pre_OneNodeForceA
			if (P0.h_RU0[k] - P0.h_RU0[P.N - 2] > 0)
			{
#ifndef pre_FreeCell
				fs[12] += fr[0];
				fs[13] += fr[1];
#endif // !pre_FreeCell

#ifdef pre_FreeCell
				fs[12] += P.h_BPDfi[i] * fr[0];
				fs[13] += P.h_BPDfi[i] * fr[1];
#endif // pre_FreeCell		
				//std::cerr << "FBA " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 2] << "\n";
			}
#endif // !pre_OneNodeForceA
			//std::cerr << "A " << i << " " << sqrt(fr[2] * fr[2] + fr[3] * fr[3]) << "\n";
			//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
			//++nA;				
		}
		r[0] /= double(P.NBPT[0] - 1);
		_1d_V = 1.0 / (MC_pi * r[0] * r[0]);
		fs[4] *= _1d_V;		fs[5] *= _1d_V;
		fs[6] *= _1d_V;		fs[7] *= _1d_V;
		fs[16] = 0.5 * (fs[4] + fs[7]);
#ifndef pre_OneNodeForceA
#ifdef pre_FreeCell
		fs[12] *= sumfi[0];
		fs[13] *= sumfi[0];
#endif // pre_FreeCell
#endif // !pre_OneNodeForceA

		for (i = P.NBPT[0]; i < npb2 - 1; ++i)
		{
			k = P.h_BP[i];
			fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + i];
			fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + i];
			fr[2] = Padd.h_FResult[(ResultFRNum * j + 2) * npb2 + i];
			fr[3] = Padd.h_FResult[(ResultFRNum * j + 3) * npb2 + i];

			fs[2] += fr[0];
			fs[3] += fr[1];
			fs[8] += fr[0] * fr[2];
			fs[9] += fr[1] * fr[2];
			fs[10] += fr[0] * fr[3];
			fs[11] += fr[1] * fr[3];
			r[1] += sqrt(fr[2] * fr[2] + fr[3] * fr[3]);
#ifdef pre_OneNodeForceB
			if (i == P.iBP[1])
			{
#ifndef pre_FreeCell
				fs[14] = fr[0];
				fs[15] = fr[1];
#endif // !pre_FreeCell

#ifdef pre_FreeCell
				fs[14] = P.h_BPDfi[i] * fr[0];
				fs[15] = P.h_BPDfi[i] * fr[1];
#endif // pre_FreeCell		
				//std::cerr << "FBA " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 2] << "\n";
			}
#endif // pre_OneNodeForceB
#ifndef pre_OneNodeForceB			
			if (P0.h_RU0[k] - P0.h_RU0[P.N - 1] < 0)
			{
#ifndef pre_FreeCell
				fs[14] += fr[0];
				fs[15] += fr[1];
#endif // !pre_FreeCell

#ifdef pre_FreeCell
				fs[14] += P.h_BPDfi[i] * fr[0];
				fs[15] += P.h_BPDfi[i] * fr[1];
#endif // pre_FreeCell
			}
#endif // !pre_OneNodeForceB
			//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
			//++nA;				
		}
		r[1] /= double(P.NBPT[1] - 1);
		_1d_V = 1.0 / (MC_pi * r[1] * r[1]);
		fs[8] *= _1d_V;		fs[9] *= _1d_V;
		fs[10] *= _1d_V;	fs[11] *= _1d_V;
		fs[17] = 0.5 * (fs[8] + fs[11]);
#ifdef pre_FreeCell
		fs[14] *= sumfi[1];
		fs[15] *= sumfi[1];
#endif // pre_FreeCell
		for (m = 0; m < varNum; ++m)
			fs[m] -= Padd.FSAB_aver[m];

		file.precision(10);
		file << j << " " << j * Po.dt;
		for (m = 0; m < varNum; ++m)
			file << " " << fs[m];

		fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[0]] - Padd.FABi_aver[0];
		fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[0]] - Padd.FABi_aver[1];
		fr[2] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2];
		fr[3] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[1]] - Padd.FABi_aver[3];
		file << " " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << "\n";

		//std::cerr << "FR " << r[0] << " " << r[1] << "\n";
		//std::cin.get();
	}
	file.close();
	//std::cin.get();
	fprintf(stderr, "Finish Save FR\n", 0);
}

void SaveTXTParticles(p_data& P, p0_data& P0, potential_data& Po, pNet_data &Pnet, char* Name)
{
	std::cerr << "P " << P.NBP << " " << P.NBPT[0] << " " << P.NBPT[1] << " " << P.NBPT[2] << "\n";
	uint_fast32_t i, j = 0, k, mmm;
	std::ofstream file;
	file.open(Name, std::ios::out);
	file << P.N << "\n";
	file << "Type X Y Ekx Eky Vx Vy Fx Fy\n";
	for (i = 0; i < P.N; ++i)
	{
		mmm = 0;
		for (j = 0; j < P.NBP; ++j)
			if (P.h_BP[j] == i)
			{
				if (j < P.NBPT[0])
					file << "C2 ";
				else if (j < P.NBPT[0] + P.NBPT[1])
					file << "C3 ";
				else if (j < P.NBPT[0] + P.NBPT[1] + P.NBPT[2])
				{
					file << "C4 ";
					//std::cerr << "BP " << i << " " << j << " " << P.h_BP[j] << "\n";
				}
					
				mmm = 1;
			}
		if (i > P.N - Pnet.AdN && mmm == 0)
		{
			file << "C5 ";
			mmm = 1;
		}
		if(mmm==0)			
			file << "C1 ";

		file << P0.h_RU0[i] + P.h_U[i] << " " << P0.h_RU0[i + P.N] + P.h_U[i + P.N]
			<< " " << 0.5 * Po.m * P.h_V[i] * P.h_V[i] << " " << 0.5 * Po.m * P.h_V[i + P.N] * P.h_V[i + P.N]//<<"\n";
			<< " " << P.h_V[i] << " " << P.h_V[i + P.N] 
			<< " " << P.h_F[i] << " " << P.h_F[i + P.N] << "\n"; 
		//std::cin.get();
	}
	file.close();
	std::cerr << "Save\n";
}

void SaveLammpsDATAParticles(p_data& P, p0_data& P0, potential_data& Po, pNet_data& Pnet, pAdd_data& Padd, char* Name)
{

	/*double maxE = -1e+30;
	for (int i = 0; i < P.NI; ++i)
	{
		if (maxE < Padd.h_Ebound[i]-Padd.h_Ebound0[i])
			maxE = Padd.h_Ebound[i]-Padd.h_Ebound0[i];
	}
	std::cerr << "MaxE " << maxE << "\n";/**/

	uint_fast32_t i, j = 0, k, mmm, ii, jj, pnid2 = P.NI / 2;
	std::ofstream file;
	file.open(Name, std::ios::out);
	double drx, dry, f, ek;
	//file << P.N << "\n";
	memset(Padd.h_LammpsSumF, 0, P.N * sizeof(float));
	j = 0;
	for (i = 0; i < P.NI; ++i)
	{
		ii = P.h_In[i];
		jj = P.h_In[i + P.NI];
		if (ii < jj)
		{
			
			drx = P0.h_RU0[jj] + P.h_U[jj] + P0.h_RU0[ii] + P.h_U[ii];
			dry = P0.h_RU0[jj + P.N] + P.h_U[jj + P.N] + P0.h_RU0[ii + P.N] + P.h_U[ii + P.N];
			f = Padd.h_Fbound[i] - Padd.h_Fbound0[i];
			Padd.h_LammpsAddParticles[j] = 0.5 * drx;
			Padd.h_LammpsAddParticles[j + pnid2] = 0.5 * dry;

			Padd.h_LammpsAddParticles[j + 2 * pnid2] = f;	


#ifdef pre_SaveLammpsEnergy
			Padd.h_LammpsAddParticles[j + 3 * pnid2] = Padd.h_Ebound[i] - Padd.h_Ebound0[i];
#endif // pre_SaveLammpsEnergy
#ifndef pre_SaveLammpsEnergy
			Padd.h_LammpsAddParticles[j + 3 * pnid2] = 0;
#endif // !pre_SaveLammpsEnergy	
#ifdef pre_SaveLammpsUx
			Padd.h_LammpsAddParticles[j + 3 * pnid2] = Padd.h_Ubound[i] - Padd.h_Ubound0[i];
#endif // pre_SaveLammpsUx
	
			Padd.h_LammpsSumF[ii] += f;
			Padd.h_LammpsSumF[jj] += f;
			//if(f)
			++j;
		}
		
	}
	for (i = 0; i < P.N; ++i)
	{
		if(P.h_ShIn[i + P.N] > 0)
			Padd.h_LammpsSumF[i] *= 1.0 / double(P.h_ShIn[i + P.N]);
		else
			Padd.h_LammpsSumF[i] = 0;
		//std::cerr << "FI " << P.h_ShIn[i + P.N] << " " << Padd.h_LammpsSumF[i] << "\n"; std::cin.get();
	}
	
	file << "LAMMPS Description T0="<<Po.dt<<"  (1st line of file)\n\n";
	file << P.N + pnid2 << " atoms\n" << P.NI << " bonds\n" << 0 << " angles\n" << 0 << " dihedrals\n" << 0 << " impropers\n";
	file << 5 + 1 << " atom types\n" << 1 << " bond types\n";
	file << -2.0 << " " << 2.0 << " xlo xhi\n" << -2.0 << " " << 2.0 << " ylo yhi\n";
	file << "Masses\n\n" << 1 << " " << 1.0 << "\n" << 2 << " " << 1.0 << "\n" << 3 << " " << 1.0 << "\n" << 4 << " " << 1.0 << "\n" << 5 << " " << 1.0 << "\n" << 6 << " " << 0.1 << "\n\n";
	//file << "Nonbond Coeffs\n" << 1 << " " << 1.0 << "\n" << 2 << " " << 1.0 << "\n" << 3 << " " << 1.0 << "\n" << 4 << " " << 1.0 << "\n" << 5 << " " << 1.0 << "\n";
	file << "Bond Coeffs\n\n" << 1 << " " << 1.0 << "\n\n";
	file << "Atoms\n\n";
	for (i = 0; i < P.N; ++i)
	{
		file << i+1 << " ";
		mmm = 0;
		for (j = 0; j < P.NBP; ++j)
			if (P.h_BP[j] == i)
			{
				if (j < P.NBPT[0])
					file << "2 ";
				else if (j < P.NBPT[0] + P.NBPT[1])
					file << "3 ";
				else if (j < P.NBPT[0] + P.NBPT[1] + P.NBPT[2])
					file << "4 ";
				mmm = 1;
			}
		if (i > P.N - Pnet.AdN && mmm == 0)
		{
			file << "5 ";
			mmm = 1;
		}
		if (mmm == 0)
			file << "1 ";

		file << P0.h_RU0[i] + P.h_U[i] << " " << P0.h_RU0[i + P.N] + P.h_U[i + P.N] << " " << 0.0 << "\n";
			//<< " " << 0.5 * Po.m * P.h_V[i] * P.h_V[i] << " " << 0.5 * Po.m * P.h_V[i + P.N] * P.h_V[i + P.N]//<<"\n";
			//<< " " << P.h_V[i] << " " << P.h_V[i + P.N]
			//<< " " << P.h_F[i] << " " << P.h_F[i + P.N] << "\n";
	}
	for (i = 0; i < pnid2; ++i)
	{
		file << i + P.N + 1 << " 6 " << Padd.h_LammpsAddParticles[i] << " " << Padd.h_LammpsAddParticles[i + pnid2] << " " << 0.0 << "\n";		
	}
	file << "Velocities\n\n";
	for (i = 0; i < P.N; ++i)
	{
		ek = P.h_IM[i] * (P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N]);
		file << i + 1 << " " << Padd.h_LammpsSumF[i] << " " << ek << " " << 0.0 << "\n";
		//std::cerr << "FI " << Padd.h_Fbound[i] << " " << Padd.h_LammpsSumF[i] << "\n"; std::cin.get();
		
		
	}
	for (i = 0; i < pnid2; ++i)
	{
		file << i + P.N + 1 << " " << Padd.h_LammpsAddParticles[i + 2 * pnid2] << " " << Padd.h_LammpsAddParticles[i + 3 * pnid2] << " " << 0.0 << "\n";
	}
	file << "\nBonds\n\n";
	ii = 1;
	j = 1;
	for (i = 0; i < P.NI; ++i)
	{
		if (P.h_In[i] < P.h_In[i + P.NI])
		{
			file << ii << " " << 1 << " " << P.h_In[i] + 1 << " " << P.N + j << "\n";
			file << ii + 1 << " " << 1 << " " << P.h_In[i + P.NI] + 1 << " " << P.N + j << "\n";
			ii+=2;
			++j;
		}			
	}	
	file.close();
	std::cerr << "Save PBM\n";
	//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
}
