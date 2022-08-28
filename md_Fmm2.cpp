#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>

void calculate_Fminmax2(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po)
{
	int i, j, k, m;
	const unsigned int varNum = 18;
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1], nImp[2] = { 0, 0 }, nImpi[2] = { 0,0 }, stepBmax, stepBmin;
	double fs[varNum], fr[4], r[2], _1d_V, R0[3] = { 0.03987475, Padd.Eps0 * 0.03987475, 0.03987475 }, Impuls[varNum], fsmm[2 * varNum], Impulsi[4], fsimm[8], sumfi[2] = { 0,0 };;
	//double fAmm[4] = { 1e30, -1e30, 1e30, -1e30 }, fBmm[4] = { 1e30, -1e30, 1e30, -1e30 }, intfA[2] = { 0,0 }, intfB[2] = { 0,0 }, 
	//	AvA[4] = { 0, 0, 0, 0 }, AvB[4] = { 0, 0, 0, 0 }, _1d_nA, _1d_nB, _1d_n, ImpulsA = 0, ImpulsB = 0, Avi[8] = { 0,0,0,0,0,0,0,0 }, f, _1d_FAB_aver[4], Impulsi[4] = { 0,0,0,0 };
	bool impA = true, impB = true, impBStart = false, impBiStart = false;
#ifdef pre_FreeCell
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
#endif // pre_FreeCell

	
	//_1d_FAB_aver[0] = 1.0 / Padd.FAB_aver[0]; _1d_FAB_aver[1] = 1.0 / Padd.FAB_aver[1];
	//_1d_FAB_aver[2] = 1.0 / Padd.FAB_aver[2]; _1d_FAB_aver[3] = 1.0 / Padd.FAB_aver[3];
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	//for (m = 0; m < varNum; ++m)
	//	aver[m] = sqrt(Padd.FSAB_aver[m + varNum] - Padd.FSAB_aver[m] * Padd.FSAB_aver[m]);
	//for (m = 0; m < 4; ++m)
	//	averi[m] = sqrt(Padd.FABi_aver[m + 4] - Padd.FABi_aver[m] * Padd.FABi_aver[m]);
	std::cerr << "AAAA " << sumfi[0] << " " << sumfi[1] << "\n";
	std::cerr << "Faver ";
	for (m = 0; m < varNum; ++m)
		std::cerr << Padd.FSAB_aver[m] << " " << Padd.FSAB_aver[m + varNum] << " | ";
	std::cerr << "\n";

	std::cerr << "Faveri ";
	for (m = 0; m < 4; ++m)
		std::cerr << Padd.FABi_aver[m] << " " << Padd.FABi_aver[m + 4] << " | ";
	std::cerr << "\n";

	memset(Impuls, 0, varNum * sizeof(double));
	memset(Impulsi, 0, 4 * sizeof(double));
	for (m = 0; m < varNum; ++m)
	{
		fsmm[m] = 1e30;
		fsmm[m + varNum] = -1e30;
	}
	for (m = 0; m < 4; ++m)
	{
		fsimm[m] = 1e30;
		fsimm[m + 4] = -1e30;
	}

	for (j = 0; j < Padd.time/ResultFRSave; ++j)
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
			//if(j%10000==0)std::cerr << "FA " << i <<" "<<j << " | " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << "\n";
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
				//if (i == P.iBP[0] && j % 1000 == 0) { std::cerr << "FA " << i << " " << fr[0] << " " << fr[1] << "\n"; }
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
			//if (j % 10000 == 0)std::cerr << "FB " << i << " " << j << " | " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << "\n";

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
		//if (j % 1000 == 0) { std::cerr << "U " << j << " " << fs[12] << " " << fs[13] << " | " << fs[14] << " " << fs[15] << "\n"; }
		for (m = 0; m < varNum; ++m)
			fs[m] -= Padd.FSAB_aver[m];

		if (nImp[0] < Padd.ImpulsSteps)
		{
			Impuls[0] += fs[0] * Po.dt;
			Impuls[1] += fs[1] * Po.dt;
			for (m = 4; m < 8; ++m)
				Impuls[m] += fs[m] * Po.dt;

			++nImp[0];
		}
#ifdef pre_ReleaseHalfSine
		if (!impBStart && fs[14] > 10.0 * Padd.FSAB_aver[14 + varNum])
#endif // pre_ReleaseHalfSine
#ifndef pre_ReleaseHalfSine
		if (!impBStart && fs[14] < -5.0 * Padd.FSAB_aver[14 + varNum])
#endif // !pre_ReleaseHalfSine
		{
			impBStart = true;
			std::cerr << "StartImpulsB " << j << " " << j * Po.dt <<" " << j * Po.dt / 0.016235485 << " | " << fs[14] << " "<< Padd.FSAB_aver[14 + varNum] << "\n";
			//std::cin.get();
		}
		if (impBStart && nImp[1] < Padd.ImpulsSteps)//Diff
		{
			Impuls[2] += fs[2] * Po.dt;
			Impuls[3] += fs[3] * Po.dt;
			for (m = 9; m < 12; ++m)
				Impuls[m] += fs[m] * Po.dt;

			++nImp[1];
		}

		fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[0]] - Padd.FABi_aver[0];
		fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[0]] - Padd.FABi_aver[1];
		fr[2] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2];
		fr[3] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[1]] - Padd.FABi_aver[3];

		if (nImpi[0] < Padd.ImpulsSteps)
		{
			Impulsi[0] += fr[0] * Po.dt;
			Impulsi[1] += fr[1] * Po.dt;
			++nImpi[0];
		}
#ifdef pre_ReleaseHalfSine
		if (!impBiStart && fr[2] > 5.0 * Padd.FABi_aver[2 + 4])
#endif // pre_ReleaseHalfSine
#ifndef pre_ReleaseHalfSine
		if (!impBiStart && fr[2] < -5.0 * Padd.FABi_aver[2 + 4])
#endif // !pre_ReleaseHalfSine		
		{
			impBiStart = true;
			std::cerr << "StartImpulsBi " << j << " " << j * Po.dt << "\n";
			/*if (j * Po.dt < 0.018)
			{
				std::wcerr << "ERR! " << j * Po.dt << "\n";
				std::cin.get();
			}/**/
		}
		//std::cerr << "FBi " << Padd.h_FResult[2 * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2] << "\n";
		if (impBiStart && nImpi[1] < Padd.ImpulsSteps)//Diff
		{
			Impulsi[2] += fr[2] * Po.dt;
			Impulsi[3] += fr[3] * Po.dt;
			++nImpi[1];
		}/**/

		if (j < Padd.ImpulsSteps)
		{
			if (fsimm[0] > fr[0]) fsimm[0] = fr[0];
			else if (fsimm[0 + 4] < fr[0]) fsimm[0 + 4] = fr[0];
			if (fsimm[1] > fr[1]) fsimm[1] = fr[1];
			else if (fsimm[1 + 4] < fr[1]) fsimm[1 + 4] = fr[1];
		}
		if (impBiStart && nImpi[1] < Padd.ImpulsSteps)
		{
			if (fsimm[2] > fr[2]) fsimm[2] = fr[2];
			else if (fsimm[2 + 4] < fr[2]) fsimm[2 + 4] = fr[2];
			if (fsimm[3] > fr[3]) fsimm[3] = fr[3];
			else if (fsimm[3 + 4] < fr[3]) fsimm[3 + 4] = fr[3];
		}

		if (j < Padd.ImpulsSteps)
		{
			if (fsmm[0] > fs[0]) fsmm[0] = fs[0];
			else if (fsmm[0 + varNum] < fs[0]) fsmm[0 + varNum] = fs[0];
			if (fsmm[1] > fs[1]) fsmm[1] = fs[1];
			else if (fsmm[1 + varNum] < fs[1]) fsmm[1 + varNum] = fs[1];

			if (fsmm[4] > fs[4]) fsmm[4] = fs[4];
			else if (fsmm[4 + varNum] < fs[4]) fsmm[4 + varNum] = fs[4];
			if (fsmm[5] > fs[5]) fsmm[5] = fs[5];
			else if (fsmm[5 + varNum] < fs[5]) fsmm[5 + varNum] = fs[5];
			if (fsmm[6] > fs[6]) fsmm[6] = fs[6];
			else if (fsmm[6 + varNum] < fs[6]) fsmm[6 + varNum] = fs[6];
			if (fsmm[7] > fs[7]) fsmm[7] = fs[7];
			else if (fsmm[7 + varNum] < fs[7]) fsmm[7 + varNum] = fs[7];

			if (fsmm[12] > fs[12]) fsmm[12] = fs[12];
			else if (fsmm[12 + varNum] < fs[12]) fsmm[12 + varNum] = fs[12];
			if (fsmm[13] > fs[13]) fsmm[13] = fs[13];
			else if (fsmm[13 + varNum] < fs[13]) fsmm[13 + varNum] = fs[13];

			if (fsmm[16] > fs[16]) fsmm[16] = fs[16];
			else if (fsmm[16 + varNum] < fs[16]) fsmm[16 + varNum] = fs[16];
		}

		if (impBStart && nImp[1] < Padd.ImpulsSteps)
		{
			if (fsmm[2] > fs[2]) fsmm[2] = fs[2];
			else if (fsmm[2 + varNum] < fs[2]) fsmm[2 + varNum] = fs[2];
			if (fsmm[3] > fs[3]) fsmm[3] = fs[3];
			else if (fsmm[3 + varNum] < fs[3]) fsmm[3 + varNum] = fs[3];

			if (fsmm[8] > fs[8]) fsmm[8] = fs[8];
			else if (fsmm[8 + varNum] < fs[8]) fsmm[8 + varNum] = fs[8];
			if (fsmm[9] > fs[9]) fsmm[9] = fs[9];
			else if (fsmm[9 + varNum] < fs[9]) fsmm[9 + varNum] = fs[9];
			if (fsmm[10] > fs[10]) fsmm[10] = fs[10];
			else if (fsmm[10 + varNum] < fs[10]) fsmm[10 + varNum] = fs[10];
			if (fsmm[11] > fs[11]) fsmm[11] = fs[11];
			else if (fsmm[11 + varNum] < fs[11]) fsmm[11 + varNum] = fs[11];

			if (fsmm[14] > fs[14]) { fsmm[14] = fs[14]; stepBmin = j; }
			else if (fsmm[14 + varNum] < fs[14]) { fsmm[14 + varNum] = fs[14]; stepBmax = j; }
			if (fsmm[15] > fs[15]) fsmm[15] = fs[15]; 
			else if (fsmm[15 + varNum] < fs[15]) fsmm[15 + varNum] = fs[15];

			if (fsmm[17] > fs[17]) fsmm[17] = fs[17];
			else if (fsmm[17 + varNum] < fs[17]) fsmm[17 + varNum] = fs[17];
		}

		/*for (m = 0; m < varNum; ++m)
		{
			if (fsmm[m] > fs[m]) fsmm[m] = fs[m];
			if (fsmm[m + varNum] < fs[m]) fsmm[m + varNum] = fs[m];
		}/**/

		//std::cerr << "Fmm step " << j << " " << r[0] << " " << r[1] << "\n";
		//std::cin.get();
	}

	/*for (j = 0; j < Padd.time; ++j)
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
			//if(j%10000==0)std::cerr << "FA " << i <<" "<<j << " | " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << "\n";
			fs[0] += fr[0];
			fs[1] += fr[1];
			fs[4] += fr[0] * fr[2];
			fs[5] += fr[1] * fr[2];
			fs[6] += fr[0] * fr[3];
			fs[7] += fr[1] * fr[3];
			r[0] += sqrt(fr[2] * fr[2] + fr[3] * fr[3]);
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
			//std::cerr << "A " << i << " " << sqrt(fr[2] * fr[2] + fr[3] * fr[3]) << "\n";
			//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";
			//++nA;
		}
		r[0] /= double(P.NBPT[0] - 1);
		_1d_V = 1.0 / (MC_pi * r[0] * r[0]);
		fs[4] *= _1d_V;		fs[5] *= _1d_V;
		fs[6] *= _1d_V;		fs[7] *= _1d_V;
		fs[16] = 0.5 * (fs[4] + fs[7]);
#ifdef pre_FreeCell
		fs[12] *= sumfi[0];
		fs[13] *= sumfi[0];
#endif // pre_FreeCell

		for (i = P.NBPT[0]; i < npb2 - 1; ++i)
		{
			k = P.h_BP[i];
			fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + i];
			fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + i];
			fr[2] = Padd.h_FResult[(ResultFRNum * j + 2) * npb2 + i];
			fr[3] = Padd.h_FResult[(ResultFRNum * j + 3) * npb2 + i];
			//if (j % 10000 == 0)std::cerr << "FB " << i << " " << j << " | " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << "\n";

			fs[2] += fr[0];
			fs[3] += fr[1];
			fs[8] += fr[0] * fr[2];
			fs[9] += fr[1] * fr[2];
			fs[10] += fr[0] * fr[3];
			fs[11] += fr[1] * fr[3];
			r[1] += sqrt(fr[2] * fr[2] + fr[3] * fr[3]);
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
				//std::cerr << "FBB " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 1] << "\n";
			}
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
		if (fabs(fsmm[14] - fs[14]) < 1e-5)std::cerr << "Min 14 " << fsmm[14] << " " << j << " | " << fsmm[14] - fs[14] << "\n";
		if (fabs(fsmm[14 + varNum] - fs[14]) < 1e-5)std::cerr << "Max 14 " << fsmm[14 + varNum] << " " << j << " | " << fsmm[14 + varNum] - fs[14] << "\n";

	}/**/

	/*if (nImp[1] < Padd.ImpulsSteps)
	{
		std::wcerr << "ERR! " << nImp[1] << " " << Padd.ImpulsSteps << " " << nImp[1] * Po.dt << "\n";
		std::cin.get();
	}/**/
	std::cerr << "Fmm " << Padd.V << " Min " << stepBmin << "(" << fsmm[14] << ")" << " Max " << stepBmax << "(" << fsmm[14 + varNum] << ")" << "\n";
	std::cerr << "Fmm " << Padd.V << " Min " << stepBmin * Po.dt / 0.016235485 << "(" << fsmm[14] << ")" << " Max " << stepBmax * Po.dt / 0.016235485 << "(" << fsmm[14 + varNum] << ")" << "\n";
	//std::cin.get();
	for (m = 0; m < 2 * varNum; ++m)
	{
		std::cerr << fsmm[m] << " ";
		if (m == 1) std::cerr << "| ";
		else if (m == 3) std::cerr << "| ";
		else if (m == 7) std::cerr << "| ";
		else if (m == 11) std::cerr << "| ";
		else if (m == 15) std::cerr << "| ";
		else if (m == varNum - 1) std::cerr << "\n";
		else if (m == varNum + 1) std::cerr << "| ";
		else if (m == varNum + 3) std::cerr << "| ";
		else if (m == varNum + 7) std::cerr << "| ";
		else if (m == varNum + 11) std::cerr << "| ";
		else if (m == varNum + 15) std::cerr << "| ";
	}
	std::cerr << "\nFmmi " << Padd.V << "\n";
	for (m = 0; m < 8; ++m)
	{
		std::cerr << fsimm[m] << " ";
		if (m == 1) std::cerr << "| ";
		else if (m == 3) std::cerr << "\n";
		else if (m == 5) std::cerr << "| ";
	}
	std::cerr << "\n";
	//std::cerr<<"FF "<<Avi[0]

	//std::cerr << "Fmm " << Padd.V << " " << fAmm[0] << " " << fAmm[1] << " " << fBmm[0] << " " << fBmm[1] << "\n";
	for (m = 0; m < 2 * varNum; ++m)
	{
		Padd.h_Fmm[ResultFmmNum * Padd.StepV + m] = fsmm[m];
	}
	for (m = 0; m < varNum; ++m)
	{
		Padd.h_Fmm[ResultFmmNum * Padd.StepV + 2 * varNum + m] = Impuls[m];
	}
	std::cerr << "Fi " << ResultFmmNum * Padd.StepV + 3 * varNum << "\n";
	for (m = 0; m < 8; ++m)
	{
		Padd.h_Fmm[ResultFmmNum * Padd.StepV + 3 * varNum + m] = fsimm[m];
	}
	for (m = 0; m < 4; ++m)
	{
		Padd.h_Fmm[ResultFmmNum * Padd.StepV + 3 * varNum + 8 + m] = Impulsi[m];
	}
#ifndef pre_FreeCell
	std::cerr << "RESULT " << Padd.V << " " << fsmm[varNum + 12] << " " << fsmm[14] << " | " << -fsmm[14] / fsmm[varNum + 12] << "\n\n";
#endif // !pre_FreeCell
#ifdef pre_FreeCell
	std::cerr << "RESULT " << Padd.V << " " << fsmm[12] << " " << fsmm[14] << " " << fsmm[varNum + 12] << " " << fsmm[varNum + 14] << " | " << fsmm[14] / fsmm[12] << "\n\n";
#endif // pre_FreeCell
	
	//std::cin.get();

}

void calculate_Faver2(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po)
{
	int i, j, k, m;
	const unsigned int varNum = 18;
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1], nImp[2] = { 0, 0 }, nImpi[2] = { 0,0 };
	double fs[varNum], fr[4], r[2], _1d_V, R0[3] = { 0.03987475, Padd.Eps0 * 0.03987475, 0.03987475 }, _1d_n, Av[2 * varNum], Avi[8], sumfi[2] = { 0,0 };
	//double fAmm[4] = { 1e30, -1e30, 1e30, -1e30 }, fBmm[4] = { 1e30, -1e30, 1e30, -1e30 }, intfA[2] = { 0,0 }, intfB[2] = { 0,0 }, 
	//	AvA[4] = { 0, 0, 0, 0 }, AvB[4] = { 0, 0, 0, 0 }, _1d_nA, _1d_nB, _1d_n, ImpulsA = 0, ImpulsB = 0, Avi[8] = { 0,0,0,0,0,0,0,0 }, f, _1d_FAB_aver[4], Impulsi[4] = { 0,0,0,0 };
	bool impA = true, impB = true, impBStart = false, impBiStart = false;

	//_1d_FAB_aver[0] = 1.0 / Padd.FAB_aver[0]; _1d_FAB_aver[1] = 1.0 / Padd.FAB_aver[1];
	//_1d_FAB_aver[2] = 1.0 / Padd.FAB_aver[2]; _1d_FAB_aver[3] = 1.0 / Padd.FAB_aver[3];
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	//std::cerr << "Faver " << Padd.FSAB_aver[0] << " " << Padd.FSAB_aver[1] << " " << Padd.FSAB_aver[2] << " " << Padd.FSAB_aver[3] << "\n";
#ifdef pre_ConnectCellSurface
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
	std::cerr << "Sumfi " << sumfi[0] << " " << sumfi[1] << "\n";
#endif // pre_ConnectCellSurface
	//std::cerr << "cFaver2 " << Padd.time << "\n";
	memset(Av, 0, 2 * varNum * sizeof(double));
	for (j = 0; j < Padd.time; ++j)
	{
		//std::cerr << "AAA! 0\n";
		memset(fs, 0, varNum * sizeof(double));
		r[0] = 0; r[1] = 0;
		for (i = 0; i < P.NBPT[0] - 1; ++i)
		{
			k = P.h_BP[i];
			//std::cerr << "AA " << i<<" "<< P.NBPT[0]<<" "<< P.h_BP[i] << "\n";
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
			//std::cerr << "A " << k << "\n";
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
				//fs[12] += fr[0];
				//fs[13] += fr[1];
#endif // pre_FreeCell		
				//std::cerr << "FBA " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 2] << "\n";
			}
#endif // !pre_OneNodeForceA	
			//if (i == P.iBP[0] && j%1000==0){ std::cerr << "A_FA " << i << " " << fr[0] << " " << fr[1] << "\n"; }
			//std::cerr << "A " << i << " " << ResultFRNum * j * npb2 << " | " << fr[0] << " " << fr[1] << " " << fr[2] << " " << fr[3] << " " << sqrt(fr[2] * fr[2] + fr[3] * fr[3]) << "\n";
			//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
			//++nA;				
		}
		//std::cerr << "1 " << r[0] << " " << P.NBPT[0] << "\n";
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
		//std::cerr << "AAA! 1\n";
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
				//std::cerr << "fs1415 " << fr[0] << " " << fr[1] << " | " << P.h_BPDfi[i] << " | " << fr[2] << " " << fr[3] << "\n";
#endif // pre_FreeCell
			}
#endif // !pre_OneNodeForceB
			//std::cerr << "FBB " << i << " " << fr[0] << " " << fr[1] << " | " << P0.h_RU0[k] << " " << P0.h_RU0[P.N - 1] << "\n";

		//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
		//++nA;				
		}
		//std::cerr << "2 " << r[1] << " " << P.NBPT[1] << "\n";
		r[1] /= double(P.NBPT[1] - 1);
		_1d_V = 1.0 / (MC_pi * r[1] * r[1]);
		fs[8] *= _1d_V;		fs[9] *= _1d_V;
		fs[10] *= _1d_V;	fs[11] *= _1d_V;
		fs[17] = 0.5 * (fs[8] + fs[11]);

#ifndef pre_OneNodeForceB			
#ifdef pre_FreeCell
		fs[14] *= sumfi[1];
		fs[15] *= sumfi[1];
#endif // pre_FreeCell
#endif // !pre_OneNodeForceB
		//if (j > 10000){std::cerr << "AAA! " << j << " " << fs[14] << "\n"; std::cin.get();}
		//if (j % 1000 == 0) { std::cerr << "AU " << j << " " << fs[12] << " " << fs[13] << " | " << fs[14] << " " << fs[15] << "\n"; }

		for (m = 0; m < varNum; ++m)
		{
			Av[m] += fs[m];
			Av[m + varNum] += (fs[m] * fs[m]);
		}

		fr[0] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[0]];
		fr[1] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[0]];
		fr[2] = Padd.h_FResult[ResultFRNum * j * npb2 + P.iBP[1]];
		fr[3] = Padd.h_FResult[(ResultFRNum * j + 1) * npb2 + P.iBP[1]];
		for (m = 0; m < 4; ++m)
		{
			Avi[m] += fr[m];
			Avi[m + 4] += (fr[m] * fr[m]);
		}
		//std::cerr << "Fmm step " << j << " " << fs[0] << " " << fs[1] << " " << fs[2] << " " << fs[3] << " | " << fs[4] << " " << fs[7] << " " << fs[8] << " " << fs[11] << "\n";
		//std::cin.get();
	}
	//std::cerr << "TEST " << Av[0] << " " << Av[5] << " " << Av[6] << " " << Av[7] << "\n";
	//std::cerr << "TEST " << Av[0 + varNum] << " " << Av[5 + varNum] << " " << Av[6 + varNum] << " " << Av[7 + varNum] << "\n";
	_1d_n = 1.0 / double(Padd.time);
	for (m = 0; m < 2 * varNum; ++m)
		Av[m] *= _1d_n;

	for (m = 0; m < 8; ++m)
		Avi[m] *= _1d_n;

	double Disp;
	for (m = 0; m < varNum; ++m)
		Padd.FSAB_aver[m] = Av[m];
	for (m = varNum; m < 2 * varNum; ++m)
	{
		Disp = Av[m] - Av[m - varNum] * Av[m - varNum];
		//std::cerr << "AAA " << Av[m] << " " << Av[m - varNum] << " " << sqrt(fabs(Disp)) / fabs(Av[m - varNum]) << " | " << Disp << " " << sqrt(Disp) << "\n\n";
		Padd.FSAB_aver[m] = (sqrt(fabs(Disp)) / fabs(Av[m - varNum]) > 1e-10) ? sqrt(fabs(Disp)) : 1e-8 * fabs(Av[m - varNum]);
		if (fabs(Padd.FSAB_aver[m]) < 1e-14)Padd.FSAB_aver[m] = 1e-9;
	}
	//std::cin.get();
	for (m = 0; m < 4; ++m)
		Padd.FABi_aver[m] = Avi[m];
	for (m = 4; m < 8; ++m)
	{
		Disp = Avi[m] - Avi[m - 4] * Avi[m - 4];
		Padd.FABi_aver[m] = (sqrt(fabs(Disp)) / fabs(Av[m - 4]) > 1e-10) ? sqrt(fabs(Disp)) : 1e-8 * fabs(Avi[m - 4]);
		if (fabs(Padd.FABi_aver[m]) < 1e-14)Padd.FABi_aver[m] = 1e-9;
	}


	//std::cerr << "TEST " << Av[0] << " " << Av[5] << " " << Av[6] << " " << Av[7] << "\n";
	//std::cerr << "TEST " << Av[0 + varNum] << " " << Av[5 + varNum] << " " << Av[6 + varNum] << " " << Av[7 + varNum] << "\n";
	//std::cerr << "TEST " << Av[0 + varNum] - Av[0] * Av[0] << " " << Av[5 + varNum] - Av[5] * Av[5] << " " << Av[6 + varNum] - Av[6] * Av[6] << " " << Av[7 + varNum] - Av[7] * Av[7] << "\n";
#ifndef pre_FreeCell
	std::cerr << "TEST " << Av[14] << " " << Av[14 + varNum] << " " << Padd.FSAB_aver[14] << " " << Padd.FSAB_aver[14 + varNum] << "\n";
#endif // !pre_FreeCell
#ifdef pre_FreeCell
	std::cerr << "TEST " << Av[14] << " " << Av[14 + varNum] << " " << Padd.FSAB_aver[14] << " " << Padd.FSAB_aver[14 + varNum] << "\n";
#endif // pre_FreeCell


	
	//std::cerr << "TEST " << 1.0/sumfi[0] << " " << 1.0/sumfi[1] << " " << Padd.FSAB_aver[14] << " " << Padd.FSAB_aver[14 + varNum] << "\n";

	std::cerr << "Aver " << Padd.V << "\n";
	std::cerr.precision(10);
	for (m = 0; m < varNum; ++m)
	{
		std::cerr << Av[m] << " ";
		if (m == 1) std::cerr << "| ";
		else if (m == 3) std::cerr << "| ";
		else if (m == 7) std::cerr << "| ";
		else if (m == 11) std::cerr << "| ";
		else if (m == 15) std::cerr << "| ";
		else if (m == varNum - 1) std::cerr << "\n";
		else if (m == varNum + 1) std::cerr << "| ";
		else if (m == varNum + 3) std::cerr << "| ";
		else if (m == varNum + 7) std::cerr << "| ";
		else if (m == varNum + 11) std::cerr << "| ";
		else if (m == varNum + 15) std::cerr << "| ";
	}
	for (m = 0; m < varNum; ++m)
	{
		std::cerr << sqrt((Av[m + varNum] - Av[m] * Av[m])) << " ";
		if (m == 1) std::cerr << "| ";
		else if (m == 3) std::cerr << "| ";
		else if (m == 7) std::cerr << "| ";
		else if (m == varNum - 1) std::cerr << "\n";
	}
	std::cerr << "Averi " << Avi[0] << " " << Avi[1] << " " << Avi[2] << " " << Avi[3] << "\n";
	//std::cin.get();
}

/*0.5
AverA 0.01 159.73 8.18073 0.254245 0.000667182
AverB 0.01 - 81.5424 3.8631 0.0662594 0.000148842/**/