#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>

void calculate_Fminmax(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po)
{
	int i, j, k;
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1], nA = 0, nB = 0, ni[2] = { 0,0 };
	double fA[2], fB[2], fAmm[4] = { 1e30, -1e30, 1e30, -1e30 }, fBmm[4] = { 1e30, -1e30, 1e30, -1e30 }, intfA[2] = { 0,0 }, intfB[2] = { 0,0 }, 
		AvA[4] = { 0, 0, 0, 0 }, AvB[4] = { 0, 0, 0, 0 }, _1d_nA, _1d_nB, _1d_n, ImpulsA = 0, ImpulsB = 0, Avi[8] = { 0,0,0,0,0,0,0,0 }, f, _1d_FAB_aver[4], Impulsi[4] = { 0,0,0,0 };
	bool impA = true, impB = true, impBStart = false, impBiStart = false;

	_1d_FAB_aver[0] = 1.0 / Padd.FSAB_aver[0]; _1d_FAB_aver[1] = 1.0 / Padd.FSAB_aver[1];
	_1d_FAB_aver[2] = 1.0 / Padd.FSAB_aver[2]; _1d_FAB_aver[3] = 1.0 / Padd.FSAB_aver[3];
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	std::cerr << "Faver " << Padd.FSAB_aver[0] << " " << Padd.FSAB_aver[1] << " " << Padd.FSAB_aver[2] << " " << Padd.FSAB_aver[3] << "\n";
	for (j = 0; j < Padd.time; ++j)
	{
		fA[0] = 0; fA[1] = 0;
		for (i = 0; i < P.NBPT[0]-1; ++i)
		{
			k = P.h_BP[i];
			if (P0.h_RU0[k] + P0.h_RU0[P.N-2] > 0)
			{
				fA[0] += Padd.h_FResult[2 * j * npb2 + i];
				fA[1] += Padd.h_FResult[(2 * j + 1) * npb2 + i];
				//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
				//++nA;
			}
			
		}
		AvA[0] += fA[0];
		AvA[1] += fA[1];
		AvA[2] += fA[0] * fA[0];
		AvA[3] += fA[1] * fA[1];
		//f = Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		Avi[0] += Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		Avi[1] += Padd.h_FResult[2 * j * npb2 + P.iBP[0]] * Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		//std::cerr << "f " << f << " " << f * f <<" "<< Avi[1] << "\n";
		//if (fA[0] > Padd.FAB_aver[0] - 0.03 && impA)//Diff
		if(nA < Padd.ImpulsSteps)
		{
			ImpulsA += fabs(fA[0] - Padd.FSAB_aver[0]) * Po.dt;
			++nA;
			Impulsi[0] += fabs(Padd.h_FResult[2 * j * npb2 + P.iBP[0]] - Padd.FABi_aver[0]) * Po.dt;
			++ni[0];
		}
		/*if (fA[0] > -0.01 && impA)//0%
		{
			ImpulsA += fA[0] * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 2.56085 - 0.01 && impA)//1%
		{
			ImpulsA += (fA[0] - 2.56085) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 27.2725 - 0.01 && impA)//10%
		{
			ImpulsA += (fA[0] - 27.2725) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 57.5282 - 0.01 && impA)//20%
		{
			ImpulsA += (fA[0] - 57.5282) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 89.8472 - 0.01 && impA)//30%
		{
			ImpulsA += (fA[0] - 89.8472) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 123.869 - 0.02 && impA)//40%
		{
			ImpulsA += (fA[0] - 123.869) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 159.73 - 0.02 && impA)//50%
		{
			ImpulsA += (fA[0] - 159.73) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 262.049 - 0.02 && impA)//2x50%
		{
			ImpulsA += (fA[0] - 262.049) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 175.587 - 0.01 && impA)//50% a0.005 Cb 
		{

			ImpulsA += (fA[0] - 175.587) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 159.72 -0.01 && impA)//50% 159.836 159.846
		{
			
			ImpulsA += (fA[0] - 159.72) * Po.dt;
			++nA;
		}/**/
		/*if (fA[0] > 149.985 - 0.01 && impA)//50% 0.002
		{

			ImpulsA += (fA[0] - 149.985) * Po.dt;
			++nA;
		}/**/
		/*else
		{
			impA = false;
		}/**/
		fB[0] = 0; fB[1] = 0;
		for (i = P.NBPT[0]; i < npb2-1; ++i)
		{
			k = P.h_BP[i];
			if (P0.h_RU0[k] - 0.121052175 < 0)
			{
				fB[0] += Padd.h_FResult[2 * j * npb2 + i];
				fB[1] += Padd.h_FResult[(2 * j + 1) * npb2 + i];
				//std::cerr << "FBB " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
				//++nB;
			}
		}
		AvB[0] += fB[0];
		AvB[1] += fB[1];
		AvB[2] += fB[0] * fB[0];
		AvB[3] += fB[1] * fB[1];
		Avi[4] += Padd.h_FResult[2 * j * npb2 + P.iBP[1]];
		Avi[5] += Padd.h_FResult[2 * j * npb2 + P.iBP[1]] * Padd.h_FResult[2 * j * npb2 + P.iBP[1]];
		
		//if (fB[0] < Padd.FAB_aver[2] + 0.02 && impB)//Diff
		if (!impBStart && fB[0] - Padd.FSAB_aver[2] < -1e-2)
		{
			impBStart = true;
			std::cerr << "StartImpulsB " << j << " " << j * Po.dt << "\n";
		}
		if (impBStart && nB < Padd.ImpulsSteps)//Diff
		{
			ImpulsB += fabs(fB[0] - Padd.FSAB_aver[2]) * Po.dt;
			++nB;
		}

		if (!impBiStart && Padd.h_FResult[2 * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2] < -1e-3)
		{
			impBiStart = true;
			std::cerr << "StartImpulsBi " << j << " " << j * Po.dt << "\n";
		}
		//std::cerr << "FBi " << Padd.h_FResult[2 * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2] << "\n";
		if (impBiStart && nB < Padd.ImpulsSteps)//Diff
		{
			Impulsi[1] += fabs(Padd.h_FResult[2 * j * npb2 + P.iBP[1]] - Padd.FABi_aver[2]) * Po.dt;
			++ni[1];
		}
		
		/*if (fB[0] < 0.01 && impB)//0%
		{
			ImpulsB += fB[0] * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -1.17717 + 0.02 && impB)//1%
		{
			ImpulsB += (fB[0] + 1.17717) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -12.9065 + 0.02 && impB)//10%
		{
			ImpulsB += (fB[0] + 12.9065) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -27.8541 + 0.02 && impB)//20%
		{
			ImpulsB += (fB[0] + 27.8541) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -44.3987 + 0.02 && impB)//30%
		{
			ImpulsB += (fB[0] + 44.3987) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -62.291 + 0.02 && impB)//40%
		{
			ImpulsB += (fB[0] + 62.291) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -81.5419 + 0.02 && impB)//50%
		{
			ImpulsB += (fB[0] + 81.5419) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -289.796 + 0.04 && impB)//2x50%
		{
			ImpulsB += (fB[0] + 289.796) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -97.236 + 0.02 && impB)//50% -81.4975 81.4298
		{
			ImpulsB += (fB[0] + 97.236) * Po.dt;
			++nB;
		}/**/
		/*if (fB[0] < -81.5513 + 0.03 && impB)//50% -81.4975 81.4298
		{
			ImpulsB += (fB[0] + 81.5513) * Po.dt;
			++nB;
		}/**/		
		/*else
		{
			impB = false;
		}/**/
		//std::cerr << "F " << fA[0] << " " << fA[1] << " | " << fB[0] << " " << fB[1] << "\n";
		if (fAmm[0] > fA[0]) fAmm[0] = fA[0];
		if (fAmm[1] < fA[0]) fAmm[1] = fA[0];
		if (fBmm[0] > fB[0]) fBmm[0] = fB[0];
		if (fBmm[1] < fB[0]) fBmm[1] = fB[0];

		if (fAmm[2] > fA[1]) fAmm[2] = fA[1];
		if (fAmm[3] < fA[1]) fAmm[3] = fA[1];
		if (fBmm[2] > fB[1]) fBmm[2] = fB[1];
		if (fBmm[3] < fB[1]) fBmm[3] = fB[1];

		intfA[0] += fabs(fA[0]) * Po.dt;
		intfA[1] += fabs(fA[1]) * Po.dt;

		intfB[0] += fabs(fB[0]) * Po.dt;
		intfB[1] += fabs(fB[1]) * Po.dt;
		//std::cin.get();
	}	
	_1d_n = 1.0 / double(Padd.time);
	for (i = 0; i < 4; ++i)
	{
		AvA[i] *= _1d_n;
		AvB[i] *= _1d_n;
	}
	for (i = 0; i < 8; ++i)	
		Avi[i] *= _1d_n;
	//std::cerr<<"FF "<<Avi[0]
	
	std::cerr << "Fmm " << Padd.V << " " << fAmm[0] << " " << fAmm[1] << " " << fBmm[0] << " " << fBmm[1] << "\n";
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 0] = fAmm[0];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 1] = fAmm[1];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 2] = fBmm[0];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 3] = fBmm[1];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 4] = fAmm[2];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 5] = fAmm[3];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 6] = fBmm[2];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 7] = fBmm[3];

	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 8] = intfA[0];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 9] = intfB[0];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 10] = intfA[1];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 11] = intfB[1];

	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 12] = ImpulsA;
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 13] = ImpulsB;/**/
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 14] = Impulsi[0];
	Padd.h_Fmm[ResultFmmNum * Padd.StepV + 15] = Impulsi[1];/**/
	std::cerr << "AverA " << Padd.V << " " << AvA[0] << " " << AvA[1] << " " << AvA[2] - AvA[0] * AvA[0] << " " << AvA[3] - AvA[1] * AvA[1] << "\n";
	//std::cerr << "AverA " << Padd.V << " " << AvA[0] << " " << AvA[1] << " " << AvA[2] << " " << AvA[3]<< "\n";
	std::cerr << "AverB " << Padd.V << " " << AvB[0] << " " << AvB[1] << " " << AvB[2] - AvB[0] * AvB[0] << " " << AvB[3] - AvB[1] * AvB[1] << "\n";

	std::cerr << "AveriA " << Padd.V << " " << Avi[0] << " " << Avi[1] - Avi[0] * Avi[0] << "  | " << Avi[4] << " " << Avi[5] - Avi[4] * Avi[4] << "\n";
	//std::cerr << "AverB " << Padd.V << " " << AvB[0] << " " << AvB[1] << " " << AvB[2] << " " << AvB[3] << "\n";
	//Padd.h_Fmm[20 * Padd.StepV + 12] = intfB[1];
	std::cerr << std::setprecision(8) << "Impuls " << nA << " " << nB << " " << ImpulsA << " " << ImpulsB << " | " << ni[0] << " " << ni[1] << " " << Impulsi[0] << " " << Impulsi[1] << " | " << nA * Po.dt << " " << nB * Po.dt << "\n";
	//std::cin.get();

}

void calculate_Faver(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po)
{
	int i, j, k;
	unsigned int npb2 = P.NBPT[0] + P.NBPT[1], nA = 0, nB = 0;
	double fA[2], fB[2], AvA[4] = { 0, 0, 0, 0 }, AvB[4] = { 0, 0, 0, 0 }, _1d_nA, _1d_nB, _1d_n, Avi[8] = { 0,0,0,0,0,0,0,0 };
	bool impA = true, impB = true;
	//double ekx, eky, _1d_ekxmax, _1d_ekymax, _1d_ekmax, xc, yc, eks;
	std::cerr << "T " << Padd.time << "\n";
	for (j = 0; j < Padd.time; ++j)
	{
		fA[0] = 0; fA[1] = 0;
		for (i = 0; i < P.NBPT[0]; ++i)
		{
			k = P.h_BP[i];
			if (P0.h_RU0[k] + 0.121052175 > 0)
			{
				fA[0] += Padd.h_FResult[2 * j * npb2 + i];
				fA[1] += Padd.h_FResult[(2 * j + 1) * npb2 + i];
				//std::cerr << "FBA " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
				//++nA;
			}

		}
		AvA[0] += fA[0];
		AvA[1] += fA[1];
		AvA[2] += fA[0] * fA[0];
		AvA[3] += fA[1] * fA[1];
		//f = Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		Avi[0] += Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		Avi[1] += Padd.h_FResult[2 * j * npb2 + P.iBP[0]] * Padd.h_FResult[2 * j * npb2 + P.iBP[0]];
		
		fB[0] = 0; fB[1] = 0;
		for (i = P.NBPT[0]; i < npb2; ++i)
		{
			k = P.h_BP[i];
			if (P0.h_RU0[k] - 0.121052175 < 0)
			{
				fB[0] += Padd.h_FResult[2 * j * npb2 + i];
				fB[1] += Padd.h_FResult[(2 * j + 1) * npb2 + i];
				//std::cerr << "FBB " << i << " " << Padd.h_FResult[2 * j * npb2 + i] << " " << Padd.h_FResult[2 * (j + 1) * npb2 + i] << "\n";				
				//++nB;
			}
		}
		AvB[0] += fB[0];
		AvB[1] += fB[1];
		AvB[2] += fB[0] * fB[0];
		AvB[3] += fB[1] * fB[1];
		Avi[4] += Padd.h_FResult[2 * j * npb2 + P.iBP[1]];
		Avi[5] += Padd.h_FResult[2 * j * npb2 + P.iBP[1]] * Padd.h_FResult[2 * j * npb2 + P.iBP[1]];		
		
		//std::cin.get();
	}
	_1d_n = 1.0 / double(Padd.time);
	for (i = 0; i < 4; ++i)
	{
		AvA[i] *= _1d_n;
		AvB[i] *= _1d_n;
	}
	for (i = 0; i < 8; ++i)
		Avi[i] *= _1d_n;
	//std::cerr<<"FF "<<Avi[0]

	//std::cerr << "Fmm " << Padd.V << " " << fAmm[0] << " " << fAmm[1] << " " << fBmm[0] << " " << fBmm[1] << "\n";
	
	std::cerr << "AverA " << Padd.V << " " << AvA[0] << " " << AvA[1] << " " << AvA[2] - AvA[0] * AvA[0] << " " << AvA[3] - AvA[1] * AvA[1] << "\n";
	//std::cerr << "AverA " << Padd.V << " " << AvA[0] << " " << AvA[1] << " " << AvA[2] << " " << AvA[3]<< "\n";
	std::cerr << "AverB " << Padd.V << " " << AvB[0] << " " << AvB[1] << " " << AvB[2] - AvB[0] * AvB[0] << " " << AvB[3] - AvB[1] * AvB[1] << "\n";

	Padd.FSAB_aver[0] = AvA[0];
	Padd.FSAB_aver[1] = AvA[1];
	Padd.FSAB_aver[2] = AvB[0];
	Padd.FSAB_aver[3] = AvB[1];

	Padd.FABi_aver[0] = Avi[0];
	Padd.FABi_aver[1] = 0;
	Padd.FABi_aver[2] = Avi[4];
	Padd.FABi_aver[3] = 0;

	std::cerr << "Averi " << Padd.V << " " << Padd.FABi_aver[0] << " " << Padd.FABi_aver[2] << " " << Avi[1] - Avi[0] * Avi[0] << " " << Avi[5] - Avi[4] * Avi[4] << "\n";

	//std::cerr << "AveriA " << Padd.V << " " << Avi[0] << " " << Avi[1] - Avi[0] * Avi[0] << "  | " << Avi[4] << " " << Avi[5] - Avi[4] * Avi[4] << "\n";
	//std::cerr << "AverB " << Padd.V << " " << AvB[0] << " " << AvB[1] << " " << AvB[2] << " " << AvB[3] << "\n";
	//Padd.h_Fmm[20 * Padd.StepV + 12] = intfB[1];
	//std::cerr << std::setprecision(8) << "Impuls " << nA << " " << nB << " " << ImpulsA << " " << ImpulsB << " | " << nA * Po.dt << " " << nB * Po.dt << "\n";
	//std::cin.get();

}

/*0.5
AverA 0.01 159.73 8.18073 0.254245 0.000667182
AverB 0.01 - 81.5424 3.8631 0.0662594 0.000148842/**/