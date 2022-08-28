#include "md_data_types.h"
#include "md.h"

void initArrays(p_data& P, p0_data& P0, pAdd_data& Padd, pNet_data& Pnet, EnergyDist_data& ED)
{
	P.h_F = nullptr;
	P.h_V = nullptr;
	P.h_U = nullptr;	
	P.h_Ir0 = nullptr;
	P.h_BPR = nullptr;
	P.h_IM = nullptr;
	P.h_1d_IM = nullptr;

	P.d_F = nullptr;
	P.d_V = nullptr;
	P.d_U = nullptr;
	P.d_Ir0 = nullptr;
	P.d_BPR = nullptr;
	P.d_1d_IM = nullptr;

	P.h_In = nullptr;
	P.h_BP = nullptr;
	P.h_ShIn = nullptr;
	P.h_VisR = nullptr;

	P.d_In = nullptr;
	P.d_BP = nullptr;
	P.d_ShIn = nullptr;
	P.h_BPDfi = nullptr;
	P.d_VisR = nullptr;

	P0.h_RU0 = nullptr;
	P0.d_RU0 = nullptr;
	P0.d_U0 = nullptr;

	Padd.d_Fmm = nullptr;
	Padd.d_FResult = nullptr;
	Padd.h_V = nullptr;
	Padd.h_Fmm = nullptr;
	Padd.h_FResult = nullptr;
	Padd.d_Fbound = nullptr;
	Padd.h_Fbound = nullptr;
	Padd.h_LammpsAddParticles = nullptr;
	Padd.h_LammpsSumF = nullptr;
	Padd.h_Fbound0 = nullptr;	
	Padd.d_Ebound = nullptr;
	//Padd.h_Ebound0 = nullptr;
	Padd.h_Ebound = nullptr;
	Padd.h_Ebound0 = nullptr;
	Padd.h_Ek0 = nullptr;
	Padd.d_Esum = nullptr;
	Padd.h_Esum = nullptr;
	Padd.h_Ubound = nullptr;
	Padd.h_Ubound0 = nullptr;
	Padd.d_Ubound = nullptr;

	Pnet.h_S = nullptr;
	Pnet.h_Sc = nullptr;
		
	ED.h_Step = nullptr;
	ED.h_E = nullptr;
	ED.h_sE = nullptr;
	ED.h_Ebound0 = nullptr;
	ED.h_Ek0 = nullptr;

	Padd.EF.h_EFb = nullptr;
	Padd.EF.h_EFb0 = nullptr;
	Padd.EF.h_CEF = nullptr;

	Padd.EF.d_EFb = nullptr;
	Padd.EF.d_EFb0 = nullptr;
	Padd.EF.d_CEF = nullptr;
	Padd.EF.EFMinMax = nullptr;
	Padd.EF.CEFAverage = nullptr;
	Padd.EF.EFcell0 = nullptr;
	Padd.EF.EFcell = nullptr;
}
void deleteArrays(p_data& P, p0_data &P0, pAdd_data &Padd, pNet_data &Pnet, EnergyDist_data& ED)
{	
	free(P.h_F);
	free(P.h_V);
	free(P.h_U);
	free(P.h_Ir0);
	free(P.h_BPR);
	free(P.h_IM);
	free(P.h_1d_IM);
	free(P.h_VisR);
	P.h_F = nullptr;
	P.h_V = nullptr;
	P.h_U = nullptr;
	P.h_Ir0 = nullptr;
	P.h_BPR = nullptr;
	P.h_IM = nullptr;
	P.h_1d_IM = nullptr;
	P.h_VisR = nullptr;


	cudaFree(P.d_F); 
	cudaFree(P.d_V);
	cudaFree(P.d_U);
	cudaFree(P.d_Ir0);
	cudaFree(P.d_1d_IM);
	cudaFree(P.d_BPR);
	cudaFree(P.d_VisR);
	P.d_F = nullptr;
	P.d_V = nullptr;
	P.d_U = nullptr;
	P.d_Ir0 = nullptr;
	P.d_1d_IM = nullptr;
	P.d_BPR = nullptr;
	P.d_VisR = nullptr;

	free(P.h_In);
	free(P.h_BP);
	free(P.h_ShIn);
	P.h_In = nullptr;
	P.h_BP = nullptr;
	P.h_ShIn = nullptr;
	
	cudaFree(P.d_In);
	cudaFree(P.d_BP);
	cudaFree(P.d_ShIn);
	P.d_In = nullptr;
	P.d_BP = nullptr;
	P.d_ShIn = nullptr;
//#ifdef pre_ConnectCellSurface
	free(P.h_BPDfi);
	P.h_BPDfi = nullptr;
//#endif // pre_ConnectCellSurface

	
	free(P0.h_RU0);
	P0.h_RU0 = nullptr;

	cudaFree(P0.d_RU0);
	cudaFree(P0.d_U0);	
	P0.d_RU0 = nullptr;
	P0.d_U0 = nullptr;

	free(Padd.h_V);
	free(Padd.h_Fmm);
	free(Padd.h_FResult);
	
	Padd.h_V = nullptr;
	Padd.h_Fmm = nullptr;
	Padd.h_FResult = nullptr;
	
	cudaFree(Padd.d_Fmm);
	cudaFree(Padd.d_FResult);
	
	Padd.d_Fmm = nullptr;
	Padd.d_FResult = nullptr;
	
//#ifdef pre_SaveLammps
	free(Padd.h_Fbound);
	Padd.h_Fbound = nullptr;
	free(Padd.h_LammpsAddParticles);
	Padd.h_LammpsAddParticles = nullptr;
	free(Padd.h_LammpsSumF);
	Padd.h_LammpsSumF = nullptr;
	free(Padd.h_Fbound0);
	Padd.h_Fbound0 = nullptr;
	cudaFree(Padd.d_Fbound);
	Padd.d_Fbound = nullptr;
//#endif // pre_SaveLammps
	free(Padd.h_Ebound0);
	free(Padd.h_Ek0);
	Padd.h_Ebound0 = nullptr;
	Padd.h_Ek0 = nullptr;
	cudaFree(Padd.d_Esum);
	Padd.d_Esum = nullptr;
	free(Padd.h_Esum);	
	Padd.h_Esum = nullptr;

	free(Padd.h_Ubound);
	Padd.h_Ubound = nullptr;	
	free(Padd.h_Ubound0);
	Padd.h_Ubound0 = nullptr;
	cudaFree(Padd.d_Ubound);
	Padd.d_Ubound = nullptr;

	free(Pnet.h_S);
	free(Pnet.h_Sc);
	Pnet.h_S = nullptr;
	Pnet.h_Sc = nullptr;


	//free(Padd.h_Ebound0); 
	//Padd.h_Ebound0 = nullptr;
	free(Padd.h_Ebound);
	Padd.h_Ebound = nullptr;
	cudaFree(Padd.d_Ebound);
	Padd.d_Ebound = nullptr;


	free(ED.h_Step);
	free(ED.h_E);
	free(ED.h_sE);
	free(ED.h_Ebound0);
	free(ED.h_Ek0);
	ED.h_Step = nullptr;
	ED.h_E = nullptr;
	ED.h_sE = nullptr;
	ED.h_Ebound0 = nullptr;
	ED.h_Ek0 = nullptr;

	free(Padd.EF.h_EFb);
	free(Padd.EF.h_EFb0);
	free(Padd.EF.h_CEF);
	cudaFree(Padd.EF.d_EFb);
	cudaFree(Padd.EF.d_EFb0);
	cudaFree(Padd.EF.d_CEF);

	Padd.EF.h_EFb = nullptr;
	Padd.EF.h_EFb0 = nullptr;
	Padd.EF.h_CEF = nullptr;

	Padd.EF.d_EFb = nullptr;
	Padd.EF.d_EFb0 = nullptr;
	Padd.EF.d_CEF = nullptr;
	free(Padd.EF.EFMinMax);
	Padd.EF.EFMinMax = nullptr;
	free(Padd.EF.CEFAverage);
	Padd.EF.CEFAverage = nullptr;

	free(Padd.EF.EFcell0);
	free(Padd.EF.EFcell);
	Padd.EF.EFcell0 = nullptr;
	Padd.EF.EFcell = nullptr;

	//curandDestroyGenerator(Padd.gen);
}