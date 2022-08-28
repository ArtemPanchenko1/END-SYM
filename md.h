#pragma once
#include <cuda_runtime.h>
#include "md_data_types.h"
#include "md_phys_constants.h"

const unsigned int SMEMDIM = 1024;
const double ReadCoordinatesCoefficient = 1e-3/ length_const;
const double RCC = ReadCoordinatesCoefficient;
const double RCC2 = ReadCoordinatesCoefficient * ReadCoordinatesCoefficient;
//const float rfiber_const = 1e-7 / length_const;
//const float rowater_const = 1e+3 / density_const;
#define pre_gammavis 1.7811f;
//#define IMatrixSize 7
//#define StepsToGPU 1000
//#define StepsToHost 10000
#define StepsMax 50000000
#define ResultFRSave 1
#define def_a_aver 2e-3 * ReadCoordinatesCoefficient
#define pre_RatioToCell 0.2
//#define def_a_aver 1e-2
#define pre_logveloscale
//#define pre_nonlinearC
//#define pre_XVfb
//#define pre_YVfb
//#define pre_FreeEnd
#define TwoDirectionMove
//#define pre_MoveCell
//#define pre_RotateCell
//#define pre_Relaxation
//#define pre_FireRelaxation
//#define pre_Viscocity
//#define pre_CylinderViscocity
//#define pre_CylinderViscocity1
//#define pre_CylinderViscocity2
//#define pre_CylinderViscocity3
//#define pre_CylinderViscocityShapovalov
//#define pre_CylinderViscocityLindstrom
//#define pre_CylinderDragFluidResistance
//#define pre_SaveLammps
//#define pre_SaveLammpsEnergy
//#define pre_SaveLammpsPoint
//#define pre_SaveLammpsU
//#define pre_SaveLammpsUx
#define pre_SineImpuls
//#define pre_ReleaseHalfSine
//#define pre_SquareImpuls
//#define pre_OneNodeContract
//#define pre_OneNodeContractRelax
//#define pre_OneNodeForceA
//#define pre_FreeCell
#define pre_deleteClose
//#define pre_ConnectCellSurface
//#define pre_FreeCellHalf
//#define pre_SaveEnergyData
//#define pre_LoadEnergyData
//#define pre_SaveEnergyDataRelax
//#define pre_OnlyOneCell
//#define pre_OneCellEFdistribution
//#define pre_OneCellEdistribution
//#define pre_OneCellFdistribution
//#define pre_OneCellMaterialdistribution

//#define pre_SaveFR
//#define pre_CalcFullEnergy
//#define pre_CalcFullKEnergy
#define pre_AlignCell 
#define pre_ReadNodes3
//#define pre_GetNetworkParameters
#ifdef pre_SaveLammpsUx
#ifdef pre_SaveLammpsEnergy
No simmultaniusly
#endif // pre_SaveLammpsEnergy
#endif // pre_SaveLammpsUx


const unsigned int StepsEntire = 1;

const unsigned int StreamsNumber  = 2;
const unsigned int RepeatEnsemble = 10000;
void createLattice(p_data& P, p0_data& P0, l_data& L, pAdd_data& Padd, potential_data& Po);
void renewLattice(p_data& P, p0_data& P0);
void deleteArrays(p_data& P, p0_data& P0, pAdd_data& Padd, pNet_data& Pnet, EnergyDist_data& ED);
void initArrays(p_data& P, p0_data& P0, pAdd_data& Padd, pNet_data& Pnet, EnergyDist_data& ED);

//void setSpotTemperatureNormal(p_data& P, p0_data& P0, param_data& Pr, potential_data& Po);
//void setSpotTemperatureNormal2(p_data& P, p0_data& P0, param_data& Pr, potential_data& Po);
void setSpotTemperatureNormal(p_data& P, p0_data& P0, param_data& Pr, potential_data& Po, pAdd_data& Padd);
void setSinTemperatureNormal(p_data& P, p0_data& P0, param_data& Pr, potential_data& Po, pAdd_data& Padd, l_data& L);

void createIMatrix(pAdd_data& Padd);
void calculateForcesIM(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L);
void calculateIncrements(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd);

void calculateGPUSteps(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet);
void calculateGPUStepsContractRelax(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet);
void calculateGPUStepsAverage(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet);
void calculateGPUStepsContractRelaxFIRE(p_data& P, p0_data& P0, potential_data& Po, pAdd_data& Padd, l_data& L, pNet_data& Pnet, firerelax_data& Fire);
void SaveTXTGraphsFmm(pAdd_data& Padd, potential_data& Po, pNet_data& Pnet);
void SaveTXTGraphsFR(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po, int savestep = 1);
void SaveTXTParticles(p_data& P, p0_data& P0, potential_data& Po, pNet_data& Pnet, char* Name);
void SaveLammpsDATAParticles(p_data& P, p0_data& P0, potential_data& Po, pNet_data& Pnet, pAdd_data& Padd, char* Name);
void ReadLattice(pNet_data& Pnet, potential_data& Po, char* filename1, char* filename2);
void SplitNet(p_data& P, p0_data& P0, l_data& L, pAdd_data& Padd, potential_data& Po, pNet_data& Pnet);

void calculate_Fminmax(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po);
void calculate_Fminmax2(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po);
void calculate_Faver(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po);
void calculate_Faver2(pAdd_data& Padd, p_data& P, p0_data& P0, potential_data& Po);


void SaveAllData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, int dn);
void LoadAllData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, int dn);
void reloadLattice(p_data& P, p0_data& P0);
void SaveEnergyDataStart(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, char* Name, unsigned int step);
void SaveEnergyDataStep(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, char* Name, unsigned int step);
void LoadEnergyDataStep(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED, char* Name);
void CalcEnergyDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED, unsigned int step);
void SumEnergyDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED);
void SaveEnergyTineDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED);
void SaveEnergyRTineDistribution(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po, EnergyDist_data& ED);

void EFInitData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po);
void EFMinMaxData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po);
void SaveTXTGraphsEFmm(pAdd_data& Padd, potential_data& Po, pNet_data& Pnet);
void EFAverageData(p_data& P, p0_data& P0, pNet_data& Pnet, pAdd_data& Padd, potential_data& Po);
__global__ void d_distrubuteBoundToCircle(const int* __restrict__ In, const float* __restrict__ U, const float* __restrict__ RU0, const float* __restrict__ EFbound, const float* __restrict__ EFbound0, float* CEF, const unsigned int n, const unsigned int ni, const unsigned int nT, const double X0, const double Y0, const double _1d_DR);
__global__ void d_calculateForcesI_EF(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ F, float* __restrict__ EFbound, const unsigned int n, const unsigned int ni, const float P_c);


__global__ void d_createLattice(float* __restrict__ RU0, float* __restrict__ U0, int* __restrict__ Ri, const int n, const float L0X, const float Eps0);

__device__ float dd_nonlinearC(const double rmada, const double c);

__global__ void d_calculateForcesI(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float P_c);
__global__ void d_calculateForcesIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Fbound, const unsigned int n, const unsigned int ni, const float P_c);
__global__ void d_calculateEnergyIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Ebound, const unsigned int n, const unsigned int ni, const float P_c);
__global__ void d_calculateUIBound(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, float* __restrict__ Ubound, const unsigned int n, const unsigned int ni, const float P_c);

__global__ void d_calculateBorders(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1);
__global__ void d_calculateBordersVis(const float* __restrict__ _1d_Mass, const float* __restrict__ VisR, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1);
__global__ void d_calculateBorders(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1, const int ibp);
__global__ void d_calculateBordersFix(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1);
__global__ void d_calculateBordersMove(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, const float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1);
__global__ void d_calculateBordersMove(const float* __restrict__ _1d_Mass, const int* __restrict__ BP, const float* __restrict__ BPR, float* __restrict__ F, float* __restrict__ V, const float* __restrict__ U, float* __restrict__ FR,
	const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step, const float P_vis, const float V1, const int ibp);

__global__ void d_calculateIncrements(const float* __restrict__ _1d_Mass, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vism);
__global__ void d_calculateIncrementsVis(const float* __restrict__ _1d_Mass, const float* __restrict__ VisR, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vis);
//__global__ void d_calculateIncrementsMove(const float* __restrict__ _1d_Mass, const float* __restrict__ F, const float* __restrict__ FB, float* __restrict__ V, float* __restrict__ U,
//	const unsigned int n, const float P_dt, const float P_vis, const unsigned int Step);

__global__ void d_calculateKineticEnergy(const float* __restrict__ VU, const float* __restrict__ VV,
	float* __restrict__ aEk, unsigned int offset, unsigned int n);

__global__ void d_calculateKineticEnergy_precision(const float* __restrict__ VU, const float* __restrict__ VV,
	const float* __restrict__ FU, const float* __restrict__ FV, float* __restrict__ aEk, unsigned int offset,
	unsigned int n, const float P_dtm);


__global__ void d_PrintFR(const float* __restrict__ FR, const unsigned int n, const unsigned int nbp, const unsigned int nbpt1, const unsigned int nbpt2, const unsigned int nbpt3, const unsigned int Step);


__global__ void d_getEnergyEntire(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ _1d_Mass, const float* __restrict__ U, const float* __restrict__ V, float* Esum, const unsigned int n, const unsigned int ni, const float P_c);
__device__ float d_calculateKEnergyi(const float* __restrict__ V, const unsigned int i, const unsigned int n, const float m);
__device__ float d_calculatePEnergyIk(const int* __restrict__ In, const float* __restrict__ Ir0, const float* __restrict__ U, const unsigned int n, const unsigned int ni, const int ks, const int kmax, const float P_c);

__global__ void d_calculateIncrementsFIRE(const float* __restrict__ _1d_Mass, const float* __restrict__ F, float* __restrict__ V, float* __restrict__ U,
	const unsigned int n, const float P_dt, const float P_vis, const float F_alpha);
__global__ void d_calculateDecrementsHalfStepFIRE(const float* __restrict__ V, float* __restrict__ U, const unsigned int n, const float P_dt);
__global__ void d_FdotVEntire(const float* __restrict__ V, const float* __restrict__ F, float* FdotV, const unsigned int n);

__global__ void d_calculateVIscosForces(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid);
__global__ void d_calculateVIscosForces2(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_1d_hfreefiber);
__global__ void d_calculateVIscosForces3(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_1d_hfreefiber);
__global__ void d_calculateVIscosForcesShapovalov(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_CShfreefiber);
__global__ void d_calculateVIscosForcesLindstrom(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_rfiber, const float Po_1d_rfiber, const float Po_roliquid);
