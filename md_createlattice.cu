#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math_functions.h>
#include <iostream>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
//#include "lattice_constans.h"



void createLattice(p_data& P, p0_data& P0, l_data& L, pAdd_data &Padd, potential_data &Po)
{   
    /*
    //fprintf(stderr, "Start createLattice\n");
    P.h_F = nullptr;
    P.h_V = nullptr;
    P.h_U = nullptr;
    
    P.d_F = nullptr;
    P.d_V = nullptr;
    P.d_U = nullptr;

        
    P0.h_RU0 = nullptr;  
    P0.h_Ri = nullptr;

    P0.d_RU0 = nullptr;
    P0.d_U0 = nullptr;    
    P0.d_Ri = nullptr;


    Padd.h_Fmm = nullptr;
    Padd.h_V = nullptr;

    Padd.d_Fmm = nullptr;
    Padd.d_Fstx = nullptr;
    Padd.d_Fsty = nullptr;
    Padd.d_Fetx = nullptr;
    Padd.d_Fety = nullptr;
    Padd.d_cmax = nullptr;
    Padd.d_cmin = nullptr;

    //Padd.h_S = nullptr;         
    //Padd.h_Sc = nullptr;

    L.LV.x = Po.a;
    L.LV.y = 0;
    
    L.n.x = P.N;
    L.n.y = 1;

    L.PS.x = L.LV.x * float(L.n.x);
    L.PS.y = L.LV.y * float(L.n.y);
            
    fprintf(stderr, "Sample size L/a %f %f | N %i\n", L.PS.x / Po.a, L.PS.y / Po.a, P.N);
    P0.N = P.N;
    P0._1d_N = P._1d_N;

    Padd.bloks = ceil(P.N / (SMEMDIM)) + 1;

    /*int memmax = (8 * Padd.bloks4 * sizeof(float) + (11 + NGPUEk * 2) * 2 * P.N * sizeof(float)) / (1024 * 1024);
    if (memmax > 7500) 
    {
        fprintf(stderr, "Error Max Memory! %i\n", memmax); return;
    }/**/
    //P._1d_N = 1e-10;

    /*
    fprintf(stderr, "Start create Arrays %i %e | %u\n", P0.N, P._1d_N, Padd.bloks);
    P.h_U = (float*)malloc(2 * P.N * sizeof(float));
    P.h_V = (float*)malloc(2 * P.N * sizeof(float));
    P.h_F = (float*)malloc(2 * P.N * sizeof(float));     

    P0.h_Ri = (int*)malloc(2 * P.N * sizeof(int));
    P0.h_RU0 = (float*)malloc(2 * P0.N * sizeof(float));

    Padd.h_Fmm = (float*)malloc(8 * (Padd.StepV1 + Padd.StepV2) * sizeof(float));
    Padd.h_V = (float*)malloc( (Padd.StepV1 + Padd.StepV2) * sizeof(float));


#ifdef  pre_XVfb	
    Padd.MaxTimeStep = (2.0 * Padd.MaxShift / Padd.V01 + 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vl) / Po.dt + 1;
    Padd.MaxTimeStep = SMEMDIM * (Padd.MaxTimeStep / SMEMDIM + 1);
#endif //  pre_XVfb

#ifdef pre_YVfb
    Padd.MaxTimeStep = (2.0 * Padd.MaxShift / Padd.V01 + 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vt) / Po.dt + 1;
    Padd.MaxTimeStep = SMEMDIM * (Padd.MaxTimeStep / SMEMDIM + 1);    
#endif
    Padd.blokst = Padd.MaxTimeStep / (SMEMDIM);
    unsigned int memmax = (8 * (Padd.StepV1 + Padd.StepV2) * sizeof(float) + 2 * Padd.MaxTimeStep * sizeof(float) + 2 * Padd.blokst * sizeof(float)) / (1024 * 1024);
    fprintf(stderr, "Max memory GPU %u Mb | %u %e %e\n", memmax, Padd.MaxTimeStep, 1.1 * Po.a * (1.0 + Padd.Eps0) * P.N / Padd.Vt, Padd.Eps0);
    //std::cin.get();
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fmm, 8 * (Padd.StepV1 + Padd.StepV2) * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fstx, Padd.MaxTimeStep * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fsty, Padd.MaxTimeStep * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fetx, Padd.MaxTimeStep * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_Fety, Padd.MaxTimeStep * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_cmax, Padd.blokst * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&Padd.d_cmin, Padd.blokst * sizeof(float)));
    Padd.h_Fstx = (float*)malloc(Padd.MaxTimeStep * sizeof(float));
    Padd.h_Fsty = (float*)malloc(Padd.MaxTimeStep * sizeof(float));
    Padd.h_Fetx = (float*)malloc(Padd.MaxTimeStep * sizeof(float));
    Padd.h_Fety = (float*)malloc(Padd.MaxTimeStep * sizeof(float));
    //P.N = 1000;
       
    fprintf(stderr, "Finish create Arrays\n");

    //fprintf(stderr, "Start create cuda Arrays\n");
    
   
    HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 2 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 2 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 2 * P.N * sizeof(float)));    

    HANDLE_ERROR(cudaMalloc((void**)&P0.d_Ri, 2 * P0.N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&P0.d_RU0, 2 * P0.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&P0.d_U0, 2 * P0.N * sizeof(float)));

    //fprintf(stderr, "Start createLattice_kernel\n");
    //d_createLattice <<<Padd.bloks, SMEMDIM >>> (P0.d_RU0, P0.d_Ri, P0.N, L.LV.x);
    d_createLattice << <Padd.bloks, SMEMDIM >> > (P0.d_RU0, P0.d_U0, P0.d_Ri, P0.N, L.LV.x, Padd.Eps0);

    //fprintf(stderr, "Finish createLattice_kernel\n");/**/

    //fprintf(stderr, "Start memset\n");
    //HANDLE_ERROR(cudaMemset((void*)P.d_U, 0, 2 * P.N * sizeof(float)));
    //fprintf(stderr, "Finish memset\n");
/*
    //fprintf(stderr, "Start memcpy\n");
    HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(P0.h_RU0, P0.d_RU0, 2 * P0.N * sizeof(float), cudaMemcpyDeviceToHost));
    //fprintf(stderr, "Finish memcpy\n");

    //for (int i = 0; i < P0.N; ++i)    
    //    fprintf(stderr, "P %i %f %f | %f %f\n", i, P0.RU0X[i], P0.RU0Y[i], P0.RV0X[i], P0.RV0Y[i]);
    /*

    L.iCenter = L.n.x / 2;
    L.rCenter.x = P0.h_RU0[L.iCenter];
    L.V = P0.N * L.LV.x;
    L._1d_V = 1.0 / L.V;
    
    ///curandCreateGenerator(&Padd.gen, CURAND_RNG_PSEUDO_MTGP32);
    ///curandSetPseudoRandomGeneratorSeed(Padd.gen, time(NULL));

    fprintf(stderr, "Finish createLattice %i\n", P.N);  /**/  
}

__global__ void d_createLattice(float* __restrict__ RU0, float* __restrict__ U0, int* __restrict__ Ri, const int n, const float L0X, const float Eps0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x, ix, iy;
    //printf("GPU0 %i %i | %i %f \n", i, N, nX, _1d_nX);
    float fi;// , fix, fiy;
    while (i < n)
    {
        fi = __int2float_rn(i);
          
        RU0[i] =     fi*L0X;
        RU0[i + n] = 0;

        U0[i] = fi * Eps0 * L0X;
        U0[i + n] = 0;

        Ri[i] = __float2int_rn(fi);
        Ri[i + n] = 0;
        //printf("GPU0 %i %i | %f %f %f %f | \%i %i\n", i, N, RV0X[i], RV0Y[i], RU0X[i], RU0Y[i], RiX[i], RiY[i]);
        i += blockDim.x * gridDim.x;
    }  
    return;
}

void renewLattice(p_data& P, p0_data& P0)
{         
    //HANDLE_ERROR(cudaMemset((void*)P.d_U, 0, 2 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 2 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)P.d_U, 0, 2 * P.N * sizeof(float)));
    //HANDLE_ERROR(cudaMemcpy((void*)P.d_U, (void*)P0.d_U0, 2 * P.N * sizeof(float), cudaMemcpyDeviceToDevice));   
    //fprintf(stderr, "Finish renewLattice\n");
}