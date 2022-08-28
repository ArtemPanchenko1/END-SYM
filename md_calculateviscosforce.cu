#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>

__global__ void d_calculateVIscosForces(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drm, _1d_drm, re;
	float2 dr, t, v, vv, ff, fsum;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsum.x = 0;
		fsum.y = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			dr.x = Ir0[k] + U[j] - U[i];
			dr.y = Ir0[k + ni] + U[j + n] - U[i + n];
			drm = __fsqrt_rn(dr.x * dr.x + dr.y * dr.y);
			_1d_drm = __frcp_rn(drm); //__frsqrt_rn(drx * drx + dry * dry);
			t.x = dr.x * _1d_drm;
			t.y = dr.y * _1d_drm;
			//n.x = t.y; n.y = -t.x;
			v.x = 0.5f * (V[j] + V[i]);
			v.y = 0.5f * (V[j + n] + V[i + n]);
			vv.x = v.x * t.x + v.y * t.y;
			vv.y = v.x * t.y - v.y * t.x;//__fsqrt_rn(v.x * v.x + v.y * v.y - vv.x * vv.x);

			ff.x = 4.0f * Po_rfiber * __fsqrt_rn(MCf_pi * Po_mu * Po_roliquid * drm * fabsf(vv.x)) * vv.x;
			re = fabsf(vv.y) * Po_rfiber * Po_roliquid * Po_1d_mu;
			if (vv.y > 35) printf("ErrorVis %i %i %i | %e %e | %e %e | %e %e \n", idx, i, j, V[i], V[i + n], vv.x, vv.y, re, (2.0f * __logf(1.7811f * 0.25f * re)));
			ff.y = 8.0f * MCf_pi * Po_mu * drm * vv.y * __frcp_rn(1.0f - 2.0f * logf(1.7811f * 0.25f * re));

			fsum.x -= t.x * ff.x + t.y * ff.y;
			fsum.y -= t.y * ff.x - t.x * ff.y;
		}
		F[idx] += 0.5f * fsum.x;
		F[idx + n] += 0.5f * fsum.y;
		idx += blockDim.x * gridDim.x;
	}	
}

__global__ void d_calculateVIscosForces2(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_1d_hfreefiber)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drm, _1d_drm, re;
	float2 dr, t, v, vv, ff, fsum;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsum.x = 0;
		fsum.y = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			dr.x = Ir0[k] + U[j] - U[i];
			dr.y = Ir0[k + ni] + U[j + n] - U[i + n];
			drm = __fsqrt_rn(dr.x * dr.x + dr.y * dr.y);
			_1d_drm = __frcp_rn(drm); //__frsqrt_rn(drx * drx + dry * dry);
			t.x = dr.x * _1d_drm;
			t.y = dr.y * _1d_drm;
			//n.x = t.y; n.y = -t.x;
			v.x = 0.5f * (V[j] + V[i]);
			v.y = 0.5f * (V[j + n] + V[i + n]);
			vv.x = v.x * t.x + v.y * t.y;
			vv.y = v.x * t.y - v.y * t.x;//__fsqrt_rn(v.x * v.x + v.y * v.y - vv.x * vv.x);

			//ff.x = 4.0f * Po_rfiber * __fsqrt_rn(MCf_pi * Po_mu * Po_roliquid * drm * fabsf(vv.x)) * vv.x;
			ff.x = MCf_pi * Po_rfiber * drm * Po_1d_hfreefiber * Po_mu * vv.x;
			re = fabsf(vv.y) * Po_rfiber * Po_roliquid * Po_1d_mu;
			//if (vv.y > 35) printf("ErrorVis %i %i %i | %e %e | %e %e | %e %e \n", idx, i, j, V[i], V[i + n], vv.x, vv.y, re, (2.0f * __logf(1.7811f * 0.25f * re)));
			ff.y = 8.0f * MCf_pi * Po_mu * drm * vv.y * __frcp_rn(1.0f - 2.0f * logf(1.7811f * 0.25f * re));
#ifdef pre_CylinderDragFluidResistance
			ff.y += 1.1f * Po_roliquid * Po_rfiber * drm * fabsf(vv.y) * vv.y;
#endif // pre_CylinderDragFluidResistance

			fsum.x -= t.x * ff.x + t.y * ff.y;
			fsum.y -= t.y * ff.x - t.x * ff.y;
		}
		F[idx] += 0.5f * fsum.x;
		F[idx + n] += 0.5f * fsum.y;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_calculateVIscosForces3(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_1d_hfreefiber)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drm, _1d_drm, re, c, m_1d_dv, vabs;
	float2 dr, t, vi, vj, ff, fsum;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsum.x = 0;
		fsum.y = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			dr.x = Ir0[k] + U[j] - U[i];
			dr.y = Ir0[k + ni] + U[j + n] - U[i + n];
			drm = __fsqrt_rn(dr.x * dr.x + dr.y * dr.y);
			_1d_drm = __frcp_rn(drm); //__frsqrt_rn(drx * drx + dry * dry);
			t.x = dr.x * _1d_drm;
			t.y = dr.y * _1d_drm;

			vi.x = V[i] * t.x + V[i + n] * t.y;
			vi.y = V[i] * t.y - V[i + n] * t.x;
			vj.x = V[j] * t.x + V[j + n] * t.y;
			vj.y = V[j] * t.y - V[j + n] * t.x;
			c = 0.5f * MCf_pi * Po_rfiber * drm * Po_1d_hfreefiber * Po_mu;
			vabs = fabsf(vi.x) + fabsf(vj.x);			
			if (fabsf(vi.x - vj.x) > 1e-9)
				m_1d_dv = vi.x * __frcp_rn(vi.x - vj.x);
			else 
				m_1d_dv = 0;
			if (m_1d_dv > 0 && m_1d_dv < 1)
			{
				ff.x = c * vi.x * m_1d_dv;
			}
			else if (vabs > 1e-9)
			{
				ff.x = c * (vi.x + vj.x) * fabsf(vi.x) * __frcp_rn(vabs);
			}
			else
			{
				ff.x = 0.5f * c * (vi.x + vj.x);
			}
			re = 0.5f * vabs * Po_rfiber * Po_roliquid * Po_1d_mu;
			ff.y = 8.0f * MCf_pi * MCf_1d6 * Po_mu * drm * (2.0f * vi.y + vj.y) * __frcp_rn(1.0f - 2.0f * logf(1.7811f * 0.25f * re));

			fsum.x -= t.x * ff.x + t.y * ff.y;
			fsum.y -= t.y * ff.x - t.x * ff.y;
		}
		F[idx] += fsum.x;
		F[idx + n] += fsum.y;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_calculateVIscosForcesShapovalov(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_1d_mu, const float Po_rfiber, const float Po_roliquid, const float Po_CShfreefiber)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drm, _1d_drm, _1d_re, c, m_1d_dv, vabs;
	float2 dr, t, vi, vj, ff, fsum;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsum.x = 0;
		fsum.y = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			dr.x = Ir0[k] + U[j] - U[i];
			dr.y = Ir0[k + ni] + U[j + n] - U[i + n];
			drm = __fsqrt_rn(dr.x * dr.x + dr.y * dr.y);
			_1d_drm = __frcp_rn(drm); //__frsqrt_rn(drx * drx + dry * dry);
			t.x = dr.x * _1d_drm;
			t.y = dr.y * _1d_drm;

			vi.x = V[i] * t.x + V[i + n] * t.y;
			vi.y = V[i] * t.y - V[i + n] * t.x;
			vj.x = V[j] * t.x + V[j + n] * t.y;
			vj.y = V[j] * t.y - V[j + n] * t.x;

			c = MCf_pi * Po_rfiber * drm * Po_mu * Po_CShfreefiber;
			vabs = fabsf(vi.x) + fabsf(vj.x);
			if (fabsf(vi.x - vj.x) > 1e-9)
				m_1d_dv = vi.x * __frcp_rn(vi.x - vj.x);
			else
				m_1d_dv = 0;
			if (m_1d_dv > 0 && m_1d_dv < 1)
			{
				ff.x = c * vi.x * m_1d_dv;
			}
			else if (vabs > 1e-9)
			{
				ff.x = c * (vi.x + vj.x) * fabsf(vi.x) * __frcp_rn(vabs);
			}
			else
			{
				ff.x = 0.5f * c * (vi.x + vj.x);
			}
			_1d_re = __frcp_rn(0.5f * vabs * Po_rfiber * Po_roliquid * Po_1d_mu);
			ff.y = 4.0f * MCf_pi * MCf_1d6 * Po_mu * drm * (2.0f * vi.y + vj.y) * __frcp_rn(logf(7.4f * _1d_re));
			//ff.x = 0;
			//ff.y = 0;
			fsum.x -= t.x * ff.x + t.y * ff.y;
			fsum.y -= t.y * ff.x - t.x * ff.y;
		}
		F[idx] += fsum.x;
		F[idx + n] += fsum.y;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_calculateVIscosForcesLindstrom(const int* __restrict__ In, const float* __restrict__ Ir0, const int* __restrict__ ShIn, const float* __restrict__ U, const float* __restrict__ V, float* __restrict__ F, const unsigned int n, const unsigned int ni, const float Po_mu, const float Po_rfiber, const float Po_1d_rfiber, const float Po_roliquid)
{
	//int vpx, vpy, j1, j2, mj1, mj2;
	int i, j, k, ks, kmax;
	float drm, _1d_drm, a,b,e,L,XA, YA, YC, c1, c2, c3, re, c, m_1d_dv, vabs;
	float2 dr, t, vi, vj, ff, fsum;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	while (idx < n)
	{
		//printf("FI %i %i %i\n", idx, ShIn[idx], ShIn[idx + n]);
		ks = ShIn[idx];
		kmax = ks + ShIn[idx + n];
		fsum.x = 0;
		fsum.y = 0;
		for (k = ks; k < kmax; ++k)
		{
			i = In[k];
			j = In[k + ni];

			dr.x = Ir0[k] + U[j] - U[i];
			dr.y = Ir0[k + ni] + U[j + n] - U[i + n];
			drm = __fsqrt_rn(dr.x * dr.x + dr.y * dr.y);
			_1d_drm = __frcp_rn(drm); //__frsqrt_rn(drx * drx + dry * dry);
			t.x = dr.x * _1d_drm;
			t.y = dr.y * _1d_drm;

			vi.x = V[i] * t.x + V[i + n] * t.y;
			vi.y = V[i] * t.y - V[i + n] * t.x;
			vj.x = V[j] * t.x + V[j + n] * t.y;
			vj.y = V[j] * t.y - V[j + n] * t.x;

			a = 0.5f * drm;
			b = 0.806451612903225f * Po_rfiber * __fsqrt_rn(logf(0.5f * drm * Po_1d_rfiber));
			e = __fsqrt_rn(1.0f - 4.0f * b * b * _1d_drm * _1d_drm);
			L = logf((1.0f + e) * __frcp_rn(1.0f - e));
			c1 = 4.0f * MCf_pi * MCf_1d3 * e * e * e * drm * Po_mu;
			c2 = __frcp_rn(-2.0f * e + (1.0f + e * e) * L);
			c3 = __frcp_rn(2.0f * e + (3.0f * e * e - 1.0f) * L);
			XA = 6.0f * c1 * c2;
			YA = 12.0f * c1 * c3;
			YC = c1 * (2.0f - e * e) * c2;

			vabs = fabsf(vi.x) + fabsf(vj.x);
			if (vabs > 1e-9)
			{
				ff.x = 0.5f * XA * (vi.x + vj.x) * fabsf(vi.x) * __frcp_rn(vabs);
			}
			else
			{
				ff.x = 0.25f * XA * (vi.x + vj.x);
			}
			//ff.x = XA * 0.25f * (vi.x + vj.x);
			ff.y = YA * (vi.y + vj.y) * 0.25f - YC * (vj.y - vi.y);
			//if(idx==568)printf("IM %f %f | %f %f %f %f %f %f %f\n", drm, Po_rfiber, a, b, e, L, XA, YA, YC);
			//c = MCf_pi * 2.0f * Po_rfiber * drm * Po_mu * Po_CShfreefiber;
			//c = 3.0f * XA;
			//re = (fabsf(vi.y) + fabsf(vj.y)) * Po_rfiber * Po_roliquid * Po_1d_mu;
			//ff.y = 4.0f * MCf_pi * Po_mu * drm * (2.0f * vi.y + vj.y) * __frcp_rn(logf(7.4f * re));
			//ff.y = 0;
			fsum.x -= t.x * ff.x + t.y * ff.y;
			fsum.y -= t.y * ff.x - t.x * ff.y;
		}
		F[idx] += fsum.x;
		F[idx + n] += fsum.y;
		idx += blockDim.x * gridDim.x;
	}
}