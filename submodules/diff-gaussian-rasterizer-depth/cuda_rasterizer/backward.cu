/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <typename T>
__device__ int argMax(T a, T b, T c)
{
	if (a >= b && a >= c)
	{
		return 0;
	}
	else if (b >= a && b >= c)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

template <typename T>
__forceinline__ __device__ int argMin(T a, T b, T c)
{
	if (a <= b && a <= c)
	{
		return 0;
	}
	else if (b <= a && b <= c)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

__device__ void computeNormal_ScaleMax_backward(const glm::vec3 scale, float scale_mod, const glm::vec4 rot, float4 *normal_scale)
{
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	int normal_aixs = argMin(scale.x, scale.y, scale.z);
	int scale_max_axis = argMax(scale.x, scale.y, scale.z);
	normal_scale->x = R[0][normal_aixs];
	normal_scale->y = R[1][normal_aixs];
	normal_scale->z = R[2][normal_aixs];
	normal_scale->w = scale[scale_max_axis] * scale_mod;
}

__device__ float computeDistanceToPlane_backward(const float3 pNormal, const float3 pPoint, const float3 ray)
{
	float t = (pPoint.x * pNormal.x + pPoint.y * pNormal.y + pPoint.z * pNormal.z) /
			  (ray.x * pNormal.x + ray.y * pNormal.y + ray.z * pNormal.z + 1e-8);
	return t;
}

__forceinline__ __device__ float absdot(const float3 p1, const float3 p2)
{
	return abs(p1.x * p2.x + p1.y * p2.y + p1.z * p2.z);
}

__forceinline__ __device__ float dot(const float3 p1, const float3 p2)
{
	return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}
__forceinline__ __device__ float3 ndc2ray(uint2 pix, float fx, float fy, float cx, float cy)
{
	float3 ray = {(pix.x - cx) / fx, (pix.y - cy) / fy, 1.0};
	float ray_norm = 1 / sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
	ray.x *= ray_norm;
	ray.y *= ray_norm;
	ray.z *= ray_norm;
	return ray;
}

__forceinline__ __device__ void propagateRotationGrad(glm::vec4 rotation, float *dL_dq0, float *dL_dq1, float *dL_dq2, float *dL_dq3, int axis)
{
	float q0 = rotation.x, q1 = rotation.y, q2 = rotation.z, q3 = rotation.w;
	if (axis == 0)
	{
		dL_dq0[0] = 0;
		dL_dq0[1] = 2 * q3;
		dL_dq0[2] = -2 * q2;
		dL_dq1[0] = 0;
		dL_dq1[1] = 2 * q2;
		dL_dq1[2] = 2 * q3;
		dL_dq2[0] = -4 * q2;
		dL_dq2[1] = 2 * q1;
		dL_dq2[2] = -2 * q0;
		dL_dq3[0] = -4 * q3;
		dL_dq3[1] = 2 * q0;
		dL_dq3[2] = 2 * q1;
	}
	else if (axis == 1)
	{
		dL_dq0[0] = -2 * q3;
		dL_dq0[1] = 0;
		dL_dq0[2] = 2 * q1;
		dL_dq1[0] = 2 * q2;
		dL_dq1[1] = -4 * q1;
		dL_dq1[2] = 2 * q0;
		dL_dq2[0] = 2 * q1;
		dL_dq2[1] = 0;
		dL_dq2[2] = 2 * q3;
		dL_dq3[0] = -2 * q0;
		dL_dq3[1] = -4 * q3;
		dL_dq3[2] = 2 * q2;
	}
	else
	{
		dL_dq0[0] = 2 * q2;
		dL_dq0[1] = -2 * q1;
		dL_dq0[2] = 0;
		dL_dq1[0] = 2 * q3;
		dL_dq1[1] = -2 * q0;
		dL_dq1[2] = -4 * q1;
		dL_dq2[0] = 2 * q0;
		dL_dq2[1] = 2 * q3;
		dL_dq2[2] = -4 * q2;
		dL_dq3[0] = 2 * q1;
		dL_dq3[1] = 2 * q2;
		dL_dq3[2] = 0;
	}
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, const bool *clamped, const glm::vec3 *dL_dcolor, glm::vec3 *dL_dmeans, glm::vec3 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
								 const float3 *means,
								 const int *radii,
								 const float *cov3Ds,
								 const float h_x, float h_y,
								 const float tan_fovx, float tan_fovy,
								 const float *view_matrix,
								 const float *dL_dconics,
								 float3 *dL_dmeans,
								 float *dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;
	// Reading location of 3D covariance for this Gaussian
	const float *cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]};
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
							0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
							0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
					(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
					(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
					(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
					(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
					(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
					(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({dL_dtx, dL_dty, dL_dtz}, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	// if (idx == 9)
	// {
	// 	// denom2inv * (-c * c * dL_dconic.x
	// 	printf("denom2inv: %f, dL_dconic_x: %f, dL_dconic_y: %f,dL_dconic_x: %f\n", denom2inv, dL_dconic.x, dL_dconic.y, dL_dconic.z);
	// 	// W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02
	// 	printf("dL_da: %f, dL_db: %f, dL_dc: %f\n", dL_da, dL_db, dL_dc);
	// 	printf("W20: %f, dL_dT00: %f, W21: %f, dL_dT01: %f, W22: %f, dL_dT02: %f\n",
	// 		   W[2][0], dL_dT00, W[2][1], dL_dT01, W[2][2], dL_dT02);
	// 	printf("viewmatrix: %f, %f, %f, %f\n", view_matrix[0], view_matrix[4], view_matrix[8], view_matrix[12]);
	// 	printf("h_x: %f, tz2: %f, dL_dJ02: %f\n", h_x, tz2, dL_dJ02);
	// 	printf("tx: %f, ty: %f, tz: %f\n", t.x, t.y, t.z);
	// 	printf("preprocess before dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z);
	// 	printf("dL_dmean_x: %f, dL_dmean_y: %f, dL_dmean_z: %f\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
	// }
	dL_dmeans[idx].x += dL_dmean.x;
	dL_dmeans[idx].y += dL_dmean.y;
	dL_dmeans[idx].z += dL_dmean.z;
	// if (idx == 9)
	// {
	// 	printf("preprocess dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z);
	// }
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float *dL_dcov3Ds, glm::vec3 *dL_dscales, glm::vec4 *dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot; // / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3 *dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4 *dL_drot = (float4 *)(dL_drots + idx);
	*dL_drot = float4{dL_drot->x + dL_dq.x, dL_drot->y + dL_dq.y, dL_drot->z + dL_dq.z, dL_drot->w + dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3 *means,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *proj,
	const glm::vec3 *campos,
	const float3 *dL_dmean2D,
	glm::vec3 *dL_dmeans,
	float *dL_dcolor,
	float *dL_dcov3D,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;
	// if (idx == 9)
	// {
	// 	printf("preprocess-1 dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z);
	// }
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3 *)means, *campos, shs, clamped, (glm::vec3 *)dL_dcolor, (glm::vec3 *)dL_dmeans, (glm::vec3 *)dL_dsh);
	// if (idx == 9)
	// {
	// 	printf("preprocess-2 dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z);
	// }
	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		float fx, float fy,
		const float cx, const float cy,
		const float scale_mod,
		const float normal_threshold,
		const float depth_threshold,
		const float *viewmatrix,
		const glm::vec3 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const float3 *__restrict__ means3D,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float4 *__restrict__ conic_opacity,
		const float *__restrict__ colors,
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_dpixel_depths,
		const int *__restrict__ hit_image,
		const float3 *__restrict__ hit_normal_c,
		const float3 *__restrict__ hit_point_c,
		float3 *__restrict__ dL_dmean2D,
		float4 *__restrict__ dL_dconic2D,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_dmeans3D,
		float *__restrict__ dL_drotation)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = {0};
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = {0};

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			const int global_id = collected_id[j];
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;

			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];

				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
	// hit gaussian depth loss grad
	if (inside && hit_image[pix_id] >= 0)
	{
		int gaussian_id = hit_image[pix_id];

		int gaussian_id3 = gaussian_id * 3;
		int gaussian_id4 = gaussian_id * 4;
		// dL_dmeans3D
		float4 hit_normal_scale;
		float3 pix_ray = ndc2ray(pix, fx, fy, cx, cy);
		// computeNormal_ScaleMax_backward(((glm::vec3 *)scales)[gaussian_id], scale_mod,
		// 								((glm::vec4 *)rotations)[gaussian_id], &hit_normal_scale);
		const float scale_max = fmax(fmax(scales[gaussian_id].x, scales[gaussian_id].y), scales[gaussian_id].z);
		float3 gaussian_normal_c = hit_normal_c[pix_id];
		float3 points_xyz_w = means3D[gaussian_id];
		float3 points_xyz_c = transformPoint4x3(points_xyz_w, viewmatrix);
		float3 hit_point = hit_point_c[pix_id];
		// float t = computeDistanceToPlane_backward(gaussian_normal_c, points_xyz_c, pix_ray);
		float angle_distance = absdot(gaussian_normal_c, pix_ray);
		float depth_distance = abs(hit_point.z - points_xyz_c.z);
		const float dL_ddi = dL_dpixel_depths[pix_id];
		if (depth_distance <= depth_threshold * scale_max && angle_distance >= normal_threshold)
		{
			// normal depth
			// means3D grad
			const float nr = dot(gaussian_normal_c, pix_ray) + 1e-8;
			const float inv_nr = 1 / nr;
			const float inv_nr2 = inv_nr * inv_nr;
			const float np = dot(gaussian_normal_c, points_xyz_c);
			const float dL_dpx = pix_ray.z * gaussian_normal_c.x * inv_nr;
			const float dL_dpy = pix_ray.z * gaussian_normal_c.y * inv_nr;
			const float dL_dpz = pix_ray.z * gaussian_normal_c.z * inv_nr;

			const float dL_dx = dL_ddi * (dL_dpx * viewmatrix[0] + dL_dpy * viewmatrix[1] + dL_dpz * viewmatrix[2]);
			const float dL_dy = dL_ddi * (dL_dpx * viewmatrix[4] + dL_dpy * viewmatrix[5] + dL_dpz * viewmatrix[6]);
			const float dL_dz = dL_ddi * (dL_dpx * viewmatrix[8] + dL_dpy * viewmatrix[9] + dL_dpz * viewmatrix[10]);

			atomicAdd(&(dL_dmeans3D[gaussian_id3]), dL_dx);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 1]), dL_dy);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 2]), dL_dz);
			// if (gaussian_id == 9)
			// {
			// 	printf("add dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dx, dL_dy, dL_dz);
			// 	printf("origin dL_dx: %f, dL_dy: %f, dL_dz: %f\n", dL_dmeans3D[gaussian_id3], dL_dmeans3D[gaussian_id3 + 1], dL_dmeans3D[gaussian_id3 + 2]);
			// }
			// rotation grad

			const int rotation_axis = argMin(scales[gaussian_id].x, scales[gaussian_id].y, scales[gaussian_id].z);
			const float ddi_n1_c = pix_ray.z * (nr * points_xyz_c.x - np * pix_ray.x) * inv_nr2;
			const float ddi_n2_c = pix_ray.z * (nr * points_xyz_c.y - np * pix_ray.y) * inv_nr2;
			const float ddi_n3_c = pix_ray.z * (nr * points_xyz_c.z - np * pix_ray.z) * inv_nr2;
			const float ddi_n1_w = ddi_n1_c * viewmatrix[0] + ddi_n2_c * viewmatrix[1] + ddi_n3_c * viewmatrix[2];
			const float ddi_n2_w = ddi_n1_c * viewmatrix[4] + ddi_n2_c * viewmatrix[5] + ddi_n3_c * viewmatrix[6];
			const float ddi_n3_w = ddi_n1_c * viewmatrix[8] + ddi_n2_c * viewmatrix[9] + ddi_n3_c * viewmatrix[10];
			float dn1_dq0, dn1_dq1, dn1_dq2, dn1_dq3, dn2_dq0, dn2_dq1, dn2_dq2, dn2_dq3, dn3_dq0, dn3_dq1, dn3_dq2, dn3_dq3;
			float dn_dq0[3], dn_dq1[3], dn_dq2[3], dn_dq3[3];
			propagateRotationGrad(rotations[gaussian_id], dn_dq0, dn_dq1, dn_dq2, dn_dq3, rotation_axis);
			// if (gaussian_id == 20744)
			// {
			// 	printf("pix_xy:(%d, %d)\n", pix.x, pix.y);
			// }
			// if (gaussian_id == 20744 && pix.x == 190 && pix.y == 653)
			// {
			// 	printf("pix_xy:(%d, %d)\n", pix.x, pix.y);
			// 	printf("pix_ray: %f, %f, %f. normal: %f, %f, %f. points: %f, %f, %f\n", pix_ray.x,
			// 		   pix_ray.y, pix_ray.z, gaussian_normal_c.x, gaussian_normal_c.y, gaussian_normal_c.z,
			// 		   points_xyz_c.x, points_xyz_c.y, points_xyz_c.z);
			// 	printf("ddi_dn1: %f, ddi_dn2: %f, ddi_dn3: %f\n", ddi_n1_w, ddi_n2_w, ddi_n3_w);
			// }
			const float dL_dq0 = dL_ddi * (ddi_n1_w * dn_dq0[0] + ddi_n2_w * dn_dq0[1] + ddi_n3_w * dn_dq0[2]);
			const float dL_dq1 = dL_ddi * (ddi_n1_w * dn_dq1[0] + ddi_n2_w * dn_dq1[1] + ddi_n3_w * dn_dq1[2]);
			const float dL_dq2 = dL_ddi * (ddi_n1_w * dn_dq2[0] + ddi_n2_w * dn_dq2[1] + ddi_n3_w * dn_dq2[2]);
			const float dL_dq3 = dL_ddi * (ddi_n1_w * dn_dq3[0] + ddi_n2_w * dn_dq3[1] + ddi_n3_w * dn_dq3[2]);
			atomicAdd(&(dL_drotation[gaussian_id4]), dL_dq0);
			atomicAdd(&(dL_drotation[gaussian_id4 + 1]), dL_dq1);
			atomicAdd(&(dL_drotation[gaussian_id4 + 2]), dL_dq2);
			atomicAdd(&(dL_drotation[gaussian_id4 + 3]), dL_dq3);
		}
		else
		{
			// opaque depth
			atomicAdd(&(dL_dmeans3D[gaussian_id3]), dL_ddi * viewmatrix[2]);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 1]), dL_ddi * viewmatrix[6]);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 2]), dL_ddi * viewmatrix[10]);
		}
	}
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA_flat(
		const int *tile_indices,
		const int tile_width,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		float fx, float fy,
		const float cx, const float cy,
		const float scale_mod,
		const float normal_threshold,
		const float depth_threshold,
		const float *viewmatrix,
		const glm::vec3 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const float3 *__restrict__ means3D,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float4 *__restrict__ conic_opacity,
		const float *__restrict__ colors,
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_dpixel_depths,
		const int *__restrict__ hit_image,
		const float3 *__restrict__ hit_normal_c,
		const float3 *__restrict__ hit_point_c,
		const float *color_weight_sum,
		float3 *__restrict__ dL_dmean2D,
		float4 *__restrict__ dL_dconic2D,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_dmeans3D,
		float *__restrict__ dL_drotation)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const int block_id = block.group_index().x;
	const int real_tile = tile_indices[block_id];
	const int real_tile_x = real_tile % tile_width;
	const int real_tile_y = real_tile / tile_width;
	// const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	// const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	// const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	// const uint32_t pix_id = W * pix.y + pix.x;
	// const float2 pixf = {(float)pix.x, (float)pix.y};

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {real_tile_x * BLOCK_X, real_tile_y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	// const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const uint2 range = ranges[real_tile];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	float inv_weight_sum = 1.0f;
	if (inside && color_weight_sum[pix_id] > 0)
	{
		inv_weight_sum = 1 / color_weight_sum[pix_id];
	}
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = {0};
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = {0};

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			const int global_id = collected_id[j];
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;

			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];

				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
	// hit gaussian depth loss grad
	if (inside && hit_image[pix_id] >= 0)
	{
		int gaussian_id = hit_image[pix_id];

		int gaussian_id3 = gaussian_id * 3;
		int gaussian_id4 = gaussian_id * 4;
		// dL_dmeans3D
		float4 hit_normal_scale;
		float3 pix_ray = ndc2ray(pix, fx, fy, cx, cy);
		// computeNormal_ScaleMax_backward(((glm::vec3 *)scales)[gaussian_id], scale_mod,
		// 								((glm::vec4 *)rotations)[gaussian_id], &hit_normal_scale);
		const float scale_max = fmax(fmax(scales[gaussian_id].x, scales[gaussian_id].y), scales[gaussian_id].z);
		float3 gaussian_normal_c = hit_normal_c[pix_id];
		float3 points_xyz_w = means3D[gaussian_id];
		float3 points_xyz_c = transformPoint4x3(points_xyz_w, viewmatrix);
		float3 hit_point = hit_point_c[pix_id];
		// float t = computeDistanceToPlane_backward(gaussian_normal_c, points_xyz_c, pix_ray);
		float angle_distance = absdot(gaussian_normal_c, pix_ray);
		float depth_distance = abs(hit_point.z - points_xyz_c.z);
		const float dL_ddi = dL_dpixel_depths[pix_id];
		if (depth_distance <= depth_threshold * scale_max && angle_distance >= normal_threshold)
		{
			// normal depth
			// means3D grad
			const float nr = dot(gaussian_normal_c, pix_ray) + 1e-8;
			const float inv_nr = 1 / nr;
			const float inv_nr2 = inv_nr * inv_nr;
			const float np = dot(gaussian_normal_c, points_xyz_c);
			const float dL_dpx = pix_ray.z * gaussian_normal_c.x * inv_nr;
			const float dL_dpy = pix_ray.z * gaussian_normal_c.y * inv_nr;
			const float dL_dpz = pix_ray.z * gaussian_normal_c.z * inv_nr;

			const float dL_dx = dL_ddi * (dL_dpx * viewmatrix[0] + dL_dpy * viewmatrix[1] + dL_dpz * viewmatrix[2]);
			const float dL_dy = dL_ddi * (dL_dpx * viewmatrix[4] + dL_dpy * viewmatrix[5] + dL_dpz * viewmatrix[6]);
			const float dL_dz = dL_ddi * (dL_dpx * viewmatrix[8] + dL_dpy * viewmatrix[9] + dL_dpz * viewmatrix[10]);

			atomicAdd(&(dL_dmeans3D[gaussian_id3]), dL_dx);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 1]), dL_dy);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 2]), dL_dz);

			const int rotation_axis = argMin(scales[gaussian_id].x, scales[gaussian_id].y, scales[gaussian_id].z);
			const float ddi_n1_c = pix_ray.z * (nr * points_xyz_c.x - np * pix_ray.x) * inv_nr2;
			const float ddi_n2_c = pix_ray.z * (nr * points_xyz_c.y - np * pix_ray.y) * inv_nr2;
			const float ddi_n3_c = pix_ray.z * (nr * points_xyz_c.z - np * pix_ray.z) * inv_nr2;
			const float ddi_n1_w = ddi_n1_c * viewmatrix[0] + ddi_n2_c * viewmatrix[1] + ddi_n3_c * viewmatrix[2];
			const float ddi_n2_w = ddi_n1_c * viewmatrix[4] + ddi_n2_c * viewmatrix[5] + ddi_n3_c * viewmatrix[6];
			const float ddi_n3_w = ddi_n1_c * viewmatrix[8] + ddi_n2_c * viewmatrix[9] + ddi_n3_c * viewmatrix[10];
			float dn1_dq0, dn1_dq1, dn1_dq2, dn1_dq3, dn2_dq0, dn2_dq1, dn2_dq2, dn2_dq3, dn3_dq0, dn3_dq1, dn3_dq2, dn3_dq3;
			float dn_dq0[3], dn_dq1[3], dn_dq2[3], dn_dq3[3];
			propagateRotationGrad(rotations[gaussian_id], dn_dq0, dn_dq1, dn_dq2, dn_dq3, rotation_axis);

			const float dL_dq0 = dL_ddi * (ddi_n1_w * dn_dq0[0] + ddi_n2_w * dn_dq0[1] + ddi_n3_w * dn_dq0[2]);
			const float dL_dq1 = dL_ddi * (ddi_n1_w * dn_dq1[0] + ddi_n2_w * dn_dq1[1] + ddi_n3_w * dn_dq1[2]);
			const float dL_dq2 = dL_ddi * (ddi_n1_w * dn_dq2[0] + ddi_n2_w * dn_dq2[1] + ddi_n3_w * dn_dq2[2]);
			const float dL_dq3 = dL_ddi * (ddi_n1_w * dn_dq3[0] + ddi_n2_w * dn_dq3[1] + ddi_n3_w * dn_dq3[2]);
			atomicAdd(&(dL_drotation[gaussian_id4]), dL_dq0);
			atomicAdd(&(dL_drotation[gaussian_id4 + 1]), dL_dq1);
			atomicAdd(&(dL_drotation[gaussian_id4 + 2]), dL_dq2);
			atomicAdd(&(dL_drotation[gaussian_id4 + 3]), dL_dq3);
		}
		else
		{
			// opaque depth
			atomicAdd(&(dL_dmeans3D[gaussian_id3]), dL_ddi * viewmatrix[2]);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 1]), dL_ddi * viewmatrix[6]);
			atomicAdd(&(dL_dmeans3D[gaussian_id3 + 2]), dL_ddi * viewmatrix[10]);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3 *means3D,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *cov3Ds,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3 *campos,
	const float3 *dL_dmean2D,
	const float *dL_dconic,
	glm::vec3 *dL_dmean3D,
	float *dL_dcolor,
	float *dL_dcov3D,
	float *dL_dsh,
	glm::vec3 *dL_dscale,
	glm::vec4 *dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3 *)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3 *)dL_dmean2D,
		(glm::vec3 *)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float fx, const float fy,
	const float cx, const float cy,
	const float depth_threshold,
	const float normal_threshold,
	const float scale_mod,
	const float *scales,
	const float *rotations,
	const float3 *means3D,
	const float *bg_color,
	const float2 *means2D,
	const float4 *conic_opacity,
	const float *colors,
	const float *final_Ts,
	const uint32_t *n_contrib,
	const float *dL_dpixels,
	const float *dL_dpixel_depths,
	const int *hit_image,
	const float *viewmatrix,
	const float3 *hit_normal_c,
	const float3 *hit_point_c,
	float3 *dL_dmean2D,
	float4 *dL_dconic2D,
	float *dL_dopacity,
	float *dL_dcolors,
	float *dL_dmeans3D,
	float *dL_drotation)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		fx, fy,
		cx, cy,
		scale_mod,
		normal_threshold,
		depth_threshold,
		viewmatrix,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		means3D,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		hit_image,
		hit_normal_c,
		hit_point_c,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dmeans3D,
		dL_drotation);
}

void BACKWARD::render_flat(
	const int *tile_indices,
	const int tile_num,
	const int tile_width,
	dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float fx, const float fy,
	const float cx, const float cy,
	const float depth_threhsold,
	const float normal_threshold,
	const float scale_mod,
	const float *scales,
	const float *rotations,
	const float3 *means3D,
	const float *bg_color,
	const float2 *means2D,
	const float4 *conic_opacity,
	const float *colors,
	const float *final_Ts,
	const uint32_t *n_contrib,
	const float *dL_dpixels,
	const float *dL_dpixel_depths,
	const int *hit_image,
	const float *viewmatrix,
	const float3 *hit_normal_c,
	const float3 *hit_point_c,
	const float *color_weight_sum,
	float3 *dL_dmean2D,
	float4 *dL_dconic2D,
	float *dL_dopacity,
	float *dL_dcolors,
	float *dL_dmeans3D,
	float *dL_drotation)
{
	dim3 grid{tile_num, 1, 1};
	renderCUDA_flat<NUM_CHANNELS><<<grid, block>>>(
		tile_indices,
		tile_width,
		ranges,
		point_list,
		W, H,
		fx, fy,
		cx, cy,
		scale_mod,
		normal_threshold,
		depth_threhsold,
		viewmatrix,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		means3D,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		hit_image,
		hit_normal_c,
		hit_point_c,
		color_weight_sum,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dmeans3D,
		dL_drotation);
}