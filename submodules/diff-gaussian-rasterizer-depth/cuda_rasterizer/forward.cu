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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

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
__device__ int argMin(T a, T b, T c)
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

__device__ void computeNormal_ScaleMax(const glm::vec3 scale, float scale_mod, const glm::vec4 rot, float4 *normal_scale)
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

__device__ float computeDistanceToPlane(const float3 pNormal, const float3 pPoint, const float3 ray)
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

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
					 SH_C2[0] * xy * sh[4] +
					 SH_C2[1] * yz * sh[5] +
					 SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					 SH_C2[3] * xz * sh[7] +
					 SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
						 SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						 SH_C3[1] * xy * z * sh[10] +
						 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						 SH_C3[5] * z * (xx - yy) * sh[14] +
						 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3 &mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D, const float *viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float *cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

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

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P, int D, int M,
							   const float color_sigma,
							   const float *orig_points,
							   const glm::vec3 *scales,
							   const float scale_modifier,
							   const glm::vec4 *rotations,
							   const float *opacities,
							   const float *shs,
							   bool *clamped,
							   const float *cov3D_precomp,
							   const float *colors_precomp,
							   const float *viewmatrix,
							   const float *projmatrix,
							   const int *tile_mask,
							   const glm::vec3 *cam_pos,
							   const int W, const int H,
							   const float tan_fovx, const float tan_fovy,
							   const float focal_x, const float focal_y,
							   const float cx, const float cy,
							   int *radii,
							   float2 *points_xy_image,
							   float *depths,
							   float *cov3Ds,
							   float *rgb,
							   float4 *conic_opacity,
							   const dim3 grid,
							   uint32_t *tiles_touched,
							   bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	auto idx3 = idx * 3;
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	// Perform near culling, quit if outside.
	float3 p_view;
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float *cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
	// float3 transform_normal = transformPoint4x3(float3{normals[idx3], normals[idx3 + 1], normals[idx3 + 2]}, viewmatrix);
	// normals[idx3] = transform_normal.x;
	// normals[idx3 + 1] = transform_normal.y;
	// normals[idx3 + 2] = transform_normal.z;
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(color_sigma * sqrt(max(lambda1, lambda2)));
	// float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	float2 point_image = {ndc2Pix(p_proj.x, W, cx), ndc2Pix(p_proj.y, H, cy)};

	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3 *)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
	int tiles_touch = 0;
	for (int x = rect_min.x; x < rect_max.x; x++)
	{
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			if (tile_mask[y * grid.x + x])
				tiles_touch++;
		}
	}
	tiles_touched[idx] = tiles_touch;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	test_T(
		const int W,
		const int H,
		float *__restrict__ final_T)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	if (inside)
	{
		final_T[pix_id] = 2.5;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA_test(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float fx, const float fy,
		const float scale_modify,
		const float *viewmatrix,
		const float *__restrict__ points_xyz_world,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ features,
		const float *__restrict__ depths,
		const glm::vec3 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const float4 *__restrict__ conic_opacity,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		const float opaque_threshold,
		const float hit_depth_threshold,
		const float hit_normal_threshold,
		float *__restrict__ out_color,
		float *__restrict__ out_depth,
		int *__restrict__ out_hit)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	float cx = W / 2.0, cy = H / 2.0;
	float3 pix_ray = ndc2ray(pix, fx, fy, cx, cy);
	if (inside)
	{
		final_T[pix_id] = 0;
		return;
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float fx, const float fy,
		const float cx, const float cy,
		const float scale_modify,
		const float *viewmatrix,
		const float *__restrict__ points_xyz_world,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ features,
		const float *__restrict__ depths,
		const glm::vec3 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const float4 *__restrict__ conic_opacity,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		const float opaque_threshold,
		const float hit_depth_threshold,
		const float hit_normal_threshold,
		float3 *__restrict__ hit_normal_c,
		float3 *__restrict__ hit_point_c,
		float *__restrict__ out_color,
		float *__restrict__ out_depth,
		int *__restrict__ out_hit_depth,
		int *__restrict__ out_hit_color,
		float *__restrict__ out_hit_color_weight,
		float *__restrict__ out_hit_depth_weight,
		float *__restrict__ out_T)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	float3 pix_ray = ndc2ray(pix, fx, fy, cx, cy);
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	bool render = true;
	// if (inside && render_mask[pix_id] == 0)
	// {
	// 	render = false;
	// }
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	float end_T = T;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};
	float depth_ = 0;
	bool hit_gaussian = false;
	bool hit_color_end = false;
	float opaque_depth = 0;
	float3 hit_normal{0, 0, 0};
	float hit_scale = 0;
	int hit_gaussian_id = -1;
	int hit_gaussian_color_id = -1;
	float color_weight_max = -1;
	float hit_color_weight = 0.0;
	float hit_depth_weight = 0.0;
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && render && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			const int gaussian_id = collected_id[j];
			const int gaussian_id3 = collected_id[j] * 3;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			// t = torch.sum(hit_normals * hit_points, dim=1) / (torch.sum(
			//     hit_normals * pixel_rays, dim=1
			// ) + 1e-8)
			// check hit gaussian

			float4 hit_normal_scale;
			computeNormal_ScaleMax(scales[gaussian_id], scale_modify, rotations[gaussian_id], &hit_normal_scale);
			float3 points_xyz_w{points_xyz_world[gaussian_id3], points_xyz_world[gaussian_id3 + 1], points_xyz_world[gaussian_id3 + 2]};
			float3 gaussian_normal_c = transformVec4x3(float3{hit_normal_scale.x, hit_normal_scale.y, hit_normal_scale.z}, viewmatrix);
			float3 points_xyz_c = transformPoint4x3(points_xyz_w, viewmatrix);
			float t = (points_xyz_c.x * gaussian_normal_c.x + points_xyz_c.y * gaussian_normal_c.y + points_xyz_c.z * gaussian_normal_c.z) /
					  (pix_ray.x * gaussian_normal_c.x + pix_ray.y * gaussian_normal_c.y + pix_ray.z * gaussian_normal_c.z + 1e-8);
			float3 hit_point{t * pix_ray.x, t * pix_ray.y, t * pix_ray.z};
			float3 hit_points_delta{hit_point.x - points_xyz_c.x, hit_point.y - points_xyz_c.y, hit_point.z - points_xyz_c.z};
			float angle_distance = abs(pix_ray.x * gaussian_normal_c.x +
									   pix_ray.y * gaussian_normal_c.y +
									   pix_ray.z * gaussian_normal_c.z);
			float depth_distance = abs(hit_point.z - points_xyz_c.z);
			if (!hit_gaussian && alpha >= opaque_threshold)
			{
				opaque_depth = depths[collected_id[j]];
				hit_gaussian_id = collected_id[j];
				bool opaque = true;
				hit_depth_weight = alpha * T;
				if (depth_distance <= hit_normal_scale.w * hit_depth_threshold && angle_distance >= hit_normal_threshold)
				{
					depth_ = t * pix_ray.z;
					opaque = false;
				}
				else
				{
					depth_ = opaque_depth;
				}
				hit_normal_c[pix_id] = gaussian_normal_c;
				hit_point_c[pix_id] = hit_point;
				hit_gaussian = true;
			}

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f && hit_gaussian)
			{
				done = true;
				continue;
			}
			if (test_T >= 0.0001f)
			{
				const float color_weight = alpha * T;
				// Eq. (3) from 3D Gaussian splatting paper.
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * color_weight;
				if (color_weight > color_weight_max)
				{
					color_weight_max = color_weight;
					hit_gaussian_color_id = gaussian_id;
					hit_color_weight = color_weight_max;
				}

				// Keep track of last range entry to update this
				// pixel.
				last_contributor = contributor;
				end_T = test_T;
			}
			T = test_T;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside && render)
	{
		final_T[pix_id] = end_T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = depth_;
		out_hit_depth[pix_id] = hit_gaussian_id;
		out_hit_color[pix_id] = hit_gaussian_color_id;
		out_hit_color_weight[pix_id] = hit_color_weight;
		out_hit_depth_weight[pix_id] = hit_depth_weight;
		out_T[pix_id] = end_T;
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA_withMask(
		const int *tile_indices,
		const int tile_width,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int W, int H,
		const float fx, const float fy,
		const float cx, const float cy,
		const float scale_modify,
		const float *viewmatrix,
		const float *__restrict__ points_xyz_world,
		const float2 *__restrict__ points_xy_image,
		const float *__restrict__ features,
		const float *__restrict__ depths,
		const glm::vec3 *__restrict__ scales,
		const glm::vec4 *__restrict__ rotations,
		const float4 *__restrict__ conic_opacity,
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		const float *__restrict__ bg_color,
		const float opaque_threshold,
		const float hit_depth_threshold,
		const float hit_normal_threshold,
		const float T_threshold,
		float3 *__restrict__ hit_normal_c,
		float3 *__restrict__ hit_point_c,
		float *__restrict__ out_color,
		float *__restrict__ out_depth,
		int *__restrict__ out_hit_depth,
		int *__restrict__ out_hit_color,
		float *__restrict__ out_hit_color_weight,
		float *__restrict__ out_hit_depth_weight,
		float *__restrict__ out_T,
		float *__restrict__ out_weight_sum,
		int *__restrict__ n_touched)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const int block_id = block.group_index().x;
	const int real_tile = tile_indices[block_id];
	const int real_tile_x = real_tile % tile_width;
	const int real_tile_y = real_tile / tile_width;
	// uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	// uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	// uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	// uint32_t pix_id = W * pix.y + pix.x;
	// float2 pixf = {(float)pix.x, (float)pix.y};

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = {real_tile_x * BLOCK_X, real_tile_y * BLOCK_Y};
	uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	float3 pix_ray = ndc2ray(pix, fx, fy, cx, cy);
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	bool render = true;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	uint2 range = ranges[real_tile];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	float end_T = T;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = {0};
	float depth_ = 0;
	bool hit_gaussian = false;
	bool hit_color_end = false;
	float opaque_depth = 0;
	float3 hit_normal{0, 0, 0};
	float hit_scale = 0;
	int hit_gaussian_id = -1;
	int hit_gaussian_color_id = -1;
	float color_weight_max = -1;
	float hit_color_weight = 0.0;
	float hit_depth_weight = 0.0;
	float weight_sum = 0.0f;
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && render && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			const int gaussian_id = collected_id[j];
			const int gaussian_id3 = collected_id[j] * 3;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			// t = torch.sum(hit_normals * hit_points, dim=1) / (torch.sum(
			//     hit_normals * pixel_rays, dim=1
			// ) + 1e-8)
			// check hit gaussian

			float4 hit_normal_scale;
			computeNormal_ScaleMax(scales[gaussian_id], scale_modify, rotations[gaussian_id], &hit_normal_scale);
			float3 points_xyz_w{points_xyz_world[gaussian_id3], points_xyz_world[gaussian_id3 + 1], points_xyz_world[gaussian_id3 + 2]};
			float3 gaussian_normal_c = transformVec4x3(float3{hit_normal_scale.x, hit_normal_scale.y, hit_normal_scale.z}, viewmatrix);
			float3 points_xyz_c = transformPoint4x3(points_xyz_w, viewmatrix);
			float t = (points_xyz_c.x * gaussian_normal_c.x + points_xyz_c.y * gaussian_normal_c.y + points_xyz_c.z * gaussian_normal_c.z) /
					  (pix_ray.x * gaussian_normal_c.x + pix_ray.y * gaussian_normal_c.y + pix_ray.z * gaussian_normal_c.z + 1e-8);
			float3 hit_point{t * pix_ray.x, t * pix_ray.y, t * pix_ray.z};
			float3 hit_points_delta{hit_point.x - points_xyz_c.x, hit_point.y - points_xyz_c.y, hit_point.z - points_xyz_c.z};
			float angle_distance = abs(pix_ray.x * gaussian_normal_c.x +
									   pix_ray.y * gaussian_normal_c.y +
									   pix_ray.z * gaussian_normal_c.z);
			float depth_distance = abs(hit_point.z - points_xyz_c.z);
			if (!hit_gaussian && alpha >= opaque_threshold)
			{
				opaque_depth = depths[collected_id[j]];
				hit_gaussian_id = collected_id[j];
				bool opaque = true;
				hit_depth_weight = alpha * T;
				if (depth_distance <= hit_normal_scale.w * hit_depth_threshold && angle_distance >= hit_normal_threshold)
				{
					depth_ = t * pix_ray.z;
					opaque = false;
				}
				else
				{
					depth_ = opaque_depth;
				}
				hit_normal_c[pix_id] = gaussian_normal_c;
				hit_point_c[pix_id] = hit_point;
				hit_gaussian = true;
			}

			float test_T = T * (1 - alpha);
			if (test_T < T_threshold && hit_gaussian)
			{
				done = true;
				continue;
			}
			if (test_T >= T_threshold)
			{
				const float color_weight = alpha * T;
				weight_sum += color_weight;
				// Eq. (3) from 3D Gaussian splatting paper.
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * color_weight;
				if (color_weight > color_weight_max)
				{
					color_weight_max = color_weight;
					hit_gaussian_color_id = gaussian_id;
					hit_color_weight = color_weight_max;
				}

				//?2024-12-12 增加对高斯的计数
				if(test_T > 0.5f){
					atomicAdd(&(n_touched[collected_id[j]]), 1);
				}
				// Keep track of last range entry to update this
				// pixel.
				last_contributor = contributor;
				end_T = test_T;
			}
			T = test_T;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside && render)
	{
		final_T[pix_id] = end_T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		// if (weight_sum > 0)
		// {
		// 	for (int ch = 0; ch < CHANNELS; ch++)
		// 		out_color[ch * H * W + pix_id] = C[ch] / weight_sum;
		// }
		out_depth[pix_id] = depth_;
		out_hit_depth[pix_id] = hit_gaussian_id;//对深度贡献最大的高斯id
		out_hit_color[pix_id] = hit_gaussian_color_id;//对该像素点贡献最大的高斯
		out_hit_color_weight[pix_id] = hit_color_weight;//对该像素点颜色贡献最大的高斯的贡献权重
		out_hit_depth_weight[pix_id] = hit_depth_weight;
		out_weight_sum[pix_id] = weight_sum;
		out_T[pix_id] = end_T;//保存光线穿过所有高斯点后剩余的透明度
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const float fx, const float fy,
	const float cx, const float cy,
	const float scale_modify,
	const float *viewmatrix,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float *means3D,
	const float2 *means2D,
	const float *colors,
	const float *depths,
	const float4 *conic_opacity,
	float *final_T,
	uint32_t *n_contrib,
	const float *bg_color,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float opaque_threshold,
	const float hit_depth_threshold,
	const float hit_normal_threshold,
	float3 *hit_normal_c,
	float3 *hit_point_c,
	float *out_color,
	float *out_depth,
	int *out_hit_depth,
	int *out_hit_color,
	float *out_hit_color_weight,
	float *out_hit_depth_weight,
	float *out_T)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		fx, fy,
		cx, cy, scale_modify,
		viewmatrix,
		means3D,
		means2D,
		colors,
		depths,
		scales,
		rotations,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		opaque_threshold,
		hit_depth_threshold,
		hit_normal_threshold,
		hit_normal_c,
		hit_point_c,
		out_color,
		out_depth,
		out_hit_depth,
		out_hit_color,
		out_hit_color_weight,
		out_hit_depth_weight,
		out_T);
}

void FORWARD::render_flat(
	const int *tile_indices,
	const int tile_num,
	const int tile_width,
	dim3 block,
	const float fx, const float fy,
	const float cx, const float cy,
	const float scale_modify,
	const float *viewmatrix,
	const uint2 *ranges,
	const uint32_t *point_list,
	int W, int H,
	const float *means3D,
	const float2 *means2D,
	const float *colors,
	const float *depths,
	const float4 *conic_opacity,
	float *final_T,
	uint32_t *n_contrib,
	const float *bg_color,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	const float opaque_threshold,
	const float hit_depth_threshold,
	const float hit_normal_threshold,
	const float T_threshold,
	float3 *hit_normal_c,
	float3 *hit_point_c,
	float *out_color,
	float *out_depth,
	int *out_hit_depth,
	int *out_hit_color,
	float *out_hit_color_weight,
	float *out_hit_depth_weight,
	float *out_T,
	float *out_weight_sum,
	int *n_touched)
{
	dim3 grid{tile_num, 1, 1};
	renderCUDA_withMask<NUM_CHANNELS><<<grid, block>>>(
		tile_indices,
		tile_width,
		ranges,
		point_list,
		W, H,
		fx, fy,
		cx, cy, scale_modify,
		viewmatrix,
		means3D,
		means2D,
		colors,
		depths,
		scales,
		rotations,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		opaque_threshold,
		hit_depth_threshold,
		hit_normal_threshold,
		T_threshold,
		hit_normal_c,
		hit_point_c,
		out_color,
		out_depth,
		out_hit_depth,
		out_hit_color,
		out_hit_color_weight,
		out_hit_depth_weight,
		out_T,
		out_weight_sum,
		n_touched);
}

void FORWARD::preprocess(int P, int D, int M,
						 const float color_sigma,
						 const float *means3D,
						 const glm::vec3 *scales,
						 const float scale_modifier,
						 const glm::vec4 *rotations,
						 const float *opacities,
						 const float *shs,
						 bool *clamped,
						 const float *cov3D_precomp,
						 const float *colors_precomp,
						 const float *viewmatrix,
						 const float *projmatrix,
						 const glm::vec3 *cam_pos,
						 const int *tile_mask,
						 const int W, const int H,
						 const float focal_x, const float focal_y,
						 const float tan_fovx, const float tan_fovy,
						 const float cx, const float cy,
						 int *radii,
						 float2 *means2D,
						 float *depths,
						 float *cov3Ds,
						 float *rgb,
						 float4 *conic_opacity,
						 const dim3 grid,
						 uint32_t *tiles_touched,
						 bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		color_sigma,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		tile_mask,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		cx, cy,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}