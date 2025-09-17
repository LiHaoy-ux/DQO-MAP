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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
					const glm::vec3 *cam_pos,
					const int *tile_mask,
					const int W, const int H,
					const float focal_x, const float focal_y,
					const float tan_fovx, const float tan_fovy,
					const float cx, const float cy,
					int *radii,
					float2 *points_xy_image,
					float *depths,
					float *cov3Ds,
					float *colors,
					float4 *conic_opacity,
					const dim3 grid,
					uint32_t *tiles_touched,
					bool prefiltered);

	// Main rasterization method.
	void render(
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
		float *out_T);

	void render_flat(
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
		int *n_touched);
}

#endif