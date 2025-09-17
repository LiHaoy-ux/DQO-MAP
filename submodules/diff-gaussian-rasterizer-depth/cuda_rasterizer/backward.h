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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
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
		float3 *dL_dmean2D,
		float4 *dL_dconic2D,
		float *dL_dopacity,
		float *dL_dcolors,
		float *dL_dmeans3D,
		float *dL_drotation);

	void render_flat(
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
		float *dL_drotation);

	void preprocess(
		int P, int D, int M,
		const float3 *means,
		const int *radii,
		const float *shs,
		const bool *clamped,
		const glm::vec3 *scales,
		const glm::vec4 *rotations,
		const float scale_modifier,
		const float *cov3Ds,
		const float *view,
		const float *proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3 *campos,
		const float3 *dL_dmean2D,
		const float *dL_dconics,
		glm::vec3 *dL_dmeans,
		float *dL_dcolor,
		float *dL_dcov3D,
		float *dL_dsh,
		glm::vec3 *dL_dscale,
		glm::vec4 *dL_drot);
}

#endif