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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static void markVisible(
			int P,
			float *means3D,
			float *viewmatrix,
			float *projmatrix,
			bool *present);

		static int forward(
			std::function<char *(size_t)> geometryBuffer,
			std::function<char *(size_t)> binningBuffer,
			std::function<char *(size_t)> imageBuffer,
			const int P, int D, int M,
			const float color_sigma,
			const float *background,
			const int width, int height,
			const float cx, const float cy,
			const float *means3D,
			const float *shs,
			const float *colors_precomp,
			const float *opacities,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const float *cov3D_precomp,
			const float *viewmatrix,
			const float *projmatrix,
			const float *cam_pos,
			const int *render_mask,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			const float opaque_threshold,
			const float hit_depth_threshold,
			const float hit_normal_threshold,
			const float T_threshold,
			int *tile_indices,
			float *out_color,
			float *out_depth,
			int *out_hit_depth,
			int *out_hit_color,
			float *out_hit_color_weight,
			float *out_hit_depth_weight,
			float *out_T,
			int &tile_num,
			int *radii = nullptr,
			int *n_touched=nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const int tile_num,
			const int *tile_indices,
			const float *background,
			const int width, int height,
			const float cx, const float cy,
			const float *means3D,
			const float *shs,
			const float *colors_precomp,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const float *cov3D_precomp,
			const float *viewmatrix,
			const float *projmatrix,
			const float *campos,
			const float tan_fovx, float tan_fovy,
			const float normal_threshold, const float depth_threshold,
			const int *radii,
			char *geom_buffer,
			char *binning_buffer,
			char *image_buffer,
			const float *dL_dpix,
			const float *dL_dpix_depth,
			const int *hit_image,
			float *dL_dmean2D,
			float *dL_dconic,
			float *dL_dopacity,
			float *dL_dcolor,
			float *dL_dmean3D,
			float *dL_dcov3D,
			float *dL_dsh,
			float *dL_dscale,
			float *dL_drot,
			bool debug);
	};
};

#endif