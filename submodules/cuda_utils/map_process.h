#ifndef CUDA_UTILS_MAP_PROCESS_H
#define CUDA_UTILS_MAP_PROCESS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_X 16
#define BLOCK_Y 16

void accumulate_gaussian_error_impl(
    const float *screen_color_error,
    const float *screen_depth_error,
    const float *screen_normal_error,
    const int *screen_color_index,
    const int *screen_depth_index,
    const float color_threshold,
    const float depth_threshold,
    const float normal_threshold,
    const int W, const int H, const int P,
    float *gs_color_error,
    float *gs_depth_error,
    float *gs_normal_error,
    int *color_error_counter,
    int *depth_error_counter,
    int *normal_error_counter,
    float *rescale_counter,
    const bool check_max);

void accumulate_gaussian_confidence_impl(
    const int H, const int W, const int P,
    const float *confidence_map,
    const int *gs_index_map,
    int *gs_confidence_counter,
    float *gs_confidence_max,
    float *gs_confidence_min,
    float *gs_confidence_mean);

#endif