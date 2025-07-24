#include "map_process.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__ void acuumulate_error_preprocessCUDA(
    const float *__restrict__ screen_color_error,
    const float *__restrict__ screen_depth_error,
    const float *__restrict__ screen_normal_error,
    const int *__restrict__ screen_color_index,
    const int *__restrict__ screen_depth_index,
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
    const bool check_max)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    bool inside = pix.x < W && pix.y < H;
    if (!inside)
    {
        return;
    }
    const float color_error = screen_color_error[pix_id];
    const float depth_error = screen_depth_error[pix_id];
    const float normal_error = screen_normal_error[pix_id];
    const int color_index = screen_color_index[pix_id];
    const int depth_index = screen_depth_index[pix_id];
    if (color_index >= 0 && color_index < P)
    {
        if (check_max)
        {
            atomicMax(&(gs_color_error[color_index]), color_error);
        }
        else
        {
            atomicAdd(&(gs_color_error[color_index]), color_error);
        }
        atomicAdd(&(color_error_counter[color_index]), 1);
        if (color_error > color_threshold)
        {
            atomicAdd(&(rescale_counter[color_index]), 1.0);
        }
    }
    if (depth_index >= 0 && depth_index < P)
    {
        if (check_max)
        {
            atomicMax(&(gs_depth_error[depth_index]), depth_error);
            atomicMax(&(gs_normal_error[depth_index]), normal_error);
        }
        else
        {
            atomicAdd(&(gs_depth_error[depth_index]), depth_error);
            atomicAdd(&(gs_normal_error[depth_index]), normal_error);
        }
        atomicAdd(&(depth_error_counter[depth_index]), 1);
        atomicAdd(&(normal_error_counter[depth_index]), 1);
        if (depth_error > depth_threshold)
        {
            atomicAdd(&(rescale_counter[depth_index]), 1.0);
        }
        if (normal_error > normal_threshold)
        {
            atomicAdd(&(rescale_counter[depth_index]), 1.0);
        }
    }
}

void accumulate_error_preprocess(
    dim3 tile_grid, dim3 block,
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
    const bool check_max)
{
    acuumulate_error_preprocessCUDA<<<tile_grid, block>>>(
        screen_color_error, screen_depth_error, screen_normal_error,
        screen_color_index, screen_depth_index,
        color_threshold, depth_threshold, normal_threshold,
        W, H, P,
        gs_color_error,
        gs_depth_error,
        gs_normal_error,
        color_error_counter,
        depth_error_counter,
        normal_error_counter,
        rescale_counter,
        check_max);
}

__global__ void accumulate_error_meanCUDA(
    const int P,
    const int *__restrict__ gs_color_counter,
    const int *__restrict__ gs_depth_counter,
    const int *__restrict__ gs_normal_counter,
    float *gs_color_error,
    float *gs_depth_error,
    float *gs_normal_error)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    const int color_counter = gs_color_counter[idx];
    const int depth_counter = gs_depth_counter[idx];
    const int normal_counter = gs_normal_counter[idx];
    if (color_counter > 0)
    {
        gs_color_error[idx] = gs_color_error[idx] / color_counter;
    }
    if (depth_counter > 0)
    {
        gs_depth_error[idx] = gs_depth_error[idx] / depth_counter;
    }
    if (normal_counter > 0)
    {
        gs_normal_error[idx] = gs_normal_error[idx] / normal_counter;
    }
}

void accumulate_error_mean(
    const int P,
    const int *gs_color_counter,
    const int *gs_depth_counter,
    const int *gs_normal_counter,
    float *gs_color_error,
    float *gs_depth_error,
    float *gs_normal_error)
{
    accumulate_error_meanCUDA<<<(P + 255) / 256, 256>>>(P,
                                                        gs_color_counter,
                                                        gs_depth_counter,
                                                        gs_normal_counter,
                                                        gs_color_error,
                                                        gs_depth_error,
                                                        gs_normal_error);
}

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
    const bool check_max)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // acuumulate error sum
    accumulate_error_preprocess(tile_grid, block,
                                screen_color_error,
                                screen_depth_error,
                                screen_normal_error,
                                screen_color_index,
                                screen_depth_index,
                                color_threshold,
                                depth_threshold,
                                normal_threshold,
                                W, H, P,
                                gs_color_error,
                                gs_depth_error,
                                gs_normal_error,
                                color_error_counter,
                                depth_error_counter,
                                normal_error_counter,
                                rescale_counter, check_max);
    if (!check_max)
    {
        // calculate error mean
        accumulate_error_mean(P,
                              color_error_counter,
                              depth_error_counter,
                              normal_error_counter,
                              gs_color_error,
                              gs_depth_error,
                              gs_normal_error);
    }
}

__global__ void accumulate_confidence_preprocessCUDA(
    const int H, const int W, const int P,
    const float *__restrict__ confidence_map,
    const int *__restrict__ gs_index_map,
    int *gs_confidence_counter,
    float *gs_confidence_max,
    float *gs_confidence_min,
    float *gs_confidence_mean)
{
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    bool inside = pix.x < W && pix.y < H;
    if (!inside)
    {
        return;
    }
    const float gs_confidence = confidence_map[pix_id];
    const int gs_index = gs_index_map[pix_id];
    if (gs_index < 0)
    {
        return;
    }
    if (gs_index < P)
    {
        atomicAdd(&(gs_confidence_counter[gs_index]), 1);
        atomicAdd(&(gs_confidence_mean[gs_index]), gs_confidence);
        atomicMax(&(gs_confidence_max[gs_index]), gs_confidence);
        atomicMin(&(gs_confidence_min[gs_index]), gs_confidence);
    }
}

void accumulate_confidence_preprocess(
    dim3 tile_grid, dim3 block,
    const int H, const int W, const int P,
    const float *confidence_map,
    const int *gs_index_map,
    int *gs_confidence_counter,
    float *gs_confidence_max,
    float *gs_confidence_min,
    float *gs_confidence_mean)
{
    accumulate_confidence_preprocessCUDA<<<tile_grid, block>>>(
        H, W, P,
        confidence_map, gs_index_map, gs_confidence_counter,
        gs_confidence_max, gs_confidence_min, gs_confidence_mean);
}

__global__ void accumulate_confidence_meanCUDA(
    const int P,
    const int *__restrict__ gs_confidence_counter,
    float *__restrict__ gs_confidence_max,
    float *__restrict__ gs_confidence_min,
    float *gs_confidence_mean)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    int gs_counter = gs_confidence_counter[idx];
    if (gs_counter > 0)
    {
        gs_confidence_mean[idx] = gs_confidence_mean[idx] / gs_counter;
    }
    else
    {
        gs_confidence_mean[idx] = 0;
        gs_confidence_max[idx] = 0;
        gs_confidence_min[idx] = 0;
    }
}

void accumulate_confidence_mean(
    const int P,
    const int *gs_confidence_counter,
    float *gs_confidence_mean,
    float *gs_confidence_max,
    float *gs_confidence_min)
{
    accumulate_confidence_meanCUDA<<<(P + 255) / 256, 256>>>(P,
                                                             gs_confidence_counter,
                                                             gs_confidence_max,
                                                             gs_confidence_min,
                                                             gs_confidence_mean);
}

void accumulate_gaussian_confidence_impl(
    const int H, const int W, const int P,
    const float *confidence_map,
    const int *gs_index_map,
    int *gs_confidence_counter,
    float *gs_confidence_max,
    float *gs_confidence_min,
    float *gs_confidence_mean)
{
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // acuumulate confidence sum
    accumulate_confidence_preprocess(tile_grid, block,
                                     H, W, P,
                                     confidence_map, gs_index_map,
                                     gs_confidence_counter,
                                     gs_confidence_max,
                                     gs_confidence_min,
                                     gs_confidence_mean);

    accumulate_confidence_mean(P, gs_confidence_counter, gs_confidence_mean, gs_confidence_max,
                               gs_confidence_min);
}