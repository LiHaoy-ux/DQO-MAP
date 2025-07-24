#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_utils.h"
#include <fstream>
#include <string>
#include <functional>

#include "map_process.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
accumulate_gaussian_error(
    const int H, const int W, const int P,
    const torch::Tensor &screen_color_error,
    const torch::Tensor &screen_depth_error,
    const torch::Tensor &screen_normal_error,
    const torch::Tensor &screen_color_index,
    const torch::Tensor &screen_depth_index,
    const float color_threshold,
    const float depth_threshold,
    const float normal_threshold,
    const bool check_max)
{
    auto float_opts = screen_color_error.options().dtype(torch::kFloat32);
    auto int_opts = screen_color_error.options().dtype(torch::kInt32);
    torch::Tensor gs_color_error = torch::full({P, 1}, 0.0, float_opts);
    torch::Tensor gs_depth_error = torch::full({P, 1}, 0.0, float_opts);
    torch::Tensor gs_normal_error = torch::full({P, 1}, 0.0, float_opts);

    torch::Tensor gs_color_counter = torch::full({P, 1}, 0, int_opts);
    torch::Tensor gs_depth_counter = torch::full({P, 1}, 0, int_opts);
    torch::Tensor gs_normal_counter = torch::full({P, 1}, 0, int_opts);
    torch::Tensor gs_rescale_counter = torch::full({P, 1}, 0.0, float_opts);
    accumulate_gaussian_error_impl(
        screen_color_error.contiguous().data<float>(),
        screen_depth_error.contiguous().data<float>(),
        screen_normal_error.contiguous().data<float>(),
        screen_color_index.contiguous().data<int>(),
        screen_depth_index.contiguous().data<int>(),
        color_threshold,
        depth_threshold,
        normal_threshold,
        W, H, P,
        gs_color_error.contiguous().data<float>(),
        gs_depth_error.contiguous().data<float>(),
        gs_normal_error.contiguous().data<float>(),
        gs_color_counter.contiguous().data<int>(),
        gs_depth_counter.contiguous().data<int>(),
        gs_normal_counter.contiguous().data<int>(),
        gs_rescale_counter.contiguous().data<float>(),
        check_max);
    return std::make_tuple(gs_color_error, gs_depth_error, gs_normal_error,
                           gs_rescale_counter);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
accumulate_gaussian_confidence(
    const int H, const int W, const int P,
    const torch::Tensor &gaussian_index_map,
    const torch::Tensor &gaussian_confidence_map)
{
    auto float_opts = gaussian_confidence_map.options().dtype(torch::kFloat32);
    auto int_opts = gaussian_index_map.options().dtype(torch::kInt32);
    torch::Tensor gs_confidence_mean = torch::full({P, 1}, 0.0, float_opts);
    torch::Tensor gs_confidence_max = torch::full({P, 1}, std::numeric_limits<float>::lowest(), float_opts);
    torch::Tensor gs_confidence_min = torch::full({P, 1}, std::numeric_limits<float>::max(), float_opts);
    torch::Tensor gs_confidence_counter = torch::full({P, 1}, 0, int_opts);

    accumulate_gaussian_confidence_impl(H, W, P,
                                        gaussian_confidence_map.contiguous().data<float>(),
                                        gaussian_index_map.contiguous().data<int>(),
                                        gs_confidence_counter.contiguous().data<int>(),
                                        gs_confidence_max.contiguous().data<float>(),
                                        gs_confidence_min.contiguous().data<float>(),
                                        gs_confidence_mean.contiguous().data<float>());
    return std::make_tuple(gs_confidence_max, gs_confidence_min, gs_confidence_mean);
}