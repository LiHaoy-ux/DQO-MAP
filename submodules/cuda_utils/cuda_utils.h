#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

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
    const bool check_max);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
accumulate_gaussian_confidence(
    const int H, const int W, const int P,
    const torch::Tensor &gaussian_index_map,
    const torch::Tensor &gaussian_confidence_map);
