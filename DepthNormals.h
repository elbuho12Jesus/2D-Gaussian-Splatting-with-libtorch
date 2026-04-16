#pragma once

#include <torch/torch.h>

namespace depthnorm {

// Returns world-space points for each pixel as a flattened tensor [H*W, 3].
// This follows utils/point_utils.py::depths_to_points() exactly.
//
// Inputs:
// - world_view_transform: [4,4] CUDA float32 (same as view.world_view_transform in the repo)
// - full_proj_transform : [4,4] CUDA float32 (same as view.full_proj_transform in the repo)
// - W, H               : image width/height (pixels)
// - depth_1hw          : [1,H,W] CUDA float32 depth map
//
// Output:
// - points_world_flat  : [H*W,3] CUDA float32
at::Tensor depths_to_points_like_repo(
    const at::Tensor& world_view_transform,
    const at::Tensor& full_proj_transform,
    int W,
    int H,
    const at::Tensor& depth_1hw);

// Returns world-space normal map [H,W,3].
// This follows utils/point_utils.py::depth_to_normal() exactly.
//
// Inputs:
// - world_view_transform: [4,4] CUDA float32
// - full_proj_transform : [4,4] CUDA float32
// - W, H               : image width/height (pixels)
// - depth_1hw          : [1,H,W] CUDA float32 depth map
// - eps                : small epsilon to stabilize normalization
//
// Output:
// - normal_hw3         : [H,W,3] CUDA float32, zeros on border pixels
at::Tensor depth_to_normal_like_repo(
    const at::Tensor& world_view_transform,
    const at::Tensor& full_proj_transform,
    int W,
    int H,
    const at::Tensor& depth_1hw,
    float eps = 1e-8f);

} // namespace depthnorm