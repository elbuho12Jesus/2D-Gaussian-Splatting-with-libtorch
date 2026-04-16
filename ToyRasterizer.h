#pragma once

#include <torch/torch.h>

// Differentiable toy rasterizer (no CUDA custom kernels).
// Projects 3D points to 2D with a pinhole camera and splats each point with a
// small Gaussian footprint in pixel space.
//
// Input:
//  - xyz: (N,3) float32 on CPU or CUDA (requires_grad allowed)
//  - rgb: (N,3) float32 in [0,1] on same device as xyz (requires_grad allowed)
//
// Output:
//  - image: (3,H,W) float32 on same device
//
// Notes:
//  - This is O(N * K^2) where K=(2*radius+1). Keep H,W small and radius small.
//  - We avoid hard indexing based on rounded pixels to preserve gradients.

torch::Tensor render_toy_gaussian_splat(
    const torch::Tensor& xyz,
    const torch::Tensor& rgb,
    int H,
    int W,
    float fx,
    float fy,
    float cx,
    float cy,
    float sigma_px,
    int radius_px = 2,
    float eps = 1e-8f
);