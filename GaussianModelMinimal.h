#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ParceBinaries.h" // ColmapPoints3DMinimal

// Minimal GaussianModel-like container for LibTorch.
// Focus: xyz (N x 3) and features_dc (N x 3) as trainable parameters.
// This is intentionally minimal: no SH rest, no opacity/scale/rotation yet.
//
// LibTorch version target: 2.11.0+cu130 (win-shared-with-deps-debug)
// Notes:
// - OptimizerParamGroup in this version expects unique_ptr<OptimizerOptions>.
// - Avoid including this header from .cu files compiled with NVCC.

struct GaussianModelMinimal {
    torch::Device device{torch::kCPU};

    // Trainable parameters
    torch::Tensor xyz;         // (N, 3) float32
    torch::Tensor features_dc; // (N, 3) float32 in [0, 1]

    // Optimizer (created after init)
    std::unique_ptr<torch::optim::Adam> optimizer;

    // Initialize from COLMAP points3D (xyz in doubles, rgb in uint8)
    // lr_xyz and lr_color are separate learning rates.
    void init_from_colmap_points(const ColmapPoints3DMinimal& pts,
                                 const torch::Device& dev,
                                 double lr_xyz = 1e-2,
                                 double lr_color = 1e-2);

    int64_t num_points() const;
};