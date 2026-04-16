#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ParceBinaries.h" // ColmapPoints3DMinimal

// Stage-1 Gaussian Model for LibTorch:
// - xyz (N,3) trainable
// - features_dc (N,3) trainable
// - opacity_logits (N,1) trainable (sigmoid in forward)
// - scaling_log (N,2) trainable (exp in forward)  [2DGS-style anisotropic in image plane]
// - rotation (N,4) trainable (normalized quaternion)
//
// Design goal: keep scale initialization swappable.

struct ScaleInitConfig {
    // Option A: constant scale (used as fallback)
    float constant_scale = 0.01f;

    // Option B: CPU KNN approximate (nearest neighbor). If N is large, consider using a spatial index.
    // If max_points_for_bruteforce < N, we fallback to constant to avoid O(N^2) time.
    int64_t max_points_for_bruteforce = 200000; // safe-ish limit, tune for your machine

    // Numerical stability
    float min_dist2 = 1e-12f;

    // Mapping from neighbor distance to sigma:
    // scaling = sqrt(dist2) * scale_multiplier
    float scale_multiplier = 1.0f;
};

// --- Scale init strategies (swappable) ---

// Option A: constant scaling for all points (returns N x 2 scaling in world units).
torch::Tensor init_scale_constant(int64_t N, float constant_scale, torch::Device device);

// Option B: nearest-neighbor distance in CPU (bruteforce) -> scaling (N x 2).
// Input xyz_cpu_f32: (N,3) float32 CPU tensor.
// Returns CPU tensor (N,2) float32.
torch::Tensor init_scale_knn_cpu_bruteforce(const torch::Tensor& xyz_cpu_f32,
                                           const ScaleInitConfig& cfg);

struct GaussianModelStage1 {
    torch::Device device{torch::kCPU};

    // Trainable parameters
    torch::Tensor xyz;            // (N,3) float32
    torch::Tensor features_dc;    // (N,3) float32 in [0,1]
    torch::Tensor opacity_logits; // (N,1) float32 (sigmoid -> opacity)
    torch::Tensor scaling_log;    // (N,2) float32 (exp -> scaling)
    torch::Tensor rotation;       // (N,4) float32 normalized (quaternion)

    std::unique_ptr<torch::optim::Adam> optimizer;

    void init_from_colmap_points(const ColmapPoints3DMinimal& pts,
                                 const torch::Device& dev,
                                 const ScaleInitConfig& scale_cfg = {},
                                 double lr_xyz = 1e-2,
                                 double lr_color = 1e-2,
                                 double lr_opacity = 1e-2,
                                 double lr_scaling = 1e-2,
                                 double lr_rotation = 1e-3);

    void save_gaussians_ply_ascii(const std::string& filename) const;

    int64_t num_points() const;

    // Convenience getters (apply activations)
    torch::Tensor get_opacity() const; // sigmoid(opacity_logits)
    torch::Tensor get_scaling() const; // exp(scaling_log)
    torch::Tensor get_rotation() const; // normalized quaternion
};