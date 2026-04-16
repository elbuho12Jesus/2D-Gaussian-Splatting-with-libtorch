#pragma once
#include <torch/torch.h>

namespace camcfg {

float focal2fov(float focal, float pixels);
torch::Tensor qvec_to_rotmat(double qw, double qx, double qy, double qz,
                             const torch::TensorOptions& opts);

torch::Tensor make_world_view_transform_like_repo(
    const torch::Tensor& R,
    const torch::Tensor& t,
    const torch::Tensor& translate,
    float scale,
    const torch::TensorOptions& opts
);

torch::Tensor make_projection_matrix_like_repo(
    float znear, float zfar, float fovX, float fovY,
    const torch::TensorOptions& opts
);

} // namespace camcfg