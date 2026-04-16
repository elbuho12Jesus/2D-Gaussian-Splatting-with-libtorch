#include "ConfigCamera.h"
#include <cmath>

namespace camcfg {

float focal2fov(float focal, float pixels) {
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

torch::Tensor qvec_to_rotmat(double qw, double qx, double qy, double qz,
                             const torch::TensorOptions& opts) {
    const double r00 = 1.0 - 2.0*(qy*qy + qz*qz);
    const double r01 = 2.0*(qx*qy - qz*qw);
    const double r02 = 2.0*(qx*qz + qy*qw);

    const double r10 = 2.0*(qx*qy + qz*qw);
    const double r11 = 1.0 - 2.0*(qx*qx + qz*qz);
    const double r12 = 2.0*(qy*qz - qx*qw);

    const double r20 = 2.0*(qx*qz - qy*qw);
    const double r21 = 2.0*(qy*qz + qx*qw);
    const double r22 = 1.0 - 2.0*(qx*qx + qy*qy);

    torch::Tensor R = torch::empty({3, 3}, opts);
    R.index_put_({0,0}, (float)r00); R.index_put_({0,1}, (float)r01); R.index_put_({0,2}, (float)r02);
    R.index_put_({1,0}, (float)r10); R.index_put_({1,1}, (float)r11); R.index_put_({1,2}, (float)r12);
    R.index_put_({2,0}, (float)r20); R.index_put_({2,1}, (float)r21); R.index_put_({2,2}, (float)r22);
    return R;
}

// Replica EXACTA de graphics_utils.getWorld2View2(...) y luego cameras.py .transpose(0,1)
torch::Tensor make_world_view_transform_like_repo(
    const torch::Tensor& R,            // [3,3] (CUDA)
    const torch::Tensor& t,            // [3]   (CUDA)
    const torch::Tensor& translate,    // [3]   (CUDA)
    float scale,
    const torch::TensorOptions& opts
) {
    // Rt (4x4)
    torch::Tensor Rt = torch::eye(4, opts);

    // Rt[:3,:3] = R^T
    Rt.slice(0, 0, 3).slice(1, 0, 3).copy_(R.transpose(0, 1));

    // Rt[:3,3] = t
    Rt.slice(0, 0, 3).select(1, 3).copy_(t);

    // C2W = inv(Rt)
    torch::Tensor C2W = torch::linalg_inv(Rt);

    // cam_center = C2W[:3,3]
    torch::Tensor cam_center = C2W.slice(0, 0, 3).select(1, 3);

    // cam_center = (cam_center + translate) * scale
    cam_center = (cam_center + translate) * scale;

    // C2W[:3,3] = cam_center
    C2W.slice(0, 0, 3).select(1, 3).copy_(cam_center);

    // Rt = inv(C2W)
    Rt = torch::linalg_inv(C2W);

    // cameras.py hace transpose(0,1)
    return Rt.transpose(0, 1).contiguous();
}

// Replica EXACTA de graphics_utils.getProjectionMatrix(...) y luego cameras.py .transpose(0,1)
torch::Tensor make_projection_matrix_like_repo(
    float znear, float zfar, float fovX, float fovY,
    const torch::TensorOptions& opts
) {
    float tanHalfFovY = std::tan(fovY * 0.5f);
    float tanHalfFovX = std::tan(fovX * 0.5f);

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4, 4}, opts);
    float z_sign = 1.0f;

    P.index_put_({0, 0}, 2.0f * znear / (right - left));
    P.index_put_({1, 1}, 2.0f * znear / (top - bottom));
    P.index_put_({0, 2}, (right + left) / (right - left));
    P.index_put_({1, 2}, (top + bottom) / (top - bottom));
    P.index_put_({3, 2}, z_sign);
    P.index_put_({2, 2}, z_sign * zfar / (zfar - znear));
    P.index_put_({2, 3}, -(zfar * znear) / (zfar - znear));

    return P.transpose(0, 1).contiguous();
}

} // namespace camcfg