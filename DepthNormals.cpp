#include "DepthNormals.h"
#include <torch/torch.h>

namespace depthnorm {

static at::Tensor make_ndc2pix(int W, int H, const at::TensorOptions& opts) {
    at::Tensor M = torch::tensor(
        {
            {(float)W / 2.0f, 0.0f,            0.0f,            (float)W / 2.0f},
            {0.0f,            (float)H / 2.0f, 0.0f,            (float)H / 2.0f},
            {0.0f,            0.0f,            0.0f,            1.0f}
        },
        opts
    );
    return M.transpose(0, 1).contiguous(); // [4,3]
}

at::Tensor depths_to_points_like_repo(
    const at::Tensor& world_view_transform,
    const at::Tensor& full_proj_transform,
    int W,
    int H,
    const at::Tensor& depth_1hw
) {
    auto opts = depth_1hw.options();

    at::Tensor c2w = torch::linalg_inv(world_view_transform.transpose(0, 1)).contiguous(); // [4,4]
    at::Tensor ndc2pix = make_ndc2pix(W, H, opts); // [4,3]

    at::Tensor projection_matrix = c2w.transpose(0, 1).matmul(full_proj_transform).contiguous(); // [4,4]

    at::Tensor intrins = projection_matrix.matmul(ndc2pix)
        .slice(0, 0, 3).slice(1, 0, 3)
        .transpose(0, 1)
        .contiguous(); // [3,3]

    // ---------------------------
    // FIX: construir grid_x/grid_y como en indexing='xy'
    // grid_x[v,u] = u, grid_y[v,u] = v, ambos [H,W]
    // ---------------------------
    at::Tensor xs = torch::arange(W, opts).to(opts.dtype()); // [W]
    at::Tensor ys = torch::arange(H, opts).to(opts.dtype()); // [H]

    at::Tensor grid_x = xs.view({1, W}).expand({H, W}).contiguous(); // [H,W]
    at::Tensor grid_y = ys.view({H, 1}).expand({H, W}).contiguous(); // [H,W]

    at::Tensor ones = torch::ones({H, W}, opts);
    at::Tensor pix = torch::stack({grid_x, grid_y, ones}, -1).view({-1, 3}); // [H*W,3]

    at::Tensor intrins_inv_T = torch::linalg_inv(intrins).transpose(0, 1).contiguous(); // [3,3]
    at::Tensor c2w_R_T = c2w.slice(0, 0, 3).slice(1, 0, 3).transpose(0, 1).contiguous(); // [3,3]

    at::Tensor rays_d = pix.matmul(intrins_inv_T).matmul(c2w_R_T); // [H*W,3]
    at::Tensor rays_o = c2w.slice(0, 0, 3).select(1, 3).contiguous(); // [3]

    at::Tensor depth_flat = depth_1hw.view({-1, 1}); // [H*W,1]
    at::Tensor points_world = depth_flat * rays_d + rays_o.view({1, 3}); // [H*W,3]

    return points_world;
}

at::Tensor depth_to_normal_like_repo(
    const at::Tensor& world_view_transform,
    const at::Tensor& full_proj_transform,
    int W,
    int H,
    const at::Tensor& depth_1hw,
    float eps
) {
    at::Tensor points = depths_to_points_like_repo(world_view_transform, full_proj_transform, W, H, depth_1hw)
        .view({H, W, 3}); // [H,W,3]

    at::Tensor output = torch::zeros_like(points);

    at::Tensor dx =
        points.index({torch::indexing::Slice(2, H), torch::indexing::Slice(1, W - 1), torch::indexing::Slice()}) -
        points.index({torch::indexing::Slice(0, H - 2), torch::indexing::Slice(1, W - 1), torch::indexing::Slice()});

    at::Tensor dy =
        points.index({torch::indexing::Slice(1, H - 1), torch::indexing::Slice(2, W), torch::indexing::Slice()}) -
        points.index({torch::indexing::Slice(1, H - 1), torch::indexing::Slice(0, W - 2), torch::indexing::Slice()});

    at::Tensor n = torch::cross(dx, dy, -1);
    at::Tensor norm = torch::sqrt((n * n).sum(-1, true) + eps);
    at::Tensor normal_map = n / norm;

    output.index_put_(
        {torch::indexing::Slice(1, H - 1), torch::indexing::Slice(1, W - 1), torch::indexing::Slice()},
        normal_map
    );

    return output;
}

} // namespace depthnorm