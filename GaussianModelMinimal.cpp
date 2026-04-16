#include "GaussianModelMinimal.h"

void GaussianModelMinimal::init_from_colmap_points(const ColmapPoints3DMinimal& pts,
                                                   const torch::Device& dev,
                                                   double lr_xyz,
                                                   double lr_color) {
    const int64_t N = static_cast<int64_t>(pts.error.size());
    if (N <= 0) throw std::runtime_error("points3D: N == 0");
    if (static_cast<int64_t>(pts.xyz.size()) != N * 3) throw std::runtime_error("points3D: xyz size mismatch");
    if (static_cast<int64_t>(pts.rgb.size()) != N * 3) throw std::runtime_error("points3D: rgb size mismatch");

    device = dev;

    // xyz: double -> float32
    auto xyz_cpu = torch::from_blob((void*)pts.xyz.data(), {N, 3}, torch::kFloat64)
                       .clone()
                       .to(torch::kFloat32);

    // rgb: uint8 -> float32 in [0,1]
    auto rgb_u8 = torch::from_blob((void*)pts.rgb.data(), {N, 3}, torch::kUInt8).clone();
    auto rgb_cpu = rgb_u8.to(torch::kFloat32).div(255.0);

    xyz = xyz_cpu.to(device);
    features_dc = rgb_cpu.to(device);

    // Make them trainable (leaf tensors)
    xyz.requires_grad_(true);
    features_dc.requires_grad_(true);

    // Build two param groups with separate learning rates.
    // In LibTorch 2.11, OptimizerParamGroup requires unique_ptr<OptimizerOptions>.
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.reserve(2);

    {
        auto opt_xyz = std::make_unique<torch::optim::AdamOptions>(lr_xyz);
        groups.emplace_back(std::vector<torch::Tensor>{xyz}, std::move(opt_xyz));
    }
    {
        auto opt_col = std::make_unique<torch::optim::AdamOptions>(lr_color);
        groups.emplace_back(std::vector<torch::Tensor>{features_dc}, std::move(opt_col));
    }

    optimizer = std::make_unique<torch::optim::Adam>(std::move(groups));
}

int64_t GaussianModelMinimal::num_points() const {
    return xyz.defined() ? xyz.size(0) : 0;
}