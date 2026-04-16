#include "ToyRasterizer.h"

#include <cmath>
#include <stdexcept>

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
    int radius_px,
    float eps
) {
    TORCH_CHECK(xyz.defined() && rgb.defined(), "xyz/rgb must be defined");
    TORCH_CHECK(xyz.dtype() == torch::kFloat32, "xyz must be float32");
    TORCH_CHECK(rgb.dtype() == torch::kFloat32, "rgb must be float32");
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be (N,3)");
    TORCH_CHECK(rgb.dim() == 2 && rgb.size(1) == 3, "rgb must be (N,3)");
    TORCH_CHECK(xyz.size(0) == rgb.size(0), "xyz/rgb N mismatch");
    TORCH_CHECK(H > 0 && W > 0, "Invalid H/W");
    TORCH_CHECK(sigma_px > 0.0f, "sigma_px must be > 0");
    TORCH_CHECK(radius_px >= 0, "radius_px must be >= 0");

    const auto dev = xyz.device();
    TORCH_CHECK(rgb.device() == dev, "rgb must be on same device as xyz");

    const int64_t N = xyz.size(0);

    // Output accumulators: (H,W,3) and (H,W,1)
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto acc_rgb = torch::zeros({H, W, 3}, opts);
    auto acc_w   = torch::zeros({H, W, 1}, opts);

    // Camera: identity R,t. We only use xyz directly in camera space.
    // Project
    auto X = xyz.select(1, 0);
    auto Y = xyz.select(1, 1);
    auto Z = xyz.select(1, 2);

    // Avoid division by 0
    auto Zc = torch::clamp_min(Z, 1e-4f);

    auto u = fx * (X / Zc) + cx; // (N)
    auto v = fy * (Y / Zc) + cy; // (N)

    // Build a small pixel offset grid for the patch
    // dx,dy in [-radius, radius]
    const int K = 2 * radius_px + 1;
    auto dx = torch::arange(-radius_px, radius_px + 1, opts).view({1, K}); // (1,K)
    auto dy = torch::arange(-radius_px, radius_px + 1, opts).view({K, 1}); // (K,1)

    // Precompute gaussian denominator
    const float inv_2sigma2 = 1.0f / (2.0f * sigma_px * sigma_px);

    // Loop over points (slow but simple). For early correctness tests only.
    for (int64_t i = 0; i < N; ++i) {
        // Continuous center (ui,vi)
        auto ui = u[i];
        auto vi = v[i];

        // Compute integer anchor (floor) just to bound the patch. This does NOT affect gradients much;
        // the weights remain continuous w.r.t ui,vi.
        int32_t u0 = static_cast<int32_t>(std::floor(ui.item<float>()));
        int32_t v0 = static_cast<int32_t>(std::floor(vi.item<float>()));

        int32_t x_start = u0 - radius_px;
        int32_t y_start = v0 - radius_px;

        int32_t x_end = u0 + radius_px;
        int32_t y_end = v0 + radius_px;

        // Skip if fully out of bounds
        if (x_end < 0 || x_start >= W || y_end < 0 || y_start >= H) continue;

        // Clamp bounds
        x_start = std::max<int32_t>(x_start, 0);
        y_start = std::max<int32_t>(y_start, 0);
        x_end = std::min<int32_t>(x_end, W - 1);
        y_end = std::min<int32_t>(y_end, H - 1);

        // Create local pixel coordinate tensors for the valid window.
        // px: (1,Wx), py: (Hy,1)
        auto px = torch::arange(x_start, x_end + 1, opts).view({1, -1});
        auto py = torch::arange(y_start, y_end + 1, opts).view({-1, 1});

        // Continuous deltas
        auto du = px - ui; // (1,Wx)
        auto dv = py - vi; // (Hy,1)

        // Squared distance grid via broadcast: (Hy,Wx)
        auto d2 = dv.pow(2) + du.pow(2);

        // Gaussian weights
        auto w = torch::exp(-d2 * inv_2sigma2).unsqueeze(-1); // (Hy,Wx,1)

        // Color
        auto ci = rgb[i].view({1, 1, 3}); // (1,1,3)

        // Accumulate
        acc_rgb.index({torch::indexing::Slice(y_start, y_end + 1), torch::indexing::Slice(x_start, x_end + 1), torch::indexing::Slice()})
            .add_(w * ci);
        acc_w.index({torch::indexing::Slice(y_start, y_end + 1), torch::indexing::Slice(x_start, x_end + 1), 0})
            .add_(w.squeeze(-1));
    }

    // Normalize
    auto img_hwc = acc_rgb / (acc_w + eps);

    // Convert to (3,H,W)
    auto img_chw = img_hwc.permute({2, 0, 1}).contiguous();
    return img_chw;
}