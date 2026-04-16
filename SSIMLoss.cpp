#include "SSIMLoss.h"
#include <cmath>

namespace photoloss {

at::Tensor l1_loss(const at::Tensor& network_output, const at::Tensor& gt) {
    return (network_output - gt).abs().mean();
}

// gaussian(window_size, sigma=1.5)
static at::Tensor gaussian_1d(int window_size, float sigma, const at::TensorOptions& opts) {
    // Python: exp(-(x - window_size//2)^2 / (2*sigma^2)) for x in range(window_size)
    int half = window_size / 2;
    auto xs = torch::arange(window_size, opts).to(torch::kFloat32);
    xs = xs - (float)half;
    auto g = torch::exp(-(xs * xs) / (2.0f * sigma * sigma));
    return g / g.sum();
}

// create_window(window_size, channel)
static at::Tensor create_window(int window_size, int channel, const at::TensorOptions& opts) {
    auto _1D = gaussian_1d(window_size, 1.5f, opts).unsqueeze(1);           // [K,1]
    auto _2D = _1D.matmul(_1D.transpose(0, 1)).to(torch::kFloat32);         // [K,K]
    auto w = _2D.unsqueeze(0).unsqueeze(0);                                 // [1,1,K,K]
    // expand(channel,1,K,K) contiguous
    return w.expand({channel, 1, window_size, window_size}).contiguous();   // [C,1,K,K]
}

static at::Tensor _ssim(const at::Tensor& img1, const at::Tensor& img2,
                        const at::Tensor& window,
                        int window_size, int channel, bool size_average) {
    using namespace torch::nn::functional;

    auto mu1 = conv2d(img1, window, Conv2dFuncOptions().padding(window_size / 2).groups(channel));
    auto mu2 = conv2d(img2, window, Conv2dFuncOptions().padding(window_size / 2).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = conv2d(img1 * img1, window, Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq;
    auto sigma2_sq = conv2d(img2 * img2, window, Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq;
    auto sigma12   = conv2d(img1 * img2, window, Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2;

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    auto ssim_map = ((2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average) {
        return ssim_map.mean();
    } else {
        // Python: mean(1).mean(1).mean(1)
        return ssim_map.mean(1).mean(1).mean(1);
    }
}

at::Tensor ssim(const at::Tensor& img1, const at::Tensor& img2,
                int window_size, bool size_average) {
    TORCH_CHECK(img1.sizes() == img2.sizes(), "ssim: img1 and img2 sizes must match");
    TORCH_CHECK(img1.dim() == 4, "ssim: expected NCHW tensors (like Python)");
    int channel = (int)img1.size(1);

    auto window = create_window(window_size, channel, img1.options());
    // En Python hacen window.type_as(img1) y si es cuda lo mueven al device; aquí window ya está en img1.options()
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

} // namespace photoloss