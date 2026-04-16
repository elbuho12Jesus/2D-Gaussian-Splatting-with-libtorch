#pragma once
#include <torch/torch.h>

namespace photoloss {

// Equivalente a loss_utils.l1_loss: mean(abs(x - y))
at::Tensor l1_loss(const at::Tensor& network_output, const at::Tensor& gt);

// Equivalente a loss_utils.ssim(img1, img2, window_size=11, size_average=true)
// IMPORTANTE: img1/img2 deben estar en NCHW (como en Python cuando llaman ssim),
// pero si tú tienes CHW en main, simplemente añade .unsqueeze(0) antes.
at::Tensor ssim(const at::Tensor& img1, const at::Tensor& img2,
                int window_size = 11, bool size_average = true);

} // namespace photoloss