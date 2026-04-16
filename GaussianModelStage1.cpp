#include "GaussianModelStage1.h"
#include "spatial.h"   // distCUDA2(points)
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <algorithm> // std::min

torch::Tensor init_scale_constant(int64_t N, float constant_scale, torch::Device device) {
    // Return scaling (N,2) on requested device.
    auto s = torch::full({N, 2}, constant_scale, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return s;
}

torch::Tensor init_scale_knn_cpu_bruteforce(const torch::Tensor& xyz_cpu_f32,
                                           const ScaleInitConfig& cfg) {
    TORCH_CHECK(xyz_cpu_f32.device().is_cpu(), "xyz_cpu_f32 must be on CPU");
    TORCH_CHECK(xyz_cpu_f32.dtype() == torch::kFloat32, "xyz_cpu_f32 must be float32");
    TORCH_CHECK(xyz_cpu_f32.dim() == 2 && xyz_cpu_f32.size(1) == 3, "xyz_cpu_f32 must be (N,3)");

    const int64_t N = xyz_cpu_f32.size(0);

    // Guard: bruteforce is O(N^2)
    if (N > cfg.max_points_for_bruteforce) {
        // Fallback to constant scale on CPU
        return torch::full({N, 2}, cfg.constant_scale, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    }

    // Access raw data
    const float* p = xyz_cpu_f32.contiguous().data_ptr<float>();

    std::vector<float> scales;
    scales.resize(static_cast<size_t>(N) * 2);

    for (int64_t i = 0; i < N; ++i) {
        const float xi = p[i * 3 + 0];
        const float yi = p[i * 3 + 1];
        const float zi = p[i * 3 + 2];

        float best_d2 = std::numeric_limits<float>::infinity();

        // Find nearest neighbor (exclude self)
        for (int64_t j = 0; j < N; ++j) {
            if (j == i) continue;
            const float xj = p[j * 3 + 0];
            const float yj = p[j * 3 + 1];
            const float zj = p[j * 3 + 2];

            const float dx = xi - xj;
            const float dy = yi - yj;
            const float dz = zi - zj;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) best_d2 = d2;
        }

        if (!std::isfinite(best_d2)) {
            best_d2 = cfg.min_dist2;
        }
        best_d2 = std::max(best_d2, cfg.min_dist2);

        const float sigma = std::sqrt(best_d2) * cfg.scale_multiplier;

        // 2DGS: keep two in-plane scales; you can later adjust mapping.
        scales[static_cast<size_t>(i) * 2 + 0] = sigma;
        scales[static_cast<size_t>(i) * 2 + 1] = sigma;
    }

    auto out = torch::from_blob(scales.data(), {N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    return out;
}

torch::Tensor init_scale_knn_simpleknn_cuda(const torch::Tensor& xyz_cpu_f32,
                                            const torch::Device& device,
                                            const ScaleInitConfig& cfg)
{
    TORCH_CHECK(device.is_cuda(), "device must be CUDA for simple-knn");
    TORCH_CHECK(xyz_cpu_f32.device().is_cpu(), "xyz_cpu_f32 must be on CPU");
    TORCH_CHECK(xyz_cpu_f32.dtype() == torch::kFloat32, "xyz_cpu_f32 must be float32");
    TORCH_CHECK(xyz_cpu_f32.dim() == 2 && xyz_cpu_f32.size(1) == 3, "xyz_cpu_f32 must be (N,3)");

    // Move points to GPU (same device as training)
    torch::Tensor xyz_cuda = xyz_cpu_f32.to(device).contiguous();

    // meanDists: (N) float32 CUDA, mean of 3 NN squared distances (repo-faithful)
    torch::Tensor meanDists = distCUDA2(xyz_cuda);

    // Safety clamp (avoid sqrt(0) and NaNs); cfg.min_dist2 should be small like 1e-7
    meanDists = torch::clamp_min(meanDists, cfg.min_dist2);

    // Convert to sigma; repo typically uses sqrt(meanDist2) * multiplier
    torch::Tensor sigma = torch::sqrt(meanDists) * cfg.scale_multiplier; // (N)

    // 2DGS expects (N,2) in-plane scales (you already do that)
    torch::Tensor scaling = sigma.unsqueeze(1).repeat({1, 2});            // (N,2) CUDA
    return scaling;
}

void GaussianModelStage1::init_from_colmap_points(const ColmapPoints3DMinimal& pts,
                                                  const torch::Device& dev,
                                                  const ScaleInitConfig& scale_cfg,
                                                  double lr_xyz,
                                                  double lr_color,
                                                  double lr_opacity,
                                                  double lr_scaling,
                                                  double lr_rotation) {
    const int64_t N = static_cast<int64_t>(pts.error.size());
    if (N <= 0) throw std::runtime_error("points3D: N == 0");
    if (static_cast<int64_t>(pts.xyz.size()) != N * 3) throw std::runtime_error("points3D: xyz size mismatch");
    if (static_cast<int64_t>(pts.rgb.size()) != N * 3) throw std::runtime_error("points3D: rgb size mismatch");

    device = dev;

    // --- Base tensors on CPU ---

    auto xyz_cpu = torch::from_blob((void*)pts.xyz.data(), {N, 3}, torch::kFloat64)
                       .clone()
                       .to(torch::kFloat32);

    auto rgb_u8 = torch::from_blob((void*)pts.rgb.data(), {N, 3}, torch::kUInt8).clone();
    auto rgb_cpu = rgb_u8.to(torch::kFloat32).div(255.0);

    // --- Scale init (swappable strategy) ---      
    torch::Tensor scaling_log_init;  // declara UNA sola vez
    if (device.is_cuda()) {
        torch::Tensor scaling = init_scale_knn_simpleknn_cuda(xyz_cpu, device, scale_cfg); // (N,2) CUDA
        scaling = torch::clamp_min(scaling, 1e-8f);
        scaling_log_init = torch::log(scaling); // CUDA
    } else {
        torch::Tensor scaling_cpu = init_scale_knn_cpu_bruteforce(xyz_cpu, scale_cfg); // (N,2) CPU
        scaling_cpu = torch::clamp_min(scaling_cpu, 1e-8f);
        scaling_log_init = torch::log(scaling_cpu).to(device); // CPU->device
    }

    // --- Rotation init ---
    // Random quaternion then normalize.
    auto rot_cpu = torch::rand({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    rot_cpu = torch::nn::functional::normalize(rot_cpu, torch::nn::functional::NormalizeFuncOptions().dim(1));

    // --- Opacity init ---
    // Store logits so that sigmoid(logit)=0.1 initially.
    // inverse_sigmoid(p) = log(p/(1-p))
    const float p0 = 0.1f;
    const float logit0 = std::log(p0 / (1.0f - p0));
    auto opacity_logits_cpu = torch::full({N, 1}, logit0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // --- Move to device and set requires_grad ---
    xyz = xyz_cpu.to(device);
    features_dc = rgb_cpu.to(device);

    // scaling_log ya está en el device correcto por el if/else anterior.
    // (si quieres asegurar contiguidad, puedes hacer scaling_log = scaling_log.contiguous();)
    scaling_log = scaling_log_init;

    rotation = rot_cpu.to(device);
    opacity_logits = opacity_logits_cpu.to(device);

    xyz.requires_grad_(true);
    features_dc.requires_grad_(true);
    scaling_log.requires_grad_(true);
    rotation.requires_grad_(true);
    opacity_logits.requires_grad_(true);

    // --- Optimizer: 5 param groups with separate LRs ---
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.reserve(5);

    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr_xyz);
        groups.emplace_back(std::vector<torch::Tensor>{xyz}, std::move(opt));
    }
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr_color);
        groups.emplace_back(std::vector<torch::Tensor>{features_dc}, std::move(opt));
    }
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr_opacity);
        groups.emplace_back(std::vector<torch::Tensor>{opacity_logits}, std::move(opt));
    }
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr_scaling);
        groups.emplace_back(std::vector<torch::Tensor>{scaling_log}, std::move(opt));
    }
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(lr_rotation);
        groups.emplace_back(std::vector<torch::Tensor>{rotation}, std::move(opt));
    }

    optimizer = std::make_unique<torch::optim::Adam>(std::move(groups));
}

void GaussianModelStage1::save_gaussians_ply_ascii(const std::string& filename) const
{
    TORCH_CHECK(xyz.defined(), "xyz not defined");
    TORCH_CHECK(features_dc.defined(), "features_dc not defined");
    TORCH_CHECK(opacity_logits.defined(), "opacity_logits not defined");
    TORCH_CHECK(scaling_log.defined(), "scaling_log not defined");
    TORCH_CHECK(rotation.defined(), "rotation not defined");

    auto xyz_cpu = xyz.detach().to(torch::kCPU).contiguous();                         // (N,3)
    auto dc_cpu  = features_dc.detach().to(torch::kCPU).contiguous();                 // (N,3)
    auto op_cpu  = torch::sigmoid(opacity_logits.detach()).to(torch::kCPU).contiguous(); // (N,1)
    auto sc_cpu  = torch::exp(scaling_log.detach()).to(torch::kCPU).contiguous();     // (N,2) o (N,3)
    auto rot_cpu = torch::nn::functional::normalize(
                       rotation.detach(),
                       torch::nn::functional::NormalizeFuncOptions().dim(1))
                       .to(torch::kCPU).contiguous();                                 // (N,4)

    const int64_t N = xyz_cpu.size(0);
    TORCH_CHECK(xyz_cpu.sizes() == torch::IntArrayRef({N, 3}), "xyz must be (N,3)");
    TORCH_CHECK(dc_cpu.sizes()  == torch::IntArrayRef({N, 3}), "features_dc must be (N,3)");
    TORCH_CHECK(op_cpu.sizes()  == torch::IntArrayRef({N, 1}), "opacity must be (N,1)");
    TORCH_CHECK(rot_cpu.sizes() == torch::IntArrayRef({N, 4}), "rotation must be (N,4)");
    TORCH_CHECK(sc_cpu.dim() == 2 && sc_cpu.size(0) == N, "scaling must be (N,2) or (N,3)");

    const int64_t S = sc_cpu.size(1);
    TORCH_CHECK(S == 2 || S == 3, "scaling_log must be (N,2) or (N,3)");

    std::ofstream f(filename, std::ios::out);
    if (!f) throw std::runtime_error("Could not open PLY for writing: " + filename);

    // ---- Header (ASCII PLY) ----
    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << N << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    f << "property float opacity\n";
    f << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    f << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    f << "end_header\n";

    const float* p_xyz = xyz_cpu.data_ptr<float>();
    const float* p_dc  = dc_cpu.data_ptr<float>();
    const float* p_op  = op_cpu.data_ptr<float>();
    const float* p_sc  = sc_cpu.data_ptr<float>();
    const float* p_rot = rot_cpu.data_ptr<float>();

    f << std::setprecision(9);

    for (int64_t i = 0; i < N; ++i) {
        const float x = p_xyz[i * 3 + 0];
        const float y = p_xyz[i * 3 + 1];
        const float z = p_xyz[i * 3 + 2];

        const float dc0 = p_dc[i * 3 + 0];
        const float dc1 = p_dc[i * 3 + 1];
        const float dc2 = p_dc[i * 3 + 2];

        const float opacity = p_op[i * 1 + 0];

        const float s0 = p_sc[i * S + 0];
        const float s1 = p_sc[i * S + 1];
        const float s2 = (S == 3) ? p_sc[i * S + 2] : std::min(s0, s1); // 2D->3D fallback

        const float q0 = p_rot[i * 4 + 0];
        const float q1 = p_rot[i * 4 + 1];
        const float q2 = p_rot[i * 4 + 2];
        const float q3 = p_rot[i * 4 + 3];

        f << x << " " << y << " " << z << " "
          << dc0 << " " << dc1 << " " << dc2 << " "
          << opacity << " "
          << s0 << " " << s1 << " " << s2 << " "
          << q0 << " " << q1 << " " << q2 << " " << q3 << "\n";
    }
}

int64_t GaussianModelStage1::num_points() const {
    return xyz.defined() ? xyz.size(0) : 0;
}

torch::Tensor GaussianModelStage1::get_opacity() const {
    return torch::sigmoid(opacity_logits);
}

torch::Tensor GaussianModelStage1::get_scaling() const {
    return torch::exp(scaling_log);
}

torch::Tensor GaussianModelStage1::get_rotation() const {
    return torch::nn::functional::normalize(rotation,
        torch::nn::functional::NormalizeFuncOptions().dim(1));
}