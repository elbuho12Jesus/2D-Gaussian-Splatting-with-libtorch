#include <iostream>
#include <random>
#include <cmath>
#include <torch/torch.h>

#include "ParceBinaries.h"
#include "LoadImages.h"
#include "GaussianModelStage1.h"
#include "CudaRasterizerAutograd.h"
#include "ConfigCamera.h"
#include "DepthNormals.h"   // <-- NUEVO
#include "SSIMLoss.h"

static const Camera* findCameraById(const std::vector<Camera>& cams, int cam_id) {
    for (const auto& c : cams) if (c.camera_id == cam_id) return &c;
    return nullptr;
}

int main() {
    try {
        std::string base = "C:/Users/elbuh/Documents/dataSets/GS/360_extra_scenes/flowers/";
        std::string sparse = base + "sparse/0/";
        std::string images_dir = base + "images/";

        auto pts  = readColmapPoints3DMinimal(sparse + "points3D.bin");
        auto cams = readColmapCameras(sparse + "cameras.bin");
        auto imgs = readColmapImages(sparse + "images.bin");

        torch::Device dev(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        if (!dev.is_cuda()) {
            std::cerr << "Necesitas CUDA para este rasterizador.\n";
            return 1;
        }
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        // Init gaussians
        ScaleInitConfig scale_cfg;
        scale_cfg.constant_scale = 0.01f;
        scale_cfg.max_points_for_bruteforce = 200000;
        scale_cfg.min_dist2 = 1e-12f;
        scale_cfg.scale_multiplier = 1.0f;

        GaussianModelStage1 gm;
        gm.init_from_colmap_points(pts, dev, scale_cfg);

        std::cout << "Modelo cargado con " << gm.num_points() << " Gaussianas.\n";
        std::cout << "Camaras: " << cams.size() << " | Imagenes: " << imgs.size() << "\n";

        // background como repo (ajusta si white_background)
        torch::Tensor background = torch::zeros({3}, opts);
        torch::Tensor empty0 = torch::empty({0}, opts);

        float znear = 0.01f;
        float zfar  = 100.0f;

        // Loss params (pon aquí tus valores reales)
        float lambda_dssim  = 0.2f;  // opt.lambda_dssim (SSIM pendiente)
        float lambda_dist   = 0.01f; // opt.lambda_dist
        float lambda_normal = 0.05f; // opt.lambda_normal
        float depth_ratio   = 1.0f;  // pipe.depth_ratio (1=median, 0=expected)

        int iterations = 8000; // así ya entra normal_loss después de 7000

        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> cam_dist(0, (int)imgs.size() - 1);

        for (int iter = 1; iter <= iterations; ++iter) {
            gm.optimizer->zero_grad();

            // Pick random camera
            const ImagePose& im = imgs[cam_dist(rng)];
            const Camera* cam = findCameraById(cams, im.camera_id);
            if (!cam) throw std::runtime_error("camera_id no encontrado: " + std::to_string(im.camera_id));

            // Intrinsics -> FoV (igual que repo, vía focal2fov)
            uint64_t W = cam->width;
            uint64_t H = cam->height;

            float fx, fy;
            if (cam->model_id == 1) {        // PINHOLE
                fx = (float)cam->params[0];
                fy = (float)cam->params[1];
            } else if (cam->model_id == 0) { // SIMPLE_PINHOLE
                fx = fy = (float)cam->params[0];
            } else {
                throw std::runtime_error("Modelo de camara no soportado para FoV (usa PINHOLE/SIMPLE_PINHOLE).");
            }

            float FoVx = camcfg::focal2fov(fx, (float)W);
            float FoVy = camcfg::focal2fov(fy, (float)H);
            float tanfovx = std::tan(FoVx * 0.5f);
            float tanfovy = std::tan(FoVy * 0.5f);

            // Extrinsics (COLMAP world->cam)
            torch::Tensor R = camcfg::qvec_to_rotmat(im.qw, im.qx, im.qy, im.qz, opts);
            torch::Tensor t = torch::tensor({(float)im.tx, (float)im.ty, (float)im.tz}, opts);

            torch::Tensor translate = torch::zeros({3}, opts);
            float scale = 1.0f;

            torch::Tensor world_view_transform = camcfg::make_world_view_transform_like_repo(R, t, translate, scale, opts);
            torch::Tensor projection_matrix = camcfg::make_projection_matrix_like_repo(znear, zfar, FoVx, FoVy, opts);
            torch::Tensor full_proj_transform = world_view_transform.matmul(projection_matrix);

            torch::Tensor campos = torch::linalg_inv(world_view_transform)
                .slice(0, 3, 4).slice(1, 0, 3).squeeze(0).contiguous();

            // Load GT
            std::string img_path = images_dir + im.name;
            torch::Tensor gt_cpu = loadImageWithOpenCV(img_path); // [3,H,W] CPU
            torch::Tensor gt = gt_cpu.to(dev).clamp(0.0, 1.0);
            int HH = (int)gt.size(1);
            int WW = (int)gt.size(2);

            // screenspace_points (solo para mantener compatibilidad conceptual con el repo)
            torch::Tensor means2D = torch::zeros_like(gm.xyz, gm.xyz.options()).set_requires_grad(true);
            (void)means2D;

            // 2DGS: scales [P,2]
            torch::Tensor scales_2d = gm.get_scaling().slice(1, 0, 2).contiguous();

            // Render
            auto outputs = GaussianRasterizerFunction::apply(
                background,
                gm.xyz,
                gm.features_dc,
                gm.get_opacity(),
                scales_2d,
                gm.get_rotation(),
                1.0f,
                empty0,
                world_view_transform,
                full_proj_transform,
                tanfovx, tanfovy,
                HH, WW,
                empty0,
                0,
                campos,
                false,
                false
            );

            torch::Tensor image  = outputs[0]; // [3,HH,WW]
            torch::Tensor allmap = outputs[1]; // [7,HH,WW]

            // recomendado si quieres asegurar rango para SSIM
            //image = torch::clamp(image, 0.0, 1.0);

            // Ll1 idéntico
            torch::Tensor Ll1 = photoloss::l1_loss(image, gt);

            // SSIM idéntico (NCHW)
            torch::Tensor ssim_val = photoloss::ssim(image.unsqueeze(0), gt.unsqueeze(0)); // escalar

            torch::Tensor photometric =
                (1.0f - lambda_dssim) * Ll1 + lambda_dssim * (1.0f - ssim_val);

            // Distortion loss (allmap[6:7]) después de 3000
            torch::Tensor rend_dist = allmap.slice(0, 6, 7); // [1,HH,WW]
            float lambda_dist_eff = (iter > 3000) ? lambda_dist : 0.0f;
            torch::Tensor dist_loss = lambda_dist_eff * rend_dist.mean();

            // Normal loss después de 7000 (idéntico al repo)
            float lambda_normal_eff = (iter > 7000) ? lambda_normal : 0.0f;
            torch::Tensor normal_loss = torch::zeros({}, opts);

            if (lambda_normal_eff > 0.0f) {
                // allmap channels
                torch::Tensor rend_alpha = allmap.slice(0, 1, 2);        // [1,HH,WW]
                torch::Tensor rend_normal_view = allmap.slice(0, 2, 5);  // [3,HH,WW]
                torch::Tensor depth_median = torch::nan_to_num(allmap.slice(0, 5, 6), 0.0, 0.0, 0.0); // [1,HH,WW]

                torch::Tensor depth_expected = allmap.slice(0, 0, 1) / (rend_alpha + 1e-8);
                depth_expected = torch::nan_to_num(depth_expected, 0.0, 0.0, 0.0);

                torch::Tensor surf_depth = depth_expected * (1.0f - depth_ratio) + depth_median * depth_ratio; // [1,HH,WW]

                // surf_normal = depth_to_normal(view, surf_depth) -> [HH,WW,3] en WORLD space
                torch::Tensor surf_normal_hw3 = depthnorm::depth_to_normal_like_repo(
                    world_view_transform, full_proj_transform,
                    WW, HH, surf_depth
                );

                // permute a [3,HH,WW]
                torch::Tensor surf_normal = surf_normal_hw3.permute({2, 0, 1}).contiguous();

                // surf_normal *= rend_alpha.detach()
                surf_normal = surf_normal * rend_alpha.detach();

                // rend_normal to WORLD: (H,W,3) @ (world_view_transform[:3,:3].T)
                torch::Tensor Rw = world_view_transform.slice(0, 0, 3).slice(1, 0, 3).transpose(0, 1).contiguous();
                torch::Tensor rend_normal_world = rend_normal_view
                    .permute({1, 2, 0})
                    .matmul(Rw)
                    .permute({2, 0, 1})
                    .contiguous();

                torch::Tensor normal_error = (1.0 - (rend_normal_world * surf_normal).sum(0)).unsqueeze(0);
                normal_loss = lambda_normal_eff * normal_error.mean();
            }

            torch::Tensor total_loss = photometric + dist_loss + normal_loss;
            total_loss.backward();
            gm.optimizer->step();

            if (iter % 10 == 0) {
                std::cout << "Iter " << iter
                          << " | Ll1=" << Ll1.item<float>()
                          << " | dist=" << dist_loss.item<float>()
                          << " | normal=" << normal_loss.item<float>()
                          << " | total=" << total_loss.item<float>()
                          << " | img=" << im.name
                          << "\n";
            }
        }

        std::cout << "Training loop terminado.\n";
        gm.save_gaussians_ply_ascii("output/final_point_cloud.ply");
    }
    catch (const std::exception& e) {
        std::cerr << "Error Critico: " << e.what() << "\n";
        return 1;
    }
    return 0;
}