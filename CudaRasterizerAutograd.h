#pragma once

#include <torch/torch.h>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <tuple>

class GaussianRasterizerFunction : public torch::autograd::Function<GaussianRasterizerFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        
        // Tensores requeridos
        torch::Tensor background,
        torch::Tensor means3D,
        torch::Tensor colors,
        torch::Tensor opacity,
        torch::Tensor scales,
        torch::Tensor rotations,
        float scale_modifier,
        torch::Tensor transMat_precomp,
        torch::Tensor viewmatrix,
        torch::Tensor projmatrix,
        float tan_fovx, 
        float tan_fovy,
        int image_height,
        int image_width,
        torch::Tensor sh,
        int degree,
        torch::Tensor campos,
        bool prefiltered,
        bool debug
    ) {
        int P = means3D.size(0);
        int H = image_height;
        int W = image_width;

        auto float_opts = means3D.options().dtype(torch::kFloat32);
        
        // Inicializar tensores de salida (NUM_CHANNELS viene de config.h, suele ser 3)
        torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
        torch::Tensor out_others = torch::zeros({3 + 3 + 1, H, W}, float_opts); // [Normales(3) + Distancia(3) + Alpha(1)]
        torch::Tensor radii = torch::zeros({P}, means3D.options().dtype(torch::kInt32));

        // Buffers para el Radix Sort
        torch::Tensor geomBuffer = torch::empty({0}, means3D.options().dtype(torch::kByte));
        torch::Tensor binningBuffer = torch::empty({0}, means3D.options().dtype(torch::kByte));
        torch::Tensor imageBuffer = torch::empty({0}, means3D.options().dtype(torch::kByte));

        auto resizeFunctional = [](torch::Tensor& t) {
            return [&t](size_t N) {
                t.resize_({(long long)N});
                return reinterpret_cast<char*>(t.contiguous().data_ptr());
            };
        };

        if (P != 0) {
            int M = 0;
            if (sh.defined() && sh.size(0) != 0) {
                M = sh.size(1);
            }

            CudaRasterizer::Rasterizer::forward(
                resizeFunctional(geomBuffer),
                resizeFunctional(binningBuffer),
                resizeFunctional(imageBuffer),
                P, degree, M,
                background.contiguous().data_ptr<float>(),
                W, H,
                means3D.contiguous().data_ptr<float>(),
                sh.defined() ? sh.contiguous().data_ptr<float>() : nullptr,
                colors.defined() ? colors.contiguous().data_ptr<float>() : nullptr,
                opacity.contiguous().data_ptr<float>(),
                scales.defined() ? scales.contiguous().data_ptr<float>() : nullptr,
                scale_modifier,
                rotations.defined() ? rotations.contiguous().data_ptr<float>() : nullptr,
                transMat_precomp.defined() ? transMat_precomp.contiguous().data_ptr<float>() : nullptr,
                viewmatrix.contiguous().data_ptr<float>(),
                projmatrix.contiguous().data_ptr<float>(),
                campos.contiguous().data_ptr<float>(),
                tan_fovx, tan_fovy, prefiltered,
                out_color.contiguous().data_ptr<float>(),
                out_others.contiguous().data_ptr<float>(),
                radii.contiguous().data_ptr<int>(),
                debug
            );
        }

        // Guardar tensores para Backward
        ctx->save_for_backward({
            background, means3D, radii, colors, scales, rotations, transMat_precomp,
            viewmatrix, projmatrix, sh, campos, geomBuffer, binningBuffer, imageBuffer
        });

        // Guardar escalares
        ctx->saved_data["scale_modifier"] = scale_modifier;
        ctx->saved_data["tan_fovx"] = tan_fovx;
        ctx->saved_data["tan_fovy"] = tan_fovy;
        ctx->saved_data["degree"] = degree;
        ctx->saved_data["debug"] = debug;
        ctx->saved_data["image_height"] = image_height;
        ctx->saved_data["image_width"] = image_width;

        return {out_color, out_others, radii};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // Recuperar gradientes que vienen de loss.backward()
        torch::Tensor dL_dout_color = grad_outputs[0];
        torch::Tensor dL_dout_others = grad_outputs[1];

        auto saved = ctx->get_saved_variables();
        auto background = saved[0];
        auto means3D = saved[1];
        auto radii = saved[2];
        auto colors = saved[3];
        auto scales = saved[4];
        auto rotations = saved[5];
        auto transMat_precomp = saved[6];
        auto viewmatrix = saved[7];
        auto projmatrix = saved[8];
        auto sh = saved[9];
        auto campos = saved[10];
        auto geomBuffer = saved[11];
        auto binningBuffer = saved[12];
        auto imageBuffer = saved[13];

        float scale_modifier = ctx->saved_data["scale_modifier"].toDouble();
        float tan_fovx = ctx->saved_data["tan_fovx"].toDouble();
        float tan_fovy = ctx->saved_data["tan_fovy"].toDouble();
        int degree = ctx->saved_data["degree"].toInt();
        bool debug = ctx->saved_data["debug"].toBool();
        int H = ctx->saved_data["image_height"].toInt();
        int W = ctx->saved_data["image_width"].toInt();

        int P = means3D.size(0);
        int M = 0;
        if (sh.defined() && sh.size(0) != 0) {
            M = sh.size(1);
        }
        
        int R = 0; // Dependiendo de la versión de 2DGS, a veces calculan radios extra. 0 suele ser el default.

        // Tensores de gradientes de salida
        auto dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
        auto dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
        auto dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
        auto dL_dnormal = torch::zeros({P, 3}, means3D.options());
        auto dL_dopacity = torch::zeros({P, 1}, means3D.options());
        auto dL_dtransMat = torch::zeros({P, 9}, means3D.options());
        auto dL_dsh = torch::zeros({P, M, 3}, means3D.options());
        auto dL_dscales = torch::zeros({P, 2}, means3D.options()); // ¡Atención: 2 para 2DGS!
        auto dL_drotations = torch::zeros({P, 4}, means3D.options());

        if (P != 0) {
            CudaRasterizer::Rasterizer::backward(
                P, degree, M, R,
                background.contiguous().data_ptr<float>(),
                W, H,
                means3D.contiguous().data_ptr<float>(),
                sh.defined() ? sh.contiguous().data_ptr<float>() : nullptr,
                colors.defined() ? colors.contiguous().data_ptr<float>() : nullptr,
                scales.defined() ? scales.contiguous().data_ptr<float>() : nullptr,
                scale_modifier,
                rotations.defined() ? rotations.contiguous().data_ptr<float>() : nullptr,
                transMat_precomp.defined() ? transMat_precomp.contiguous().data_ptr<float>() : nullptr,
                viewmatrix.contiguous().data_ptr<float>(),
                projmatrix.contiguous().data_ptr<float>(),
                campos.contiguous().data_ptr<float>(),
                tan_fovx, tan_fovy,
                radii.contiguous().data_ptr<int>(),
                reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
                reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
                reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
                dL_dout_color.contiguous().data_ptr<float>(),
                dL_dout_others.contiguous().data_ptr<float>(),
                dL_dmeans2D.contiguous().data_ptr<float>(),
                dL_dnormal.contiguous().data_ptr<float>(),
                dL_dopacity.contiguous().data_ptr<float>(),
                dL_dcolors.contiguous().data_ptr<float>(),
                dL_dmeans3D.contiguous().data_ptr<float>(),
                dL_dtransMat.contiguous().data_ptr<float>(),
                dL_dsh.contiguous().data_ptr<float>(),
                dL_dscales.contiguous().data_ptr<float>(),
                dL_drotations.contiguous().data_ptr<float>(),
                debug
            );
        }

        // Retornar en el mismo orden que el forward:
        return {
            torch::Tensor(), // background
            dL_dmeans3D,     // means3D
            dL_dcolors,      // colors
            dL_dopacity,     // opacity
            dL_dscales,      // scales
            dL_drotations,   // rotations
            torch::Tensor(), // scale_modifier
            dL_dtransMat,    // transMat_precomp
            torch::Tensor(), // viewmatrix
            torch::Tensor(), // projmatrix
            torch::Tensor(), // tan_fovx
            torch::Tensor(), // tan_fovy
            torch::Tensor(), // image_height
            torch::Tensor(), // image_width
            dL_dsh,          // sh
            torch::Tensor(), // degree
            torch::Tensor(), // campos
            torch::Tensor(), // prefiltered
            torch::Tensor()  // debug
        };
    }
};