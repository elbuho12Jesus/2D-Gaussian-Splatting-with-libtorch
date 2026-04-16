#include "LoadImages.h"

torch::Tensor loadImageWithOpenCV(const std::string& image_path) {
    // 1. Leer imagen
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Error de OpenCV: Imagen no encontrada.");

    // 2. Convertir BGR a RGB y luego a Flotantes (0.0 - 1.0)
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32FC3, 1.0f / 255.0f);

    // 3. CRÍTICO: Asegurar que la memoria de OpenCV sea un bloque físico continuo
    if (!img_rgb.isContinuous()) {
        img_rgb = img_rgb.clone(); // Obliga a reorganizar la RAM sin espacios vacíos
    }

    // 4. CRÍTICO: Castear las dimensiones explícitamente a 64 bits para PyTorch
    int64_t height = static_cast<int64_t>(img_rgb.rows);
    int64_t width = static_cast<int64_t>(img_rgb.cols);
    int64_t channels = 3;

    // 5. Crear el tensor usando ptr<float>() en lugar de .data
    torch::Tensor tensor = torch::from_blob(
        img_rgb.ptr<float>(),
        { height, width, channels },
        torch::kFloat32
    );

    // 6. Separar la permutación y el clonado para aislar la memoria
    torch::Tensor permuted_tensor = tensor.permute({ 2, 0, 1 });

    // Al hacer clone(), PyTorch crea su propia memoria RAM y ya no depende de OpenCV
    return permuted_tensor.clone();
}