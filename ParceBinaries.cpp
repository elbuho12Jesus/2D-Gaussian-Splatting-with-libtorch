#include "ParceBinaries.h"

// NOTE:
// - We intentionally do NOT define a second template (readBinaryData) here.
// - Use readBinary<T>(std::ifstream&) from the header everywhere.

std::vector<Camera> readColmapCameras(const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir cameras.bin: " + bin_path);

    const uint64_t num_cameras = readBinary<uint64_t>(file);
    std::vector<Camera> cameras;
    cameras.reserve(static_cast<size_t>(num_cameras));

    for (uint64_t i = 0; i < num_cameras; ++i) {
        Camera cam;
        cam.camera_id = readBinary<int>(file);
        cam.model_id  = readBinary<int>(file);
        cam.width     = readBinary<uint64_t>(file);
        cam.height    = readBinary<uint64_t>(file);

        // Number of intrinsic parameters depends on camera model.
        // Minimal safe approach: map known COLMAP models.
        // If you only use PINHOLE (most common), it's 4 params: fx, fy, cx, cy.
        int num_params = 0;
        switch (cam.model_id) {
            case 0: // SIMPLE_PINHOLE
                num_params = 3; // f, cx, cy
                break;
            case 1: // PINHOLE
                num_params = 4; // fx, fy, cx, cy
                break;
            case 2: // SIMPLE_RADIAL
                num_params = 4; // f, cx, cy, k
                break;
            case 3: // RADIAL
                num_params = 5; // f, cx, cy, k1, k2
                break;
            case 4: // OPENCV
                num_params = 8; // fx, fy, cx, cy, k1, k2, p1, p2
                break;
            case 5: // OPENCV_FISHEYE
                num_params = 8;
                break;
            case 6: // FULL_OPENCV
                num_params = 12;
                break;
            case 7: // FOV
                num_params = 5;
                break;
            case 8: // SIMPLE_RADIAL_FISHEYE
                num_params = 4;
                break;
            case 9: // RADIAL_FISHEYE
                num_params = 5;
                break;
            case 10: // THIN_PRISM_FISHEYE
                num_params = 12;
                break;
            default:
                // If unknown, you can fail fast to avoid mis-parsing.
                throw std::runtime_error("Camera model_id desconocido en cameras.bin: " + std::to_string(cam.model_id));
        }

        cam.params.clear();
        cam.params.reserve(static_cast<size_t>(num_params));
        for (int p = 0; p < num_params; ++p) {
            cam.params.push_back(readBinary<double>(file));
        }

        cameras.push_back(std::move(cam));
    }

    return cameras;
}

std::vector<ImagePose> readColmapImages(const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir images.bin: " + bin_path);

    const uint64_t num_images = readBinary<uint64_t>(file);
    std::vector<ImagePose> images;
    images.reserve(static_cast<size_t>(num_images));

    for (uint64_t i = 0; i < num_images; ++i) {
        ImagePose img;
        img.image_id = readBinary<int>(file);

        // Quaternion (qw, qx, qy, qz)
        img.qw = readBinary<double>(file);
        img.qx = readBinary<double>(file);
        img.qy = readBinary<double>(file);
        img.qz = readBinary<double>(file);

        // Translation
        img.tx = readBinary<double>(file);
        img.ty = readBinary<double>(file);
        img.tz = readBinary<double>(file);

        img.camera_id = readBinary<int>(file);       

        // Image name (null-terminated string)
        img.name.clear();
        char c;
        while (file.read(&c, 1) && c != '\0') img.name += c;

        // points2D observations (we ignore, but MUST read)
        const uint64_t num_points2D = readBinary<uint64_t>(file);
        for (uint64_t j = 0; j < num_points2D; ++j) {
            (void)readBinary<double>(file);   // x
            (void)readBinary<double>(file);   // y
            (void)readBinary<int64_t>(file); // point3D_id (puede ser -1)
        }

        images.push_back(std::move(img));
    }

    return images;
}

ColmapPoints3DMinimal readColmapPoints3DMinimal(const std::string& points3d_bin_path) {
    std::ifstream f(points3d_bin_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("No se pudo abrir points3D.bin: " + points3d_bin_path);
    }

    const uint64_t num_points = readBinary<uint64_t>(f);

    ColmapPoints3DMinimal out;
    out.xyz.reserve(static_cast<size_t>(num_points) * 3);
    out.rgb.reserve(static_cast<size_t>(num_points) * 3);
    out.error.reserve(static_cast<size_t>(num_points));

    for (uint64_t i = 0; i < num_points; ++i) {
        (void)readBinary<uint64_t>(f); // point3D_id

        const double x = readBinary<double>(f);
        const double y = readBinary<double>(f);
        const double z = readBinary<double>(f);

        const uint8_t r = readBinary<uint8_t>(f);
        const uint8_t g = readBinary<uint8_t>(f);
        const uint8_t b = readBinary<uint8_t>(f);

        const double err = readBinary<double>(f);

        out.xyz.push_back(x);
        out.xyz.push_back(y);
        out.xyz.push_back(z);

        out.rgb.push_back(r);
        out.rgb.push_back(g);
        out.rgb.push_back(b);

        out.error.push_back(err);

        const uint64_t track_len = readBinary<uint64_t>(f);
        for (uint64_t t = 0; t < track_len; ++t) {
            (void)readBinary<int>(f); // image_id
            (void)readBinary<int>(f); // point2D_idx
        }
    }

    return out;
}