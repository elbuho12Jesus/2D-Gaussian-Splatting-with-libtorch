#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
// -------------------------
// Data structs
// -------------------------

struct Camera {
    int camera_id;
    int model_id;
    uint64_t width;
    uint64_t height;
    std::vector<double> params; // intrinsics depend on model
};

struct ImagePose {
    int image_id;
    double qw, qx, qy, qz; // quaternion
    double tx, ty, tz;     // translation
    int camera_id;
    std::string name;      // image filename
};

// Minimal set for Gaussian Splatting init
struct ColmapPoints3DMinimal {
    std::vector<double> xyz;   // N*3
    std::vector<uint8_t> rgb;  // N*3 (0..255)
    std::vector<double> error; // N
};

// -------------------------
// Binary reader (template)
// -------------------------

template <typename T>
inline T readBinary(std::ifstream& f) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) {
        throw std::runtime_error("Binary read failed (unexpected EOF)");
    }
    return v;
}

// -------------------------
// COLMAP parsers
// -------------------------

std::vector<Camera> readColmapCameras(const std::string& bin_path);
std::vector<ImagePose> readColmapImages(const std::string& bin_path);
ColmapPoints3DMinimal readColmapPoints3DMinimal(const std::string& points3d_bin_path);