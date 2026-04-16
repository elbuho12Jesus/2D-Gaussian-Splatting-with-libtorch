#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <string>

torch::Tensor loadImageWithOpenCV(const std::string& image_path);