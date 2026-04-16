cmake_minimum_required(VERSION 3.22)
project(GS_with_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA archs (portable) ----
# Puedes override desde CLI: -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

# ---- User-configurable paths (no hardcode obligatorio) ----
# En Windows puedes setear Torch_DIR / OpenCV_DIR desde cmake-gui o CLI.
# En Ubuntu normalmente usarás -DCMAKE_PREFIX_PATH=/path/to/libtorch
set(Torch_DIR "" CACHE PATH "Path to libtorch/share/cmake/Torch (optional if using CMAKE_PREFIX_PATH)")
set(OpenCV_DIR "" CACHE PATH "Path to OpenCVConfig.cmake folder (optional)")

# ---- Dependencies ----
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# -------------------------
# simple-knn sources
# -------------------------
set(SIMPLE_KNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda_knn)
set(SIMPLE_KNN_SOURCES
  ${SIMPLE_KNN_DIR}/simple_knn.cu
  ${SIMPLE_KNN_DIR}/spatial.cpp
)

# -------------------------
# Rasterizer sources
# -------------------------
set(RASTERIZER_SOURCES
  cuda_rasterizer/backward.cu
  cuda_rasterizer/forward.cu
  cuda_rasterizer/rasterizer_impl.cu
)

# -------------------------
# Target
# -------------------------
add_executable(gs_app
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LoadImages.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ParceBinaries.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/GaussianModelMinimal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GaussianModelStage1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigCamera.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DepthNormals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SSIMLoss.cpp
  ${SIMPLE_KNN_SOURCES}
  ${RASTERIZER_SOURCES}
)

# Includes
target_include_directories(gs_app PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${SIMPLE_KNN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
)

# Link
target_link_libraries(gs_app PRIVATE
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  CUDA::cudart
)

# ---- Platform/compiler specific flags ----
if(MSVC)
  # CCCL (CUDA 13+) requires standard conforming preprocessor
  target_compile_options(gs_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  target_compile_options(gs_app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  target_compile_definitions(gs_app PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Needed for multi-TU CUDA projects
set_target_properties(gs_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")