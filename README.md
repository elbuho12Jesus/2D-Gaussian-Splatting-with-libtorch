# compilation
## Windows
cmake -S . -B build -DTorch_DIR="C:/libtorch/share/cmake/Torch" -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"
cmake --build build --config Debug

## Ubuntu
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/libtorch
cmake --build build -j