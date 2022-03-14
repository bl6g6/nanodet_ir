mkdir -p build-1126
pushd build-1126
cmake \
-DCMAKE_TOOLCHAIN_FILE=/home/masike/prefix/1126/etc_toolchains_rv1126.toolchain.cmake \
-DOpenCV_DIR=/home/masike/prefix/1126/opencv/share/OpenCV \
-DCMAKE_BUILD_TYPE=Release \
..
make -j16
popd
