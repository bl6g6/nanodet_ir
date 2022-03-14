mkdir -p build-9211_0720
pushd build-9211_0720
cmake  -DCMAKE_TOOLCHAIN_FILE=/home/masike/mylib/toolChain/9211_linux/etc_toolchains_9211.toolchain.cmake \
-Dncnn_DIR=/home/masike/prefix/9211_linux/ncnn/0720_nostd/lib/cmake/ncnn \
-DCMAKE_BUILD_TYPE=Release \
..
make -j32
popd
