mkdir -p build
pushd build
cmake -Dncnn_DIR=/home/masike/prefix/x86/ncnn/0720/lib/cmake/ncnn \
..
make -j8
popd
