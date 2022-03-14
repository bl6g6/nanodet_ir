cmake_minimum_required(VERSION 3.9)
project(nanodet-mnn)

set(CMAKE_CXX_STANDARD 17)

# find_package(OpenCV REQUIRED PATHS "/work/dependence/opencv/opencv-3.4.3/build")
find_package(OpenCV REQUIRED)
include_directories(/home/masike/prefix/1126/mnn/mnn-1.2.5/include/)

add_executable(nanodet-mnn main.cpp nanodet_mnn.cpp)
target_link_libraries(
   nanodet-mnn
   ${MNN_DIR}/home/masike/prefix/1126/mnn/mnn-1.2.5/lib/libMNN.so
   ${OpenCV_LIBS}
)
