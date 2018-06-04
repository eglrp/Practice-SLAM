cmake_minimum_required(VERSION 3.0)

project(BasicVO)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})


add_executable(BasicVO main.cpp)
target_link_libraries(BasicVO ${OpenCV_LIBS})