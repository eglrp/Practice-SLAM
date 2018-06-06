cmake_minimum_required(VERSION 3.5)
project(DrawTrajectory)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# set OpenCV
set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})

# set bin path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(DrawTrajectory main.cpp)
target_link_libraries(DrawTrajectory ${OpenCV_LIBS})