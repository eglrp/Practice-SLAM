cmake_minimum_required(VERSION 3.0)

project(BasicVO)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# set output
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# set include dir
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_BINARY_DIR}/lib)


# OpenCV
set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})

add_subdirectory(src)


add_executable(BasicVO main.cpp)
target_link_libraries(BasicVO ${OpenCV_LIBS} vo)