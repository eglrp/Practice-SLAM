cmake_minimum_required(VERSION 3.5)
project(Position_esitmation_2d2d)

set(CMAKE_CXX_STANDARD 11)

# set Opencv
set(OpenCV_DIR /usr/local/opencv3/share/OpenCV)
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})

add_executable(Position_esitmation_2d2d main.cpp)
target_link_libraries(Position_esitmation_2d2d ${OpenCV_LIBS})