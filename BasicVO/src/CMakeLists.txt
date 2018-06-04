cmake_minimum_required(VERSION 3.0)

add_library(vo PinholeCamera.cpp VisualOdometry.cpp)

target_link_libraries(vo ${OpenCV_LIBS})