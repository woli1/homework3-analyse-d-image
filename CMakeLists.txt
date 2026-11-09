cmake_minimum_required(VERSION 3.10)
project(TP_contours)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TP_contours main.cpp)
target_link_libraries(TP_contours ${OpenCV_LIBS})