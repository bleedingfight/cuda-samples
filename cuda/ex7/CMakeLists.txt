cmake_minimum_required(VERSION 3.10)
project(ex7)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
add_executable(ex7 main.cpp)
target_link_libraries(ex7 ${OpenCV_LIBS})