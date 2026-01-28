cmake_minimum_required(VERSION 3.10)
project(ex3)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_C_COMPILER g++)
cuda_add_executable(
        ex3 main.cu
)
target_link_libraries(ex3 ${CUDA_LIBRARIES})