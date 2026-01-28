cmake_minimum_required(VERSION 3.11)
project(ex2)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_C_COMPILER g++)
cuda_add_executable(
        ex2 main.cu
)
target_link_libraries(ex2 ${CUDA_LIBRARIES})
