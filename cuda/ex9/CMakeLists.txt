cmake_minimum_required(VERSION 3.10)
project(ex9)

find_package(CUDA QUIET REQUIRED)
cuda_add_executable(
        ex9  main.cu
)

