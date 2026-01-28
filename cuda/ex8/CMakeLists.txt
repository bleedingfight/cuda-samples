cmake_minimum_required(VERSION 3.10)
find_package(CUDA QUIET REQUIRED)
cuda_add_executable(
        ex8  main.cu
    )

