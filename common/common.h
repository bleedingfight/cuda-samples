#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#pragma once
#define DURATION_NANO(start)                                                   \
    std::chrono::duration_cast<std::chrono::nanoseconds>(                      \
        std::chrono::high_resolution_clock::now() - start)                     \
        .count()
#define CHECK(err)                                                             \
    do {                                                                       \
        cudaError_t err_code = err;                                            \
        if (err_code != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_code));                             \
            exit(err_code);                                                    \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "CUBLAS error in " << #call << " at line "            \
                      << __LINE__ << ": " << getCublasErrorString(status)      \
                      << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CU_CHECK(apiFuncCall)                                                  \
    do {                                                                       \
        CUresult _status = apiFuncCall;                                        \
        if (_status != CUDA_SUCCESS) {                                         \
            const char *_errorString;                                          \
            cuGetErrorString(_status, &_errorString);                          \
            std::cerr << "CUDA Driver API call failed at line " << __LINE__    \
                      << " with error: " << _errorString << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CURAND(call)                                                     \
    do {                                                                       \
        curandStatus_t error = call;                                           \
        if (error != CURAND_STATUS_SUCCESS) {                                  \
            printf("CURAND error %d at %s:%d\n", error, __FILE__, __LINE__);   \
            return;                                                            \
        }                                                                      \
    } while (0)
// 辅助函数，将CUBLAS错误码转换为可读的字符串
const char *getCublasErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "Unknown CUBLAS error";
    }
}
template <typename T> void initialData(T *ip, const int size) {
    int i;

    for (i = 0; i < size; i++) {
        ip[i] = static_cast<T>((rand() & 0xFF) / 10.0f);
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-5;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5) {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

void printData(const char *msg, int *in, const int size) {
    printf("%s: ", msg);

    for (int i = 0; i < size; i++) {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}
template <typename T> void printData(T *in, const int size) {
    for (int i = 0; i < size; i++) {
        printf("%3.0f ", static_cast<float>(in[i]));
    }

    printf("\n");
    return;
}
/*
 * This example helps to visualize the relationship between thread/block IDs and
 * offsets into data. For each CUDA thread, this example displays the
 * intra-block thread ID, the inter-block block ID, the global coordinate of a
 * thread, the calculated offset into input data, and the input data at that
 * offset.
 */

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }

    printf("\n");
    return;
}
void initData(float *data, const int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 */

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}
