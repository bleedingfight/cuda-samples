#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <mma.h>

// 错误检查函数
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// 性能测试函数
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                        cudaStream_t stream, int num_repeats = 100,
                        int num_warmups = 100) {
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i) {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time / num_repeats;
}

// GPU实现：16x32 * 32x16 矩阵乘法
template <typename T1, typename T2>
__global__ void wmma_gemm_16_32(T1 const* A, T1 const* B, T2* C,
                               uint32_t lda, uint32_t ldb, uint32_t ldc) {
    // 声明fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T1,
                          nvcuda::wmma::col_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T1,
                          nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, T2>
        acc_frag;

    // 初始化累加器为0
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    // 在K维度上循环（K=32，需要两次16x16的运算）
    for (int ki = 0; ki < 32; ki += 16) {
        // 计算当前块的内存地址
        T1 const* matrix_a_ptr = A + ki * lda;  // A矩阵的第ki列
        T1 const* matrix_b_ptr = B + ki;        // B矩阵的第ki行

        // 加载数据到fragment
        nvcuda::wmma::load_matrix_sync(a_frag, matrix_a_ptr, lda);
        nvcuda::wmma::load_matrix_sync(b_frag, matrix_b_ptr, ldb);

        // 执行矩阵乘法并累加
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 存储结果
    nvcuda::wmma::store_matrix_sync(C, acc_frag, ldc, nvcuda::wmma::mem_col_major);
}

// CPU实现：矩阵乘法
template <typename T1, typename T2>
void cpu_gemm_16_32(T1 const* A, T1 const* B, T2* C) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            T2 sum = 0;
            for (int k = 0; k < 32; ++k) {
                // A是16x32的矩阵，按列主序存储，所以A[k * 16 + i]表示A(i,k)
                // B是32x16的矩阵，按列主序存储，所以B[j * 32 + k]表示B(k,j)
                sum += static_cast<T2>(A[k * 16 + i]) * 
                      static_cast<T2>(B[j * 32 + k]);
            }
            // C是16x16的矩阵，按列主序存储，所以C[j * 16 + i]表示C(i,j)
            C[j * 16 + i] = sum;
        }
    }
}

// 初始化数据函数
void fill_random_half(__half* arr, size_t n, std::default_random_engine& e) {
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = __float2half(uniform_dist(e));
    }
}

// 比较结果函数
bool compare_results(float const* arr1, float const* arr2, size_t n, float tolerance = 1e-3) {
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(arr1[i] - arr2[i]);
        float max_val = std::max(std::abs(arr1[i]), std::abs(arr2[i]));
        if (diff > tolerance * max_val) {
            std::cout << "Mismatch at index " << i << ": CPU=" << arr1[i] 
                     << ", GPU=" << arr2[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 设置矩阵维度
    constexpr int M = 16;  // 矩阵A的行数
    constexpr int K = 32;  // 矩阵A的列数，也是矩阵B的行数
    constexpr int N = 16;  // 矩阵B的列数

    // 创建CUDA流
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // 分配和初始化主机内存
    std::vector<__half> h_A(M * K);  // 16x32矩阵
    std::vector<__half> h_B(K * N);  // 32x16矩阵
    std::vector<float> h_C_cpu(M * N);  // CPU结果
    std::vector<float> h_C_gpu(M * N);  // GPU结果

    // 随机数生成
    std::default_random_engine random_engine(0);
    fill_random_half(h_A.data(), h_A.size(), random_engine);
    fill_random_half(h_B.data(), h_B.size(), random_engine);

    // 分配设备内存
    __half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 将数据拷贝到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), 
                               cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), 
                               cudaMemcpyHostToDevice));

    // 执行CPU版本
    cpu_gemm_16_32(h_A.data(), h_B.data(), h_C_cpu.data());

    // 执行GPU版本
    dim3 grid(1);
    dim3 block(32);  // 一个warp
    wmma_gemm_16_32<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, M);
    CHECK_LAST_CUDA_ERROR();

    // 将结果拷贝回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), 
                               cudaMemcpyDeviceToHost));

    // 比较结果
    bool results_match = compare_results(h_C_cpu.data(), h_C_gpu.data(), M * N);
    if (results_match) {
        std::cout << "CPU和GPU结果匹配！" << std::endl;
    } else {
        std::cout << "CPU和GPU结果不匹配！" << std::endl;
    }

    // 测试性能
    std::function<void(cudaStream_t)> const gpu_function{
        std::bind(wmma_gemm_16_32<__half, float>, d_A, d_B, d_C, M, K, M)};
    float const latency = measure_performance(gpu_function, stream);
    std::cout << "GPU执行时间: " << latency << " ms" << std::endl;

    // 清理资源
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}