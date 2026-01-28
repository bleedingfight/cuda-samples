#include "../common/common.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#define NVRTC_CHECK(Name, x)                                                   \
    do {                                                                       \
        nvrtcResult result = x;                                                \
        if (result != NVRTC_SUCCESS) {                                         \
            std::cerr << "\nerror: " << Name << " failed with error "          \
                      << nvrtcGetErrorString(result);                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/**
 * @brief 生成均匀分布数据的模板函数
 * * @tparam T 数据类型 (支持 int, float, double 等)
 * @param p_data 指向数据的指针
 * @param n 数据量
 * @param lower 均匀分布下限
 * @param upper 均匀分布上限
 * @param seed 随机数种子，默认为 0
 */
template <typename T>
void generate_uniform_data(T *p_data, const size_t n, T lower, T upper,
                           unsigned int seed = 0) {
    if (p_data == nullptr || n == 0)
        return;

    // 选择随机数引擎 (MT19937 是目前最常用的高质量伪随机数生成器)
    std::mt19937 gen(seed);

    // 根据类型 T 自动选择分布器
    // 如果 T 是浮点型，使用 uniform_real_distribution
    // 如果 T 是整型，使用 uniform_int_distribution
    using Distribution =
        typename std::conditional<std::is_floating_point<T>::value,
                                  std::uniform_real_distribution<T>,
                                  std::uniform_int_distribution<T>>::type;

    Distribution dist(lower, upper);

    // 填充数据
    for (size_t i = 0; i < n; ++i) {
        p_data[i] = dist(gen);
    }
}
/**
 * @brief 读取整个文件内容
 * @param filePath 文件路径
 * @return std::string 文件内容字符串（若打开失败则返回空字符串）
 */
std::string readFileContents(const std::string &filePath) {
    std::ifstream fileStream(filePath);

    // 检查文件是否成功打开
    if (!fileStream.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return "";
    }

    // 使用 stringstream 读取整个缓冲区
    std::stringstream buffer;
    buffer << fileStream.rdbuf();

    return buffer.str();
}
template <typename T>
__global__ void vector_add(T *d_out, const T *d_in1, const T *d_in2,
                           const size_t N) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = d_in1[idx] + d_in2[idx];
    }
}
template <typename T>
void vector_add_cpu(T *h_out, const T *h_in1, const T *h_in2, const size_t N) {
    for (int i = 0; i < N; i++) {
        h_out[i] = h_in1[i] + h_in2[i];
    }
}
void loadSourceRun(std::string &filename) {
    const char *cu_src = readFileContents(filename).c_str();
    nvrtcResult prog;
    nvrtcCreateProgram(
        &prog,           // [输出] 指向程序句柄的指针
        cu_src,          // [输入] CUDA 源代码字符串
        "vector_add.cu", // [输入]
                         // 程序名称（可选，用于报错提示，通常传文件名）
        0,       // [输入] 包含的头文件数量
        nullptr, // [输入] 头文件内容（字符串数组）
        nullptr  // [输入] 代码中 #include 的名称
    );
    std::vector<const char *> opts;
    opts.push_back("--gpu-architecture=compute_90"); // 设置你的 GPU 架构
    opts.push_back("--use_fast_math");
    opts.push_back("-O3");

    nvrtcResult res =
        nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (compileRes != NVRTC_SUCCESS) {
        // 即使失败，也要获取日志，因为日志里写了报错原因（类似编译器报错）
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);

        std::cerr << "编译失败！错误日志如下：\n" << log << std::endl;
        exit(1);
    } else {
        std::cout << "NVRTC 编译成功！" << std::endl;
    }
}
int main(int argc, char **argv) {
    const size_t n = 1 << 20;
    size_t nbytes = sizeof(float) * n;
    float *h_in1 = new float[n];
    float *h_in2 = new float[n];
    float *h_out = new float[n];
    float *h_gpu = new float[n];
    generate_uniform_data(h_in1, n, 0.f, 1.f, 1);
    generate_uniform_data(h_in2, n, 0.f, 1.f, 2);

    float *d_in1, *d_in2, *d_out;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in1), nbytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_in2), nbytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_out), nbytes));
    CHECK(cudaMemcpy(d_in1, h_in1, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in2, h_in2, nbytes, cudaMemcpyHostToDevice));

    dim3 block = 1024;
    dim3 grid = (n + block.x - 1) / block.x;
    vector_add<<<grid, block>>>(d_out, d_in1, d_in2, n);
    CHECK(cudaMemcpy(h_gpu, d_out, nbytes, cudaMemcpyDeviceToHost));
    vector_add_cpu(h_out, h_in1, h_in2, n);
    for (int i = 0; i < 10; i++) {
        if (std::abs(h_out[i] - h_gpu[i]) > 1e-4) {
            std::cout << "Error\n";
        }
    }
    CHECK(cudaFree(d_in1));
    CHECK(cudaFree(d_in2));
    CHECK(cudaFree(d_out));
    delete[] h_in1;
    delete[] h_in2;
    delete[] h_out;
    delete[] h_gpu;
}
