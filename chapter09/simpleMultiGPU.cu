#include "../common/common.h"
#include <chrono>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

/*
 * A simple example of a multi-GPU CUDA application implementing a vector sum.
 * Note that all communication and computation is done asynchronously in order
 * to overlap computation across multiple devices, and that this requires
 * allocating page-locked host memory associated with a specific device.
 */

__global__ void iKernel(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

void sumOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    int ngpus;

    printf("> starting %s", argv[0]);

    CHECK(cudaGetDeviceCount(&ngpus));
    printf(" CUDA-capable devices: %i\n", ngpus);

    int ishift = 24;

    if (argc > 2) ishift = atoi(argv[2]);

    int size = 1 << ishift;

    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            exit(1);
        }

        ngpus  = atoi(argv[1]);
    }

    int    iSize  = size / ngpus;
    size_t iBytes = iSize * sizeof(float);

    printf("> total array size %d M, using %d devices with each device "
            "handling %d M\n", size / 1024 / 1024, ngpus, iSize / 1024 / 1024);

    // 分配指针数组指向ngpus个gpu的指针
    float** d_A = new float*[ngpus];
    float** d_B = new float*[ngpus];
    float** d_C = new float*[ngpus];


    float** h_A = new float*[ngpus];
    float** h_B = new float*[ngpus];
    float** hostRef = new float*[ngpus];
    float** gpuRef = new float*[ngpus];


    cudaStream_t *stream = new cudaStream_t[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        // set current device
        CHECK(cudaSetDevice(i));

        // 为不同GPU分配他们需要的内存
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), iBytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), iBytes));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), iBytes));

        // 为每个GPU分配锁页内存（CPU内存）
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_A[i]),     iBytes));
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_B[i]),     iBytes));
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostRef[i]), iBytes));
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&gpuRef[i]),  iBytes));

        // 为每个GPU创建流
        CHECK(cudaStreamCreate(&stream[i]));
    }

    dim3 block (512);
    dim3 grid  ((iSize + block.x - 1) / block.x);

    // 为不同的GPU初始化锁页内存上的数据
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        initialData(h_A[i], iSize);
        initialData(h_B[i], iSize);
    }

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 开启异步传输数据
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        // 将数据异步拷贝到GPU
        CHECK(cudaMemcpyAsync(d_A[i], h_A[i], iBytes, cudaMemcpyHostToDevice,
                              stream[i]));
        CHECK(cudaMemcpyAsync(d_B[i], h_B[i], iBytes, cudaMemcpyHostToDevice,
                              stream[i]));

        iKernel<<<grid, block, 0, stream[i]>>>(d_A[i], d_B[i], d_C[i], iSize);

        CHECK(cudaMemcpyAsync(gpuRef[i], d_C[i], iBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
    }

    // 同步流
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(stream[i]));
    }

    // 统计多GPU下计算耗时
    auto elaps = DURATION_NANO(start);
    printf("%d GPU timer elapsed: %ld ns \n", ngpus, elaps);

    // 检查GPU计算结果是否正确
    for (int i = 0; i < ngpus; i++)
    {
        //Set device
        CHECK(cudaSetDevice(i));
        sumOnHost(h_A[i], h_B[i], hostRef[i], iSize);
        checkResult(hostRef[i], gpuRef[i], iSize);
    }

    // 清理内存
    for (int i = 0; i < ngpus; i++)
    {
        // 先选中GPU
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(d_A[i]));
        CHECK(cudaFree(d_B[i]));
        CHECK(cudaFree(d_C[i]));

        CHECK(cudaFreeHost(h_A[i]));
        CHECK(cudaFreeHost(h_B[i]));
        CHECK(cudaFreeHost(hostRef[i]));

        CHECK(cudaFreeHost(gpuRef[i]));
        CHECK(cudaStreamDestroy(stream[i]));

        CHECK(cudaDeviceReset());
    }

    delete [] d_A;
    delete [] d_B;
    delete [] d_C;
    delete [] h_A;
    delete [] h_B;

    delete [] hostRef;
    delete [] gpuRef;
    delete [] stream;
    return EXIT_SUCCESS;
}
