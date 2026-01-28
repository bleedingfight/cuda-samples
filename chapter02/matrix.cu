#include "../common/device_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
int main(void) {
    int N = 20 * (1 << 20);
    float *x, *y, *d_x, *d_y;
    //   host上分配空间
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    // device上分配空间
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 移动host中的数据到device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 511) / 512, 512>>>(N, 2.0f, d_x, d_y);

    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = max(maxError, abs(y[i] - 4.0f));
    }

    cout << "Max error: " << maxError << "\n";
    cout << "Effective Bandwidth (GB/s): " << N * 4 * 3 / milliseconds / 1e6
         << "\n";
}
