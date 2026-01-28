#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
/*
 * An example of using shared memory to transpose square thread coordinates
 * of a CUDA grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 */

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

// 按行写入SMEM，按行从SMEM写入GDRAM
__global__ void writeRowSmemReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

// 按列将数据写入SMEM，按列将SMEM的数据写入GDRAM
__global__ void writeSmemColReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
// 按行将数据写入SMEM，按列将数据读到GDRAM
__global__ void writeRowSmemReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// 动态共享内存按行写入SMEM，按行读
__global__ void writeRowSmemReadColDyn(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    // shared memory store operation
    tile[row_idx] = row_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

// 按行写入静态SMEM，按列读到GDRAM
__global__ void writeRowSmemReadColPad(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from thread index to global memory offset
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// 动态一维共享内存
__global__ void writeRowReadColDynPad(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size 2048
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1)
        iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
           block.y);

    // allocate device memory
    int *d_C;
    CHECK(cudaMalloc(reinterpret_cast<int **>(&d_C), nBytes));
    int *gpuRef = new int[nx * ny];

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeSmemColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem by col read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeRowSmemReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem by read row   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeRowSmemReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem by row read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeRowSmemReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem row read col dyn", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeRowSmemReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem row read col pad", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    writeRowReadColDynPad<<<grid, block,
                            (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iprintf)
        printData("write smem row read col DP ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(cudaFree(d_C));
    delete[] gpuRef;

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
