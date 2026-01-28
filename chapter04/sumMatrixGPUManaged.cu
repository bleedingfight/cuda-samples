#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

/*
 * This example demonstrates the use of CUDA managed memory to implement matrix
 * addition. In this example, arbitrary pointers can be dereferenced on the host
 * and device. CUDA will automatically manage the transfer of data to and from
 * the GPU as needed by the application. There is no need for the programmer to
 * use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
 * explicitly transferring data. In addition, because CUDA managed memory is not
 * forced to reside in a single place it can be transferred to the optimal
 * memory space and not require round-trips over the PCIe bus every time a
 * cross-device reference is performed (as is required with zero copy and UVA).
 */

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged((void **)&A, nBytes));
    CHECK(cudaMallocManaged((void **)&B, nBytes));
    CHECK(cudaMallocManaged((void **)&gpuRef,  nBytes);  );
    CHECK(cudaMallocManaged((void **)&hostRef, nBytes););

    // initialize data at host side
    auto start = std::chrono::high_resolution_clock::now();
    initialData(A, nxy);
    initialData(B, nxy);
    auto elaps = DURATION_NANO(start);
    printf("initialization: \t %ld nano sec\n", elaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    start = std::chrono::high_resolution_clock::now();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    elaps = DURATION_NANO(start);
    printf("sumMatrix on host:\t %ld nano sec\n", elaps);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);

    // after warm-up, time with unified memory
    start = std::chrono::high_resolution_clock::now();

    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);

    CHECK(cudaDeviceSynchronize());
    elaps = DURATION_NANO(start);
    printf("sumMatrix on gpu :\t %ld nano sec <<<(%d,%d), (%d,%d)>>> \n", elaps,
            grid.x, grid.y, block.x, block.y);

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
