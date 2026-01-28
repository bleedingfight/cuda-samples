#include <cstdio>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cuda.h>
#include <stdio.h>
#include "../common/common.h"
std::string loadPTXFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (file.is_open()) {
        std::string ptx = std::string((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
        file.close();
        return ptx;
    } else {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        exit(1);
    }
}

int main(int argc,char** argv){
    cuInit(0);
    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));
    // 创建执行上下文
        CUcontext context;
    CU_CHECK(cuCtxCreate(&context, 0, device));
    CUmodule module;
    const char *cu_bin_file = "vec_add_scalar.ptx";
    const char *kernel_name = "_Z14vec_add_scalarPffi";
    float value = 1.f;
    std::string ptx = loadPTXFromFile(cu_bin_file);

    CU_CHECK(cuModuleLoad(&module, cu_bin_file));
    // CU_CHECK(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0)); // 从ptx字符串数据加载
    // 或者
    // cuModuleLoadData(&module, myCubinData);
    CUfunction kernel;
    CU_CHECK(cuModuleGetFunction(&kernel, module, kernel_name));
    int n = 1<<20;
    float *h_input = new float[n];
    std::fill(h_input,h_input+n,2.f);
    size_t size = n * sizeof(float);

    CUdeviceptr deviceData,d_input;
    CU_CHECK(cuMemAlloc(&deviceData, size));
    CU_CHECK(cuMemAlloc(&d_input, size));

    CU_CHECK(cuMemcpyHtoD(deviceData,h_input,size));

    void *kernelParams[] = {&deviceData,&value, &n};
    unsigned int blockDimX = 128;
    unsigned int blockDimY = 1;
    unsigned int blockDimZ = 1;

    unsigned int gridDimX = (n+blockDimX-1)/blockDimX;
    unsigned int gridDimY = 1;
    unsigned int gridDimZ = 1;

    CU_CHECK(cuLaunchKernel(kernel, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, 0, kernelParams, 0));
    cuCtxSynchronize();

    float *hData = new float[n];
    cuMemcpyDtoH(hData, deviceData, size);
    for(int i=0;i<10;i++)
        std::cout<<"hData["<<i<<"]="<<hData[i]<<"\n";
    cuModuleUnload(module);
    cuCtxDestroy(context);
    cuMemFree(deviceData);
}

