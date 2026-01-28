#include "../common/common.h"
#include "../common/device_kernels.cuh"
#include <stdio.h>

int main(int argc, char **argv) {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}
