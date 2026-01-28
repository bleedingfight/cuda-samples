#include "../common/device_kernels.cuh"
int main() {
    hello<<<2, 10>>>();
    //    cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}
