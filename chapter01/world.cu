#include<stdio.h>
__global__ void hello(void)
{
    printf("hello world\n");
}
int main()
{
    hello<<<2,10>>>();
//    cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}
