#include<stdio.h>
#define BLOCK_NUMS 16
#define THREAD_NUMS 1
__global__ void hello()
{
    printf("Hello World:I'am a thread in block%d\n",blockIdx.x);
}
int main(){
    hello<<<BLOCK_NUMS,THREAD_NUMS>>>();
    cudaDeviceSynchronize();
    printf("This all");
    return 1;
}
