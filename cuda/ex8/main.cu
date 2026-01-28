#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(call){                                          \
    const cudaError_t error = call;                           \
    if (error != cudaSuccess){                                \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,      \
                cudaGetErrorString(error));                   \
        exit(1);                                              \
    }                                                         \
}
//初始化数组中每个元素的值为它的下标
void initialInt(int* ip,int size){
    for(int i = 0;i<size;i++)
        ip[i] = i;
}
void printMatrix(int* C,const int nx,const int ny)
{
    int *ic = C;
    printf("\nMatrix:(%d,%d)\n");
    for(int iy = 0;iy<ny;iy++)
    {
        for(int ix = 0;ix<nx;ix++)
            printf("%3d",ic[ix]);
        ic += nx;
        printf("\n");
    }
    printf("\n");
}
__global__ void printThreadIndex(int *A,const int nx,const int ny){
    int ix = threadIdx.x+blockIdx.x*blockDim.x;
    int iy = threadIdx.y+blockIdx.y*blockDim.y;
    unsigned int idx = iy*nx+ix;
    printf("thread_id (%d,%d),block_id (%d,%d),coordinate (%d,%d),global index: %2d val: %2d\n",threadIdx,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}
int main(int argc,char **argv)
{
    printf("%s String...\n",argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d:%s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));
    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy*sizeof(float);
    int *h_A;
    //分配能安放48个float的空间
    h_A = (int *)malloc(nBytes);
    initialInt(h_A,nxy);
    printMatrix(h_A,nx,nxy);
    int *d_MatA;
    cudaMalloc((void **)&d_MatA,nBytes);
    cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
    printThreadIndex<<<grid,block>>>(d_MatA,nx,ny);
    cudaDeviceSynchronize();
    cudaFree(d_MatA);
    free(h_A);
    cudaDeviceReset();
    return 0;

}