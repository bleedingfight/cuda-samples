#include <stdio.h>
#include <time.h>
// 两个向量求和
void sumArrayOnHost(float* A,float* B, float* C,const int N)
{
    for(int i=0;i<N;i++)
        C[i] = A[i]+B[i];
}
//初始化长度为size的向量
void initialData(float* ip,int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        ip[i] = (float)(rand()&0xFF)/10.f;
    }
}
int main(int argc,char** argv) {
    int nElem = 1024;
    size_t nBytes = nElem*sizeof(float);
    float *h_A,*h_B,*h_C,gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    float *d_A,*d_B,*d_C;
    cudaMalloc((float**)&d_A,nBytes);
    cudaMalloc((float**)&d_B,nBytes);
    cudaMalloc((float**)&d_C,nBytes);
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(d_C,h_C,nBytes,cudaMemcpyDeviceToHost);
    initialData(h_A,nElem);
    initialData(h_B,nElem);
    sumArrayOnHost(h_A,h_B,h_C,nElem);
    cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for(int i=0;i<nBytes;i++)
        printif("%d",*(gpuRef+i));
    free(gpuRef);
    return 0;
}
