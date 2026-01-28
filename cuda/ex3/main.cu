#include <iostream>
__global__ void add(int n,int* a,int* b)
{
    for(int i=0;i<n;i++)
        b[i] = a[i]+b[i];

}
int main() {
    int N = 1<<20;
    int* a;
    int* b;
    cudaMallocManaged(&a,N* sizeof(int));
    cudaMallocManaged(&b,N* sizeof(int));
    for(int i = 0;i<N;i++)
    {
        a[i] = 1;
        b[i] = 1;
    }
    add<<<16,1>>>(N,a,b);
    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    return 0;
}