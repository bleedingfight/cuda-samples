#include<bits/stdc++.h>
__global__ void add(int n,int* a,int *b)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for(int i=index;i<n;i+=stride)
		b[i] = a[i]+b[i];
}
int main(void)
{
	int N = 1<<20;
	int* a;
	int* b;
	cudaMallocManaged(&a,N*sizeof(int));
	cudaMallocManaged(&b,N*sizeof(int));
	for(int i=0;i<N;i++)
	{
		a[i] = 1;
		b[i] = 2;
	}
	add<<<1,256>>>(N,a,b);
	cudaDeviceSynchronize();
	cudaFree(a);
	cudaFree(b);
	return 0;
}
