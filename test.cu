#include<bits/stdc++.h>
__global__ void add(int n,int* a,int *b)
{
    //将数组a和b中的元素求和放到b
	for(int i=0;i<n;i++)
		b[i] = a[i]+b[i];
}
int main(void)
{
	int N = 1<<20;
	int* a;
	int* b;
	cudaMallocManaged(&a,N*sizeof(int));//分配足够的空间存储a和b
	cudaMallocManaged(&b,N*sizeof(int));
    // 初始化数组a和b中的元素
	for(int i=0;i<N;i++)
	{
		a[i] = 1;
		b[i] = 2;
	}
    //调用kernal函数执行计算
	add<<<1,1>>>(N,a,b);
    //同步操作
	cudaDeviceSynchronize();
    //释放GPU空间
	cudaFree(a);
	cudaFree(b);
	return 0;
}
