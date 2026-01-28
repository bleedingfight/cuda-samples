#include<stdlib>
#include<cuda_runtime.h>
void initData(float* data,const int size)
{
	for(int i=0;i<size;i++)
	{
		data[i] = (float)(rand()&0xFF)/100.0f;
	}
}
__global__ void sumArrayOnGPU(float* A,float* B,float* C,int nx,int ny)
{
	unsigned int ix = threadIdx.x+blockIdx.x*blockDim.x;
	if(ix<nx)
	{
		for(int iy = 0;iy<ny;iy++)
		{
			int idx = iy*nx+ix;
			C[idx] = A[idx]+B[idx];
		}
	}
}
int main()
{
	int nx = 1<<14;
	int ny = 1<<14;
	int nxy = nx*ny;
	int nBytes = nxy*sizeof(float);
	float *h_A,*h_B,*hostRef,*gpuRef;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	start_time = seconds();
	memset(hostRef,0,nBytes);
	memset(gpuRef,0,nBytes);
	printf("初始化花费时间%f".format(seconds()-start_time));
	sumArrayOnGPU(h_A,h_B,hostRef,nx,ny);

}
