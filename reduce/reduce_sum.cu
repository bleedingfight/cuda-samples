#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#define BLOCK 1024
__global__ void reduce_sum_kernel(float* d_out,float*d_in,const int n){
    auto idx = blockDim.x*blockIdx.x;
    for(int span = blockDim.x/2;span>1;span/=2){
	if(threadIdx.x<span){
	    d_in[idx+threadIdx.x] += d_in[idx+threadIdx.x+span];
	}
    }
    __syncthreads();
    if(idx==0){
	d_out[blockIdx.x] = d_in[idx];
    }

}

float reduce_sum_cpu(float*h_in,const int N){
    float value = 0.f;
    for(int i=0;i<N;i++){
	value += h_in[i];
    }
    return value;
}

float reduce_sum_cuda(float *h_in,const int N){
    float *d_in,*d_out;
    size_t nbytes = N*sizeof(float);
    int grid = (N+BLOCK-1)/BLOCK;
    size_t obytes = grid*sizeof(float);
    float *h_out = new float[BLOCK];
    float data = 0.f;
    cudaMalloc(reinterpret_cast<void**>(&d_in),nbytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),obytes);
    cudaMemcpy(d_in,h_in,nbytes,cudaMemcpyHostToDevice);
    reduce_sum_kernel<<<grid,BLOCK>>>(d_out,d_in,N);
    cudaMemcpy(h_out,d_out,obytes,cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);
    data = reduce_sum_cpu(h_out,BLOCK);
    delete [] h_out;
    return data;


}
int main(){
    int N = 1<<12;
    float *h_in = new float[N];
    std::fill(h_in,h_in+N,1);
    auto h_out = reduce_sum_cpu(h_in, N);
    // auto d_out = N;
    auto d_out = reduce_sum_cuda(h_in,N);
    if(abs(h_out-d_out)<1e-6){
	std::cout<<"计算正确\n";
    }else{
	std::cout<<"计算错误\n";
    }
    delete []h_in;
}
