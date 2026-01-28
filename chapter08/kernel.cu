#include <cuda_runtime.h>
__global__ void vec_add_scalar(float *data,const float v, int n) {
    auto idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<n)
		data[idx]+=v;
}

