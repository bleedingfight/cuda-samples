#include <mma.h> 
#include <iostream>
using namespace nvcuda; 
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
__global__ void wmma_kernel(half *a, half *b, float *c) { 
    //Declare the fragments 
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag; 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    // Initialize the output to zero 
    wmma::fill_fragment(c_frag, 0.0f);
    // Load the inputs 
    wmma::load_matrix_sync(a_frag, a, 16); 
    wmma::load_matrix_sync(b_frag, b, 16);
    // Perform the matrix multiplication 
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); 
    // Store the output 
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
template<typename T>
void transpose(T*out,T*in,const int rows,const int cols){
    for(int row=0;row<rows;row++){
        for(int col = 0;col<cols;col++){
            out[col*cols+row] = in[row*cols+col];
        }
    }
}
template<typename T>
void fill_range(T*data,const int rows,const int cols,int init = 0){
    for(int r=0;r<rows;r++){
        for(int c=0;c<cols;c++){
            data[r*cols+c] = init++;
        }
    }
}
__global__ void float_to_half(__half*out ,float* in,const int N){
    auto idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        out[idx] = __float2half(in[idx]);
    }

}
int main(){
    const int M = 16;
    const int N = 16;
    const int K = 16;
    float *h_a = new float[M*K];
    float *h_b = new float[K*N];
    float *h_c = new float[M*N];

    __half *h_a_half = new __half[M*K];

    float *d_a,*d_b;
    __half *d_a_half,*d_b_half;
    fill_range(h_a,M,K,0);
    transpose(h_b,h_a,M,K);

    float *d_c;

    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_a),M*K*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_b),K*N*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_a_half),M*K*sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_b_half),K*N*sizeof(__half)));

    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_c),M*N*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,M*K*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b,h_b,K*N*sizeof(float),cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERROR(cudaMemcpy(d_a_half,h_a_half, M*K*sizeof(__half),cudaMemcpyHostToDevice));


    float_to_half<<<16,16>>>(d_a_half,d_a,M*K);
    float_to_half<<<16,16>>>(d_b_half,d_b,K*N);

	dim3 gridDim = {4,1,1};
	dim3 blockDim = {128,1,1};
    wmma_kernel<<<gridDim,blockDim>>>(d_a_half, d_b_half, d_c);
    CHECK_CUDA_ERROR(cudaMemcpy(h_c,d_c,M*N*sizeof(__half),cudaMemcpyDeviceToHost));

    for(int m=0;m<M;m++){
        for(int k=0;k<N;k++){
            std::cout<<h_c[m*K+k]<<" ";
        }
        std::cout<<"\n";
    }
    delete [] h_a;
    delete [] h_b;
    delete [] h_c;

}

