#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 8;
int N = 12;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = static_cast<float>(static_cast<int>((dr / rMax) * 10.0));
    }

    *outX = X;
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = static_cast<float>(static_cast<int>((dr / rMax) * 10.0));
        }
    }

    *outA = A;
}
void hostGemv(float *h_out,float *A,float *X,float *Y,const float alpha,const float beta,const int M,const int N){
    for(int r = 0;r<M;r++){
        float s = 0;
        for(int c=0;c<N;c++){
            s+=alpha*A[r*N+c]*X[c]+beta*Y[c];
        }
        h_out[r] = s;
    }

}

int main(int argc, char **argv)
{
    int i;
    float *A, *dA;
    float *X, *dX;
    float *Y, *dY;
    float *h_out = new float[M];
    float beta;
    float alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // 创建cublas handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // 分配对应的内存
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N));
    CHECK(cudaMalloc((void **)&dY, sizeof(float) * M));

    // 从host内存X拷贝float类型的数据N个元素到device的dX位置，拷贝的时候源数据是一个一个连续拷贝到目的位置
    CHECK_CUBLAS(cublasSetVector(N, sizeof(float), X, 1, dX, 1));
    CHECK_CUBLAS(cublasSetVector(M, sizeof(float), Y, 1, dY, 1));
    // 从host拷贝矩阵A的数据到device dA位置，拷贝的时候
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));

    // Execute the matrix-vector multiplication
    hostGemv(h_out,A,X,Y,alpha,beta,M,N);
    // alpha*dA[MxN]*dX+beta*dY 
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, dA, M, dX, 1,
                             &beta, dY, 1));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetVector(M, sizeof(float), dY, 1, Y, 1));
    for (i = 0; i < M; i++)
    {
        printf("cublas %2.2f cpu = %2.2f \n", Y[i],h_out[i]);
    }

    printf("...\n");

    free(A);
    free(X);
    free(Y);
    delete [] h_out;

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dY));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
