#include <cuda_runtime.h>
typedef struct { int width; int height; float* elements; } Matrix;
#define BLOCK_SIZE 16
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void MatMul(const Matrix A, const Matrix B, Matrix C);
oid MatMul_CPU(const Matrix A, const Matrix B, Matrix C);
