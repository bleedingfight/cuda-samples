#include <stdio.h>
#define L 32
#define M 32
#define N 32
int main() {
    int i, j, k, a[L][M][N], b[L][M][N], c[L][M][N];
    for (i = 0; i < L; i++)
        for (j = 0; j < M; j++)
            for (k = 0; k < N; k++) {
                a[i][j][k] = -1;
                b[i][j][k] = c[i][j][k] = i + j + k;
            }
#pragma acc kernels
    for (i = 0; i < L; i++)
        for (j = 0; j < M; j++)
            for (k = 0; k < N; k++)
                a[i][j][k] = b[i][j][k] + c[i][j][k];
    printf("a[L-1][M-1][N-1] = %d\n", a[L - 1][M - 1][N - 1]);
    return 0;
}
