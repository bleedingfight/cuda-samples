#include <algorithm>
#include <numeric>
#include <stdio.h>
#define N 1024
int main() {
    int i, a[N], b[N], c[N];
    std::fill(a, a + N, 0);
    std::iota(b, b + N, 0);
    std::iota(c, c + N, 0);
#pragma acc kernels
    {
#pragma acc loop
        for (i = 0; i < N; i++)
            a[i] = b[i] + c[i];
        /* #pragma acc loop */
#pragma acc loop independent
        for (i = 1; i < N; i++)
            b[i] = b[i - 1];
    }
    printf("b[%d] = %d\n", 2, b[2]);
    return 0;
}
