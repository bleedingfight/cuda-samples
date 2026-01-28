#include <algorithm>
#include <numeric>
#include <stdio.h>
#define N 1024
int main() {
    int a[N], b[N], c[N];
    std::iota(a, a + N, 0);
    std::fill(b, b + N, 1);
    std::fill(c, c + N, 2);
#pragma acc kernels
#pragma acc loop
    for (int i = 0; i < N; i++) {
        a[a[i]] = c[i];
    }
    printf("a[%d]=%d\n", 2, c[2]);
}
