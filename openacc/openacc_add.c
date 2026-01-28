#include <stdio.h>
#define N 256
int main() {
    int i, a[N], b[N], c[N];
    for (i = 0; i < N; i++) {
        a[i] = 0;
        b[i] = c[i] = i;
    }
#pragma acc kernels
    for (i = 0; i < N; i++)
        a[i] = b[i] + c[i];
    printf("a[N-1] = %d\n", a[N - 1]);
    return 0;
}
