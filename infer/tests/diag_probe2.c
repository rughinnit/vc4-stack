/* diag_probe2.c - B = identity, so C[r][c] = A[r][c] for c < K.
 * usage: diag_probe2 M N K
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    float *A = malloc((size_t)M * K * 4), *P = malloc((size_t)M * K * 4);
    float *B = calloc((size_t)K * N, 4), *C = malloc((size_t)M * N * 4);
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (float)(r * 100 + k);
    for (int k = 0; k < K && k < N; ++k) B[(size_t)k * N + k] = 1.0f;
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * K + k) * 8 + r] = A[(size_t)(rb * 8 + r) * K + k];
    vc4_buf bp, bb, bc;
    vc4_alloc(ctx, (size_t)M * K * 4, &bp);
    vc4_alloc(ctx, (size_t)K * N * 4, &bb);
    vc4_alloc(ctx, (size_t)M * N * 4, &bc);
    vc4_upload(ctx, &bp, P, (size_t)M * K * 4);
    vc4_upload(ctx, &bb, B, (size_t)K * N * 4);
    memset(C, 0xCD, (size_t)M * N * 4);
    vc4_upload(ctx, &bc, C, (size_t)M * N * 4);
    vc4_status st = vc4__gemm_run(ctx, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                  vc4_bus_addr(ctx, &bb), vc4_bus_addr(ctx, &bc),
                                  vc4_bus_addr(ctx, &bp));
    printf("gemm_run(%d,%d,%d) -> %s\n", M, N, K, vc4_strerror(st));
    if (st != VC4_OK) { vc4_shutdown(ctx); gpu_unlock(lock); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, (size_t)M * N * 4);
    for (int r = 0; r < M && r < 8; ++r) {
        printf("  r%d got:", r);
        for (int c = 0; c < K; ++c) printf(" %4.0f", C[(size_t)r * N + c]);
        printf("   want:");
        for (int c = 0; c < K; ++c) printf(" %4d", r * 100 + c);
        printf("\n");
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
