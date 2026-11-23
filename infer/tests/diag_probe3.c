/* diag_probe3.c - B[k][c] = c, A = delta at (k0, all rows).
 * k-shift   -> C[r][k0-8] = k0-8
 * col-shift -> C[r][k0]   = k0+8
 * usage: diag_probe3 M N K k0
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int M = argc > 1 ? atoi(argv[1]) : 8;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int k0 = argc > 4 ? atoi(argv[4]) : 8;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    float *P = calloc((size_t)M * K, 4), *B = malloc((size_t)K * N * 4);
    float *C = malloc((size_t)M * N * 4);
    for (int rb = 0; rb < M / 8; ++rb)
        for (int r = 0; r < 8; ++r) P[((size_t)rb * K + k0) * 8 + r] = 1.0f;
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < N; ++c) B[(size_t)k * N + c] = (float)c;
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
    printf("gemm_run(%d,%d,%d) k0=%d -> %s\n", M, N, K, k0, vc4_strerror(st));
    if (st != VC4_OK) { vc4_shutdown(ctx); gpu_unlock(lock); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, (size_t)M * N * 4);
    for (int r = 0; r < 4; ++r) {
        printf("  r%d:", r);
        for (int c = 0; c < 24; ++c) printf(" %4.0f", C[(size_t)r * N + c]);
        printf("\n");
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
