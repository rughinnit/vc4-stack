/* diag_gemm10.c - drive the asm10 GEMM with a CPU-built panel.
 * usage: diag_gemm10 M N K [mode]
 *   mode 0 = A all ones (panel ones), 1 = A = delta at k=0, 2 = A = delta at k=1
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int M = argc > 1 ? atoi(argv[1]) : 16;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int mode = argc > 4 ? atoi(argv[4]) : 0;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(ctx ? NULL : &ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    size_t np = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *P = malloc(np * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * K + k) * 8 + r] =
                    (mode == 0) ? 1.0f : ((k == mode - 1) ? 1.0f : 0.0f);
    for (size_t i = 0; i < nb; ++i) B[i] = 1.0f;
    vc4_buf bp, bb, bc;
    vc4_alloc(ctx, np * 4, &bp);
    vc4_alloc(ctx, nb * 4, &bb);
    vc4_alloc(ctx, nc * 4, &bc);
    vc4_upload(ctx, &bp, P, np * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    memset(C, 0xCD, nc * 4);
    vc4_upload(ctx, &bc, C, nc * 4);
    vc4_status st = vc4__gemm_run(ctx, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                  vc4_bus_addr(ctx, &bb), vc4_bus_addr(ctx, &bc),
                                  vc4_bus_addr(ctx, &bp));
    printf("gemm_run(%d,%d,%d) -> %s\n", M, N, K, vc4_strerror(st));
    if (st != VC4_OK) { vc4_shutdown(ctx); gpu_unlock(lock); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, nc * 4);
    for (int r = 0; r < 4 && r < M; ++r) {
        printf("  C[%d][0..3] =", r);
        for (int n = 0; n < 4; ++n)
            printf(" %.9g(%08x)", C[(size_t)r * N + n],
                   *(unsigned *)&C[(size_t)r * N + n]);
        printf("\n");
    }
    for (int r = 0; r < M; ++r) {
        printf("  r%2d:", r);
        for (int c = 0; c < N; ++c) putchar(C[(size_t)r * N + c] == (float)K ? '#' : '.');
        printf("  %.9g %08x\n", C[(size_t)r * N], *(unsigned *)&C[(size_t)r * N]);
    }
    long nz = 0;
    for (size_t i = 0; i < nc; ++i) if (C[i] != 0.0f) nz++;
    printf("nonzero=%ld/%zu\n", nz, nc);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
