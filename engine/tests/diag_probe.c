/* diag_probe.c - probe the asm10 GEMM's A mapping with delta panels.
 * usage: diag_probe M N K [r0]
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

static vc4_ctx *ctx;
static vc4_buf bp, bb, bc;
static float *P, *B, *C;
static int M, N, K;

static void pack_delta(int k0, int r0)     /* r0 < 0: all rows */
{
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * K + k) * 8 + r] =
                    (k == k0 && (r0 < 0 || r == r0)) ? 1.0f : 0.0f;
    vc4_upload(ctx, &bp, P, (size_t)M * K * 4);
}

static int run(void)
{
    memset(C, 0xCD, (size_t)M * N * 4);
    vc4_upload(ctx, &bc, C, (size_t)M * N * 4);
    vc4_status st = vc4__gemm_run(ctx, (uint32_t)M, (uint32_t)N, (uint32_t)K,
                                  vc4_bus_addr(ctx, &bb), vc4_bus_addr(ctx, &bc),
                                  vc4_bus_addr(ctx, &bp));
    if (st != VC4_OK) { printf("gemm: %s\n", vc4_strerror(st)); return -1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bc, C, (size_t)M * N * 4);
    return 0;
}

int main(int argc, char **argv)
{
    M = argc > 1 ? atoi(argv[1]) : 8;
    N = argc > 2 ? atoi(argv[2]) : 64;
    K = argc > 3 ? atoi(argv[3]) : 16;
    int r0 = argc > 4 ? atoi(argv[4]) : -1;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }
    P = malloc((size_t)M * K * 4);
    B = malloc((size_t)K * N * 4);
    C = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)K * N; ++i) B[i] = 1.0f;
    vc4_alloc(ctx, (size_t)M * K * 4, &bp);
    vc4_alloc(ctx, (size_t)K * N * 4, &bb);
    vc4_alloc(ctx, (size_t)M * N * 4, &bc);
    vc4_upload(ctx, &bb, B, (size_t)K * N * 4);

    if (r0 < 0) {
        printf("M=%d N=%d K=%d: delta at k0 -> C[r][0] should be 1 for every r\n",
               M, N, K);
        for (int k0 = 0; k0 < K; ++k0) {
            pack_delta(k0, -1);
            if (run()) return 1;
            printf("  k0=%2d:", k0);
            for (int r = 0; r < M && r < 16; ++r)
                printf(" %5.1f", C[(size_t)r * N]);
            printf("\n");
        }
    } else {
        printf("M=%d N=%d K=%d r0=%d: delta at (r0,k0) -> only row %d\n",
               M, N, K, r0, r0);
        for (int k0 = 0; k0 < K; ++k0) {
            pack_delta(k0, r0);
            if (run()) return 1;
            printf("  k0=%2d:", k0);
            for (int r = 0; r < M && r < 16; ++r)
                printf(" %5.1f", C[(size_t)r * N]);
            printf("\n");
        }
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return 0;
}
