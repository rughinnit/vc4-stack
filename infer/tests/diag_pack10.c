/* diag_pack10.c - dump the asm10 panel builder and compare with the reference.
 * usage: diag_pack10 [K] [M]
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int K = argc > 1 ? atoi(argv[1]) : 16;
    int M = argc > 2 ? atoi(argv[2]) : 16;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    size_t n = (size_t)M * K;
    float *A = malloc(n * 4), *P = malloc(n * 4);
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            A[(size_t)r * K + k] = (float)(r * 1000 + k);
    vc4_buf ba, bp;
    vc4_alloc(ctx, n * 4, &ba);
    vc4_alloc(ctx, n * 4, &bp);
    vc4_upload(ctx, &ba, A, n * 4);
    vc4_status st = vc4__pack_run(ctx, vc4_bus_addr(ctx, &ba), (uint32_t)K,
                                  vc4_bus_addr(ctx, &bp), (uint32_t)M / 8);
    printf("pack_run(K=%d, blocks=%d) -> %s\n", K, M / 8, vc4_strerror(st));
    if (st != VC4_OK) { vc4_shutdown(ctx); gpu_unlock(lock); return 1; }
    vc4__l2_flush(ctx);
    vc4_download(ctx, &bp, P, n * 4);
    long bad = 0;
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r) {
                float ref = A[(size_t)(rb * 8 + r) * K + k];
                float got = P[((size_t)rb * K + k) * 8 + r];
                if (got != ref) {
                    if (bad < 6)
                        printf("  rb=%d k=%d r=%d got %.1f ref %.1f\n", rb, k, r,
                               got, ref);
                    bad++;
                }
            }
    printf("panel: %ld/%zu mismatches -> %s\n", bad, n, bad ? "FAIL" : "PASS");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return bad ? 1 : 0;
}
