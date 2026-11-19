/* diag_gemm11.c - asm10 GEMM vs a CPU oracle, with a CPU-built panel.
 * usage: diag_gemm11 M N K [nqpu]
 */
#include "vc4infer.h"
#include "util.h"
#include "../src/vc4_int.h"

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 16;
    int N = argc > 2 ? atoi(argv[2]) : 64;
    int K = argc > 3 ? atoi(argv[3]) : 16;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init failed\n"); return 2; }

    size_t np = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
    float *A = malloc(np * 4), *P = malloc(np * 4), *B = malloc(nb * 4);
    float *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    for (size_t i = 0; i < np; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();
    for (int rb = 0; rb < M / 8; ++rb)
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < 8; ++r)
                P[((size_t)rb * K + k) * 8 + r] = A[(size_t)(rb * 8 + r) * K + k];
    ref_gemm(A, B, R, M, N, K);

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
    long bad = 0;
    for (int r = 0; r < M; ++r) {
        long rowbad = 0;
        for (int c = 0; c < N; ++c) {
            double e = fabs(C[(size_t)r * N + c] - R[(size_t)r * N + c]);
            double den = fabs(R[(size_t)r * N + c]) > 1.0 ? fabs(R[(size_t)r * N + c]) : 1.0;
            if (e > 1e-4 * den) {
                if (rowbad == 0)
                    printf("  r%2d c%2d got %.6g ref %.6g\n", r, c,
                           C[(size_t)r * N + c], R[(size_t)r * N + c]);
                rowbad++;
            }
        }
        if (rowbad) printf("  row %2d: %ld/%d bad\n", r, rowbad, N);
        bad += rowbad;
    }
    printf("gemm: %ld/%zu mismatches -> %s\n", bad, nc, bad ? "FAIL" : "PASS");
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return bad ? 1 : 0;
}
