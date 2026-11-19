/* test_matmul.c - vc4_matmul vs a double-precision CPU oracle.
 *
 * Every output element of every run is compared.  The CPU is the verifier
 * only: nothing here feeds a CPU result back into the library.
 *
 * usage: test_matmul [reps]
 */
#include "vc4infer.h"
#include "util.h"

#define TOL 1e-4        /* relative, as in asm3 check_gemm */

static int one_shape(vc4_ctx *ctx, int M, int N, int K, int reps)
{
    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    float *A = malloc(na * 4), *B = malloc(nb * 4), *C = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    if (!A || !B || !C || !R)
        return 1;
    for (size_t i = 0; i < na; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();

    vc4_buf ba, bb, bc;
    if (vc4_alloc(ctx, na * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, nb * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bc) != VC4_OK) {
        printf("FAIL: arena alloc for %dx%dx%d\n", M, N, K);
        return 1;
    }
    vc4_upload(ctx, &ba, A, na * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    memset(C, 0xCD, nc * 4);
    vc4_upload(ctx, &bc, C, nc * 4);

    ref_gemm(A, B, R, M, N, K);

    int clean = 0;
    double best_gflops = 0, best_ms = 0;
    for (int rep = 0; rep < reps; ++rep) {
        /* rewrite C so a stale frame cannot masquerade as a result */
        memset(C, 0, nc * 4);
        vc4_upload(ctx, &bc, C, nc * 4);

        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        vc4_status st = vc4_matmul(ctx, &ba, &bb, &bc, (uint32_t)M, (uint32_t)N,
                                   (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: vc4_matmul(%d,%d,%d) -> %s (%s)\n", M, N, K,
                   vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);

        vc4_download(ctx, &bc, C, nc * 4);
        check_t ck = { 0, 0, 0, 1 };
        for (size_t i = 0; i < nc; ++i)
            check_elem(&ck, C[i], R[i], TOL, i < 8);

        double ms = (double)(t1 - t0) / 1e6;
        double pack_ms = (double)(s1.pack_ns - s0.pack_ns) / 1e6;
        double kern_ms = (double)(s1.kernel_ns - s0.kernel_ns) / 1e6;
        double gpu_ms = pack_ms + kern_ms;
        double gf = 2.0 * M * N * K / (gpu_ms * 1e6);
        if (gf > best_gflops) {
            best_gflops = gf;
            best_ms = gpu_ms;
        }
        if (ck.mismatches == 0)
            clean++;
        printf("  M=%3d N=%3d K=%3d rep%d: mismatches=%ld/%zu maxabs=%.3g maxrel=%.2g "
               "| gpu %.3f ms = pack %.3f + gemm %.3f  %.2f GFLOP/s "
               "(gemm-only %.2f, wall %.3f ms)\n",
               M, N, K, rep, ck.mismatches, nc, ck.max_abs, ck.max_rel,
               gpu_ms, pack_ms, kern_ms, gf,
               kern_ms > 0 ? 2.0 * M * N * K / (kern_ms * 1e6) : 0.0, ms);
    }
    printf("%s %dx%dx%d: %d/%d clean, best %.3f ms = %.2f GFLOP/s\n",
           clean == reps ? "PASS" : "FAIL", M, N, K, clean, reps, best_ms,
           best_gflops);

    vc4_arena_used(ctx);            /* keep the bump cursor where it is */
    free(A); free(B); free(C); free(R);
    return clean == reps ? 0 : 1;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 5;

    int lock = gpu_lock();
    if (lock < 0)
        return 2;

    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s\n", vc4_strerror(st));
        gpu_unlock(lock);
        return 2;
    }
    printf("libvc4infer %s  arena %.1f MiB  reps=%d\n",
           VC4INFER_VERSION_STRING, (double)vc4_arena_bytes(ctx) / (1 << 20), reps);

    int rc = 0;
    /* square sizes with a cached kernel, then awkward sizes that exercise the
       16-alignment padding, then a skinny one (M=16, one QPU) */
    rc |= one_shape(ctx, 128, 128, 128, reps);
    rc |= one_shape(ctx, 256, 256, 256, reps);
    rc |= one_shape(ctx, 64, 48, 32, reps);
    rc |= one_shape(ctx, 20, 36, 44, reps);
    rc |= one_shape(ctx, 16, 16, 16, reps);
    rc |= one_shape(ctx, 512, 64, 288, reps);

    vc4_stats s;
    vc4_get_stats(ctx, &s);
    printf("stats: launches=%llu pack=%.2f ms kernel=%.2f ms flops=%.3f GFLOP\n",
           (unsigned long long)s.launches, s.pack_ns / 1e6, s.kernel_ns / 1e6,
           s.flops / 1e9);
    vc4_print_supported_ops(stdout);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    return rc;
}
