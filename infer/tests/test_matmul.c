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

/* matmul -> activation: the relu epilogue of the 8x64 asm10 tile.
 *
 * Checked three ways on the same inputs: against the float64 oracle (product,
 * then max(x,0)), against the two-launch path bit for bit, and after the
 * autotuner has chosen a launch for the shape -- the QPU count changes the row
 * blocks a QPU owns, not the body, so the answer must not move.
 */
static int one_shape_act(vc4_ctx *ctx, int M, int N, int K, int reps, int tune)
{
    const size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    float *A = malloc(na * 4), *B = malloc(nb * 4);
    float *Cf = malloc(nc * 4), *C2 = malloc(nc * 4), *Cm = malloc(nc * 4);
    double *R = malloc(nc * sizeof(double));
    if (!A || !B || !Cf || !C2 || !Cm || !R)
        return 1;
    for (size_t i = 0; i < na; ++i) A[i] = frand();
    for (size_t i = 0; i < nb; ++i) B[i] = frand();

    vc4_buf ba, bb, bf, b2, bm;
    if (vc4_alloc(ctx, na * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, nb * 4, &bb) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bf) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &b2) != VC4_OK ||
        vc4_alloc(ctx, nc * 4, &bm) != VC4_OK) {
        printf("FAIL: arena alloc for %dx%dx%d act\n", M, N, K);
        return 1;
    }
    vc4_upload(ctx, &ba, A, na * 4);
    vc4_upload(ctx, &bb, B, nb * 4);
    ref_gemm(A, B, R, M, N, K);
    for (size_t i = 0; i < nc; ++i)
        R[i] = R[i] > 0.0 ? R[i] : 0.0;

    if (tune) {
        vc4_tune_shape sh;
        vc4_tune_result tr;
        vc4_tune_opts to;
        memset(&sh, 0, sizeof(sh));
        memset(&tr, 0, sizeof(tr));
        memset(&to, 0, sizeof(to));
        sh.m = (uint32_t)M;
        sh.n = (uint32_t)N;
        sh.k = (uint32_t)K;
        to.reps = 2;
        vc4_status ts = vc4_autotune(ctx, VC4_OP_MATMUL, &sh, &to, &tr);
        if (ts != VC4_OK) {
            printf("FAIL: vc4_autotune(%dx%dx%d): %s (%s)\n", M, N, K,
                   vc4_strerror(ts), vc4_last_error(ctx));
            return 1;
        }
        printf("  autotune %dx%dx%d: pack_q %u gemm_q %u %.3f ms vs default "
               "%.3f ms, max_abs %.3g, %s\n", M, N, K, tr.params.pack_q,
               tr.params.gemm_q, tr.best_ms, tr.default_ms, tr.max_abs,
               tr.from_cache ? "from cache" : "measured");
    }

    int clean = 0, exact = 0;
    for (int rep = 0; rep < reps; ++rep) {
        memset(Cf, 0, nc * 4);
        vc4_upload(ctx, &bf, Cf, nc * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        vc4_status st = vc4_matmul_act(ctx, &ba, &bb, &bf, (uint32_t)M,
                                       (uint32_t)N, (uint32_t)K,
                                       VC4_LAYOUT_ROW_MAJOR, VC4_ACT_RELU);
        vc4_get_stats(ctx, &s1);
        if (st != VC4_OK) {
            printf("FAIL: vc4_matmul_act(%d,%d,%d) -> %s (%s)\n", M, N, K,
                   vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        const unsigned long long l_fused = s1.launches - s0.launches;

        vc4_get_stats(ctx, &s0);
        st = vc4_matmul(ctx, &ba, &bb, &bm, (uint32_t)M, (uint32_t)N,
                        (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
        if (st == VC4_OK)
            st = vc4_elementwise(ctx, VC4_OP_RELU, &bm, NULL, &b2, nc);
        vc4_get_stats(ctx, &s1);
        if (st != VC4_OK) {
            printf("FAIL: two-launch matmul+relu -> %s (%s)\n",
                   vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        const unsigned long long l_two = s1.launches - s0.launches;

        vc4_download(ctx, &bf, Cf, nc * 4);
        vc4_download(ctx, &b2, C2, nc * 4);
        size_t diff = 0;
        for (size_t i = 0; i < nc; ++i)
            if (Cf[i] != C2[i])
                ++diff;
        check_t ck = { 0, 0, 0, 1 };
        for (size_t i = 0; i < nc; ++i)
            check_elem(&ck, Cf[i], R[i], TOL, i < 8);
        printf("  M=%3d N=%3d K=%3d relu rep%d: mismatches=%ld/%zu maxabs=%.3g "
               "| vs two launches: %zu differ | launches %llu vs %llu\n",
               M, N, K, rep, ck.mismatches, nc, ck.max_abs, diff, l_fused,
               l_two);
        if (ck.mismatches == 0)
            ++clean;
        if (diff == 0)
            ++exact;
        if (l_fused >= l_two) {
            printf("FAIL: the relu epilogue did not save a launch (%llu vs "
                   "%llu)\n", l_fused, l_two);
            return 1;
        }
    }
    printf("%s %dx%dx%d relu: %d/%d in tolerance, %d/%d bit-identical to two "
           "launches\n", (clean == reps && exact == reps) ? "PASS" : "FAIL",
           M, N, K, clean, reps, exact, reps);
    free(A); free(B); free(Cf); free(C2); free(Cm); free(R);
    return (clean == reps && exact == reps) ? 0 : 1;
}

/* an activation the tile cannot express must be refused, never ignored */
static int refuse_act(vc4_ctx *ctx, vc4_act act, const char *name)
{
    float A[64 * 64] = { 0 }, B[64 * 64] = { 0 };
    vc4_buf ba, bb, bc;
    if (vc4_alloc(ctx, sizeof(A), &ba) != VC4_OK ||
        vc4_alloc(ctx, sizeof(B), &bb) != VC4_OK ||
        vc4_alloc(ctx, sizeof(A), &bc) != VC4_OK)
        return 1;
    vc4_upload(ctx, &ba, A, sizeof(A));
    vc4_upload(ctx, &bb, B, sizeof(B));
    vc4_status st = vc4_matmul_act(ctx, &ba, &bb, &bc, 64, 64, 64,
                                   VC4_LAYOUT_ROW_MAJOR, act);
    if (st != VC4_ERR_UNSUPPORTED) {
        printf("FAIL: matmul+%s returned %s, expected VC4_ERR_UNSUPPORTED\n",
               name, vc4_strerror(st));
        return 1;
    }
    printf("PASS matmul+%s is refused: %s\n", name, vc4_last_error(ctx));
    return 0;
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
    /* the relu epilogue: default launch, then the autotuned one */
    rc |= one_shape_act(ctx, 64, 64, 64, reps, 0);
    rc |= one_shape_act(ctx, 20, 36, 44, reps, 0);
    rc |= one_shape_act(ctx, 128, 128, 128, reps, 1);
    rc |= refuse_act(ctx, (vc4_act)7, "act7");

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
