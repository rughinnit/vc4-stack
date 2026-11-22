/* bench_split.c - where a matmul's GPU time goes: A-panel pack vs the GEMM.
 *
 * vc4_matmul() launches twice: the asm10 packer builds the 8-row k-major A
 * panel, then the asm10 GEMM runs.  The engine's timers already separate them
 * (ctx->st.pack_ns and ctx->st.kernel_ns), so this harness calls the two
 * internal entry points with the same arguments the public call would use and
 * reports each median and the GEMM-only GFLOP/s.
 *
 *   bench_split [M N K] [reps]
 *
 * With no arguments it measures the three shapes used by docs/BENCHMARKS.md
 * (128/256/512 cubed).  The arithmetic is 2*M*N*K over the *padded* shape the
 * kernel actually runs (M rounded to 8, N to 64, K to 16), which is what the
 * kernel executes and therefore what its rate should be quoted against.
 *
 * This is a development harness: it includes src/vc4_int.h and calls
 * vc4__pack_run_q / vc4__gemm_run_q directly, like the diag_* programs.
 */
#include "vc4infer.h"
#include "../src/vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int one(vc4_ctx *ctx, int M, int N, int K, int reps)
{
    const uint32_t Mp = vc4__align8((uint32_t)M);
    const uint32_t Np = vc4__align64((uint32_t)N);
    const uint32_t Kp = vc4__align16((uint32_t)K);
    vc4_buf A, B, C, P;
    double pack[32], gemm[32], wall[32];

    if (reps < 1 || reps > 32)
        reps = 5;
    if (vc4_alloc(ctx, (size_t)Mp * Kp * 4u, &A) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Kp * Np * 4u, &B) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Mp * Np * 4u, &C) != VC4_OK)
        return -1;
    vc4__scratch_reset(ctx);
    float *panel = vc4__scratch_alloc(ctx, (size_t)Mp * Kp * 4u, &P);
    if (!panel)
        return -1;
    float *ones = malloc((size_t)Mp * Kp * 4u);
    if (!ones)
        return -1;
    for (size_t i = 0, n = (size_t)Mp * Kp; i < n; ++i)
        ones[i] = 1.0f;
    if (vc4_upload(ctx, &A, ones, (size_t)Mp * Kp * 4u) != VC4_OK)
        return -1;
    free(ones);

    /* one warm-up pair: blob generation is not part of a steady-state timing */
    if (vc4__pack_run_q(ctx, vc4_bus_addr(ctx, &A), Kp,
                        ctx->gpu_bus + P.off, Mp / 8u, 0) != VC4_OK ||
        vc4__gemm_run_q(ctx, Mp, Np, Kp, vc4_bus_addr(ctx, &B),
                        vc4_bus_addr(ctx, &C), ctx->gpu_bus + P.off, 0,
                        VC4_ACT_NONE) != VC4_OK)
        return -1;

    for (int r = 0; r < reps; ++r) {
        uint64_t p0 = ctx->st.pack_ns, k0 = ctx->st.kernel_ns;
        double t0 = now_ms();
        if (vc4__pack_run_q(ctx, vc4_bus_addr(ctx, &A), Kp,
                            ctx->gpu_bus + P.off, Mp / 8u, 0) != VC4_OK)
            return -1;
        double t1 = now_ms();
        if (vc4__gemm_run_q(ctx, Mp, Np, Kp, vc4_bus_addr(ctx, &B),
                            vc4_bus_addr(ctx, &C), ctx->gpu_bus + P.off, 0,
                            VC4_ACT_NONE) != VC4_OK)
            return -1;
        double t2 = now_ms();
        pack[r] = (double)(ctx->st.pack_ns - p0) / 1e6;
        gemm[r] = (double)(ctx->st.kernel_ns - k0) / 1e6;
        wall[r] = t2 - t0;
        (void)t1;
    }
    double flops = 2.0 * (double)Mp * (double)Np * (double)Kp;
    double pm = median(pack, reps), gm = median(gemm, reps);
    printf("%4dx%4dx%4d  padded %ux%ux%u  pack %.3f ms (%4.1f%%)  "
           "gemm %.3f ms  %.2f GFLOP/s gemm-only  %.2f total  wall %.3f ms\n",
           M, N, K, Mp, Np, Kp, pm, 100.0 * pm / (pm + gm), gm,
           flops / (gm * 1e6), flops / ((gm + pm) * 1e6), median(wall, reps));
    vc4__scratch_reset(ctx);
    return 0;
}

int main(int argc, char **argv)
{
    vc4_ctx *ctx = NULL;
    int reps = 5;

    if (argc == 1) {
        if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
            fprintf(stderr, "vc4_init failed\n");
            return 1;
        }
        one(ctx, 128, 128, 128, reps);
        one(ctx, 256, 256, 256, reps);
        one(ctx, 512, 512, 512, reps);
        vc4_shutdown(ctx);
        return 0;
    }
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s [M N K [reps]]\n", argv[0]);
        return 2;
    }
    if (argc == 5)
        reps = atoi(argv[4]);
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        fprintf(stderr, "vc4_init failed\n");
        return 1;
    }
    int rc = one(ctx, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), reps);
    vc4_shutdown(ctx);
    return rc == 0 ? 0 : 1;
}
