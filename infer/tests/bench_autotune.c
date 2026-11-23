/* bench_autotune.c - before/after for the per-shape autotuner, on silicon.
 *
 * For each case it measures the op the way the engine runs it today (the
 * default launch), then calls vc4_autotune for that shape, then measures the
 * same op again now that the executor consults the record.  The autotuner's
 * own correctness check is printed with the result: `max|diff|` against the
 * default launch's output and the number of elements outside the recorded
 * atol = rtol = 5e-4 (docs/ACCURACY.md).
 *
 * Run it against an empty cache directory to see the cold-search cost, then
 * again against the same directory to see the cached path:
 *
 *   VC4INFER_CACHE=$HOME/scratch/kcache-autotune VC4INFER_DAEMON=1 \
 *       tools/thermal-guard.sh --max 71 -- ./build/bench_autotune 5
 *
 * The timings are medians of `reps` runs after one warm-up, the same protocol
 * the tuner uses, so the before/after numbers are comparable even though the
 * tuner measured the candidates first.
 */
#define _GNU_SOURCE
#include "vc4infer.h"
#include "gpu_gate.h"

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

static void fill(float *p, size_t n, uint32_t seed)
{
    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (float)((int)((x >> 16) & 0xFFu) - 128) / 64.0f;
    }
}

static vc4_ctx *ctx;

static int upload(float *host, size_t n, vc4_buf *out)
{
    if (vc4_alloc(ctx, n * 4u, out) != VC4_OK)
        return -1;
    return vc4_upload(ctx, out, host, n * 4u) == VC4_OK ? 0 : -1;
}

typedef vc4_status (*op_fn)(void *);

static double time_op(op_fn op, void *arg, int reps, vc4_status *last)
{
    double v[16];
    vc4_status st = op(arg);
    if (st != VC4_OK) {
        *last = st;
        return -1.0;
    }
    for (int i = 0; i < reps; ++i) {
        double t0 = now_ms();
        st = op(arg);
        v[i] = now_ms() - t0;
        if (st != VC4_OK) {
            *last = st;
            return -1.0;
        }
    }
    *last = VC4_OK;
    return median(v, reps);
}

/* --------------------------------------------------------------- matmul */

struct mm_case {
    vc4_buf a, b, c;
    uint32_t M, N, K;
};

static vc4_status mm_run(void *p)
{
    struct mm_case *m = p;
    return vc4_matmul(ctx, &m->a, &m->b, &m->c, m->M, m->N, m->K,
                      VC4_LAYOUT_ROW_MAJOR);
}

static void case_matmul(uint32_t M, uint32_t N, uint32_t K, int reps)
{
    struct mm_case c;
    vc4_tune_shape sh;
    vc4_tune_result r;
    vc4_stats s0, s1;
    vc4_status st;
    double before, after, t0, search_ms;
    float *host;

    memset(&c, 0, sizeof(c));
    c.M = M;
    c.N = N;
    c.K = K;
    host = malloc((size_t)M * K * 4u);
    fill(host, (size_t)M * K, 1u);
    if (upload(host, (size_t)M * K, &c.a) != 0)
        goto fail;
    free(host);
    host = malloc((size_t)K * N * 4u);
    fill(host, (size_t)K * N, 2u);
    if (upload(host, (size_t)K * N, &c.b) != 0)
        goto fail;
    free(host);
    host = malloc((size_t)M * N * 4u);
    fill(host, (size_t)M * N, 0u);
    if (upload(host, (size_t)M * N, &c.c) != 0)
        goto fail;

    vc4_reset_stats(ctx);
    before = time_op(mm_run, &c, reps, &st);
    vc4_get_stats(ctx, &s0);
    if (st != VC4_OK) {
        printf("matmul %ux%ux%u: default failed: %s\n", M, N, K,
               vc4_strerror(st));
        goto out;
    }

    memset(&sh, 0, sizeof(sh));
    sh.m = M;
    sh.n = N;
    sh.k = K;
    t0 = now_ms();
    st = vc4_autotune(ctx, VC4_OP_MATMUL, &sh, NULL, &r);
    search_ms = now_ms() - t0;
    if (st != VC4_OK) {
        printf("matmul %ux%ux%u: autotune failed: %s (%s)\n", M, N, K,
               vc4_strerror(st), vc4_last_error(ctx));
        goto out;
    }

    vc4_reset_stats(ctx);
    after = time_op(mm_run, &c, reps, &st);
    vc4_get_stats(ctx, &s1);

    printf("matmul %ux%ux%u\n", M, N, K);
    printf("  params     pack_q=%u gemm_q=%u kernel=%s%s\n", r.params.pack_q,
           r.params.gemm_q, r.kernel, r.from_cache ? " (from cache)" : "");
    printf("  default    %.3f ms\n", before);
    printf("  tuned      %.3f ms   gain %.2fx\n", after, before / after);
    printf("  check      max|diff| %.3g, %u over 5e-4, %u sets measured\n",
           r.max_abs, r.over_tol, r.candidates);
    printf("  launches   %llu before, %llu after\n",
           (unsigned long long)s0.launches, (unsigned long long)s1.launches);
    printf("  search     %.2f s wall, %.3f ms for the chosen launch\n",
           search_ms / 1e3, r.best_ms);
out:
    free(host);
    return;
fail:
    printf("matmul %ux%ux%u: arena allocation failed\n", M, N, K);
    free(host);
}

/* ----------------------------------------------------------------- conv */

struct cv_case {
    vc4_buf in, wt, bias, out;
    int Cin, Cout, H, W, pad, ho, wo;
};

static vc4_status cv_run(void *p)
{
    struct cv_case *c = p;
    int ho = 0, wo = 0;
    return vc4_conv2d(ctx, &c->in, c->Cin, c->H, c->W, &c->wt, c->Cout, 3, 3,
                      &c->bias, 1, c->pad, &c->out, &ho, &wo);
}

static void case_conv(int Cin, int Cout, int H, int W, int pad, int reps)
{
    struct cv_case c;
    vc4_tune_shape sh;
    vc4_tune_result r;
    vc4_status st;
    double before, after, t0, search_ms;
    float *host;
    size_t nin = (size_t)Cin * H * W, nw = (size_t)Cout * Cin * 9;

    memset(&c, 0, sizeof(c));
    c.Cin = Cin;
    c.Cout = Cout;
    c.H = H;
    c.W = W;
    c.pad = pad;
    c.ho = H + 2 * pad - 2;
    c.wo = W + 2 * pad - 2;

    host = malloc(nin * 4u);
    fill(host, nin, 3u);
    if (upload(host, nin, &c.in) != 0)
        goto fail;
    free(host);
    host = malloc(nw * 4u);
    fill(host, nw, 4u);
    if (upload(host, nw, &c.wt) != 0)
        goto fail;
    free(host);
    host = malloc((size_t)Cout * 4u);
    fill(host, (size_t)Cout, 5u);
    if (upload(host, (size_t)Cout, &c.bias) != 0)
        goto fail;
    free(host);
    host = malloc((size_t)Cout * c.ho * c.wo * 4u);
    if (vc4_alloc(ctx, (size_t)Cout * c.ho * c.wo * 4u, &c.out) != VC4_OK) {
        printf("conv %dx%dx%dx%d: out allocation failed\n", Cin, H, W, Cout);
        goto out;
    }

    before = time_op(cv_run, &c, reps, &st);
    if (st != VC4_OK) {
        printf("conv %dx%dx%dx%d: default failed: %s\n", Cin, H, W, Cout,
               vc4_strerror(st));
        goto out;
    }

    memset(&sh, 0, sizeof(sh));
    sh.cin = (uint32_t)Cin;
    sh.cout = (uint32_t)Cout;
    sh.h = (uint32_t)H;
    sh.w = (uint32_t)W;
    sh.kh = sh.kw = 3;
    sh.pad = (uint32_t)pad;
    sh.stride = 1;
    sh.flags = VC4_TUNE_F_BIAS;
    t0 = now_ms();
    st = vc4_autotune(ctx, VC4_OP_CONV2D, &sh, NULL, &r);
    search_ms = now_ms() - t0;
    if (st != VC4_OK) {
        printf("conv %dx%dx%dx%d: autotune failed: %s (%s)\n", Cin, H, W,
               Cout, vc4_strerror(st), vc4_last_error(ctx));
        goto out;
    }

    after = time_op(cv_run, &c, reps, &st);

    printf("conv2d %dx%dx%d -> %d ch, 3x3 pad %d (out %dx%dx%d)\n", Cin, H, W,
           Cout, pad, Cout, c.ho, c.wo);
    printf("  params     im2col_q=%u gemm_q=%u kernel=%s%s\n",
           r.params.im2col_q, r.params.gemm_q, r.kernel,
           r.from_cache ? " (from cache)" : "");
    printf("  default    %.3f ms\n", before);
    printf("  tuned      %.3f ms   gain %.2fx\n", after, before / after);
    printf("  check      max|diff| %.3g, %u over 5e-4, %u sets measured\n",
           r.max_abs, r.over_tol, r.candidates);
    printf("  search     %.2f s wall, %.3f ms for the chosen launch\n",
           search_ms / 1e3, r.best_ms);
out:
    free(host);
    return;
fail:
    printf("conv %dx%dx%dx%d: arena allocation failed\n", Cin, H, W, Cout);
    free(host);
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int reps = argc > 1 ? atoi(argv[1]) : 5;
    const char *cache = getenv("VC4INFER_CACHE");

    if (reps < 1)
        reps = 1;
    if (reps > 16)
        reps = 16;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        fprintf(stderr, "vc4_init failed\n");
        return 1;
    }
    printf("backend: %s\n", getenv("VC4INFER_DAEMON") ? "vc4d" : "direct");
    printf("autotune cache: %s\n", cache ? cache : "(default)");
    printf("reps: %d (median, after one warm-up)\n\n", reps);

    case_matmul(128, 128, 128, reps);
    case_matmul(256, 256, 256, reps);
    case_matmul(512, 512, 512, reps);
    case_conv(32, 32, 32, 32, 1, reps);

    vc4_shutdown(ctx);
    return 0;
}
