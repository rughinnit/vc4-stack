/* test_gated.c - the five kernels that used to be gated (BUGS.md B4/B5).
 *
 * Each subcommand is one entry point and one element-wise oracle built from
 * the *documented semantics*, never from the stride arithmetic the call is
 * built with:
 *
 *   gather   out[o] = tab[o] < 0 ? fill : arena[tab[o]]   (arena byte offsets)
 *   blit     out[0..bytes) == in[0..bytes)                (no overlap)
 *   reduce   out[o] = op(a[o*red .. o*red+red))
 *   cumsum   out[r][c] = sum_{c'<=c} a[r][c']
 *   sel3     where / clip3 / prelu
 *
 * Shapes are chosen to break the obvious defects: an output count that is not
 * a multiple of the 256-element group (so the padded tail has to be right), a
 * gather table with negative entries (the fill path), a reduce whose `outer`
 * crosses 256 in both directions, a cumsum of more than one row and more than
 * one group of columns, and a WHERE condition that is negative, zero, -0.0 and
 * nonzero positive (the sign-bit test selected the false operand for all of
 * them).
 *
 * One entry point per process (run_tests.sh calls it five times) so a wedge
 * can be attributed.  usage: test_gated gather|blit|reduce|cumsum|sel3
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

static int g_fail;
static vc4_ctx *g_ctx;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d ", __FILE__, __LINE__);                         \
            printf(__VA_ARGS__);                                               \
            printf(" [%s]\n", vc4_last_error(g_ctx));                          \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

/* A kernel that is still gated reports its refusal and does not fail the run:
 * the element check below is what turns green the moment the gate is removed. */
static int gated(vc4_status st, const char *what)
{
    const char *m = vc4_last_error(g_ctx);
    if (st == VC4_ERR_UNSUPPORTED && m &&
        strstr(m, "not verified on this board")) {
        printf("GATED %-26s %s\n", what, m);
        return 1;
    }
    return 0;
}

static float *alloc_host(size_t n)
{
    float *p = malloc(n * 4);
    if (!p) {
        printf("FAIL: out of host memory\n");
        exit(1);
    }
    return p;
}

static vc4_buf mk_buf(size_t bytes)
{
    vc4_buf b = { 0, 0 };
    if (vc4_alloc(g_ctx, bytes, &b) != VC4_OK) {
        printf("FAIL: vc4_alloc(%zu): %s\n", bytes, vc4_last_error(g_ctx));
        exit(1);
    }
    return b;
}

/* ---------------------------------------------------------------- gather */
static void t_gather(void)
{
    static const size_t shapes[] = { 1, 16, 17, 255, 256, 257, 1000 };
    const float fill = -7.5f;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        const size_t n = shapes[s];
        /* the source is 2n floats so a table can reach a region the output
         * does not overlap */
        const size_t nsrc = 2 * n + 16;
        float *src = alloc_host(nsrc);
        int32_t *tab = (int32_t *)alloc_host(n);  /* 4 B each, same size */
        float *out = alloc_host(n);
        float *want = alloc_host(n);
        for (size_t i = 0; i < nsrc; ++i)
            src[i] = (float)(i + 1) + 0.25f;
        vc4_buf bs = mk_buf(nsrc * 4), bt = mk_buf(n * 4), bo = mk_buf(n * 4);
        for (size_t i = 0; i < n; ++i) {
            /* reverse permutation, a negative entry every 7th, plus a repeat */
            size_t k = n - 1 - i;
            int neg = (i % 7) == 3;
            tab[i] = neg ? -1 : (int32_t)(bs.off + (uint32_t)k * 4);
            want[i] = neg ? fill : src[k];
        }
        vc4_upload(g_ctx, &bs, src, nsrc * 4);
        vc4_upload(g_ctx, &bt, tab, n * 4);
        uint64_t t0 = now_ns();
        vc4_status st = vc4_gather(g_ctx, &bt, &bo, n, fill);
        uint64_t ns = now_ns() - t0;
        if (gated(st, "gather"))
            { free(src); free(tab); free(out); free(want); continue; }
        CHECK(st == VC4_OK, "gather n=%zu: %s", n, vc4_strerror(st));
        if (st == VC4_OK) {
            vc4_download(g_ctx, &bo, out, n * 4);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i)
                if (memcmp(&out[i], &want[i], 4) != 0)
                    ++bad;
            printf("%s gather  n=%-5zu %4zu/%zu bad  %7.3f ms\n",
                   bad ? "FAIL" : "PASS", n, bad, n, ns / 1e6);
            if (bad) {
                for (size_t i = 0; i < n && i < 40; ++i)
                    if (memcmp(&out[i], &want[i], 4) != 0)
                        printf("   [%zu] got %.7g want %.7g\n", i,
                               (double)out[i], (double)want[i]);
                ++g_fail;
            }
        }
        free(src); free(tab); free(out); free(want);
    }
}

/* ------------------------------------------------------------------ blit */
static void t_blit(void)
{
    static const size_t shapes[] = { 64, 256, 448, 1024, 832, 6400 };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        const size_t bytes = shapes[s];
        const size_t n = bytes / 4;
        float *a = alloc_host(n), *out = alloc_host(n);
        for (size_t i = 0; i < n; ++i)
            a[i] = (float)(i * 3 + 1) * 0.5f;
        memset(out, 0xA5, bytes);
        vc4_buf ba = mk_buf(bytes), bo = mk_buf(bytes);
        vc4_upload(g_ctx, &ba, a, bytes);
        vc4_upload(g_ctx, &bo, out, bytes);
        uint64_t t0 = now_ns();
        vc4_status st = vc4_blit(g_ctx, &ba, &bo, bytes);
        uint64_t ns = now_ns() - t0;
        if (gated(st, "blit"))
            { free(a); free(out); continue; }
        CHECK(st == VC4_OK, "blit bytes=%zu: %s", bytes, vc4_strerror(st));
        if (st == VC4_OK) {
            vc4_download(g_ctx, &bo, out, bytes);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i)
                if (memcmp(&out[i], &a[i], 4) != 0)
                    ++bad;
            printf("%s blit    %5zu B  %4zu/%zu bad  %7.3f ms\n",
                   bad ? "FAIL" : "PASS", bytes, bad, n, ns / 1e6);
            if (bad) {
                for (size_t i = 0; i < n && i < 40; ++i)
                    if (memcmp(&out[i], &a[i], 4) != 0)
                        printf("   [%zu] got %.7g want %.7g\n", i,
                               (double)out[i], (double)a[i]);
                ++g_fail;
            }
        }
        free(a); free(out);
    }
}

/* ---------------------------------------------------------------- reduce */
/* one output, in double, with the same association the kernel is allowed */
static double red_ref_one(const float *a, size_t red, int op, long *argi)
{
    double v;
    size_t best = 0;
    if (op == VC4_RED_SUM || op == VC4_RED_MEAN) {
        v = 0.0;
        for (size_t i = 0; i < red; ++i)
            v += (double)a[i];
        return op == VC4_RED_MEAN ? v / (double)red : v;
    }
    v = a[0];
    for (size_t i = 1; i < red; ++i) {
        int better = (op == VC4_RED_MAX || op == VC4_RED_ARGMAX)
                         ? (a[i] > (float)v) : (a[i] < (float)v);
        if (better) {
            v = a[i];
            best = i;
        }
    }
    *argi = (long)best;
    return v;
}

static void t_reduce(void)
{
    static const uint32_t outers[] = { 1, 7, 16, 255, 256, 300, 1024 };
    static const uint32_t reds[] = { 1, 3, 16, 64 };
    static const int ops[] = { VC4_RED_SUM, VC4_RED_MEAN, VC4_RED_MAX,
                               VC4_RED_MIN, VC4_RED_ARGMAX, VC4_RED_ARGMIN };
    static const char *nm[] = { "sum", "mean", "max", "min", "argmax",
                                "argmin" };
    for (size_t oi = 0; oi < sizeof(outers) / sizeof(outers[0]); ++oi) {
        for (size_t ri = 0; ri < sizeof(reds) / sizeof(reds[0]); ++ri) {
            const uint32_t outer = outers[oi], red = reds[ri];
            const size_t n = (size_t)outer * red;
            float *a = alloc_host(n), *out = alloc_host(outer);
            int32_t *iout = (int32_t *)out;
            for (size_t i = 0; i < n; ++i) {
                /* distinct values: no argmax ties, and a wide dynamic range
                 * makes an uninitialised accumulator visible */
                a[i] = (float)((int32_t)(i * 37 % 1013) - 500) * 0.125f
                       + (float)(i % 7) * 1e-3f;
                if (i % 5 == 0)
                    a[i] = -a[i];
            }
            vc4_buf ba = mk_buf(n * 4), bo = mk_buf(outer * 4);
            vc4_upload(g_ctx, &ba, a, n * 4);
            for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); ++k) {
                const int op = ops[k];
                uint64_t t0 = now_ns();
                vc4_status st = vc4_reduce(g_ctx, &ba, &bo, outer, red, op);
                uint64_t ns = now_ns() - t0;
                if (gated(st, "reduce"))
                    continue;
                CHECK(st == VC4_OK, "reduce %s outer=%u red=%u: %s", nm[k],
                      outer, red, vc4_strerror(st));
                if (st != VC4_OK)
                    continue;
                vc4_download(g_ctx, &bo, out, outer * 4);
                size_t bad = 0;
                double worst = 0;
                for (uint32_t o = 0; o < outer; ++o) {
                    long ai = 0;
                    double want = red_ref_one(a + (size_t)o * red, red, op,
                                              &ai);
                    int is_arg = (op == VC4_RED_ARGMAX || op == VC4_RED_ARGMIN);
                    /* red_ref_one returns the *value*; the arg ops store the
                     * index, so the oracle compares against `ai` (comparing
                     * the index with the value marked every arg result bad). */
                    if (is_arg)
                        want = (double)ai;
                    double got = is_arg ? (double)iout[o] : (double)out[o];
                    double tol = is_arg ? 0.0
                                        : 1e-5 + 1e-5 * (want < 0 ? -want : want);
                    /* A sum is a chain of `red` QPU fadds and the FPU
                     * truncates toward zero (docs/NUMERICS.md), so its error
                     * is bounded by red ulps of the largest partial, not by
                     * an IEEE-ordered sum.  Add that bound for Sum. */
                    if (op == VC4_RED_SUM) {
                        double scale = 0.0;
                        for (uint32_t r = 0; r < red; ++r) {
                            double v = a[(size_t)o * red + r];
                            if (v < 0)
                                v = -v;
                            if (v > scale)
                                scale = v;
                        }
                        tol += (double)red * scale * 2.4e-7;
                    }
                    double e = got - want;
                    if (e < 0)
                        e = -e;
                    if (e > worst)
                        worst = e;
                    if (e > tol)
                        ++bad;
                }
                printf("%s %-6s outer=%-5u red=%-3u %4zu/%u bad  max|err| "
                       "%.3g  %7.3f ms\n", bad ? "FAIL" : "PASS", nm[k],
                       outer, red, bad, outer, worst, ns / 1e6);
                if (bad) {
                    for (uint32_t o = 0; o < outer && o < 24; ++o) {
                        long ai = 0;
                        double want = red_ref_one(a + (size_t)o * red, red, op,
                                                  &ai);
                        int is_arg = (op == VC4_RED_ARGMAX ||
                                      op == VC4_RED_ARGMIN);
                        if (is_arg)
                            want = (double)ai;
                        printf("   [%u] got %.7g want %.7g\n", o,
                               is_arg ? (double)iout[o] : (double)out[o], want);
                    }
                    ++g_fail;
                }
            }
            free(a); free(out);
        }
    }
}

/* ---------------------------------------------------------------- cumsum */
static void t_cumsum(void)
{
    static const uint32_t shapes[][2] = { { 1, 256 }, { 2, 256 }, { 3, 512 },
                                          { 5, 1024 } };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        const uint32_t rows = shapes[s][0], cols = shapes[s][1];
        const size_t n = (size_t)rows * cols;
        float *a = alloc_host(n), *out = alloc_host(n);
        for (size_t i = 0; i < n; ++i)
            a[i] = frand() * 2.0f;
        vc4_buf ba = mk_buf(n * 4), bo = mk_buf(n * 4);
        vc4_upload(g_ctx, &ba, a, n * 4);
        uint64_t t0 = now_ns();
        vc4_status st = vc4_cumsum(g_ctx, &ba, &bo, rows, cols);
        uint64_t ns = now_ns() - t0;
        if (gated(st, "cumsum"))
            { free(a); free(out); continue; }
        CHECK(st == VC4_OK, "cumsum rows=%u cols=%u: %s", rows, cols,
              vc4_strerror(st));
        if (st == VC4_OK) {
            vc4_download(g_ctx, &bo, out, n * 4);
            size_t bad = 0;
            double worst = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                double acc = 0.0;
                for (uint32_t c = 0; c < cols; ++c) {
                    size_t i = (size_t)r * cols + c;
                    acc += (double)a[i];
                    double e = (double)out[i] - acc;
                    if (e < 0)
                        e = -e;
                    if (e > worst)
                        worst = e;
                    if (e > 1e-5 + 1e-5 * (acc < 0 ? -acc : acc))
                        ++bad;
                }
            }
            printf("%s cumsum  %ux%-5u %4zu/%zu bad  max|err| %.3g  %7.3f ms\n",
                   bad ? "FAIL" : "PASS", rows, cols, bad, n, worst, ns / 1e6);
            if (bad) {
                ++g_fail;
                for (size_t i = 0; i < n && i < 40; ++i)
                    printf("   [%zu] got %.7g\n", i, (double)out[i]);
            }
        }
        free(a); free(out);
    }
}

/* ------------------------------------------------------------------ sel3 */
static void t_sel3(void)
{
    static const size_t shapes[] = { 1, 15, 16, 17, 256, 300, 1000 };
    typedef struct { vc4_selop op; const char *nm; } selc;
    static const selc ops[] = { { VC4_SEL_WHERE, "where" },
                                { VC4_SEL_CLIP3, "clip3" },
                                { VC4_SEL_PRELU, "prelu" } };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
        const size_t n = shapes[s];
        float *a = alloc_host(n), *b = alloc_host(n), *c = alloc_host(n);
        float *out = alloc_host(n);
        for (size_t i = 0; i < n; ++i) {
            /* the WHERE condition must include 0, -0.0, positive and negative
             * nonzero; the original sign-bit test got all four wrong */
            static const float cond[] = { 0.0f, -0.0f, 1.0f, -1.0f, 2.5f,
                                          -2.5f, 1e-30f };
            a[i] = cond[i % 7];
            b[i] = (float)i + 0.5f;
            c[i] = -(float)i - 0.25f;
        }
        vc4_buf ba = mk_buf(n * 4), bb = mk_buf(n * 4), bc = mk_buf(n * 4);
        vc4_buf bo = mk_buf(n * 4);
        vc4_upload(g_ctx, &ba, a, n * 4);
        vc4_upload(g_ctx, &bb, b, n * 4);
        vc4_upload(g_ctx, &bc, c, n * 4);
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); ++k) {
            uint64_t t0 = now_ns();
            vc4_status st = vc4_select3(g_ctx, ops[k].op, &ba, &bb, &bc, &bo,
                                        n);
            uint64_t ns = now_ns() - t0;
            if (gated(st, "select3"))
                continue;
            CHECK(st == VC4_OK, "select3 %s n=%zu: %s", ops[k].nm, n,
                  vc4_strerror(st));
            if (st != VC4_OK)
                continue;
            vc4_download(g_ctx, &bo, out, n * 4);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i) {
                float want;
                if (ops[k].op == VC4_SEL_WHERE)
                    want = (a[i] != 0.0f) ? b[i] : c[i];
                else if (ops[k].op == VC4_SEL_CLIP3)
                    want = b[i] < a[i] ? a[i] : (b[i] > c[i] ? c[i] : b[i]);
                else
                    want = a[i] > 0.0f ? a[i] : a[i] * b[i];
                if (memcmp(&out[i], &want, 4) != 0)
                    ++bad;
            }
            printf("%s sel3 %-6s n=%-4zu %4zu/%zu bad  %7.3f ms\n",
                   bad ? "FAIL" : "PASS", ops[k].nm, n, bad, n, ns / 1e6);
            if (bad) {
                for (size_t i = 0; i < n && i < 32; ++i) {
                    float want;
                    if (ops[k].op == VC4_SEL_WHERE)
                        want = (a[i] != 0.0f) ? b[i] : c[i];
                    else if (ops[k].op == VC4_SEL_CLIP3)
                        want = b[i] < a[i] ? a[i] : (b[i] > c[i] ? c[i] : b[i]);
                    else
                        want = a[i] > 0.0f ? a[i] : a[i] * b[i];
                    if (memcmp(&out[i], &want, 4) != 0)
                        printf("   [%zu] a=%.7g b=%.7g c=%.7g got %.7g want "
                               "%.7g\n", i, (double)a[i], (double)b[i],
                               (double)c[i], (double)out[i], (double)want);
                }
                ++g_fail;
            }
        }
        free(a); free(b); free(c); free(out);
    }
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    if (argc != 2) {
        printf("usage: test_gated gather|blit|reduce|cumsum|sel3\n");
        return 2;
    }
    if (vc4_init(&g_ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        return 1;
    }
    printf("libvc4infer %s  gated-kernel check: %s\n",
           VC4INFER_VERSION_STRING, argv[1]);
    if (!strcmp(argv[1], "gather"))
        t_gather();
    else if (!strcmp(argv[1], "blit"))
        t_blit();
    else if (!strcmp(argv[1], "reduce"))
        t_reduce();
    else if (!strcmp(argv[1], "cumsum"))
        t_cumsum();
    else if (!strcmp(argv[1], "sel3"))
        t_sel3();
    else {
        printf("unknown test %s\n", argv[1]);
        vc4_shutdown(g_ctx);
        return 2;
    }
    vc4_shutdown(g_ctx);
    printf("%s\n", g_fail ? "GATED-KERNEL TEST FAILED"
                          : "ALL CHECKS PASSED");
    return g_fail ? 1 : 0;
}
