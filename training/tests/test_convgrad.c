/* test_convgrad.c - finite-difference gradient checks for the conv and pool
 * backward kernels, plus direct CPU-oracle comparisons of col2im / maxpool.
 *
 * The CPU is a checker only: every forward value used by a finite difference
 * comes from a QPU kernel, and every analytic gradient is a QPU result.
 *
 * The QPU FPU truncates and has no FMA, so the agreement is limited by fp32
 * forward noise, not by the backward formulas.  Acceptance: relative error
 * <= 1e-3 for elements whose finite difference is above the noise floor
 * (|fd| > 1e-2), with the count of smaller elements reported separately.
 *
 * usage: test_convgrad [reps]
 */
#include "vc4infer.h"
#include "util.h"
#include <math.h>

typedef struct {
    double max_rel, max_abs, worst_g, worst_fd;
    double err2, ref2;
    long   bad, small;
} gc_t;

static void gc_add(gc_t *g, double a, double fd, double tol)
{
    double err = fabs(a - fd);
    double den = fabs(fd);
    if (den > 1e-2) {
        double rel = err / den;
        if (rel > g->max_rel) { g->max_rel = rel; g->worst_g = a; g->worst_fd = fd; }
    } else {
        g->small++;
    }
    if (err > g->max_abs) g->max_abs = err;
    if (den > 1e-2 && err > tol * den + 1e-4) g->bad++;
    g->err2 += err * err;
    g->ref2 += (double)fd * fd;
}

static void gc_print(const char *name, const gc_t *g, int n, double tol)
{
    double l2 = sqrt(g->err2) / (sqrt(g->ref2) > 0 ? sqrt(g->ref2) : 1);
    printf("%s %-26s n=%d rel_L2=%.3g max_rel=%.3g max_abs=%.3g over_tol=%ld"
           " small(fd<=1e-2)=%ld worst g=%.6g fd=%.6g\n",
           (g->bad == 0 && l2 <= 5 * tol) ? "PASS" : "FAIL", name, n, l2,
           g->max_rel, g->max_abs, g->bad, g->small, g->worst_g, g->worst_fd);
}

/* ---------------- scalar loss = sum(dy * out) ---------------- */
static double conv_loss(vc4_ctx *ctx, const vc4_buf *in, int Cin, int H, int W,
                        const vc4_buf *wt, int Cout, int KH, int KW, int pad,
                        vc4_buf *out, const float *dy, int ho, int wo)
{
    int a, b;
    if (vc4_conv2d(ctx, in, Cin, H, W, wt, Cout, KH, KW, NULL, 1, pad,
                   out, &a, &b) != VC4_OK)
        return NAN;
    float *o = malloc((size_t)Cout * ho * wo * 4);
    vc4_download(ctx, out, o, (size_t)Cout * ho * wo * 4);
    double s = 0;
    for (size_t i = 0; i < (size_t)Cout * ho * wo; ++i)
        s += (double)o[i] * dy[i];
    free(o);
    return s;
}

static int t_conv(vc4_ctx *ctx, int H, int W, int pad)
{
    const int Cin = 16, Cout = 16, KH = 3, KW = 3;
    const int ho = H + 2 * pad - KH + 1, wo = W + 2 * pad - KW + 1;
    const int K = Cin * KH * KW;
    printf("-- conv %dx%d pad=%d Cin=%d Cout=%d -> %dx%d\n",
           H, W, pad, Cin, Cout, ho, wo);
    const size_t nx = (size_t)Cin * H * W, nw = (size_t)Cout * K;
    float *X = malloc(nx * 4), *Wt = malloc(nw * 4), *dy = malloc((size_t)Cout * ho * wo * 4);
    for (size_t i = 0; i < nx; ++i) X[i] = frand();
    for (size_t i = 0; i < nw; ++i) Wt[i] = frand() * 0.5f;
    for (size_t i = 0; i < (size_t)Cout * ho * wo; ++i) dy[i] = frand();

    vc4_buf bX, bW, bY, bdW, bdX;
    if (vc4_alloc(ctx, nx * 4, &bX) != VC4_OK ||
        vc4_alloc(ctx, nw * 4, &bW) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Cout * ho * wo * 4, &bY) != VC4_OK ||
        vc4_alloc(ctx, nw * 4, &bdW) != VC4_OK ||
        vc4_alloc(ctx, nx * 4, &bdX) != VC4_OK)
        return 1;
    vc4_upload(ctx, &bX, X, nx * 4);
    vc4_upload(ctx, &bW, Wt, nw * 4);
    vc4_upload(ctx, &bY, dy, (size_t)Cout * ho * wo * 4);

    vc4_status st = vc4_conv2d_backward(ctx, &bX, Cin, H, W, &bW, Cout, KH, KW,
                                        &bY, pad, &bdW, &bdX);
    if (st != VC4_OK) {
        printf("FAIL conv_bwd: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    float *gW = malloc(nw * 4), *gX = malloc(nx * 4);
    vc4_download(ctx, &bdW, gW, nw * 4);
    vc4_download(ctx, &bdX, gX, nx * 4);

    /* ---- direct double-precision oracle (tight tolerance) ---- */
    {
        double *oW = calloc(nw, 8), *oX = calloc(nx, 8);
        for (int co = 0; co < Cout; ++co)
            for (int y = 0; y < ho; ++y)
                for (int x = 0; x < wo; ++x) {
                    double g = dy[((size_t)co * ho + y) * wo + x];
                    for (int ci = 0; ci < Cin; ++ci)
                        for (int ky = 0; ky < KH; ++ky)
                            for (int kx = 0; kx < KW; ++kx) {
                                int iy = y - pad + ky, ix = x - pad + kx;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    continue;
                                double xv = X[((size_t)ci * H + iy) * W + ix];
                                oW[(((size_t)co * Cin + ci) * KH + ky) * KW + kx] += g * xv;
                                oX[((size_t)ci * H + iy) * W + ix] +=
                                    g * Wt[(((size_t)co * Cin + ci) * KH + ky) * KW + kx];
                            }
                }
        double e2w = 0, r2w = 0, e2x = 0, r2x = 0, mw = 0, mx = 0;
        for (size_t i = 0; i < nw; ++i) {
            double e = fabs(gW[i] - oW[i]);
            e2w += e * e; r2w += oW[i] * oW[i];
            if (e > mw) mw = e;
        }
        for (size_t i = 0; i < nx; ++i) {
            double e = fabs(gX[i] - oX[i]);
            e2x += e * e; r2x += oX[i] * oX[i];
            if (e > mx) mx = e;
        }
        double l2w = sqrt(e2w) / sqrt(r2w), l2x = sqrt(e2x) / sqrt(r2x);
        printf("%s conv dW vs oracle: rel_L2=%.3g max_abs=%.3g\n",
               (l2w < 2e-5 && mw < 2e-4) ? "PASS" : "FAIL", l2w, mw);
        printf("%s conv dX vs oracle: rel_L2=%.3g max_abs=%.3g\n",
               (l2x < 2e-5 && mx < 2e-4) ? "PASS" : "FAIL", l2x, mx);
        if (!(l2w < 2e-5 && mw < 2e-4 && l2x < 2e-5 && mx < 2e-4))
            return 1;
        free(oW); free(oX);
    }

    /* ---- central differences of the scalar loss ---- */
    const double h = 0.05, tol = 1e-3;
    gc_t gw = {0}, gx = {0};
    for (size_t i = 0; i < nw; ++i) {
        float save = Wt[i];
        Wt[i] = save + (float)h;
        vc4_upload(ctx, &bW, Wt, nw * 4);
        double lp = conv_loss(ctx, &bX, Cin, H, W, &bW, Cout, KH, KW, pad, &bY, dy, ho, wo);
        Wt[i] = save - (float)h;
        vc4_upload(ctx, &bW, Wt, nw * 4);
        double lm = conv_loss(ctx, &bX, Cin, H, W, &bW, Cout, KH, KW, pad, &bY, dy, ho, wo);
        Wt[i] = save;
        gc_add(&gw, gW[i], (lp - lm) / (2 * h), tol);
    }
    vc4_upload(ctx, &bW, Wt, nw * 4);
    for (size_t i = 0; i < nx; ++i) {
        float save = X[i];
        X[i] = save + (float)h;
        vc4_upload(ctx, &bX, X, nx * 4);
        double lp = conv_loss(ctx, &bX, Cin, H, W, &bW, Cout, KH, KW, pad, &bY, dy, ho, wo);
        X[i] = save - (float)h;
        vc4_upload(ctx, &bX, X, nx * 4);
        double lm = conv_loss(ctx, &bX, Cin, H, W, &bW, Cout, KH, KW, pad, &bY, dy, ho, wo);
        X[i] = save;
        gc_add(&gx, gX[i], (lp - lm) / (2 * h), tol);
    }
    vc4_upload(ctx, &bX, X, nx * 4);
    char nm[64];
    snprintf(nm, sizeof(nm), "conv dW %dx%d pad%d", H, W, pad);
    gc_print(nm, &gw, (int)nw, tol);
    snprintf(nm, sizeof(nm), "conv dX %dx%d pad%d", H, W, pad);
    gc_print(nm, &gx, (int)nx, tol);
    free(X); free(Wt); free(dy); free(gW); free(gX);
    return gw.bad != 0 || gx.bad != 0;
}

/* ---------------- max pool forward + backward ---------------- */
static double pool_loss(vc4_ctx *ctx, const vc4_buf *in, int C, int H, int W,
                        int op, vc4_buf *out, const float *dy, int ho)
{
    int a, b;
    if (vc4_maxpool2x2(ctx, in, C, H, W, op, out, &a, &b) != VC4_OK)
        return NAN;
    float *o = malloc((size_t)C * ho * op * 4);
    vc4_download(ctx, out, o, (size_t)C * ho * op * 4);
    double s = 0;
    for (size_t i = 0; i < (size_t)C * ho * op; ++i)
        s += (double)o[i] * dy[i];
    free(o);
    return s;
}

static int t_pool(vc4_ctx *ctx, int C, int H, int W)
{
    const int Ho = H / 2, Wo = W / 2, op = (Wo + 15) & ~15;
    printf("-- pool %dx%d C=%d -> %dx%d pitch %d\n", H, W, C, Ho, Wo, op);
    size_t n = (size_t)C * H * W, np = (size_t)C * Ho * op;
    float *x = malloc(n * 4 + 64), *dy = malloc(np * 4);
    for (size_t i = 0; i < n; ++i) x[i] = frand();
    memset(x + n, 0, 64);
    for (size_t i = 0; i < np; ++i) dy[i] = frand();
    vc4_buf bX, bP, bdP, bdX;
    if (vc4_alloc(ctx, n * 4 + 64, &bX) != VC4_OK ||
        vc4_alloc(ctx, np * 4, &bP) != VC4_OK ||
        vc4_alloc(ctx, np * 4, &bdP) != VC4_OK ||
        vc4_alloc(ctx, n * 4, &bdX) != VC4_OK)
        return 1;
    vc4_upload(ctx, &bX, x, n * 4 + 64);
    vc4_upload(ctx, &bdP, dy, np * 4);
    int ho, wo;
    vc4_status st = vc4_maxpool2x2(ctx, &bX, C, H, W, op, &bP, &ho, &wo);
    if (st != VC4_OK) { printf("FAIL pool fwd %s\n", vc4_strerror(st)); return 1; }
    st = vc4_maxpool2x2_backward(ctx, &bX, &bdP, C, H, W, op, &bdX);
    if (st != VC4_OK) { printf("FAIL pool bwd %s\n", vc4_strerror(st)); return 1; }
    float *g = malloc(n * 4);
    vc4_download(ctx, &bdX, g, n * 4);

    /* oracle */
    {
        check_t ck = {0};
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int px = 0; px < W; ++px) {
                    int gy = y & ~1, gx = px & ~1;
                    float mx = x[((size_t)c * H + gy) * W + gx];
                    for (int a = 0; a < 2; ++a)
                        for (int b = 0; b < 2; ++b) {
                            float v = x[((size_t)c * H + gy + a) * W + gx + b];
                            if (v > mx) mx = v;
                        }
                    double ref = (x[((size_t)c * H + y) * W + px] == mx)
                        ? dy[((size_t)c * Ho + y / 2) * op + px / 2] : 0.0;
                    check_elem(&ck, g[((size_t)c * H + y) * W + px], ref, 1e-6, 1);
                }
        printf("%s pool bwd vs oracle: mism=%ld max_abs=%.3g\n",
               ck.mismatches ? "FAIL" : "PASS", ck.mismatches, ck.max_abs);
        if (ck.mismatches) return 1;
    }

    /* maxpool is piecewise linear: a central difference crosses a kink when
       the 2x2 argmax changes inside +-h.  Those elements have no finite
       difference that equals the analytic subgradient, so they are counted
       and excluded (the direct QPU-vs-oracle comparison above covers every
       element exactly, kinks included). */
    const double h = 0.05, tol = 1e-3;
    gc_t gg = {0};
    long kink = 0;
    for (size_t i = 0; i < n; ++i) {
        int c = (int)(i / ((size_t)H * W));
        int y = (int)((i / W) % H), px = (int)(i % W);
        int gy = y & ~1, gx = px & ~1;
        float mx = x[(size_t)(c * H + gy) * W + gx];
        double second = -1e30;
        int nmax = 0;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                float v = x[(size_t)(c * H + gy + a) * W + gx + b];
                if (v > mx) mx = v;
            }
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                float v = x[(size_t)(c * H + gy + a) * W + gx + b];
                if (v == mx) nmax++;
                else if (v > second) second = v;
            }
        int is_max = (x[i] == mx);
        double margin = is_max ? ((nmax > 1) ? 0.0 : (double)mx - second)
                               : (double)mx - (double)x[i];
        if (margin <= 2 * h) {       /* argmax can change within +-h */
            kink++;
            continue;
        }
        float save = x[i];
        x[i] = save + (float)h;
        vc4_upload(ctx, &bX, x, n * 4 + 64);
        double lp = pool_loss(ctx, &bX, C, H, W, op, &bP, dy, Ho);
        x[i] = save - (float)h;
        vc4_upload(ctx, &bX, x, n * 4 + 64);
        double lm = pool_loss(ctx, &bX, C, H, W, op, &bP, dy, Ho);
        x[i] = save;
        gc_add(&gg, g[i], (lp - lm) / (2 * h), tol);
    }
    vc4_upload(ctx, &bX, x, n * 4 + 64);
    char nm[64];
    snprintf(nm, sizeof(nm), "maxpool bwd %dx%d", H, W);
    gc_print(nm, &gg, (int)n, tol);
    printf("     maxpool FD: %ld of %zu elements skipped (argmax changes within "
           "+-%.2f; the oracle check above covers them)\n", kink, n, h);
    free(x); free(dy); free(g);
    return gg.bad != 0;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 1;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n"); gpu_unlock(lock); return 2;
    }
    printf("libvc4infer %s conv/pool gradient checks (h=0.05, tol=1e-3)\n",
           VC4INFER_VERSION_STRING);
    int rc = 0;
    for (int r = 0; r < reps; ++r) {
        rc |= t_conv(ctx, 6, 6, 0);
        rc |= t_conv(ctx, 6, 6, 1);
        rc |= t_pool(ctx, 16, 16, 16);
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "CONV GRADCHECK FAILED\n" : "CONV GRADCHECK PASSED\n");
    return rc;
}
