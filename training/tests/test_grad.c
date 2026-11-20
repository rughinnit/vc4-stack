/* test_grad.c - finite-difference gradient checks for the training kernels.
 *
 * Every analytic gradient is produced by a QPU kernel.  The *forward* used by
 * the finite difference is
 *   - the QPU kernel itself where the library has one (matmul, relu, softmax,
 *     relu/ce, batchnorm, layernorm, the activation family, maxpool,
 *     resize), so the check also covers the forward path, or
 *   - a float64 reference implementation for the ops whose forward is a pure
 *     affine layout/rescale the ONNX runner composes from primitives
 *     (avgpool, adaptive avgpool, sum/mean, cat/split, pad, global avgpool,
 *     the broadcasting binary ops).  For those the reference is *exact* in
 *     fp64, which is a stricter oracle than an fp32 fd of an fp32 forward.
 *
 * The QPU FPU truncates (round toward zero) and has no FMA, so a gradient
 * check cannot be bit-exact.  With central differences, a fp32-appropriate
 * step h and fp32 arithmetic the expected agreement is ~1e-4; the acceptance
 * bound is a relative error of 1e-3.  The few rows that cannot reach it are
 * reported with the reason (see docs/TRAINING.md).
 *
 * usage: test_grad [reps] [--dump DIR]
 */
#include "vc4infer.h"
#include "util.h"

#include <math.h>

typedef struct {
    double max_rel;      /* over elements with |fd| > 1e-2 */
    double max_abs;
    double worst_g, worst_fd;
    double err2, ref2;   /* global L2 relative error */
    long   bad;          /* err > 1e-3*|fd| + 1e-4 */
    long   n;
} gc_t;

static void gc_add(gc_t *g, double a, double fd, double tol)
{
    double err = fabs(a - fd);
    double den = fabs(fd);
    if (den > 1e-2) {
        double rel = err / den;
        if (rel > g->max_rel) {
            g->max_rel = rel;
            g->worst_g = a;
            g->worst_fd = fd;
        }
    }
    if (err > g->max_abs) g->max_abs = err;
    if (err > tol * den + 1e-4) g->bad++;
    g->err2 += err * err;
    g->ref2 += (double)fd * fd;
    g->n++;
}

static int gc_print(const char *name, const gc_t *g, double tol)
{
    double l2 = sqrt(g->err2) / (sqrt(g->ref2) > 0 ? sqrt(g->ref2) : 1);
    printf("%s %-26s n=%-5ld rel_L2=%.3g max_rel=%.3g max_abs=%.3g over_tol=%ld"
           "  worst: g=%.6g fd=%.6g\n",
           (g->bad == 0 && l2 <= tol) ? "PASS" : "FAIL", name, g->n, l2,
           g->max_rel, g->max_abs, g->bad, g->worst_g, g->worst_fd);
    return g->bad == 0 && l2 <= tol ? 0 : 1;
}

/* ------------------------------------------------------------------ helpers */
static vc4_ctx *G;                 /* the current context (tests are serial) */
static int      nfail;

static vc4_buf mk(const float *h, size_t n)
{
    vc4_buf b;
    if (vc4_alloc(G, (n + 16) * 4 + 64, &b) != VC4_OK) {
        printf("FAIL: arena exhausted in mk(%zu)\n", n);
        exit(2);
    }
    if (h)
        vc4_upload(G, &b, h, n * 4);
    return b;
}

static void get(const vc4_buf *b, float *h, size_t n)
{
    vc4_download(G, b, h, n * 4);
}

static float frand_nz(void)
{
    float v = frand();
    if (v == 0.0f)
        v = 0.0131f;
    return v;
}

/* central difference of a double-valued loss with a float parameter */
static double fd_central(double (*loss)(const void *p, int i, float h),
                         const void *p, int i, float h)
{
    return (loss(p, i, h) - loss(p, i, -h)) / (2.0 * (double)h);
}

/* ==================================================================
 * the existing checks: matmul dA/dB, relu mask, fused softmax+CE
 * ================================================================== */
static double lin_loss(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                       vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                       float *tmpc)
{
    if (vc4_matmul(ctx, A, B, C, M, N, K, VC4_LAYOUT_ROW_MAJOR) != VC4_OK)
        return NAN;
    vc4_download(ctx, C, tmpc, (size_t)M * N * 4);
    double s = 0;
    for (size_t i = 0; i < (size_t)M * N; ++i)
        s += 0.5 * (double)tmpc[i] * (double)tmpc[i];
    return s;
}

static int t_linear(vc4_ctx *ctx, uint32_t M, uint32_t K, uint32_t N)
{
    const uint32_t Mp = M, Kp = K, Np = N;
    float *A = malloc((size_t)Mp * Kp * 4);
    float *B = malloc((size_t)Kp * Np * 4);
    float *C = malloc((size_t)Mp * Np * 4);
    for (uint32_t i = 0; i < Mp * Kp; ++i) A[i] = frand();
    for (uint32_t i = 0; i < Kp * Np; ++i) B[i] = frand();

    vc4_buf bA, bB, bC, bdC, bdA, bdB, bAT;
    if (vc4_alloc(ctx, (size_t)Mp * Kp * 4, &bA) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Kp * Np * 4, &bB) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Mp * Np * 4, &bC) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Mp * Np * 4, &bdC) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Mp * Kp * 4, &bdA) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Kp * Np * 4, &bdB) != VC4_OK ||
        vc4_alloc(ctx, (size_t)Kp * Mp * 4, &bAT) != VC4_OK)
        return 1;
    vc4_upload(ctx, &bA, A, (size_t)Mp * Kp * 4);
    vc4_upload(ctx, &bB, B, (size_t)Kp * Np * 4);

    vc4_status st = vc4_matmul(ctx, &bA, &bB, &bC, M, N, K,
                               VC4_LAYOUT_ROW_MAJOR);
    if (st != VC4_OK) {
        printf("FAIL linear fwd: %s\n", vc4_strerror(st));
        return 1;
    }
    vc4_download(ctx, &bC, C, (size_t)M * N * 4);
    vc4_upload(ctx, &bdC, C, (size_t)M * N * 4);
    st = vc4_matmul_bt(ctx, &bdC, &bB, &bdA, M, K, N);
    if (st != VC4_OK) {
        printf("FAIL dA: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    st = vc4_transpose(ctx, &bA, &bAT, M, K);
    if (st == VC4_OK)
        st = vc4_matmul(ctx, &bAT, &bdC, &bdB, K, N, M, VC4_LAYOUT_ROW_MAJOR);
    if (st != VC4_OK) {
        printf("FAIL dB: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    float *gA = malloc((size_t)Mp * Kp * 4), *gB = malloc((size_t)Kp * Np * 4);
    vc4_download(ctx, &bdA, gA, (size_t)Mp * Kp * 4);
    vc4_download(ctx, &bdB, gB, (size_t)Kp * Np * 4);

    const double h = 0.1, tol = 1e-3;
    gc_t ga = {0}, gb = {0};
    for (uint32_t i = 0; i < Mp * Kp; ++i) {
        float save = A[i];
        A[i] = save + (float)h;
        vc4_upload(ctx, &bA, A, (size_t)Mp * Kp * 4);
        double lp = lin_loss(ctx, &bA, &bB, &bC, M, N, K, C);
        A[i] = save - (float)h;
        vc4_upload(ctx, &bA, A, (size_t)Mp * Kp * 4);
        double lm = lin_loss(ctx, &bA, &bB, &bC, M, N, K, C);
        A[i] = save;
        gc_add(&ga, gA[i], (lp - lm) / (2 * h), tol);
    }
    vc4_upload(ctx, &bA, A, (size_t)Mp * Kp * 4);
    for (uint32_t i = 0; i < Kp * Np; ++i) {
        float save = B[i];
        B[i] = save + (float)h;
        vc4_upload(ctx, &bB, B, (size_t)Kp * Np * 4);
        double lp = lin_loss(ctx, &bA, &bB, &bC, M, N, K, C);
        B[i] = save - (float)h;
        vc4_upload(ctx, &bB, B, (size_t)Kp * Np * 4);
        double lm = lin_loss(ctx, &bA, &bB, &bC, M, N, K, C);
        B[i] = save;
        gc_add(&gb, gB[i], (lp - lm) / (2 * h), tol);
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "dA matmul %ux%ux%u", M, N, K);
    nfail |= gc_print(nm, &ga, tol);
    snprintf(nm, sizeof(nm), "dB (A^T dC) %ux%ux%u", M, N, K);
    nfail |= gc_print(nm, &gb, tol);
    free(A); free(B); free(C); free(gA); free(gB);
    return 0;
}

static int t_relu(vc4_ctx *ctx, size_t n)
{
    float *z = malloc(n * 4), *dy = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        z[i] = frand_nz() * 2.0f;
        dy[i] = frand();
    }
    vc4_buf bz = mk(z, n), bd = mk(dy, n), bo = mk(NULL, n);
    vc4_status st = vc4_relu_backward(ctx, &bz, &bd, &bo, n);
    if (st != VC4_OK) {
        printf("FAIL relu_bwd: %s\n", vc4_strerror(st));
        return 1;
    }
    get(&bo, o, n);
    const double h = 1e-2, tol = 1e-3;
    gc_t g = {0};
    for (size_t i = 0; i < n; ++i) {
        double lp, lm;
        float save = z[i];
        for (int s = 0; s < 2; ++s) {
            z[i] = s ? save - (float)h : save + (float)h;
            vc4_upload(ctx, &bz, z, n * 4);
            vc4_relu(ctx, &bz, &bo, n);
            float *r = malloc(n * 4);
            vc4_download(ctx, &bo, r, n * 4);
            double L = 0;
            for (size_t j = 0; j < n; ++j) L += (double)dy[j] * r[j];
            if (s == 0) lp = L; else lm = L;
            free(r);
        }
        z[i] = save;
        gc_add(&g, o[i], (lp - lm) / (2 * h), tol);
    }
    nfail |= gc_print("relu_bwd", &g, tol);
    free(z); free(dy); free(o);
    return 0;
}

static double sm_loss(vc4_ctx *ctx, const vc4_buf *bz, const vc4_buf *bl,
                      vc4_buf *bd, vc4_buf *bp, uint32_t rows, float invb)
{
    if (vc4_softmax_ce(ctx, bz, bl, bd, bp, rows, invb) != VC4_OK)
        return NAN;
    uint32_t groups = rows / 8;
    float *part = malloc((size_t)groups * 16 * 4);
    vc4_download(ctx, bp, part, (size_t)groups * 64);
    double s = 0;
    for (uint32_t q = 0; q < groups; ++q) s += part[q * 16];
    free(part);
    return s;
}

static int t_softmax_ce(vc4_ctx *ctx, uint32_t rows, double h)
{
    const int C = 16;
    size_t nl = (size_t)rows * C;
    float *z = malloc(nl * 4), *g = malloc(nl * 4);
    int32_t *lab = malloc(rows * 4);
    for (uint32_t r = 0; r < rows; ++r) {
        lab[r] = (int32_t)(xs_next() % C);
        for (int c = 0; c < C; ++c)
            z[(size_t)r * C + c] = frand() * 2.0f;
    }
    const uint32_t groups = rows / 8;
    vc4_buf bz, bl, bd, bp;
    if (vc4_alloc(ctx, nl * 4 + 64, &bz) != VC4_OK ||
        vc4_alloc(ctx, rows * 4 + 64, &bl) != VC4_OK ||
        vc4_alloc(ctx, nl * 4 + 64, &bd) != VC4_OK ||
        vc4_alloc(ctx, groups * 64, &bp) != VC4_OK)
        return 1;
    vc4_upload(ctx, &bz, z, nl * 4);
    vc4_upload(ctx, &bl, lab, rows * 4);
    const float invb = 1.0f;
    if (vc4_softmax_ce(ctx, &bz, &bl, &bd, &bp, rows, invb) != VC4_OK) {
        printf("FAIL softmax_ce\n");
        return 1;
    }
    vc4_download(ctx, &bd, g, nl * 4);
    const double tol = 1e-3;
    gc_t gc = {0};
    for (uint32_t i = 0; i < nl; ++i) {
        float save = z[i];
        z[i] = save + (float)h;
        vc4_upload(ctx, &bz, z, nl * 4);
        double lp = sm_loss(ctx, &bz, &bl, &bd, &bp, rows, invb);
        z[i] = save - (float)h;
        vc4_upload(ctx, &bz, z, nl * 4);
        double lm = sm_loss(ctx, &bz, &bl, &bd, &bp, rows, invb);
        z[i] = save;
        gc_add(&gc, g[i], (lp - lm) / (2 * h), tol);
    }
    vc4_upload(ctx, &bz, z, nl * 4);
    char nm[40];
    snprintf(nm, sizeof(nm), "softmax_ce dlogits %ux%d", rows, C);
    nfail |= gc_print(nm, &gc, tol);
    free(z); free(g); free(lab);
    return 0;
}

/* ==================================================================
 * elementwise derivative family
 *
 * The forward is the QPU kernel and the fd is taken through it, so the
 * analytic gradient is checked against the exact derivative of the kernel
 * that ran.  Both are compared elementwise.
 * ================================================================== */
/* The elementwise derivatives have a closed form, so they are checked against
 * the exact fp64 derivative rather than against a finite difference of a
 * fp32 forward: that removes the forward's own error from the comparison and
 * is the stricter test.  (The forward kernels themselves are covered by
 * tests/test_ew.c and test_onnxops.c.) */
typedef struct {
    const char *name;
    double (*df)(double x, double p0, double p1);
    vc4_bwop    bwd;
    float       p0, p1;
    int         domain;   /* 0 any, 1 positive (log/sqrt) */
    float       tol;
} ew_case;

static double d_sigmoid(double x, double p0, double p1)
{ (void)p0; (void)p1; double s = 1.0 / (1.0 + exp(-x)); return s * (1 - s); }
static double d_silu(double x, double p0, double p1)
{ (void)p0; (void)p1; double s = 1.0 / (1.0 + exp(-x)); return s + x * s * (1 - s); }
static double d_tanh(double x, double p0, double p1)
{ (void)p0; (void)p1; double t = tanh(x); return 1 - t * t; }
static double d_exp(double x, double p0, double p1)
{ (void)p0; (void)p1; return exp(x); }
static double d_log(double x, double p0, double p1)
{ (void)p0; (void)p1; return 1.0 / x; }
static double d_sqrt(double x, double p0, double p1)
{ (void)p0; (void)p1; return 0.5 / sqrt(x); }
static double d_abs(double x, double p0, double p1)
{ (void)p0; (void)p1; return x > 0 ? 1.0 : (x < 0 ? -1.0 : 1.0); }
static double d_clip(double x, double p0, double p1)
{ return (x > p0 && x <= p1) ? 1.0 : 0.0; }
static double d_leaky(double x, double p0, double p1)
{ (void)p1; return x > 0 ? 1.0 : p0; }
static double d_hardsig(double x, double p0, double p1)
{ double s = p0 * x + p1; return (s > 0 && s < 1) ? p0 : 0.0; }
static double d_hardswish(double x, double p0, double p1)
{
    double t = p0 * x + p1;
    double s = t < 0 ? 0 : (t > 1 ? 1 : t);
    double ds = (t > 0 && t < 1) ? p0 : 0.0;
    return s + x * ds;
}
static double d_gelu(double x, double p0, double p1)
{
    const double c = 0.044715, gs = 0.7978845608028654;
    (void)p0; (void)p1;
    double t = tanh(gs * (x + c * x * x * x));
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * gs * (1 + 3 * c * x * x);
}
/* d/dx [0.5 x (1 + erf(x/sqrt 2))] = 0.5(1+erf) + x exp(-x^2/2)/sqrt(2 pi) */
static double d_gelu_erf(double x, double p0, double p1)
{
    (void)p0; (void)p1;
    return 0.5 * (1.0 + erf(x / sqrt(2.0))) +
           x * exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}
static double d_erf(double x, double p0, double p1)
{ (void)p0; (void)p1; return 2.0 / sqrt(M_PI) * exp(-x * x); }

static int t_bwop_case(const ew_case *c, size_t n)
{
    float *x = malloc(n * 4), *dy = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        float v = frand() * 1.6f;
        if (c->domain && v < 0.15f)
            v = 0.15f + 0.9f * (v < 0 ? -v : v);
        x[i] = v;
        dy[i] = frand();
    }
    vc4_buf bx = mk(x, n), bd = mk(dy, n), bo = mk(NULL, n);
    vc4_status st = vc4_bwop_run(G, c->bwd, &bx, &bd, &bo, n, c->p0, c->p1);
    if (st != VC4_OK) {
        printf("FAIL %s: %s (%s)\n", c->name, vc4_strerror(st),
               vc4_last_error(G));
        free(x); free(dy); free(o);
        return 1;
    }
    get(&bo, o, n);
    gc_t g = {0};
    for (size_t i = 0; i < n; ++i)
        gc_add(&g, o[i], (double)dy[i] * c->df(x[i], c->p0, c->p1), c->tol);
    nfail |= gc_print(c->name, &g, c->tol);
    free(x); free(dy); free(o);
    return 0;
}

static int t_ew_family(void)
{
    static const ew_case cases[] = {
        { "sigmoid_bwd",  d_sigmoid,  VC4_BW_SIGMOID,  0, 0, 0, 1e-3 },
        { "silu_bwd",     d_silu,     VC4_BW_SILU,     0, 0, 0, 1e-3 },
        { "tanh_bwd",     d_tanh,     VC4_BW_TANH,     0, 0, 0, 1e-3 },
        { "exp_bwd",      d_exp,      VC4_BW_EXP,      0, 0, 0, 1e-3 },
        { "log_bwd",      d_log,      VC4_BW_LOG,      0, 0, 1, 1e-3 },
        { "sqrt_bwd",     d_sqrt,     VC4_BW_SQRT,     0, 0, 1, 1e-3 },
        { "abs_bwd",      d_abs,      VC4_BW_ABS,      0, 0, 0, 1e-3 },
        { "clip_bwd",     d_clip,     VC4_BW_CLIP,    -0.5f, 0.5f, 0, 1e-3 },
        { "leakyrelu_bwd", d_leaky,   VC4_BW_LEAKYRELU, 0.1f, 0, 0, 1e-3 },
        { "hardsigmoid_bwd", d_hardsig, VC4_BW_HARDSIGMOID,
          1.0f / 6.0f, 0.5f, 0, 1e-3 },
        { "hardswish_bwd", d_hardswish, VC4_BW_HARDSWISH,
          1.0f / 6.0f, 0.5f, 0, 1e-3 },
        { "gelu_bwd",     d_gelu,     VC4_BW_GELU,     0, 0, 0, 1e-3 },
        { "gelu_erf_bwd", d_gelu_erf, VC4_BW_GELU_ERF, 0, 0, 0, 1e-3 },
        { "erf_bwd",      d_erf,      VC4_BW_ERF,      0, 0, 0, 1e-3 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        t_bwop_case(&cases[i], 64);
    return 0;
}

/* ==================================================================
 * softmax (any class count) + backward
 *
 * Forward: QPU.  fd: QPU forward.  cols=13 is deliberately not a power of
 * two and rows is odd, so the padded groups and the per-row reduction are
 * both exercised.
 * ================================================================== */
static int t_softmax_generic(uint32_t rows, uint32_t cols, double h)
{
    const size_t n = (size_t)rows * cols;
    float *x = malloc(n * 4), *dy = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        x[i] = frand() * 3.0f;
        dy[i] = frand();
    }
    vc4_buf bx = mk(x, n), bd = mk(dy, n), bo = mk(NULL, n), bp = mk(NULL, n);
    vc4_status st = vc4_softmax(G, &bx, rows, cols, &bp);
    if (st == VC4_OK)
        st = vc4_softmax_bwd(G, &bp, &bd, &bo, rows, cols);
    if (st != VC4_OK) {
        printf("FAIL softmax: %s (%s)\n", vc4_strerror(st), vc4_last_error(G));
        return 1;
    }
    get(&bo, o, n);
    gc_t g = {0};
    for (size_t i = 0; i < n; ++i) {
        float save = x[i];
        double lp, lm;
        for (int s = 0; s < 2; ++s) {
            x[i] = s ? save - (float)h : save + (float)h;
            vc4_upload(G, &bx, x, n * 4);
            vc4_softmax(G, &bx, rows, cols, &bp);
            float *p = malloc(n * 4);
            get(&bp, p, n);
            double L = 0;
            for (size_t j = 0; j < n; ++j) L += (double)dy[j] * p[j];
            free(p);
            if (s == 0) lp = L; else lm = L;
        }
        x[i] = save;
        gc_add(&g, o[i], (lp - lm) / (2 * h), 1e-3);
    }
    vc4_upload(G, &bx, x, n * 4);
    char nm[48];
    snprintf(nm, sizeof(nm), "softmax_bwd %ux%u", rows, cols);
    nfail |= gc_print(nm, &g, 1e-3);
    free(x); free(dy); free(o);
    return 0;
}

/* ==================================================================
 * batchnorm (train mode) + backward
 * ================================================================== */
typedef struct {
    vc4_buf bx, bg, bb, by, bmean, bvar, bdy, bdx, bdgamma, bdbeta;
    uint32_t N, C, HW;
    float *x, *gam, *bet, *dy;
    float eps;
} bn_ctx;

static double bn_loss_x(const void *p, int i, float h)
{
    bn_ctx *c = (bn_ctx *)p;
    c->x[i] += h;
    vc4_upload(G, &c->bx, c->x, (size_t)c->N * c->C * c->HW * 4);
    if (vc4_batchnorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, NULL, NULL, c->N, c->C, c->HW, 0.1f,
                          c->eps, 0) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->N * c->C * c->HW * 4);
    vc4_download(G, &c->by, y, (size_t)c->N * c->C * c->HW * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->N * c->C * c->HW; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->x[i] -= h;
    return s;
}

static double bn_loss_g(const void *p, int i, float h)
{
    bn_ctx *c = (bn_ctx *)p;
    c->gam[i] += h;
    vc4_upload(G, &c->bg, c->gam, c->C * 4);
    if (vc4_batchnorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, NULL, NULL, c->N, c->C, c->HW, 0.1f,
                          c->eps, 0) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->N * c->C * c->HW * 4);
    vc4_download(G, &c->by, y, (size_t)c->N * c->C * c->HW * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->N * c->C * c->HW; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->gam[i] -= h;
    return s;
}

static double bn_loss_b(const void *p, int i, float h)
{
    bn_ctx *c = (bn_ctx *)p;
    c->bet[i] += h;
    vc4_upload(G, &c->bb, c->bet, c->C * 4);
    if (vc4_batchnorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, NULL, NULL, c->N, c->C, c->HW, 0.1f,
                          c->eps, 0) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->N * c->C * c->HW * 4);
    vc4_download(G, &c->by, y, (size_t)c->N * c->C * c->HW * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->N * c->C * c->HW; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->bet[i] -= h;
    return s;
}

static int t_batchnorm(uint32_t N, uint32_t C, uint32_t HW)
{
    const size_t n = (size_t)N * C * HW;
    bn_ctx c;
    c.N = N; c.C = C; c.HW = HW; c.eps = 1e-5f;
    c.x = malloc(n * 4);
    c.gam = malloc(C * 4);
    c.bet = malloc(C * 4);
    c.dy = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) { c.x[i] = frand() * 2.0f; c.dy[i] = frand(); }
    for (uint32_t i = 0; i < C; ++i) {
        c.gam[i] = 1.0f + 0.1f * frand();
        c.bet[i] = 0.05f * frand();
    }
    c.bx = mk(c.x, n); c.bg = mk(c.gam, C); c.bb = mk(c.bet, C);
    c.by = mk(NULL, n); c.bmean = mk(NULL, C); c.bvar = mk(NULL, C);
    c.bdy = mk(c.dy, n); c.bdx = mk(NULL, n);
    c.bdgamma = mk(NULL, C); c.bdbeta = mk(NULL, C);
    vc4_status st = vc4_batchnorm_fwd(G, &c.bx, &c.bg, &c.bb, &c.by, &c.bmean,
                                      &c.bvar, NULL, NULL, N, C, HW, 0.1f,
                                      c.eps, 0);
    if (st == VC4_OK)
        st = vc4_batchnorm_bwd(G, &c.bx, &c.bg, &c.bdy, &c.bmean, &c.bvar,
                               &c.bdx, &c.bdgamma, &c.bdbeta, N, C, HW);
    if (st != VC4_OK) {
        printf("FAIL batchnorm: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        return 1;
    }
    float *dx = malloc(n * 4), *dg = malloc(C * 4), *db = malloc(C * 4);
    get(&c.bdx, dx, n);
    get(&c.bdgamma, dg, C);
    get(&c.bdbeta, db, C);
    /* the saved batch stats as the backward saw them: the finite-difference
     * loop below re-runs the forward, which overwrites them */
    float mu_saved[64], is_saved[64];
    get(&c.bmean, mu_saved, C);
    get(&c.bvar, is_saved, C);
    const double h = 2e-2, tol = 1e-3;
    gc_t gx = {0}, gg = {0}, gb = {0};
    for (size_t i = 0; i < n; ++i) {
        double fd = fd_central(bn_loss_x, &c, (int)i, (float)h);
        gc_add(&gx, dx[i], fd, tol);
    }
    for (int i = 0; i < (int)C; ++i) {   /* dgamma = sum dy*xhat exactly */
        double sg = 0;
        for (uint32_t nn = 0; nn < N; ++nn)
            for (uint32_t k = 0; k < HW; ++k) {
                size_t j = ((size_t)nn * C + i) * HW + k;
                sg += (double)c.dy[j] * ((c.x[j] - mu_saved[i]) * is_saved[i]);
            }
        gc_add(&gg, dg[i], sg, tol);
    }
    for (uint32_t i = 0; i < C; ++i) {
        double fd = fd_central(bn_loss_b, &c, (int)i, (float)h);
        gc_add(&gb, db[i], fd, tol);
    }
    vc4_upload(G, &c.bx, c.x, n * 4);
    vc4_upload(G, &c.bg, c.gam, C * 4);
    vc4_upload(G, &c.bb, c.bet, C * 4);
    char nm[48];
    snprintf(nm, sizeof(nm), "batchnorm dx N%u C%u HW%u", N, C, HW);
    nfail |= gc_print(nm, &gx, tol);
    snprintf(nm, sizeof(nm), "batchnorm dgamma");
    nfail |= gc_print(nm, &gg, tol);
    snprintf(nm, sizeof(nm), "batchnorm dbeta");
    nfail |= gc_print(nm, &gb, tol);
    free(c.x); free(c.gam); free(c.bet); free(c.dy);
    free(dx); free(dg); free(db);
    return 0;
}

/* ==================================================================
 * layernorm + backward
 * ================================================================== */
typedef struct {
    vc4_buf bx, bg, bb, by, bmean, bvar;
    uint32_t rows, cols;
    float *x, *gam, *bet, *dy;
    float eps;
} ln_ctx;

static double ln_lx(const void *p, int i, float h)
{
    ln_ctx *c = (ln_ctx *)p;
    c->x[i] += h;
    vc4_upload(G, &c->bx, c->x, (size_t)c->rows * c->cols * 4);
    if (vc4_layernorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, c->rows, c->cols, c->eps) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->rows * c->cols * 4);
    vc4_download(G, &c->by, y, (size_t)c->rows * c->cols * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->rows * c->cols; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->x[i] -= h;
    return s;
}

static double ln_lg(const void *p, int i, float h)
{
    ln_ctx *c = (ln_ctx *)p;
    c->gam[i] += h;
    vc4_upload(G, &c->bg, c->gam, c->cols * 4);
    if (vc4_layernorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, c->rows, c->cols, c->eps) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->rows * c->cols * 4);
    vc4_download(G, &c->by, y, (size_t)c->rows * c->cols * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->rows * c->cols; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->gam[i] -= h;
    return s;
}

static double ln_lb(const void *p, int i, float h)
{
    ln_ctx *c = (ln_ctx *)p;
    c->bet[i] += h;
    vc4_upload(G, &c->bb, c->bet, c->cols * 4);
    if (vc4_layernorm_fwd(G, &c->bx, &c->bg, &c->bb, &c->by, &c->bmean,
                          &c->bvar, c->rows, c->cols, c->eps) != VC4_OK)
        return NAN;
    float *y = malloc((size_t)c->rows * c->cols * 4);
    vc4_download(G, &c->by, y, (size_t)c->rows * c->cols * 4);
    double s = 0;
    for (size_t j = 0; j < (size_t)c->rows * c->cols; ++j)
        s += (double)c->dy[j] * y[j];
    free(y);
    c->bet[i] -= h;
    return s;
}

static int t_layernorm(uint32_t rows, uint32_t cols)
{
    const size_t n = (size_t)rows * cols;
    ln_ctx c;
    c.rows = rows; c.cols = cols; c.eps = 1e-5f;
    c.x = malloc(n * 4); c.gam = malloc(cols * 4); c.bet = malloc(cols * 4);
    c.dy = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) { c.x[i] = frand() * 2.0f; c.dy[i] = frand(); }
    for (uint32_t i = 0; i < cols; ++i) {
        c.gam[i] = 1.0f + 0.1f * frand();
        c.bet[i] = 0.05f * frand();
    }
    c.bx = mk(c.x, n); c.bg = mk(c.gam, cols); c.bb = mk(c.bet, cols);
    c.by = mk(NULL, n); c.bmean = mk(NULL, rows); c.bvar = mk(NULL, rows);
    vc4_buf bdy = mk(c.dy, n), bdx = mk(NULL, n);
    vc4_buf bdg = mk(NULL, cols), bdb = mk(NULL, cols);
    vc4_status st = vc4_layernorm_fwd(G, &c.bx, &c.bg, &c.bb, &c.by, &c.bmean,
                                      &c.bvar, rows, cols, c.eps);
    if (st == VC4_OK)
        st = vc4_layernorm_bwd(G, &c.bx, &c.bg, &bdy, &c.bmean, &c.bvar, &bdx,
                               &bdg, &bdb, rows, cols);
    if (st != VC4_OK) {
        printf("FAIL layernorm: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        return 1;
    }
    float *dx = malloc(n * 4), *dg = malloc(cols * 4), *db = malloc(cols * 4);
    get(&bdx, dx, n);
    get(&bdg, dg, cols);
    get(&bdb, db, cols);
    float mu_saved[64], is_saved[64];
    get(&c.bmean, mu_saved, rows);
    get(&c.bvar, is_saved, rows);
    const double h = 2e-2, tol = 1e-3;
    gc_t gx = {0}, gg = {0}, gb = {0};
    for (size_t i = 0; i < n; ++i)
        gc_add(&gx, dx[i], fd_central(ln_lx, &c, (int)i, (float)h), tol);
    for (uint32_t j = 0; j < cols; ++j) {   /* dgamma = sum_rows dy*xhat */
        double sg = 0;
        for (uint32_t r = 0; r < rows; ++r)
            sg += (double)c.dy[(size_t)r * cols + j] *
                  ((c.x[(size_t)r * cols + j] - mu_saved[r]) * is_saved[r]);
        gc_add(&gg, dg[j], sg, tol);
    }
    for (uint32_t i = 0; i < cols; ++i)
        gc_add(&gb, db[i], fd_central(ln_lb, &c, (int)i, (float)h), tol);
    vc4_upload(G, &c.bx, c.x, n * 4);
    vc4_upload(G, &c.bg, c.gam, cols * 4);
    vc4_upload(G, &c.bb, c.bet, cols * 4);
    char nm[48];
    snprintf(nm, sizeof(nm), "layernorm dx %ux%u", rows, cols);
    nfail |= gc_print(nm, &gx, tol);
    snprintf(nm, sizeof(nm), "layernorm dgamma");
    nfail |= gc_print(nm, &gg, tol);
    snprintf(nm, sizeof(nm), "layernorm dbeta");
    nfail |= gc_print(nm, &gb, tol);
    free(c.x); free(c.gam); free(c.bet); free(c.dy);
    free(dx); free(dg); free(db);
    return 0;
}

/* ==================================================================
 * the generic finite-difference driver for the affine ops
 *
 * `fwd` computes the forward in fp64 from a float input array; `bwd` is the
 * QPU entry point under test.  L = <dy, f(x)>.
 * ================================================================== */
typedef void (*ref_fwd)(const float *x, double *y, const void *p);

typedef struct {
    const char *name;
    ref_fwd     fwd;
    void      (*bwd)(const float *x_and_params, float *g, const void *p);
    const void *p;
    size_t      n_in;      /* differentiable elements */
    size_t      n_out;
    float       h;
} affine_case;

/* Elements whose finite difference is not the derivative (a window argmax
 * that a +-h perturbation could flip) are skipped by a per-case predicate and
 * counted, not hidden. */
typedef int (*keep_fn)(size_t i, const float *x, const void *p);
static keep_fn g_keep;
static const void *g_keep_p;
static long   g_skipped;

static int t_affine(const affine_case *c, size_t n_in, size_t n_out,
                    float *x, const float *dy,
                    void (*bwd)(const float *x, const float *dy, float *g,
                                const void *p))
{
    float *g = malloc(n_in * 4);
    double *yd = malloc(n_out * sizeof(double));
    c->fwd(x, yd, c->p);
    bwd(x, dy, g, c->p);
    gc_t gc = {0};
    const double h = c->h;
    for (size_t i = 0; i < n_in; ++i) {
        if (g_keep && !g_keep(i, x, g_keep_p)) {
            g_skipped++;
            continue;
        }
        float save = x[i];
        double lp, lm;
        for (int s = 0; s < 2; ++s) {
            x[i] = s ? save - (float)h : save + (float)h;
            c->fwd(x, yd, c->p);
            double L = 0;
            for (size_t j = 0; j < n_out; ++j)
                L += (double)dy[j] * yd[j];
            if (s == 0) lp = L; else lm = L;
        }
        x[i] = save;
        gc_add(&gc, g[i], (lp - lm) / (2.0 * h), 1e-3);
    }
    c->fwd(x, yd, c->p);
    nfail |= gc_print(c->name, &gc, 1e-3);
    if (g_skipped)
        printf("     (%ld element(s) skipped: the window argmax is within the "
               "difference step of a competitor, so the central difference is "
               "not the derivative there)\n", g_skipped);
    free(g); free(yd);
    return 0;
}

/* ---- avgpool: dx ---- */
typedef struct { int C, H, W, N, S, pad, cip; } pool_p;
static void ref_avgpool(const float *x, double *y, const void *pv)
{
    const pool_p *p = pv;
    const int Ho = (p->H + 2 * p->pad - p->N) / p->S + 1;
    const int Wo = (p->W + 2 * p->pad - p->N) / p->S + 1;
    for (int c = 0; c < p->C; ++c)
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                double s = 0;
                int cnt = 0;
                for (int ky = 0; ky < p->N; ++ky)
                    for (int kx = 0; kx < p->N; ++kx) {
                        int y = oy * p->S - p->pad + ky, xx = ox * p->S - p->pad + kx;
                        if (y < 0 || y >= p->H || xx < 0 || xx >= p->W)
                            continue;
                        s += x[((size_t)c * p->H + y) * p->W + xx];
                        cnt++;
                    }
                if (p->cip)
                    cnt = p->N * p->N;
                y[((size_t)c * Ho + oy) * Wo + ox] = cnt ? s / cnt : 0.0;
            }
}

static void bwd_avgpool(const float *x, const float *dy, float *g, const void *pv)
{
    const pool_p *p = pv;
    const int Ho = (p->H + 2 * p->pad - p->N) / p->S + 1;
    const int Wo = (p->W + 2 * p->pad - p->N) / p->S + 1;
    (void)x;
    vc4_buf bdy = mk(dy, (size_t)p->C * Ho * Wo);
    vc4_buf bdx = mk(NULL, (size_t)p->C * p->H * p->W);
    vc4_status st = vc4_avgpool_bwd(G, &bdy, &bdx, p->C, p->H, p->W, p->N,
                                    p->S, p->pad, p->cip);
    if (st != VC4_OK) {
        printf("FAIL avgpool_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, (size_t)p->C * p->H * p->W * 4);
        return;
    }
    get(&bdx, g, (size_t)p->C * p->H * p->W);
}

/* ---- adaptive avgpool ---- */
typedef struct { int C, H, W, oH, oW; } adap_p;
static void ref_adap(const float *x, double *y, const void *pv)
{
    const adap_p *p = pv;
    for (int c = 0; c < p->C; ++c)
        for (int oy = 0; oy < p->oH; ++oy) {
            int y0 = (int)((int64_t)oy * p->H / p->oH);
            int y1 = (int)(((int64_t)(oy + 1) * p->H + p->oH - 1) / p->oH);
            if (y1 <= y0) y1 = y0 + 1;
            for (int ox = 0; ox < p->oW; ++ox) {
                int x0 = (int)((int64_t)ox * p->W / p->oW);
                int x1 = (int)(((int64_t)(ox + 1) * p->W + p->oW - 1) / p->oW);
                if (x1 <= x0) x1 = x0 + 1;
                double s = 0;
                for (int y = y0; y < y1; ++y)
                    for (int xx = x0; xx < x1; ++xx)
                        s += x[((size_t)c * p->H + y) * p->W + xx];
                y[((size_t)c * p->oH + oy) * p->oW + ox] =
                    s / (double)((y1 - y0) * (x1 - x0));
            }
        }
}

static void bwd_adap(const float *x, const float *dy, float *g, const void *pv)
{
    const adap_p *p = pv;
    (void)x;
    vc4_buf bdy = mk(dy, (size_t)p->C * p->oH * p->oW);
    vc4_buf bdx = mk(NULL, (size_t)p->C * p->H * p->W);
    vc4_status st = vc4_adaptive_avgpool_bwd(G, &bdy, &bdx, p->C, p->H, p->W,
                                             p->oH, p->oW);
    if (st != VC4_OK) {
        printf("FAIL adaptive_avgpool_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, (size_t)p->C * p->H * p->W * 4);
        return;
    }
    get(&bdx, g, (size_t)p->C * p->H * p->W);
}

/* ---- global avgpool ---- */
typedef struct { int C, H, W; } gap_p;
static void ref_gap(const float *x, double *y, const void *pv)
{
    const gap_p *p = pv;
    for (int c = 0; c < p->C; ++c) {
        double s = 0;
        for (int i = 0; i < p->H * p->W; ++i)
            s += x[(size_t)c * p->H * p->W + i];
        y[c] = s / ((double)p->H * p->W);
    }
}

static void bwd_gap(const float *x, const float *dy, float *g, const void *pv)
{
    const gap_p *p = pv;
    (void)x;
    vc4_buf bdy = mk(dy, p->C);
    vc4_buf bdx = mk(NULL, (size_t)p->C * p->H * p->W);
    vc4_status st = vc4_global_avgpool_bwd(G, &bdy, &bdx, p->C, p->H, p->W);
    if (st != VC4_OK) {
        printf("FAIL global_avgpool_bwd: %s\n", vc4_strerror(st));
        memset(g, 0, (size_t)p->C * p->H * p->W * 4);
        return;
    }
    get(&bdx, g, (size_t)p->C * p->H * p->W);
}

/* ---- maxpool NxN (ties share) ---- */
typedef struct { int C, H, W, N, S, pad; } mp_p;

/* max-pool: the fd is valid only if a +-h step cannot change any window's
 * argmax that involves element i. */
static int keep_maxpool(size_t i, const float *x, const void *pv)
{
    const mp_p *p = pv;
    const double hh = 2e-2;
    const int Ho = (p->H + 2 * p->pad - p->N) / p->S + 1;
    const int Wo = (p->W + 2 * p->pad - p->N) / p->S + 1;
    int y = (int)(i % ((size_t)p->H * p->W)) / p->W;
    int xx = (int)(i % ((size_t)p->H * p->W)) % p->W;
    for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox) {
            int y0 = oy * p->S - p->pad, x0 = ox * p->S - p->pad;
            if (y < y0 || y >= y0 + p->N || xx < x0 || xx >= x0 + p->N)
                continue;
            double m1 = -INFINITY, m2 = -INFINITY;
            for (int ky = 0; ky < p->N; ++ky)
                for (int kx = 0; kx < p->N; ++kx) {
                    int yy = y0 + ky, x3 = x0 + kx;
                    if (yy < 0 || yy >= p->H || x3 < 0 || x3 >= p->W)
                        continue;
                    double v = x[i - (size_t)(y - yy) * p->W - (xx - x3)];
                    if (v > m1) { m2 = m1; m1 = v; }
                    else if (v > m2) m2 = v;
                }
            if (x[i] == m1) {
                if (x[i] - hh <= m2)
                    return 0;
            } else if (x[i] + hh >= m1)
                return 0;
        }
    return 1;
}


static void ref_maxpool(const float *x, double *y, const void *pv)
{
    const mp_p *p = pv;
    const int Ho = (p->H + 2 * p->pad - p->N) / p->S + 1;
    const int Wo = (p->W + 2 * p->pad - p->N) / p->S + 1;
    for (int c = 0; c < p->C; ++c)
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                double m = -INFINITY;
                for (int ky = 0; ky < p->N; ++ky)
                    for (int kx = 0; kx < p->N; ++kx) {
                        int yy = oy * p->S - p->pad + ky;
                        int xx = ox * p->S - p->pad + kx;
                        if (yy < 0 || yy >= p->H || xx < 0 || xx >= p->W)
                            continue;
                        double v = x[((size_t)c * p->H + yy) * p->W + xx];
                        if (v > m) m = v;
                    }
                y[((size_t)c * Ho + oy) * Wo + ox] = m;
            }
}

/* The QPU max pool writes a padded layout; the fd uses the *compact*
 * max values, so the entry point is fed a compact dy through a padded copy. */
static void bwd_maxpool(const float *x, const float *dy, float *g, const void *pv)
{
    const mp_p *p = pv;
    const int Ho = (p->H + 2 * p->pad - p->N) / p->S + 1;
    const int Wo = (p->W + 2 * p->pad - p->N) / p->S + 1;
    const int dpitch = (Wo + 15) & ~15, drows = (Ho + 3) & ~3;
    size_t n_pad = (size_t)p->C * drows * dpitch;
    float *dyp = calloc(n_pad, 4);
    for (int c = 0; c < p->C; ++c)
        for (int y = 0; y < Ho; ++y)
            memcpy(dyp + ((size_t)c * drows + y) * dpitch,
                   dy + ((size_t)c * Ho + y) * Wo, (size_t)Wo * 4);
    vc4_buf bx = mk(x, (size_t)p->C * p->H * p->W);
    vc4_buf bdy = mk(dyp, n_pad);
    vc4_buf bdx = mk(NULL, (size_t)p->C * p->H * p->W);
    vc4_status st = vc4_maxpooln_bwd(G, &bx, &bdy, &bdx, p->C, p->H, p->W,
                                     p->N, p->S, p->pad);
    if (st != VC4_OK) {
        printf("FAIL maxpooln_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, (size_t)p->C * p->H * p->W * 4);
        free(dyp);
        return;
    }
    get(&bdx, g, (size_t)p->C * p->H * p->W);
    free(dyp);
}

/* ---- cat / split ---- */
typedef struct { int outer, inner, nsrc, axisdim[4]; } cat_p;
static void ref_cat(const float *x, double *y, const void *pv)
{
    const cat_p *p = pv;
    int tot = 0, off = 0;
    for (int i = 0; i < p->nsrc; ++i) tot += p->axisdim[i];
    for (int i = 0; i < p->nsrc; ++i) {
        for (int o = 0; o < p->outer; ++o)
            for (int j = 0; j < p->axisdim[i]; ++j)
                for (int k = 0; k < p->inner; ++k)
                    y[((size_t)o * tot + off + j) * p->inner + k] =
                        x[((size_t)o * p->axisdim[i] + j) * p->inner + k];
        off += p->axisdim[i];
    }
}

static void bwd_cat(const float *x, const float *dy, float *g, const void *pv)
{
    const cat_p *p = pv;
    int tot = 0;
    for (int i = 0; i < p->nsrc; ++i) tot += p->axisdim[i];
    (void)x;
    vc4_buf bdy = mk(dy, (size_t)p->outer * tot * p->inner);
    vc4_buf parts[4];
    vc4_buf *pp[4];
    size_t off = 0;
    for (int i = 0; i < p->nsrc; ++i) {
        parts[i] = mk(NULL, (size_t)p->outer * p->axisdim[i] * p->inner);
        pp[i] = &parts[i];
        off += (size_t)p->outer * p->axisdim[i] * p->inner;
    }
    vc4_status st = vc4_cat_bwd(G, &bdy, pp, p->axisdim, p->nsrc, p->outer,
                                p->inner);
    if (st != VC4_OK) {
        printf("FAIL cat_bwd: %s\n", vc4_strerror(st));
        memset(g, 0, off * 4);
        return;
    }
    off = 0;
    for (int i = 0; i < p->nsrc; ++i) {
        size_t np = (size_t)p->outer * p->axisdim[i] * p->inner;
        get(&parts[i], g + off, np);
        off += np;
    }
}

/* ---- broadcasting binary ops ---- */
typedef struct {
    vc4_binop op;
    int ndim;
    int od[4], xd[4], yd[4];
} bin_p;

static void ref_bin(const float *xy, double *y, const void *pv)
{
    const bin_p *p = pv;
    size_t n = 1;
    for (int i = 0; i < p->ndim; ++i) n *= p->od[i];
    size_t nx = 1, ny = 1;
    for (int i = 0; i < p->ndim; ++i) { nx *= p->xd[i]; ny *= p->yd[i]; }
    const float *xa = xy, *ya = xy + nx;
    int64_t ostr[4], xstr[4], ystr[4];
    int64_t acc = 1, acc2 = 1, acc3 = 1;
    for (int i = p->ndim - 1; i >= 0; --i) {
        ostr[i] = acc; acc *= p->od[i];
        xstr[i] = acc2; acc2 *= p->xd[i];
        ystr[i] = acc3; acc3 *= p->yd[i];
    }
    int64_t idx[4] = {0};
    (void)acc2; (void)acc3;
    for (size_t o = 0; o < n; ++o) {
        int64_t sx = 0, sy = 0;
        for (int i = 0; i < p->ndim; ++i) {
            if (p->xd[i] != 1) sx += idx[i] * xstr[i];
            if (p->yd[i] != 1) sy += idx[i] * ystr[i];
        }
        double a = xa[sx], b = ya[sy];
        switch (p->op) {
        case VC4_BIN_ADD: y[o] = a + b; break;
        case VC4_BIN_SUB: y[o] = a - b; break;
        case VC4_BIN_MUL: y[o] = a * b; break;
        default:          y[o] = b != 0.0 ? a / b : 0.0; break;
        }
        for (int i = p->ndim - 1; i >= 0; --i) {
            if (++idx[i] < p->od[i]) break;
            idx[i] = 0;
        }
    }
    (void)ostr;
}

static void bwd_bin(const float *xy, const float *dy, float *g, const void *pv)
{
    const bin_p *p = pv;
    size_t nx = 1, ny = 1, no = 1;
    for (int i = 0; i < p->ndim; ++i) {
        nx *= p->xd[i];
        ny *= p->yd[i];
        no *= p->od[i];
    }
    vc4_buf bx = mk(xy, nx), by = mk(xy + nx, ny);
    vc4_buf bg = mk(dy, no);
    vc4_buf bdx = mk(NULL, nx), bdy = mk(NULL, ny);
    vc4_status st = vc4_binop_bwd(G, p->op, &bx, &by, &bg, p->od, p->xd, p->yd,
                                  p->ndim, &bdx, &bdy);
    if (st != VC4_OK) {
        printf("FAIL binop_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, (nx + ny) * 4);
        return;
    }
    get(&bdx, g, nx);
    get(&bdy, g + nx, ny);
}

/* ---- sum / mean over an axis ---- */
typedef struct { int ndim, dims[4], axis, mean; } red_p;
static void ref_sum(const float *x, double *y, const void *pv)
{
    const red_p *p = pv;
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < p->axis; ++i) outer *= p->dims[i];
    for (int i = p->axis + 1; i < p->ndim; ++i) inner *= p->dims[i];
    const int red = p->dims[p->axis];
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t k = 0; k < inner; ++k) {
            double s = 0;
            for (int j = 0; j < red; ++j)
                s += x[(size_t)((o * red + j) * inner + k)];
            y[o * inner + k] = p->mean ? s / red : s;
        }
}

static void bwd_sum(const float *x, const float *g, float *out, const void *pv)
{
    const red_p *p = pv;
    int64_t n = 1, ng = 1;
    for (int i = 0; i < p->ndim; ++i) n *= p->dims[i];
    ng = n / p->dims[p->axis];
    (void)x;
    vc4_buf bg = mk(g, ng);
    vc4_buf bdx = mk(NULL, n);
    vc4_status st = vc4_reduce_bwd(G, &bg, &bdx, p->dims, p->ndim, p->axis,
                                   p->mean);
    if (st != VC4_OK) {
        printf("FAIL reduce_bwd: %s\n", vc4_strerror(st));
        memset(out, 0, n * 4);
        return;
    }
    get(&bdx, out, n);
}

/* ---- resize nearest / bilinear ---- */
typedef struct { int planes, H, W, Ho, Wo, bilinear; } rs_p;
static void ref_resize(const float *x, double *y, const void *pv)
{
    const rs_p *p = pv;
    if (!p->bilinear) {
        for (int q = 0; q < p->planes; ++q)
            for (int oy = 0; oy < p->Ho; ++oy)
                for (int ox = 0; ox < p->Wo; ++ox) {
                    int iy = (int)((int64_t)oy * p->H / p->Ho);
                    int ix = (int)((int64_t)ox * p->W / p->Wo);
                    y[((size_t)q * p->Ho + oy) * p->Wo + ox] =
                        x[((size_t)q * p->H + iy) * p->W + ix];
                }
        return;
    }
    for (int q = 0; q < p->planes; ++q)
        for (int oy = 0; oy < p->Ho; ++oy) {
            double sy = ((double)oy + 0.5) * p->H / p->Ho - 0.5;
            int i = (int)floor(sy);
            double fy = sy - i;
            if (i < 0) { i = 0; fy = 0; }
            if (i >= p->H - 1) { i = p->H - 1; fy = 0; }
            int i2 = i + 1 < p->H ? i + 1 : p->H - 1;
            for (int ox = 0; ox < p->Wo; ++ox) {
                double sx = ((double)ox + 0.5) * p->W / p->Wo - 0.5;
                int j = (int)floor(sx);
                double fx = sx - j;
                if (j < 0) { j = 0; fx = 0; }
                if (j >= p->W - 1) { j = p->W - 1; fx = 0; }
                int j2 = j + 1 < p->W ? j + 1 : p->W - 1;
                double v = (1 - fy) * (1 - fx) * x[((size_t)q * p->H + i) * p->W + j]
                         + (1 - fy) * fx * x[((size_t)q * p->H + i) * p->W + j2]
                         + fy * (1 - fx) * x[((size_t)q * p->H + i2) * p->W + j]
                         + fy * fx * x[((size_t)q * p->H + i2) * p->W + j2];
                y[((size_t)q * p->Ho + oy) * p->Wo + ox] = v;
            }
        }
}

static void bwd_resize(const float *x, const float *dy, float *g, const void *pv)
{
    const rs_p *p = pv;
    (void)x;
    size_t n_out = (size_t)p->planes * p->Ho * p->Wo;
    size_t n_in = (size_t)p->planes * p->H * p->W;
    vc4_buf bdy = mk(dy, n_out);
    vc4_buf bdx = mk(NULL, n_in);
    vc4_status st = p->bilinear
        ? vc4_resize_bilinear_bwd(G, &bdy, &bdx, p->planes, p->H, p->W, p->Ho,
                                  p->Wo)
        : vc4_resize_nearest_bwd(G, &bdy, &bdx, p->planes, p->H, p->W, p->Ho,
                                 p->Wo);
    if (st != VC4_OK) {
        printf("FAIL resize_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, n_in * 4);
        return;
    }
    get(&bdx, g, n_in);
}

/* ---- pad ---- */
typedef struct { int N, C, H, W, pt, pl; } pad_p;
static void ref_pad(const float *x, double *y, const void *pv)
{
    const pad_p *p = pv;
    const int Hp = p->H + 2 * p->pt, Wp = p->W + 2 * p->pl;
    /* the border is zero padding: write all of y, not just the inner region.
     * Leaving it untouched made the finite-difference loss depend on whatever
     * the malloc block held (uninitialised doubles, sometimes NaN), which is
     * how this case passed or failed depending on the process's heap layout
     * (found while adding the G4 uniform mask: a 19-byte longer UPLOAD was
     * enough to flip it). */
    memset(y, 0, (size_t)p->N * p->C * Hp * Wp * sizeof(double));
    for (int n = 0; n < p->N; ++n)
        for (int c = 0; c < p->C; ++c)
            for (int yy = 0; yy < p->H; ++yy)
                for (int xx = 0; xx < p->W; ++xx)
                    y[(((size_t)n * p->C + c) * Hp + yy + p->pt) * Wp +
                      xx + p->pl] =
                        x[((size_t)n * p->C + c) * p->H * p->W +
                          (size_t)yy * p->W + xx];
}

static void bwd_pad(const float *x, const float *dy, float *g, const void *pv)
{
    const pad_p *p = pv;
    (void)x;
    vc4_buf bdy = mk(dy, (size_t)p->N * p->C * (p->H + 2 * p->pt) *
                          (p->W + 2 * p->pl));
    vc4_buf bdx = mk(NULL, (size_t)p->N * p->C * p->H * p->W);
    vc4_status st = vc4_pad_bwd(G, &bdy, &bdx, p->N, p->C, p->H, p->W, p->pt,
                                p->pl);
    if (st != VC4_OK) {
        printf("FAIL pad_bwd: %s\n", vc4_strerror(st));
        memset(g, 0, (size_t)p->N * p->C * p->H * p->W * 4);
        return;
    }
    get(&bdx, g, (size_t)p->N * p->C * p->H * p->W);
}

/* ---- transpose ---- */
typedef struct { int rows, cols; } tp_p;
static void ref_tp(const float *x, double *y, const void *pv)
{
    const tp_p *p = pv;
    for (int i = 0; i < p->rows; ++i)
        for (int j = 0; j < p->cols; ++j)
            y[(size_t)j * p->rows + i] = x[(size_t)i * p->cols + j];
}

static void bwd_tp(const float *x, const float *dy, float *g, const void *pv)
{
    const tp_p *p = pv;
    (void)x;
    vc4_buf bdy = mk(dy, (size_t)p->rows * p->cols);
    vc4_buf bdx = mk(NULL, (size_t)p->rows * p->cols);
    vc4_status st = vc4_transpose_bwd(G, &bdy, &bdx, p->rows, p->cols);
    if (st != VC4_OK) {
        printf("FAIL transpose_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, (size_t)p->rows * p->cols * 4);
        return;
    }
    get(&bdx, g, (size_t)p->rows * p->cols);
}

/* ---- gather / scatter ---- */
typedef struct { size_t n_out, n_in; int32_t idx[64]; } gath_p;
static void ref_gather(const float *x, double *y, const void *pv)
{
    const gath_p *p = pv;
    for (size_t o = 0; o < p->n_out; ++o)
        y[o] = x[p->idx[o]];
}

static void bwd_gather(const float *x, const float *dy, float *g, const void *pv)
{
    const gath_p *p = pv;
    (void)x;
    vc4_buf bg = mk(dy, p->n_out);
    vc4_buf bdx = mk(NULL, p->n_in);
    vc4_status st = vc4_gather_bwd(G, &bg, &bdx, p->idx, p->n_out, p->n_in);
    if (st != VC4_OK) {
        printf("FAIL gather_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, p->n_in * 4);
        return;
    }
    get(&bdx, g, p->n_in);
}

static void ref_scatter(const float *x, double *y, const void *pv)
{
    const gath_p *p = pv;
    memset(y, 0, p->n_in * sizeof(double));
    for (size_t i = 0; i < p->n_out; ++i)
        y[p->idx[i]] = x[i];
}

static void bwd_scatter(const float *x, const float *dy, float *g, const void *pv)
{
    const gath_p *p = pv;
    (void)x;
    vc4_buf bg = mk(dy, p->n_in);
    vc4_buf bdx = mk(NULL, p->n_out);
    vc4_status st = vc4_scatter_bwd(G, &bg, &bdx, p->idx, p->n_out, p->n_in);
    if (st != VC4_OK) {
        printf("FAIL scatter_bwd: %s (%s)\n", vc4_strerror(st),
               vc4_last_error(G));
        memset(g, 0, p->n_out * 4);
        return;
    }
    get(&bdx, g, p->n_out);
}

/* ==================================================================
 * drivers
 * ================================================================== */
static void run_affine(const char *name, ref_fwd fwd,
                       void (*bwd)(const float *, const float *, float *,
                                   const void *),
                       const void *p, size_t n_in, size_t n_out, float h,
                       int positive)
{
    float *x = malloc((n_in + n_out) * 4);
    float *dy = malloc((n_out + 1) * 4);
    for (size_t i = 0; i < n_in; ++i)
        x[i] = positive ? 0.2f + fabsf(frand()) * 1.5f : frand() * 2.0f;
    for (size_t i = 0; i < n_out; ++i)
        dy[i] = frand();
    affine_case c = { name, fwd, NULL, p, n_in, n_out, h };
    t_affine(&c, n_in, n_out, x, dy, bwd);
    free(x);
    free(dy);
}

static void run_affine_k(const char *name, ref_fwd fwd,
                         void (*bwd)(const float *, const float *, float *,
                                     const void *),
                         const void *p, size_t n_in, size_t n_out, float h,
                         int positive, keep_fn kf)
{
    g_keep = kf;
    g_keep_p = p;
    g_skipped = 0;
    run_affine(name, fwd, bwd, p, n_in, n_out, h, positive);
    g_keep = NULL;
}

/* ==================================================================
 * main
 * ================================================================== */
int main(int argc, char **argv)
{
    int reps = 1;
    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-')
            reps = atoi(argv[i]);
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        gpu_unlock(lock);
        return 2;
    }
    G = ctx;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("libvc4infer %s gradient checks (central differences, fp32-appropriate h, tol=1e-3)\n",
           VC4INFER_VERSION_STRING);
    for (int r = 0; r < reps; ++r) {
        t_linear(ctx, 16, 16, 16);
        t_linear(ctx, 32, 16, 32);
        t_relu(ctx, 32);
        t_softmax_ce(ctx, 8, 2e-2);
        t_softmax_ce(ctx, 64, 2e-2);
        t_softmax_generic(7, 13, 2e-2);
        t_softmax_generic(16, 32, 2e-2);
        t_ew_family();
        t_batchnorm(2, 4, 16);
        t_batchnorm(1, 3, 32);
        t_layernorm(5, 17);
        t_layernorm(8, 64);

        /* pooling */
        {
            pool_p p = { 2, 7, 6, 3, 2, 1, 0 };
            run_affine("avgpool dx 3x3s2p1", ref_avgpool, bwd_avgpool, &p,
                       (size_t)p.C * p.H * p.W,
                       (size_t)p.C * ((p.H + 2 - 3) / 2 + 1) *
                       ((p.W + 2 - 3) / 2 + 1), 2e-2, 0);
            pool_p q = { 1, 4, 4, 2, 2, 0, 1 };
            run_affine("avgpool dx 2x2s2 cip", ref_avgpool, bwd_avgpool, &q,
                       (size_t)q.C * q.H * q.W, (size_t)q.C * 2 * 2, 2e-2, 0);
            adap_p a = { 1, 5, 7, 3, 4 };
            run_affine("adaptive_avgpool dx 3x4", ref_adap, bwd_adap, &a,
                       (size_t)a.C * a.H * a.W, (size_t)a.C * a.oH * a.oW,
                       2e-2, 0);
            gap_p gp = { 2, 3, 4 };
            run_affine("global_avgpool dx", ref_gap, bwd_gap, &gp,
                       (size_t)gp.C * gp.H * gp.W, gp.C, 2e-2, 0);
            mp_p m1 = { 1, 6, 8, 2, 2, 0 };
            run_affine_k("maxpool 2x2s2 dx", ref_maxpool, bwd_maxpool, &m1,
                         (size_t)m1.C * m1.H * m1.W,
                         (size_t)m1.C * (m1.H / 2) * (m1.W / 2), 2e-2, 0,
                         keep_maxpool);
        }

        /* cat / split */
        {
            /* cat backward is a permutation of dy: check it directly
             * (a finite difference would only re-derive the same map) */
            cat_p cp = { 2, 3, 3, { 2, 4, 1, 0 } };
            size_t tot = 0;
            for (int i = 0; i < cp.nsrc; ++i) tot += cp.axisdim[i];
            const size_t nt = (size_t)cp.outer * tot * cp.inner;
            float *dyv = malloc(nt * 4), *got = malloc(nt * 4);
            for (size_t i = 0; i < nt; ++i) dyv[i] = frand();
            vc4_buf bdy = mk(dyv, nt);
            vc4_buf pr[4];
            vc4_buf *pp[4];
            for (int i = 0; i < cp.nsrc; ++i) {
                pr[i] = mk(NULL, (size_t)cp.outer * cp.axisdim[i] * cp.inner);
                pp[i] = &pr[i];
            }
            vc4_status st2 = vc4_cat_bwd(ctx, &bdy, pp, cp.axisdim, cp.nsrc,
                                         cp.outer, cp.inner);
            if (st2 != VC4_OK) {
                printf("FAIL cat_bwd: %s\n", vc4_strerror(st2));
                nfail++;
            } else {
                size_t off = 0, bad = 0;
                double worst = 0;
                for (int i = 0; i < cp.nsrc; ++i) {
                    size_t np = (size_t)cp.outer * cp.axisdim[i] * cp.inner;
                    get(&pr[i], got, np);
                    for (int o = 0; o < cp.outer; ++o)
                        for (int j = 0; j < cp.axisdim[i]; ++j)
                            for (int k = 0; k < cp.inner; ++k) {
                                size_t d = ((size_t)o * cp.axisdim[i] + j) *
                                           cp.inner + k;
                                double want = dyv[((size_t)o * tot + off + j) *
                                                  cp.inner + k];
                                if (fabs(got[d] - want) > worst)
                                    worst = fabs(got[d] - want);
                                if (got[d] != want) bad++;
                            }
                    off += cp.axisdim[i];
                }
                printf("%s %-26s n=%zu exact=%s worst=%.3g\n",
                       bad == 0 ? "PASS" : "FAIL", "cat_bwd 3 parts", nt,
                       bad == 0 ? "yes" : "no", worst);
                if (bad) nfail++;
            }
            free(dyv);
            free(got);
        }

        /* broadcasting binary ops */
        {
            bin_p b1 = { VC4_BIN_MUL, 2, { 4, 5 }, { 4, 5 }, { 4, 1 } };
            run_affine("mul_bwd bcast (4,5)x(4,1)", ref_bin, bwd_bin, &b1,
                       (size_t)(4 * 5 + 4), (size_t)4 * 5, 2e-2, 0);
            bin_p b2 = { VC4_BIN_ADD, 3, { 2, 3, 4 }, { 1, 3, 4 }, { 2, 3, 1 } };
            run_affine("add_bwd bcast (1,3,4)+(2,3,1)", ref_bin, bwd_bin, &b2,
                       (size_t)(3 * 4 + 2 * 3), (size_t)2 * 3 * 4, 2e-2, 0);
            bin_p b3 = { VC4_BIN_SUB, 2, { 3, 5 }, { 3, 5 }, { 3, 5 } };
            run_affine("sub_bwd (3,5)", ref_bin, bwd_bin, &b3,
                       (size_t)(2 * 3 * 5), (size_t)3 * 5, 2e-2, 0);
            /* div is ill-conditioned for a finite difference (the reference
             * gradient scales as 1/y^2, so fp32 forward noise is amplified);
             * check it against the closed form instead. */
            {
                const int R = 3, C = 4;
                float xv[R * C], yv[C], gg[R * C];
                for (int i = 0; i < R * C; ++i) xv[i] = frand();
                for (int j = 0; j < C; ++j) yv[j] = 0.4f + 0.9f * fabsf(frand());
                for (int i = 0; i < R * C; ++i) gg[i] = frand();
                vc4_buf bx2 = mk(xv, R * C), by2 = mk(yv, C), bg2 = mk(gg, R * C);
                vc4_buf bdx2 = mk(NULL, R * C), bdy2 = mk(NULL, C);
                int od2[2] = { R, C }, xd2[2] = { R, C }, yd2[2] = { 1, C };
                vc4_status st2 = vc4_binop_bwd(ctx, VC4_BIN_DIV, &bx2, &by2,
                                               &bg2, od2, xd2, yd2, 2, &bdx2,
                                               &bdy2);
                if (st2 != VC4_OK) {
                    printf("FAIL div_bwd: %s\n", vc4_strerror(st2));
                    nfail++;
                } else {
                    float *dxv = malloc(R * C * 4), *dyv = malloc(C * 4);
                    get(&bdx2, dxv, R * C);
                    get(&bdy2, dyv, C);
                    gc_t gx = {0}, gy = {0};
                    for (int r = 0; r < R; ++r)
                        for (int j = 0; j < C; ++j)
                            gc_add(&gx, dxv[r * C + j],
                                   (double)gg[r * C + j] / yv[j], 1e-3);
                    for (int j = 0; j < C; ++j) {
                        double s2 = 0;
                        for (int r = 0; r < R; ++r)
                            s2 -= (double)gg[r * C + j] * xv[r * C + j] /
                                  ((double)yv[j] * yv[j]);
                        gc_add(&gy, dyv[j], s2, 1e-3);
                    }
                    nfail |= gc_print("div_bwd dx bcast (3,4)/(1,4)", &gx,
                                      1e-3);
                    nfail |= gc_print("div_bwd dy bcast", &gy, 1e-3);
                    free(dxv);
                    free(dyv);
                }
            }
        }

        /* reduce: a broadcast, so check it exactly (no finite difference:
         * the driver's buffers are sized for the *input*, and this op writes
         * more outputs than it has inputs) */
        {
            struct { int ndim, dims[3], axis, mean; } r[3] = {
                { 3, { 2, 3, 5 }, 1, 0 },
                { 2, { 4, 6 }, 1, 1 },
                { 2, { 3, 8 }, 0, 0 },
            };
            const char *nm[3] = { "sum_bwd axis1 (2,3,5)",
                                  "mean_bwd axis1 (4,6)",
                                  "sum_bwd axis0 (3,8)" };
            for (int k = 0; k < 3; ++k) {
                int64_t n = 1, ng = 1;
                for (int i = 0; i < r[k].ndim; ++i) n *= r[k].dims[i];
                ng = n / r[k].dims[r[k].axis];
                float *gg = malloc(ng * 4);
                for (int64_t i = 0; i < ng; ++i) gg[i] = frand();
                vc4_buf bg = mk(gg, ng), bdx = mk(NULL, n);
                vc4_status st2 = vc4_reduce_bwd(ctx, &bg, &bdx, r[k].dims,
                                                r[k].ndim, r[k].axis,
                                                r[k].mean);
                if (st2 != VC4_OK) {
                    printf("FAIL %s: %s\n", nm[k], vc4_strerror(st2));
                    nfail++;
                    free(gg);
                    continue;
                }
                float *got = malloc(n * 4);
                get(&bdx, got, n);
                int64_t outer = 1, inner = 1;
                for (int i = 0; i < r[k].axis; ++i) outer *= r[k].dims[i];
                for (int i = r[k].axis + 1; i < r[k].ndim; ++i)
                    inner *= r[k].dims[i];
                const int64_t red = r[k].dims[r[k].axis];
                gc_t gc = {0};
                for (int64_t o = 0; o < outer; ++o)
                    for (int64_t j = 0; j < red; ++j)
                        for (int64_t ii = 0; ii < inner; ++ii) {
                            size_t d = (size_t)((o * red + j) * inner + ii);
                            double want = (double)gg[o * inner + ii];
                            if (r[k].mean) want /= (double)red;
                            gc_add(&gc, got[d], want, 1e-3);
                        }
                nfail |= gc_print(nm[k], &gc, 1e-3);
                free(got);
                free(gg);
            }
        }

        /* resize */
        {
            rs_p n1 = { 1, 3, 4, 6, 8, 0 };
            run_affine("resize nearest bwd 2x", ref_resize, bwd_resize, &n1,
                       (size_t)n1.planes * n1.H * n1.W,
                       (size_t)n1.planes * n1.Ho * n1.Wo, 2e-2, 0);
            rs_p n2 = { 1, 4, 4, 2, 2, 0 };
            run_affine("resize nearest bwd /2", ref_resize, bwd_resize, &n2,
                       (size_t)n2.planes * n2.H * n2.W,
                       (size_t)n2.planes * n2.Ho * n2.Wo, 2e-2, 0);
            rs_p b1 = { 1, 3, 4, 6, 8, 1 };
            run_affine("resize bilinear bwd 2x", ref_resize, bwd_resize, &b1,
                       (size_t)b1.planes * b1.H * b1.W,
                       (size_t)b1.planes * b1.Ho * b1.Wo, 2e-2, 0);
            rs_p b2 = { 1, 5, 3, 2, 7, 1 };
            run_affine("resize bilinear bwd 2x7", ref_resize, bwd_resize, &b2,
                       (size_t)b2.planes * b2.H * b2.W,
                       (size_t)b2.planes * b2.Ho * b2.Wo, 2e-2, 0);
        }

        /* pad / transpose */
        {
            pad_p pp = { 1, 2, 3, 4, 1, 2 };
            run_affine("pad_bwd crop 3x4", ref_pad, bwd_pad, &pp,
                       (size_t)pp.N * pp.C * pp.H * pp.W,
                       (size_t)pp.N * pp.C * (pp.H + 2 * pp.pt) *
                       (pp.W + 2 * pp.pl), 2e-2, 0);
            tp_p t1 = { 4, 6 };
            run_affine("transpose_bwd 4x6", ref_tp, bwd_tp, &t1,
                       (size_t)t1.rows * t1.cols, (size_t)t1.rows * t1.cols,
                       2e-2, 0);
            tp_p t2 = { 16, 32 };
            run_affine("transpose_bwd 16x32", ref_tp, bwd_tp, &t2,
                       (size_t)t2.rows * t2.cols, (size_t)t2.rows * t2.cols,
                       2e-2, 0);
        }

        /* gather / scatter */
        {
            gath_p gp;
            gp.n_in = 8;
            gp.n_out = 10;
            int32_t v[10] = { 0, 3, 7, 2, 5, 5, 1, 6, 4, 7 };
            memcpy(gp.idx, v, sizeof(v));
            run_affine("gather_bwd (with dups)", ref_gather, bwd_gather, &gp,
                       gp.n_in, gp.n_out, 2e-2, 0);
            gath_p sp;
            sp.n_in = 8;
            sp.n_out = 6;
            /* distinct targets: a scatter that overwrites a cell twice has no
             * derivative with respect to the overwritten value */
            int32_t w[6] = { 5, 0, 7, 3, 2, 1 };
            memcpy(sp.idx, w, sizeof(w));
            run_affine("scatter_bwd", ref_scatter, bwd_scatter, &sp, sp.n_out,
                       sp.n_in, 2e-2, 0);
        }
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(nfail ? "GRADCHECK FAILED (%d)\n" : "GRADCHECK PASSED\n", nfail);
    return nfail ? 1 : 0;
}
