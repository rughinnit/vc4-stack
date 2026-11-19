/* test_grad.c - finite-difference gradient checks for the training kernels.
 *
 * The CPU in this file is a *checker only*: every forward value used by a
 * finite difference comes from a QPU kernel, and every analytic gradient is
 * produced by a QPU kernel.  The CPU recomputes the scalar loss from the
 * downloaded output (and nothing else).
 *
 * The QPU FPU truncates (round toward zero) and the QPU has no FMA, so a
 * gradient check cannot be bit-exact.  With central differences, h = 1e-2 and
 * fp32 arithmetic, the expected agreement is ~1e-4; the acceptance bound is a
 * relative error of 1e-3 (documented in docs/NUMERICS.md).
 *
 * usage: test_grad
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
}

static void gc_print(const char *name, const gc_t *g, int n, double tol)
{
    double l2 = sqrt(g->err2) / (sqrt(g->ref2) > 0 ? sqrt(g->ref2) : 1);
    printf("%s %-22s n=%d rel_L2=%.3g max_rel=%.3g max_abs=%.3g over_tol=%ld"
           "  worst: g=%.6g fd=%.6g\n",
           (g->bad == 0 && l2 <= tol) ? "PASS" : "FAIL", name, n, l2, g->max_rel,
           g->max_abs, g->bad, g->worst_g, g->worst_fd);
}

/* ---------------- linear: C = A B, L = 0.5 sum C^2 ---------------- */
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
    /* dC = C (the loss is 0.5*||C||^2) */
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

    /* L is quadratic in A and B, so a central difference is exact up to
       fp32 forward noise; a large h suppresses that noise. */
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
    snprintf(nm, sizeof(nm), "dA %ux%ux%u", M, N, K);
    gc_print(nm, &ga, (int)(Mp * Kp), tol);
    snprintf(nm, sizeof(nm), "dB (A^T dC) %ux%ux%u", M, N, K);
    gc_print(nm, &gb, (int)(Kp * Np), tol);
    free(A); free(B); free(C); free(gA); free(gB);
    return ga.bad != 0 || gb.bad != 0;
}

/* ---------------- relu backward ---------------- */
static int t_relu(vc4_ctx *ctx, size_t n)
{
    float *z = malloc(n * 4), *dy = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        z[i] = frand() * 2.0f;
        dy[i] = frand();
    }
    vc4_buf bz, bd, bo;
    vc4_alloc(ctx, n * 4, &bz); vc4_alloc(ctx, n * 4, &bd);
    vc4_alloc(ctx, n * 4, &bo);
    vc4_upload(ctx, &bz, z, n * 4);
    vc4_upload(ctx, &bd, dy, n * 4);
    vc4_status st = vc4_relu_backward(ctx, &bz, &bd, &bo, n);
    if (st != VC4_OK) {
        printf("FAIL relu_bwd: %s\n", vc4_strerror(st));
        return 1;
    }
    vc4_download(ctx, &bo, o, n * 4);
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
    gc_print("relu_bwd", &g, (int)n, tol);
    free(z); free(dy); free(o);
    return g.bad != 0;
}

/* ---------------- softmax + CE ---------------- */
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

static int t_softmax(vc4_ctx *ctx, uint32_t rows, double h)
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
    const float invb = 1.0f;   /* dlogits = p-onehot: grad of the summed loss */
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
    gc_print("softmax_ce dlogits", &gc, (int)nl, tol);
    free(z); free(g); free(lab);
    return gc.bad != 0;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 1;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        gpu_unlock(lock);
        return 2;
    }
    printf("libvc4infer %s gradient checks (finite differences, h=1e-2, tol=1e-3)\n",
           VC4INFER_VERSION_STRING);
    int rc = 0;
    for (int r = 0; r < reps; ++r) {
        rc |= t_linear(ctx, 16, 16, 16);
        rc |= t_linear(ctx, 32, 16, 32);
        rc |= t_relu(ctx, 32);
        rc |= t_softmax(ctx, 8, 2e-2);
        rc |= t_softmax(ctx, 64, 2e-2);
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "GRADCHECK FAILED\n" : "GRADCHECK PASSED\n");
    return rc;
}
