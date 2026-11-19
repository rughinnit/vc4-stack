/* test_train_ops.c - the new training primitives vs CPU oracles.
 *
 * The CPU reference here is a *checker* (compiled into the test only, never
 * linked into the library).  Every analytic quantity the test compares was
 * produced by a QPU kernel; the CPU only recomputes the expected value.
 *
 * usage: test_train_ops [reps]
 */
#include "vc4infer.h"
#include "util.h"

#include <math.h>

/* ---------------- transpose ---------------- */
static int t_transpose(vc4_ctx *ctx, uint32_t rows, uint32_t cols)
{
    size_t n = (size_t)rows * cols;
    float *a = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i)
        a[i] = frand() * 4.0f;
    vc4_buf ba, bo;
    if (vc4_alloc(ctx, n * 4, &ba) != VC4_OK ||
        vc4_alloc(ctx, n * 4, &bo) != VC4_OK)
        return 1;
    vc4_upload(ctx, &ba, a, n * 4);
    memset(o, 0, n * 4);
    vc4_upload(ctx, &bo, o, n * 4);
    vc4_status st = vc4_transpose(ctx, &ba, &bo, rows, cols);
    if (st != VC4_OK) {
        printf("FAIL transpose: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    vc4_download(ctx, &bo, o, n * 4);
    check_t c = {0};
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t col = 0; col < cols; ++col)
            check_elem(&c, o[(size_t)col * rows + r], a[(size_t)r * cols + col],
                       0.0, 1);
    printf("%s transpose %ux%u: max_abs=%.3g mismatches=%ld\n",
           c.mismatches == 0 ? "PASS" : "FAIL", rows, cols, c.max_abs,
           c.mismatches);
    free(a); free(o);
    return c.mismatches != 0;
}

/* ---------------- relu backward ---------------- */
static int t_relu_bwd(vc4_ctx *ctx, size_t n)
{
    float *z = malloc(n * 4), *dy = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) {
        z[i] = frand() * 4.0f;
        dy[i] = frand() * 2.0f;
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
    long mm = 0;
    for (size_t i = 0; i < n; ++i) {
        float ref = z[i] > 0 ? dy[i] : 0.0f;
        if (memcmp(&o[i], &ref, 4) != 0) {
            if (mm < 5)
                printf("  relu_bwd MISMATCH i=%zu z=%g dy=%g got=%g ref=%g\n",
                       i, z[i], dy[i], o[i], ref);
            mm++;
        }
    }
    printf("%s relu_bwd n=%zu: mismatches=%ld (bit-exact)\n",
           mm == 0 ? "PASS" : "FAIL", n, mm);
    free(z); free(dy); free(o);
    return mm != 0;
}

/* ---------------- rsqrt ---------------- */
static int t_rsqrt(vc4_ctx *ctx, size_t n)
{
    float *a = malloc(n * 4), *o = malloc(n * 4);
    for (size_t i = 0; i < n; ++i)
        a[i] = 1e-6f + (float)fabs(frand()) * 10.0f;
    vc4_buf ba, bo;
    vc4_alloc(ctx, n * 4, &ba); vc4_alloc(ctx, n * 4, &bo);
    vc4_upload(ctx, &ba, a, n * 4);
    vc4_status st = vc4_rsqrt(ctx, &ba, &bo, n);
    if (st != VC4_OK) {
        printf("FAIL rsqrt: %s\n", vc4_strerror(st));
        return 1;
    }
    vc4_download(ctx, &bo, o, n * 4);
    double worst = 0;
    long mm = 0;
    for (size_t i = 0; i < n; ++i) {
        double ref = 1.0 / sqrt((double)a[i]);
        double rel = fabs(o[i] - ref) / ref;
        if (rel > worst) worst = rel;
        if (rel > 1e-5) mm++;
    }
    printf("%s rsqrt n=%zu: worst_rel=%.3g mismatches=%ld (tol 1e-5)\n",
           mm == 0 ? "PASS" : "FAIL", n, worst, mm);
    free(a); free(o);
    return mm != 0;
}

/* ---------------- C = A B^T ---------------- */
static int t_matmul_bt(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K)
{
    size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    float *a = malloc(na * 4), *b = malloc(nb * 4), *c = malloc(nc * 4);
    for (size_t i = 0; i < na; ++i) a[i] = frand();
    for (size_t i = 0; i < nb; ++i) b[i] = frand();
    vc4_buf ba, bb, bc;
    vc4_alloc(ctx, na * 4, &ba); vc4_alloc(ctx, nb * 4, &bb);
    vc4_alloc(ctx, nc * 4, &bc);
    vc4_upload(ctx, &ba, a, na * 4);
    vc4_upload(ctx, &bb, b, nb * 4);
    vc4_status st = vc4_matmul_bt(ctx, &ba, &bb, &bc, M, N, K);
    if (st != VC4_OK) {
        printf("FAIL matmul_bt: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    vc4_download(ctx, &bc, c, nc * 4);
    check_t ck = {0};
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k)
                acc += (double)a[(size_t)m * K + k] * b[(size_t)n * K + k];
            check_elem(&ck, c[(size_t)m * N + n], acc, 1e-5, 1);
        }
    printf("%s matmul_bt %ux%ux%u: max_rel=%.3g mismatches=%ld\n",
           ck.mismatches == 0 ? "PASS" : "FAIL", M, N, K, ck.max_rel,
           ck.mismatches);
    free(a); free(b); free(c);
    return ck.mismatches != 0;
}

/* ---------------- softmax + CE backward ---------------- */
static int t_softmax(vc4_ctx *ctx, uint32_t rows, int pad)
{
    const int C = 16;
    const float NEG = -1e30f;
    size_t nl = (size_t)rows * C;
    float *z = malloc(nl * 4), *dl = malloc(nl * 4);
    int32_t *lab = malloc(rows * 4);
    int real = pad ? 10 : 16;
    for (uint32_t r = 0; r < rows; ++r) {
        lab[r] = (int32_t)(xs_next() % (uint32_t)real);
        for (int c = 0; c < C; ++c)
            z[(size_t)r * C + c] = c < real ? frand() * 3.0f : NEG;
    }
    const uint32_t groups = rows / 8;
    vc4_buf bz, bl, bd, bp;
    if (vc4_alloc(ctx, nl * 4 + 64, &bz) != VC4_OK ||
        vc4_alloc(ctx, rows * 4 + 64, &bl) != VC4_OK ||
        vc4_alloc(ctx, nl * 4 + 64, &bd) != VC4_OK ||
        vc4_alloc(ctx, (size_t)groups * 64, &bp) != VC4_OK)
        return 1;
    vc4_upload(ctx, &bz, z, nl * 4);
    vc4_upload(ctx, &bl, lab, rows * 4);
    float *zero = calloc((size_t)groups * 16, 4);
    vc4_upload(ctx, &bp, zero, (size_t)groups * 64);
    free(zero);
    float invb = 1.0f / (float)rows;
    vc4_status st = vc4_softmax_ce(ctx, &bz, &bl, &bd, &bp, rows, invb);
    if (st != VC4_OK) {
        printf("FAIL softmax_ce: %s (%s)\n", vc4_strerror(st), vc4_last_error(ctx));
        return 1;
    }
    vc4_download(ctx, &bd, dl, nl * 4);
    float *part = calloc((size_t)groups * 16, 4);
    vc4_download(ctx, &bp, part, (size_t)groups * 64);
    double loss = 0, corr = 0;
    for (uint32_t q = 0; q < groups; ++q) {
        loss += part[q * 16 + 0];
        corr += part[q * 16 + 1];
    }

    /* CPU oracle */
    double maxerr = 0, maxperr = 0, refloss = 0;
    long refcorr = 0, mm = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        double m = -1e30, s = 0, p[16];
        for (int c = 0; c < C; ++c) {
            double v = z[(size_t)r * C + c];
            if (v > m) m = v;
        }
        /* the kernel counts |z_label - max| <= 1e-5 (the vector-rotate
           primitive is 1-ulp lossy, so argmax is a tolerance test) */
        for (int c = 0; c < C; ++c) {
            p[c] = exp((double)z[(size_t)r * C + c] - m);
            s += p[c];
        }
        for (int c = 0; c < C; ++c) p[c] /= s;
        refloss += -log(p[lab[r]]);
        if ((double)z[(size_t)r * C + lab[r]] - m <= 1e-5 &&
            m - (double)z[(size_t)r * C + lab[r]] <= 1e-5) refcorr++;
        for (int c = 0; c < C; ++c) {
            double ref = (p[c] - (c == lab[r] ? 1.0 : 0.0)) * invb;
            double e = fabs(dl[(size_t)r * C + c] - ref);
            if (e > maxerr) maxerr = e;
            if (e > 2e-5) mm++;
        }
        double ps = 0;
        for (int c = 0; c < C; ++c) ps += dl[(size_t)r * C + c] + (c == lab[r] ? invb : 0.0);
        if (fabs(ps - invb) > 1e-5) maxperr = fmax(maxperr, fabs(ps - invb));
    }
    double lerr = fabs(loss - refloss) / (refloss > 1 ? refloss : 1);
    printf("%s softmax_ce rows=%u pad=%d: dlogits_max_abs=%.3g "
           "sum(p)_err=%.3g loss=%.6f ref=%.6f rel=%.3g correct=%.0f/%ld\n",
           (mm == 0 && lerr < 1e-5 && (long)corr == refcorr) ? "PASS" : "FAIL",
           rows, pad, maxerr, maxperr, loss, refloss, lerr, corr, refcorr);
    free(z); free(dl); free(lab); free(part);
    return (mm != 0 || lerr >= 1e-5 || (long)corr != refcorr);
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        gpu_unlock(lock);
        return 2;
    }
    printf("libvc4infer %s training primitives, arena %.1f MiB\n",
           VC4INFER_VERSION_STRING, (double)vc4_arena_bytes(ctx) / (1 << 20));
    int rc = 0;
    rc |= t_transpose(ctx, 64, 48);
    rc |= t_transpose(ctx, 32, 64);
    rc |= t_relu_bwd(ctx, 4096);
    rc |= t_rsqrt(ctx, 4096);
    rc |= t_matmul_bt(ctx, 32, 16, 64);
    rc |= t_matmul_bt(ctx, 16, 16, 16);
    rc |= t_softmax(ctx, 64, 0);
    rc |= t_softmax(ctx, 512, 1);
    vc4_print_supported_ops(stdout);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "TRAIN OPS FAILED\n" : "TRAIN OPS PASSED\n");
    return rc;
}
