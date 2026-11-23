/* test_onnxops.c - the two ops the ONNX runner needed from the engine:
 *
 *   vc4_maxpooln   generic NxN / stride-S max pool (the border is -inf,
 *                  filled by the host as bytes; all arithmetic is on the QPU)
 *   vc4_softmax_fwd forward softmax over <=16 classes (the fused training
 *                  kernel's 'p' probe, padded/compacted on the host)
 *
 * Both are checked against double-precision CPU oracles.  The oracle lives
 * in the test only; libvc4infer has no CPU compute path.
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

static int pool_case(vc4_ctx *ctx, int C, int H, int W, int N, int S, int P)
{
    const int ho = (H + 2 * P - N) / S + 1;
    const int wo = (W + 2 * P - N) / S + 1;
    const int hop = (ho + 3) & ~3;
    const int op = (wo + 15) & ~15;
    const size_t n = (size_t)C * H * W;
    const size_t no = (size_t)C * hop * op;
    float *x = malloc(n * 4), *y = malloc(no * 4), *r = malloc((size_t)C * ho * wo * 4);
    if (!x || !y || !r) { free(x); free(y); free(r); return 1; }
    for (size_t i = 0; i < n; ++i) x[i] = frand() * 2.0f;

    vc4_buf bx, by;
    vc4_status st = vc4_alloc(ctx, n * 4 + 64, &bx);
    if (st == VC4_OK) st = vc4_alloc(ctx, no * 4 + 64, &by);
    if (st != VC4_OK) {
        printf("FAIL maxpooln alloc: %s\n", vc4_strerror(st));
        free(x); free(y); free(r); return 1;
    }
    vc4_upload(ctx, &bx, x, n * 4);
    int oho = 0, owo = 0;
    uint64_t t0 = now_ns();
    st = vc4_maxpooln(ctx, &bx, C, H, W, N, S, P, &by, &oho, &owo);
    uint64_t t1 = now_ns();
    if (st != VC4_OK) {
        printf("FAIL maxpooln N=%d S=%d P=%d: %s (%s)\n", N, S, P,
               vc4_strerror(st), vc4_last_error(ctx));
        free(x); free(y); free(r); return 1;
    }
    vc4_download(ctx, &by, y, no * 4);
    if (oho != ho || owo != wo) {
        printf("FAIL maxpooln shape %dx%d, expected %dx%d\n", oho, owo, ho, wo);
        free(x); free(y); free(r); return 1;
    }
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < ho; ++i)
            for (int j = 0; j < wo; ++j) {
                float m = -INFINITY;
                for (int ky = 0; ky < N; ++ky)
                    for (int kx = 0; kx < N; ++kx) {
                        int y_ = i * S + ky - P, x_ = j * S + kx - P;
                        float v = (y_ >= 0 && y_ < H && x_ >= 0 && x_ < W)
                                      ? x[((size_t)c * H + y_) * W + x_]
                                      : -INFINITY;
                        if (v > m) m = v;
                    }
                r[((size_t)c * ho + i) * wo + j] = m;
            }
    long bad = 0;
    double maxabs = 0;
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < ho; ++i)
            for (int j = 0; j < wo; ++j) {
                float g = y[((size_t)c * hop + i) * op + j];
                float e = r[((size_t)c * ho + i) * wo + j];
                double d = fabs((double)g - (double)e);
                if (d > maxabs) maxabs = d;
                if (d != 0) ++bad;
            }
    printf("%s maxpooln %dx%d N=%d S=%d pad=%d -> %dx%d: %ld/%zu bad, "
           "maxabs=%.3g, %.2f ms\n", bad ? "FAIL" : "PASS", H, W, N, S, P,
           ho, wo, bad, (size_t)C * ho * wo, maxabs, (t1 - t0) / 1e6);
    free(x); free(y); free(r);
    return bad ? 1 : 0;
}

static int softmax_case(vc4_ctx *ctx, uint32_t rows, uint32_t cols)
{
    const size_t n = (size_t)rows * cols;
    float *z = malloc(n * 4), *y = malloc(n * 4), *r = malloc(n * 4);
    if (!z || !y || !r) { free(z); free(y); free(r); return 1; }
    for (size_t i = 0; i < n; ++i) z[i] = frand() * 8.0f;

    vc4_buf bz, by;
    vc4_status st = vc4_alloc(ctx, n * 4 + 64, &bz);
    if (st == VC4_OK) st = vc4_alloc(ctx, n * 4 + 64, &by);
    if (st != VC4_OK) {
        printf("FAIL softmax_fwd alloc: %s\n", vc4_strerror(st));
        free(z); free(y); free(r); return 1;
    }
    vc4_upload(ctx, &bz, z, n * 4);
    uint64_t t0 = now_ns();
    st = vc4_softmax_fwd(ctx, &bz, rows, cols, &by);
    uint64_t t1 = now_ns();
    if (st != VC4_OK) {
        printf("FAIL softmax_fwd %ux%u: %s (%s)\n", rows, cols,
               vc4_strerror(st), vc4_last_error(ctx));
        free(z); free(y); free(r); return 1;
    }
    vc4_download(ctx, &by, y, n * 4);
    double maxabs = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        double mx = -1e300, s = 0;
        for (uint32_t j = 0; j < cols; ++j)
            if (z[(size_t)i * cols + j] > mx) mx = z[(size_t)i * cols + j];
        for (uint32_t j = 0; j < cols; ++j) {
            r[(size_t)i * cols + j] = exp(z[(size_t)i * cols + j] - mx);
            s += r[(size_t)i * cols + j];
        }
        for (uint32_t j = 0; j < cols; ++j) {
            double e = r[(size_t)i * cols + j] / s;
            double d = fabs((double)y[(size_t)i * cols + j] - e);
            if (d > maxabs) maxabs = d;
        }
    }
    int bad = maxabs > 2e-6;
    printf("%s softmax_fwd %ux%u: maxabs=%.3g (tol 2e-6), %.2f ms\n",
           bad ? "FAIL" : "PASS", rows, cols, maxabs, (t1 - t0) / 1e6);
    free(z); free(y); free(r);
    return bad;
}

int main(void)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s (%s)\n", vc4_strerror(st),
               ctx ? vc4_last_error(ctx) : "-");
        gpu_unlock(lock);
        return 1;
    }
    int rc = 0;
    rc |= pool_case(ctx, 4, 32, 32, 2, 2, 0);
    rc |= pool_case(ctx, 3, 16, 16, 5, 1, 2);
    rc |= pool_case(ctx, 2, 16, 16, 3, 2, 1);
    rc |= softmax_case(ctx, 16, 10);
    rc |= softmax_case(ctx, 8, 16);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf("%s test_onnxops\n", rc ? "FAIL" : "PASS");
    return rc ? 1 : 0;
}
