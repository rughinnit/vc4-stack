/* test_train.c - the toy trainer: a 2-layer MLP (64-32-10) on a fixed
 * synthetic 10-class dataset, full batch, SGD or Adam.
 *
 * Strict VC4 data plane: every multiply-accumulate, every activation, the
 * fused softmax+CE gradient and the whole optimizer step run as QPU kernels.
 * The ARM only
 *   - builds the dataset and the initial weights (deterministic PRNG),
 *   - lays out the bias-augmented matrices (row copies; no arithmetic),
 *   - fills hyperparameter scalars (lr, betas, eps) into buffer words,
 *   - reads back the two numbers a GPU GEMM produced (loss, correct count)
 *     for the log line (the accuracy % is a display division).
 *
 * Model: z1 = Xa W1a  (Xa = [x, 1, 0..], W1a = [W1; b1; 0..])
 *        h  = relu(z1);  Ha = [h, 1, 0..]
 *        z2 = Ha W2a  (W2a = [W2; b2; 0..]; the 6 padding output columns are
 *        biased to -1e30 so the 16-wide softmax kernel ignores them)
 * Backward: fused softmax+CE -> dW2 = Ha^T dlogits, dHa = dlogits W2a^T,
 *           dz1 = relu_bwd(z1, dHa), dW1 = Xa^T dz1.
 * The bias rows are the augmented rows of dW1/dW2, so the bias gradient
 * comes out of the same GEMM (no extra kernel).
 *
 * usage: test_train [--opt sgd|adam] [--steps N] [--seed S]
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"

#include <math.h>

#define DIN   64
#define HID   32
#define NCLS  10
#define CPAD  16
#define DKP   (DIN + 16)     /* 80 */
#define HKP   (HID + 16)     /* 48 */
#define NTRAIN 640
#define NTEST  160

typedef struct { uint32_t xs; } rng_t;

static uint32_t rnd(rng_t *r)
{
    uint32_t x = r->xs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->xs = x;
    return x;
}
static float rf(rng_t *r)      /* [-1,1) */
{
    int32_t v = (int32_t)((rnd(r) >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

static void fill_scalar(vc4_ctx *ctx, vc4_buf *b, size_t n, float v)
{
    float *p = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) p[i] = v;
    vc4_upload(ctx, b, p, n * 4);
    free(p);
}
static void zero_device(vc4_ctx *ctx, vc4_buf *b, size_t n)
{
    float *p = calloc(n, 4);
    vc4_upload(ctx, b, p, n * 4);
    free(p);
}

typedef struct {
    vc4_buf Xa, XaT, z1, h, Ha, HaT, z2, dlogits, partial;
    vc4_buf dW2a, dHa, dz1, dW1a;
    vc4_buf g2, t1, t2, t3, ones, tot;
    vc4_buf lrneg, s_b1, s_ib1, s_b2, s_ib2, s_eps;
    vc4_buf m1, v1, m2, v2;
    vc4_buf lab_buf;
    float  *X;                 /* host copy of the raw samples */
    int32_t *lab;
    float  *h_host;            /* staging for the h download */
    float  *ha_host;           /* staging for the Ha layout copy */
} tn_t;

static vc4_status t_alloc(vc4_ctx *ctx, tn_t *t, uint32_t B)
{
    const size_t kN1 = (size_t)DKP * HKP, kN2 = (size_t)HKP * CPAD;
    const size_t maxn = kN1;
    const uint32_t G = B / 8;
    struct { vc4_buf *b; size_t sz; } a[] = {
        { &t->Xa,      (size_t)B * DKP * 4 },
        { &t->XaT,     (size_t)DKP * B * 4 },
        { &t->z1,      (size_t)B * HKP * 4 },
        { &t->h,       (size_t)B * HKP * 4 },
        { &t->Ha,      (size_t)B * HKP * 4 },
        { &t->HaT,     (size_t)HKP * B * 4 },
        { &t->z2,      (size_t)B * CPAD * 4 + 64 },
        { &t->dlogits, (size_t)B * CPAD * 4 + 64 },
        { &t->partial, (size_t)G * 64 },
        { &t->dW2a,    kN2 * 4 },
        { &t->dHa,     (size_t)B * HKP * 4 },
        { &t->dz1,     (size_t)B * HKP * 4 },
        { &t->dW1a,    kN1 * 4 },
        { &t->g2,      maxn * 4 },
        { &t->t1,      maxn * 4 },
        { &t->t2,      maxn * 4 },
        { &t->t3,      maxn * 4 },
        { &t->ones,    (size_t)B * 4 },
        { &t->tot,     CPAD * 4 + 64 },
        { &t->lrneg,   maxn * 4 },
        { &t->s_b1,    maxn * 4 },
        { &t->s_ib1,   maxn * 4 },
        { &t->s_b2,    maxn * 4 },
        { &t->s_ib2,   maxn * 4 },
        { &t->s_eps,   maxn * 4 },
        { &t->m1,      kN1 * 4 },
        { &t->v1,      kN1 * 4 },
        { &t->m2,      kN2 * 4 },
        { &t->v2,      kN2 * 4 },
        { &t->lab_buf, (size_t)B * 4 + 64 },
    };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); ++i) {
        vc4_status st = vc4_alloc(ctx, a[i].sz, a[i].b);
        if (st != VC4_OK) {
            printf("alloc %zu (%zu B) failed: %s\n", i, a[i].sz,
                   vc4_strerror(st));
            return st;
        }
    }
    t->X = malloc((size_t)B * DIN * 4);
    t->lab = malloc((size_t)B * 4);
    t->h_host = malloc((size_t)B * HKP * 4);
    t->ha_host = malloc((size_t)B * HKP * 4);
    return VC4_OK;
}

static void make_dataset(rng_t *r, tn_t *t, uint32_t B, const float *centers)
{
    for (uint32_t i = 0; i < B; ++i) {
        uint32_t c = i % NCLS;
        t->lab[i] = (int32_t)c;
        for (uint32_t d = 0; d < DIN; ++d)
            t->X[(size_t)i * DIN + d] = centers[c * DIN + d] + rf(r) * 0.35f;
    }
}

static void init_weights(rng_t *r, float *w1, float *w2)
{
    memset(w1, 0, (size_t)DKP * HKP * 4);
    memset(w2, 0, (size_t)HKP * CPAD * 4);
    float s1 = 1.0f / sqrtf((float)DIN);
    float s2 = 1.0f / sqrtf((float)HID);
    for (uint32_t i = 0; i < DIN; ++i)
        for (uint32_t o = 0; o < HID; ++o)
            w1[(size_t)i * HKP + o] = rf(r) * s1;
    for (uint32_t i = 0; i < HID; ++i)
        for (uint32_t o = 0; o < NCLS; ++o)
            w2[(size_t)i * CPAD + o] = rf(r) * s2;
    for (uint32_t o = NCLS; o < CPAD; ++o)
        w2[(size_t)HID * CPAD + o] = -1e30f;   /* padding-column bias */
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    const char *opt = "sgd";
    int steps = 300;
    uint32_t seed = 12345;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--opt") && i + 1 < argc) opt = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 0);
    }
    int use_adam = strcmp(opt, "adam") == 0;
    if (!use_adam && strcmp(opt, "sgd") != 0) { printf("bad --opt\n"); return 2; }

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        gpu_unlock(lock);
        return 2;
    }
    printf("libvc4infer %s toy trainer: %d-%d-%d, opt=%s, seed=%u, arena %.1f MiB\n",
           VC4INFER_VERSION_STRING, DIN, HID, NCLS, opt, seed,
           (double)vc4_arena_bytes(ctx) / (1 << 20));

    const uint32_t B = NTRAIN;
    tn_t tr, te;
    if (t_alloc(ctx, &tr, B) != VC4_OK || t_alloc(ctx, &te, NTEST) != VC4_OK)
        return 2;

    rng_t r = { seed };
    float *centers = malloc(NCLS * DIN * 4);
    for (uint32_t k = 0; k < NCLS; ++k)
        for (uint32_t d = 0; d < DIN; ++d)
            centers[(size_t)k * DIN + d] = rf(&r) * 1.2f;
    make_dataset(&r, &tr, B, centers);
    make_dataset(&r, &te, NTEST, centers);

    /* bias-augmented inputs, uploaded once */
    float *xa = malloc((size_t)B * DKP * 4);
    memset(xa, 0, (size_t)B * DKP * 4);
    for (uint32_t i = 0; i < B; ++i) {
        memcpy(xa + (size_t)i * DKP, tr.X + (size_t)i * DIN, DIN * 4);
        xa[(size_t)i * DKP + DIN] = 1.0f;
    }
    vc4_upload(ctx, &tr.Xa, xa, (size_t)B * DKP * 4);
    free(xa);
    float *xat = malloc((size_t)NTEST * DKP * 4);
    memset(xat, 0, (size_t)NTEST * DKP * 4);
    for (uint32_t i = 0; i < NTEST; ++i) {
        memcpy(xat + (size_t)i * DKP, te.X + (size_t)i * DIN, DIN * 4);
        xat[(size_t)i * DKP + DIN] = 1.0f;
    }
    vc4_upload(ctx, &te.Xa, xat, (size_t)NTEST * DKP * 4);
    free(xat);
    vc4_upload(ctx, &tr.lab_buf, tr.lab, B * 4);
    vc4_upload(ctx, &te.lab_buf, te.lab, NTEST * 4);

    /* weights (shared by the train and eval forward passes) */
    float *w1 = malloc((size_t)DKP * HKP * 4), *w2 = malloc((size_t)HKP * CPAD * 4);
    init_weights(&r, w1, w2);
    vc4_buf W1, W2;
    vc4_alloc(ctx, (size_t)DKP * HKP * 4, &W1);
    vc4_alloc(ctx, (size_t)HKP * CPAD * 4, &W2);
    vc4_upload(ctx, &W1, w1, (size_t)DKP * HKP * 4);
    vc4_upload(ctx, &W2, w2, (size_t)HKP * CPAD * 4);
    free(w1); free(w2);

    /* Ha layout pattern: col HID = 1, zeros elsewhere; only cols 0..HID-1
       are overwritten by the relu output each step (byte copies). */
    memset(tr.ha_host, 0, (size_t)B * HKP * 4);
    for (uint32_t i = 0; i < B; ++i) tr.ha_host[(size_t)i * HKP + HID] = 1.0f;
    vc4_upload(ctx, &tr.Ha, tr.ha_host, (size_t)B * HKP * 4);
    memset(te.ha_host, 0, (size_t)NTEST * HKP * 4);
    for (uint32_t i = 0; i < NTEST; ++i) te.ha_host[(size_t)i * HKP + HID] = 1.0f;
    vc4_upload(ctx, &te.Ha, te.ha_host, (size_t)NTEST * HKP * 4);

    /* scalar (hyperparameter) buffers */
    float *ones = malloc((size_t)B * 4);
    for (uint32_t i = 0; i < B; ++i) ones[i] = 1.0f;
    vc4_upload(ctx, &tr.ones, ones, (size_t)B * 4);
    vc4_upload(ctx, &te.ones, ones, (size_t)NTEST * 4);
    free(ones);

    const float lr = use_adam ? 0.01f : 0.5f;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    fill_scalar(ctx, &tr.s_b1, (size_t)DKP * HKP, b1);
    fill_scalar(ctx, &tr.s_ib1, (size_t)DKP * HKP, 1.0f - b1);
    fill_scalar(ctx, &tr.s_b2, (size_t)DKP * HKP, b2);
    fill_scalar(ctx, &tr.s_ib2, (size_t)DKP * HKP, 1.0f - b2);
    fill_scalar(ctx, &tr.s_eps, (size_t)DKP * HKP, eps);
    fill_scalar(ctx, &tr.lrneg, (size_t)DKP * HKP, -lr);
    zero_device(ctx, &tr.m1, (size_t)DKP * HKP);
    zero_device(ctx, &tr.v1, (size_t)DKP * HKP);
    zero_device(ctx, &tr.m2, (size_t)HKP * CPAD);
    zero_device(ctx, &tr.v2, (size_t)HKP * CPAD);

    int rc = 0;
    vc4_status st = VC4_OK;

#define CK(x) do { st = (x); if (st != VC4_OK) { \
        printf("FAIL line %d: %s (%s)\n", __LINE__, vc4_strerror(st), \
               vc4_last_error(ctx)); rc = 1; goto out; } } while (0)

    /* forward on a batch of n rows, weights W1/W2, result totals in T.tot */
#define FORWARD(T, W1v, W2v, n) do { \
        CK(vc4_matmul(ctx, &(T).Xa, &(W1v), &(T).z1, (n), HKP, DKP, \
                      VC4_LAYOUT_ROW_MAJOR)); \
        CK(vc4_relu(ctx, &(T).z1, &(T).h, (size_t)(n) * HKP)); \
        vc4_download(ctx, &(T).h, (T).h_host, (size_t)(n) * HKP * 4); \
        for (uint32_t _i = 0; _i < (n); ++_i) \
            memcpy((T).ha_host + (size_t)_i * HKP, \
                   (T).h_host + (size_t)_i * HKP, HID * 4); \
        vc4_upload(ctx, &(T).Ha, (T).ha_host, (size_t)(n) * HKP * 4); \
        CK(vc4_matmul(ctx, &(T).Ha, &(W2v), &(T).z2, (n), CPAD, HKP, \
                      VC4_LAYOUT_ROW_MAJOR)); \
        CK(vc4_softmax_ce(ctx, &(T).z2, &(T).lab_buf, &(T).dlogits, \
                          &(T).partial, (n), 1.0f / (float)(n))); \
        CK(vc4_matmul(ctx, &(T).ones, &(T).partial, &(T).tot, 1, CPAD, \
                      (uint32_t)((n) / 8), VC4_LAYOUT_ROW_MAJOR)); \
    } while (0)

    /* one optimizer step of a tensor (n floats) from gradient G */
#define OPT_STEP(W, G, M, V, n) do { \
        if (use_adam) { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &(G), &tr.g2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(M), &tr.s_b1, &tr.t1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.s_ib1, &tr.t2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.t1, &tr.t2, &(M), (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(V), &tr.s_b2, &tr.t1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.g2, &tr.s_ib2, &tr.t2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.t1, &tr.t2, &(V), (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(V), &tr.s_eps, &tr.t1, (n))); \
            CK(vc4_rsqrt(ctx, &tr.t1, &tr.t1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(M), &tr.t1, &tr.t2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.t2, &tr.lrneg, &tr.t3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(W), &tr.t3, &(W), (n))); \
        } else { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.lrneg, &tr.t1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(W), &tr.t1, &(W), (n))); \
        } \
    } while (0)

    /* -------------------------------------------------- optimizer sanity */
    {
        const size_t n = 64;
        float *h0 = malloc(n * 4), *g0 = malloc(n * 4);
        for (size_t i = 0; i < n; ++i) { h0[i] = frand(); g0[i] = frand() * 0.1f; }
        vc4_buf bW, bG, bM, bV;
        vc4_alloc(ctx, n * 4, &bW); vc4_alloc(ctx, n * 4, &bG);
        vc4_alloc(ctx, n * 4, &bM); vc4_alloc(ctx, n * 4, &bV);
        vc4_upload(ctx, &bW, h0, n * 4);
        vc4_upload(ctx, &bG, g0, n * 4);
        zero_device(ctx, &bM, n);
        zero_device(ctx, &bV, n);
        OPT_STEP(bW, bG, bM, bV, n);
        float got[64], gm[64], gv[64];
        vc4_download(ctx, &bW, got, n * 4);
        vc4_download(ctx, &bM, gm, n * 4);
        vc4_download(ctx, &bV, gv, n * 4);
        double worst = 0;
        for (size_t i = 0; i < n; ++i) {
            double m = (1 - b1) * (double)g0[i];
            double v = (1 - b2) * (double)g0[i] * g0[i];
            /* this engine's Adam puts eps inside the sqrt (see docs);
               for the tiny gradients here the two placements differ by
               O(eps/v), which is exactly what this oracle mirrors. */
            double upd = use_adam ? lr * m / sqrt(v + eps) : lr * g0[i];
            double ref = (double)h0[i] - upd;
            double e = fabs(got[i] - ref) / (fabs(ref) > 1 ? fabs(ref) : 1);
            if (e > worst) worst = e;
            if (use_adam) {
                double em = fabs(gm[i] - m) / (fabs(m) > 1e-3 ? fabs(m) : 1e-3);
                if (em > worst) worst = em;
                double ev = fabs(gv[i] - v) / (fabs(v) > 1e-6 ? fabs(v) : 1e-6);
                if (ev > worst) worst = ev;
            }
        }
        printf("%s optimizer step (%s) vs CPU oracle: worst_rel=%.3g\n",
               worst < 2e-3 ? "PASS" : "FAIL", opt, worst);
        if (worst >= 2e-3) rc = 1;
        free(h0); free(g0);
    }

    /* ------------------------------------------------------- initial eval */
    float tot0[16], tot2[16];
    FORWARD(tr, W1, W2, B);
    vc4_download(ctx, &tr.tot, tot0, 64);
    double loss0 = tot0[0] / B;
    FORWARD(te, W1, W2, NTEST);
    vc4_download(ctx, &te.tot, tot2, 64);
    printf("initial: train loss=%.6f acc=%.2f%%  test loss=%.6f acc=%.2f%%\n",
           loss0, 100.0 * tot0[1] / B, tot2[0] / NTEST, 100.0 * tot2[1] / NTEST);

    /* ----------------------------------------------------------- training */
    uint64_t t0 = now_ns();
    for (int step = 1; step <= steps; ++step) {
        if (use_adam) {
            double bc1 = 1.0 - pow((double)b1, step);
            double bc2 = 1.0 - pow((double)b2, step);
            float lrt = (float)(lr * sqrt(bc2) / bc1);
            fill_scalar(ctx, &tr.lrneg, (size_t)DKP * HKP, -lrt);
        }
        FORWARD(tr, W1, W2, B);
        CK(vc4_transpose(ctx, &tr.Ha, &tr.HaT, B, HKP));
        CK(vc4_matmul(ctx, &tr.HaT, &tr.dlogits, &tr.dW2a, HKP, CPAD, B,
                      VC4_LAYOUT_ROW_MAJOR));
        CK(vc4_matmul_bt(ctx, &tr.dlogits, &W2, &tr.dHa, B, HKP, CPAD));
        CK(vc4_relu_backward(ctx, &tr.z1, &tr.dHa, &tr.dz1, (size_t)B * HKP));
        CK(vc4_transpose(ctx, &tr.Xa, &tr.XaT, B, DKP));
        CK(vc4_matmul(ctx, &tr.XaT, &tr.dz1, &tr.dW1a, DKP, HKP, B,
                      VC4_LAYOUT_ROW_MAJOR));
        OPT_STEP(W1, tr.dW1a, tr.m1, tr.v1, (size_t)DKP * HKP);
        OPT_STEP(W2, tr.dW2a, tr.m2, tr.v2, (size_t)HKP * CPAD);

        if (step == 1 || step % 25 == 0 || step == steps) {
            vc4_download(ctx, &tr.tot, tot0, 64);
            printf("step %4d: loss=%.9f acc=%.2f%%\n", step, tot0[0] / B,
                   100.0 * tot0[1] / B);
        }
    }
    uint64_t t1 = now_ns();
    double secs = (double)(t1 - t0) / 1e9;

    /* --------------------------------------------------------- final eval */
    FORWARD(tr, W1, W2, B);
    vc4_download(ctx, &tr.tot, tot0, 64);
    double ltr = tot0[0] / B;
    FORWARD(te, W1, W2, NTEST);
    vc4_download(ctx, &te.tot, tot2, 64);
    printf("final: train loss=%.9f acc=%.2f%%  test loss=%.9f acc=%.2f%%\n",
           ltr, 100.0 * tot0[1] / B, tot2[0] / NTEST, 100.0 * tot2[1] / NTEST);
    printf("time %.3f s for %d steps, %.2f steps/s, %.0f samples/s\n",
           secs, steps, steps / secs, (double)B * steps / secs);
    if (!(ltr < loss0 * 0.2)) {
        printf("FAIL: loss did not drop 5x (%.6f -> %.6f)\n", loss0, ltr);
        rc = 1;
    }
    if (tot2[1] / NTEST < 0.9) {
        printf("FAIL: test accuracy %.2f%% < 90%%\n", 100.0 * tot2[1] / NTEST);
        rc = 1;
    }
out:
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "TRAIN FAILED\n" : "TRAIN PASSED\n");
    return rc;
}
