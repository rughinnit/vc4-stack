/* test_convnet.c - a small convolutional net trained end to end on the QPUs.
 *
 *   x (18x18, 1 channel) -> conv 3x3/1 (8 filters) -> relu -> maxpool 2x2
 *   -> flatten (8x8x8 = 512) -> fc(10 classes)
 *
 * The engine wants Cin % 16 == 0 and Cout % 16 == 0, so the trainer pads the
 * input to 16 channels and the filters to 16 on the host (zero fill + copy --
 * no arithmetic).  Only the first 8 filters and the first 512 flatten
 * features are used by the classifier; the rest exist because the GEMM tiles
 * are 16 wide.
 *
 * Strict VC4 data plane: conv2d, relu, maxpool, im2col, gemm3, col2im, the FC
 * GEMM, fused softmax+CE and the optimizer all run as QPU kernels.  The ARM
 *   - generates the dataset and the initial weights (deterministic PRNG),
 *   - lays out the per-image padded input, gathers the pooled 8x8x8 features
 *     into the FC row, scatters the FC gradient back into the pooled pitch,
 *   - fills scalar hyperparameters and prints the loss/accuracy totals the
 *     GPU produced (display division only).
 *
 * Convolution is per image (the engine API is single-image NCHW); the
 * FC/softmax run over the whole batch.
 *
 * usage: test_convnet [--opt sgd|adam] [--steps N] [--seed S] [--batch B]
 */
#include "vc4infer.h"
#include "gpu_gate.h"
#include "util.h"
#include <math.h>

#define IMG      18
#define CINPAD   16
#define COUTPAD  16
#define COUTREAL 8
#define KH       3
#define HO       16
#define POOL     8
#define PPITCH   16
#define NF       (COUTREAL * POOL * POOL)      /* 512 */
#define KFC      (NF + 16)                     /* 528 (bias column) */
#define NCLS     10
#define CPAD     16
#define NTRAIN   640
#define NTEST    160
#define NIMG     (CINPAD * IMG * IMG)
#define NZ       (COUTPAD * HO * HO)
#define NP       (COUTPAD * POOL * PPITCH)
#define NW       (COUTPAD * CINPAD * KH * KH)

typedef struct { uint32_t xs; } rng_t;
static uint32_t rnd(rng_t *r)
{
    uint32_t x = r->xs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->xs = x; return x;
}
static float rf(rng_t *r)
{
    int32_t v = (int32_t)((rnd(r) >> 9) & 0x7FFFFFu) - (1 << 22);
    return (float)v / 4194304.0f;
}

static vc4_buf slice(vc4_buf b, size_t off, size_t bytes)
{
    vc4_buf s = { b.off + (uint32_t)off, bytes };
    return s;
}
static void fill_scalar(vc4_ctx *ctx, vc4_buf *b, size_t n, float v)
{
    float *p = malloc(n * 4);
    for (size_t i = 0; i < n; ++i) p[i] = v;
    vc4_upload(ctx, b, p, n * 4);
    free(p);
}

typedef struct {
    vc4_buf X, Z, A, P, dP, dZ, dY, dWi, dX, dWacc;
    vc4_buf h, hT, logits, dlogits, partial, ones, tot;
    vc4_buf dWfc, dh, s1, s2, s3;
    vc4_buf lrneg, s_b1, s_ib1, s_b2, s_ib2, s_eps;
    vc4_buf m1, v1, m2, v2;
    vc4_buf lab;
    float  *P_host;
    float  *dyP_host;
    float  *h_host;
    float  *dh_host;
} bt_t;

static vc4_status bt_alloc(vc4_ctx *ctx, bt_t *b, uint32_t nimg, uint32_t B)
{
    const uint32_t G = B / 8;
    struct { vc4_buf *p; size_t sz; } a[] = {
        { &b->X,      (size_t)nimg * NIMG * 4 },
        { &b->Z,      (size_t)B * NZ * 4 + 64 },
        { &b->A,      (size_t)B * NZ * 4 + 64 },
        { &b->P,      (size_t)B * NP * 4 },
        { &b->dP,     (size_t)B * NP * 4 },
        { &b->dZ,     (size_t)B * NZ * 4 },
        { &b->dY,     (size_t)B * NZ * 4 },
        { &b->dWi,    (size_t)NW * 4 },
        { &b->dX,     (size_t)B * NIMG * 4 },
        { &b->dWacc,  (size_t)NW * 4 },
        { &b->h,      (size_t)B * KFC * 4 },
        { &b->hT,     (size_t)KFC * B * 4 },
        { &b->logits, (size_t)B * CPAD * 4 + 64 },
        { &b->dlogits,(size_t)B * CPAD * 4 + 64 },
        { &b->partial,(size_t)G * 64 },
        { &b->ones,   (size_t)G * 4 + 64 },
        { &b->tot,    CPAD * 4 + 64 },
        { &b->dWfc,   (size_t)KFC * CPAD * 4 },
        { &b->s1,     (size_t)KFC * CPAD * 4 },
        { &b->s2,     (size_t)KFC * CPAD * 4 },
        { &b->s3,     (size_t)KFC * CPAD * 4 },
        { &b->dh,     (size_t)B * KFC * 4 },
        { &b->lrneg,  (size_t)KFC * CPAD * 4 },
        { &b->s_b1,   (size_t)KFC * CPAD * 4 },
        { &b->s_ib1,  (size_t)KFC * CPAD * 4 },
        { &b->s_b2,   (size_t)KFC * CPAD * 4 },
        { &b->s_ib2,  (size_t)KFC * CPAD * 4 },
        { &b->s_eps,  (size_t)KFC * CPAD * 4 },
        { &b->m1,     (size_t)KFC * CPAD * 4 },
        { &b->v1,     (size_t)KFC * CPAD * 4 },
        { &b->m2,     (size_t)NW * 4 },
        { &b->v2,     (size_t)NW * 4 },
        { &b->lab,    (size_t)nimg * 4 + 64 },
    };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); ++i) {
        vc4_status st = vc4_alloc(ctx, a[i].sz, a[i].p);
        if (st != VC4_OK) {
            printf("alloc %zu (%zu B) failed: %s\n", i, a[i].sz, vc4_strerror(st));
            return st;
        }
    }
    b->P_host = malloc((size_t)NP * 4);
    b->dyP_host = calloc((size_t)NP, 4);
    b->h_host = malloc((size_t)B * KFC * 4);
    b->dh_host = malloc((size_t)B * KFC * 4);
    return VC4_OK;
}

/* deterministic 10-class dataset: a small Gaussian blob at one of ten
 * positions (5 columns x 2 rows) plus noise. */
static void make_dataset(rng_t *r, float *X, int32_t *lab, int n)
{
    static const int cx[10] = { 3, 6, 9, 12, 15, 3, 6, 9, 12, 15 };
    static const int cy[10] = { 4, 4, 4, 4, 4, 12, 12, 12, 12, 12 };
    for (int i = 0; i < n; ++i) {
        int c = i % NCLS;
        lab[i] = c;
        float *im = X + (size_t)i * IMG * IMG;
        for (int y = 0; y < IMG; ++y)
            for (int x = 0; x < IMG; ++x) {
                float dx = (float)(x - cx[c]), dy = (float)(y - cy[c]);
                im[y * IMG + x] = expf(-(dx * dx + dy * dy) / 1.3f)
                                  + rf(r) * 0.10f;
            }
    }
}

static void init_weights(rng_t *r, float *W, float *Wfc)
{
    memset(W, 0, (size_t)NW * 4);
    for (int co = 0; co < COUTREAL; ++co)
        for (int t = 0; t < KH * KH; ++t)
            W[(co * CINPAD + 0) * KH * KH + t] = rf(r) * 0.35f;
    memset(Wfc, 0, (size_t)KFC * CPAD * 4);
    float s2 = 1.0f / sqrtf((float)NF);
    for (int k = 0; k < NF; ++k)
        for (int n = 0; n < NCLS; ++n)
            Wfc[(size_t)k * CPAD + n] = rf(r) * s2;
    for (int n = NCLS; n < CPAD; ++n)
        Wfc[(size_t)NF * CPAD + n] = -1e30f;   /* padding-column bias */
}

static uint64_t fnv(const void *p, size_t n)
{
    const uint8_t *b = p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    const char *opt = "sgd";
    int steps = 300;
    uint32_t seed = 12345;
    uint32_t B = 16;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--opt") && i + 1 < argc) opt = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = (uint32_t)atoi(argv[++i]);
    }
    int use_adam = strcmp(opt, "adam") == 0;
    if (!use_adam && strcmp(opt, "sgd") != 0) { printf("bad --opt\n"); return 2; }
    if (B % 8 || B == 0 || NTRAIN % B) { printf("bad --batch (multiple of 8, divides %d)\n", NTRAIN); return 2; }

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n"); gpu_unlock(lock); return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("libvc4infer %s conv-net trainer: conv3x3(1->8)-relu-pool-fc(512->10), "
           "opt=%s seed=%u batch=%u arena %.1f MiB\n", VC4INFER_VERSION_STRING,
           opt, seed, B, (double)vc4_arena_bytes(ctx) / (1 << 20));

    bt_t tr, te;
    if (bt_alloc(ctx, &tr, NTRAIN, B) != VC4_OK ||
        bt_alloc(ctx, &te, NTEST, B) != VC4_OK)
        return 2;

    rng_t r = { seed };
    float *Xtr = malloc((size_t)NTRAIN * IMG * IMG * 4);
    float *Xte = malloc((size_t)NTEST * IMG * IMG * 4);
    int32_t *Ltr = malloc(NTRAIN * 4), *Lte = malloc(NTEST * 4);
    make_dataset(&r, Xtr, Ltr, NTRAIN);
    make_dataset(&r, Xte, Lte, NTEST);

    float *xp = calloc((size_t)NTRAIN * NIMG, 4);
    for (int i = 0; i < NTRAIN; ++i)
        memcpy(xp + (size_t)i * NIMG, Xtr + (size_t)i * IMG * IMG, IMG * IMG * 4);
    vc4_upload(ctx, &tr.X, xp, (size_t)NTRAIN * NIMG * 4);
    free(xp);
    xp = calloc((size_t)NTEST * NIMG, 4);
    for (int i = 0; i < NTEST; ++i)
        memcpy(xp + (size_t)i * NIMG, Xte + (size_t)i * IMG * IMG, IMG * IMG * 4);
    vc4_upload(ctx, &te.X, xp, (size_t)NTEST * NIMG * 4);
    free(xp);
    vc4_upload(ctx, &tr.lab, Ltr, NTRAIN * 4);
    vc4_upload(ctx, &te.lab, Lte, NTEST * 4);

    float *w0 = malloc((size_t)NW * 4), *wfc0 = malloc((size_t)KFC * CPAD * 4);
    init_weights(&r, w0, wfc0);
    vc4_buf W, Wfc;
    vc4_alloc(ctx, (size_t)NW * 4, &W);
    vc4_alloc(ctx, (size_t)KFC * CPAD * 4, &Wfc);
    vc4_upload(ctx, &W, w0, (size_t)NW * 4);
    vc4_upload(ctx, &Wfc, wfc0, (size_t)KFC * CPAD * 4);
    free(w0); free(wfc0);

    const size_t NFW = (size_t)KFC * CPAD;
    float *ones = malloc(B * 4);
    for (uint32_t i = 0; i < B; ++i) ones[i] = 1.0f;
    vc4_upload(ctx, &tr.ones, ones, (B / 8) * 4);
    vc4_upload(ctx, &te.ones, ones, (B / 8) * 4);
    free(ones);

    const float lr = use_adam ? 0.01f : 0.2f;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    fill_scalar(ctx, &tr.s_b1, NFW, b1);
    fill_scalar(ctx, &tr.s_ib1, NFW, 1.0f - b1);
    fill_scalar(ctx, &tr.s_b2, NFW, b2);
    fill_scalar(ctx, &tr.s_ib2, NFW, 1.0f - b2);
    fill_scalar(ctx, &tr.s_eps, NFW, eps);
    fill_scalar(ctx, &tr.lrneg, NFW, -lr);
    {
        float *z = calloc(NFW, 4);
        vc4_upload(ctx, &tr.m1, z, NFW * 4);
        vc4_upload(ctx, &tr.v1, z, NFW * 4);
        free(z);
        z = calloc(NW, 4);
        vc4_upload(ctx, &tr.m2, z, (size_t)NW * 4);
        vc4_upload(ctx, &tr.v2, z, (size_t)NW * 4);
        free(z);
    }

    int rc = 0;
    vc4_status st = VC4_OK;
#define CK(x) do { st = (x); if (st != VC4_OK) { \
        printf("FAIL line %d: %s <- %s (%s)\n", __LINE__, vc4_strerror(st), \
               #x, vc4_last_error(ctx)); rc = 1; goto out; } } while (0)

    /* conv + pool + FC forward for images [base, base+n) */
#define FORWARD(T, base, n, Wv, Wfcv) do { \
        for (uint32_t _i = 0; _i < (n); ++_i) { \
            vc4_buf _xi = slice((T).X, (size_t)((base) + _i) * NIMG * 4, NIMG * 4); \
            vc4_buf _zi = slice((T).Z, (size_t)_i * NZ * 4, NZ * 4); \
            vc4_buf _ai = slice((T).A, (size_t)_i * NZ * 4, NZ * 4 + 64); \
            vc4_buf _pi = slice((T).P, (size_t)_i * NP * 4, NP * 4); \
            int _ho, _wo; \
            CK(vc4_conv2d(ctx, &_xi, CINPAD, IMG, IMG, &(Wv), COUTPAD, KH, KH, \
                          NULL, 1, 0, &_zi, &_ho, &_wo)); \
            CK(vc4_relu(ctx, &_zi, &_ai, NZ)); \
            CK(vc4_maxpool2x2(ctx, &_ai, COUTPAD, HO, HO, PPITCH, &_pi, \
                              &_ho, &_wo)); \
            vc4_download(ctx, &_pi, (T).P_host, NP * 4); \
            float *_h = (T).h_host + (size_t)_i * KFC; \
            memset(_h, 0, KFC * 4); \
            for (int _c = 0; _c < COUTREAL; ++_c) \
                for (int _y = 0; _y < POOL; ++_y) \
                    memcpy(_h + _c * POOL * POOL + _y * POOL, \
                           (T).P_host + (_c * POOL + _y) * PPITCH, POOL * 4); \
            _h[NF] = 1.0f; \
        } \
        CK(vc4_upload(ctx, &(T).h, (T).h_host, (size_t)(n) * KFC * 4)); \
        { \
            vc4_buf _labs = slice((T).lab, (size_t)(base) * 4, (size_t)(n) * 4 + 64); \
            CK(vc4_matmul(ctx, &(T).h, &(Wfcv), &(T).logits, (n), CPAD, KFC, \
                          VC4_LAYOUT_ROW_MAJOR)); \
            CK(vc4_softmax_ce(ctx, &(T).logits, &_labs, &(T).dlogits, \
                              &(T).partial, (n), 1.0f / (float)(n))); \
        } \
        CK(vc4_matmul(ctx, &(T).ones, &(T).partial, &(T).tot, 1, CPAD, \
                      (uint32_t)((n) / 8), VC4_LAYOUT_ROW_MAJOR)); \
    } while (0)

#define OPT_FC(G, n) do { \
        if (use_adam) { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &(G), &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.m1, &tr.s_b1, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.s_ib1, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &tr.m1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.v1, &tr.s_b2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s1, &tr.s_ib2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &tr.v1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.v1, &tr.s_eps, &tr.s2, (n))); \
            CK(vc4_rsqrt(ctx, &tr.s2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.m1, &tr.s2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s3, &tr.lrneg, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(Wfc), &tr.s2, &(Wfc), (n))); \
        } else { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.lrneg, &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(Wfc), &tr.s1, &(Wfc), (n))); \
        } \
    } while (0)

#define OPT_CONV(G, n) do { \
        if (use_adam) { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &(G), &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.m2, &tr.s_b1, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.s_ib1, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &tr.m2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.v2, &tr.s_b2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s1, &tr.s_ib2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &tr.v2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.v2, &tr.s_eps, &tr.s2, (n))); \
            CK(vc4_rsqrt(ctx, &tr.s2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.m2, &tr.s2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s3, &tr.lrneg, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(W), &tr.s2, &(W), (n))); \
        } else { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.lrneg, &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(W), &tr.s1, &(W), (n))); \
        } \
    } while (0)

    /* ------------------------------------------------ initial eval */
    float tot[16];
    FORWARD(tr, 0, B, W, Wfc);
    vc4_download(ctx, &tr.tot, tot, 64);
    double loss0 = tot[0] / B;
    printf("initial: train loss=%.6f acc=%.2f%% (batch of %u)\n",
           loss0, 100.0 * tot[1] / B, B);

    /* ------------------------------------------------ training */
    uint64_t t0 = now_ns();
    for (int step = 1; step <= steps; ++step) {
        if (use_adam) {
            double bc1 = 1.0 - pow((double)b1, step);
            double bc2 = 1.0 - pow((double)b2, step);
            float lrt = (float)(lr * sqrt(bc2) / bc1);
            fill_scalar(ctx, &tr.lrneg, NFW, -lrt);
        }
        uint32_t base = (uint32_t)((step - 1) * (int)B) % NTRAIN;
        FORWARD(tr, base, B, W, Wfc);

        /* FC backward */
        CK(vc4_transpose(ctx, &tr.h, &tr.hT, B, KFC));
        CK(vc4_matmul(ctx, &tr.hT, &tr.dlogits, &tr.dWfc, KFC, CPAD, B,
                      VC4_LAYOUT_ROW_MAJOR));
        CK(vc4_matmul_bt(ctx, &tr.dlogits, &Wfc, &tr.dh, B, KFC, CPAD));
        vc4_download(ctx, &tr.dh, tr.dh_host, (size_t)B * KFC * 4);

        /* conv backward per image; dW accumulated with a QPU add */
        {
            static float zero[NW];
            memset(zero, 0, sizeof(zero));
            vc4_upload(ctx, &tr.dWacc, zero, sizeof(zero));
        }
        for (uint32_t i = 0; i < B; ++i) {
            vc4_buf xi = slice(tr.X, (size_t)(base + i) * NIMG * 4, NIMG * 4);
            vc4_buf ai = slice(tr.A, (size_t)i * NZ * 4, NZ * 4 + 64);
            vc4_buf zi = slice(tr.Z, (size_t)i * NZ * 4, NZ * 4);
            vc4_buf dpi = slice(tr.dP, (size_t)i * NP * 4, NP * 4);
            vc4_buf dzi = slice(tr.dZ, (size_t)i * NZ * 4, NZ * 4);
            vc4_buf dyi = slice(tr.dY, (size_t)i * NZ * 4, NZ * 4);
            vc4_buf dxi = slice(tr.dX, (size_t)i * NIMG * 4, NIMG * 4);
            memset(tr.dyP_host, 0, (size_t)NP * 4);
            const float *hh = tr.dh_host + (size_t)i * KFC;
            for (int c = 0; c < COUTREAL; ++c)
                for (int y = 0; y < POOL; ++y)
                    memcpy(tr.dyP_host + (c * POOL + y) * PPITCH,
                           hh + c * POOL * POOL + y * POOL, POOL * 4);
            CK(vc4_upload(ctx, &dpi, tr.dyP_host, (size_t)NP * 4));
            CK(vc4_maxpool2x2_backward(ctx, &ai, &dpi, COUTPAD, HO, HO,
                                       PPITCH, &dzi));
            CK(vc4_relu_backward(ctx, &zi, &dzi, &dyi, NZ));
            CK(vc4_conv2d_backward(ctx, &xi, CINPAD, IMG, IMG, &W, COUTPAD,
                                   KH, KH, &dyi, 0, &tr.dWi, &dxi));
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.dWacc, &tr.dWi,
                               &tr.dWacc, NW));
        }
        OPT_FC(tr.dWfc, NFW);
        OPT_CONV(tr.dWacc, (size_t)NW);

        if (step == 1 || step % 25 == 0 || step == steps) {
            vc4_download(ctx, &tr.tot, tot, 64);
            printf("step %4d: loss=%.9f acc=%.2f%%\n", step, tot[0] / B,
                   100.0 * tot[1] / B);
        }
    }
    uint64_t t1 = now_ns();
    double secs = (double)(t1 - t0) / 1e9;

    /* ------------------------------------------------ final eval */
    double ltr = 0, atr = 0, lte = 0, ate = 0;
    {
        /* train: average over the whole set */
        double lsum = 0, corr = 0;
        for (uint32_t base = 0; base < NTRAIN; base += B) {
            FORWARD(tr, base, B, W, Wfc);
            vc4_download(ctx, &tr.tot, tot, 64);
            lsum += tot[0]; corr += tot[1];
        }
        ltr = lsum / NTRAIN; atr = corr / NTRAIN;
        lsum = corr = 0;
        for (uint32_t base = 0; base < NTEST; base += B) {
            FORWARD(te, base, B, W, Wfc);
            vc4_download(ctx, &te.tot, tot, 64);
            lsum += tot[0]; corr += tot[1];
        }
        lte = lsum / NTEST; ate = corr / NTEST;
    }

    float *wf = malloc((size_t)NW * 4), *wff = malloc((size_t)KFC * CPAD * 4);
    vc4_download(ctx, &W, wf, (size_t)NW * 4);
    vc4_download(ctx, &Wfc, wff, (size_t)KFC * CPAD * 4);
    printf("final: train loss=%.9f acc=%.2f%%  test loss=%.9f acc=%.2f%%\n",
           ltr, 100.0 * atr, lte, 100.0 * ate);
    printf("weights hash conv=%016llx fc=%016llx\n",
           (unsigned long long)fnv(wf, (size_t)NW * 4),
           (unsigned long long)fnv(wff, (size_t)KFC * CPAD * 4));
    printf("time %.3f s for %d steps, %.2f steps/s, %.0f samples/s (batch %u)\n",
           secs, steps, steps / secs, (double)B * steps / secs, B);
    free(wf); free(wff);
    if (!(ltr < loss0 * 0.2)) {
        printf("FAIL: loss did not drop 5x (%.6f -> %.6f)\n", loss0, ltr);
        rc = 1;
    }
    if (ate < 0.9) {
        printf("FAIL: test accuracy %.2f%% < 90%%\n", 100.0 * ate);
        rc = 1;
    }
out:
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "CONV TRAIN FAILED\n" : "CONV TRAIN PASSED\n");
    return rc;
}
