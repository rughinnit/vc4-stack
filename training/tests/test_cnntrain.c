/* test_cnntrain.c - a small CNN trained end to end on the QPUs, using the
 * new training ops: batchnorm (forward + dx/dgamma/dbeta) and the generic
 * softmax + cross-entropy backward.
 *
 *   x (18x18, 1 channel, padded to 16)
 *     -> conv 3x3/1 (16 filters, 8 real)
 *     -> batchnorm (train mode, running stats updated on the QPU)
 *     -> relu -> maxpool 2x2 -> flatten (8x8x8 = 512)
 *     -> fc (16 classes) -> softmax cross-entropy
 *
 * Strict VC4 data plane: conv2d, batchnorm, relu, maxpool, im2col/gemm3/
 * col2im, the FC GEMM, softmax+CE and the optimizer all run as QPU kernels.
 * The ARM
 *   - generates the dataset and the initial weights (deterministic PRNG),
 *   - lays out the padded per-image input and gathers/scatters the pooled
 *     8x8x8 features (byte copies only),
 *   - fills scalar hyperparameters and prints the loss the GPU produced,
 *   - writes the trained weights to a file (a save, not a compute path).
 *
 * `--reps 2` trains twice from the same seed with freshly zeroed optimizer
 * state and compares the saved weight files byte for byte, which is the
 * determinism claim.
 *
 * usage: test_cnntrain [--opt sgd|adam] [--steps N] [--seed S] [--batch B]
 *                      [--reps R] [--save FILE]
 */
#include "vc4infer.h"
#include "vc4_int.h"
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
#define NCLS     3
#define CPAD     16
#define NTRAIN   64
#define NTEST    32
#define NIMG     (CINPAD * IMG * IMG)
#define NZ       (COUTPAD * HO * HO)
#define NP       (COUTPAD * POOL * PPITCH)
#define NW       (COUTPAD * CINPAD * KH * KH)

static char g_save_first[512];

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
static void zero_device(vc4_ctx *ctx, vc4_buf *b, size_t n)
{
    float *p = calloc(n, 4);
    vc4_upload(ctx, b, p, n * 4);
    free(p);
}

typedef struct {
    vc4_buf X, Z, BN, A, P, dP, dZ, dBN, dA, dY, dWi, dX, dWacc;
    vc4_buf h, hT, logits, dlogits, loss, dWfc, dh;
    vc4_buf dWfcT, hT2;
    vc4_buf gamma, beta, dgamma, dbeta, rmean, rvar, smean, sistd;
    vc4_buf dWfc2, dgamma2, dbeta2;
    vc4_buf s1, s2, s3, lrneg, s_b1, s_ib1, s_b2, s_ib2, s_eps;
    vc4_buf m1, v1, m2, v2, mg, vg, mb, vb;
    vc4_buf lab;
    float  *P_host, *dyP_host, *h_host, *dh_host;
} bt_t;

static vc4_status bt_alloc(vc4_ctx *ctx, bt_t *b, uint32_t nimg, uint32_t B)
{
    struct { vc4_buf *p; size_t sz; } a[] = {
        { &b->X,      (size_t)nimg * NIMG * 4 },
        { &b->Z,      (size_t)B * NZ * 4 + 64 },
        { &b->BN,     (size_t)B * NZ * 4 + 64 },
        { &b->A,      (size_t)B * NZ * 4 + 64 },
        { &b->P,      (size_t)B * NP * 4 },
        { &b->dP,     (size_t)B * NP * 4 },
        { &b->dZ,     (size_t)B * NZ * 4 + 64 },
        { &b->dBN,    (size_t)B * NZ * 4 + 64 },
        { &b->dA,     (size_t)B * NZ * 4 + 64 },
        { &b->dY,     (size_t)B * NZ * 4 + 64 },
        { &b->dWi,    (size_t)NW * 4 },
        { &b->dX,     (size_t)B * NIMG * 4 },
        { &b->dWacc,  (size_t)NW * 4 },
        { &b->h,      (size_t)B * KFC * 4 },
        { &b->hT,     (size_t)KFC * B * 4 },
        { &b->hT2,    (size_t)KFC * B * 4 },
        { &b->logits, (size_t)B * CPAD * 4 + 64 },
        { &b->dlogits,(size_t)B * CPAD * 4 + 64 },
        { &b->loss,   64 },
        { &b->dWfc,   (size_t)KFC * CPAD * 4 },
        { &b->dWfc2,  (size_t)KFC * CPAD * 4 },
        { &b->dh,     (size_t)B * KFC * 4 },
        { &b->gamma,  64 }, { &b->beta, 64 },
        { &b->dgamma, 64 }, { &b->dbeta, 64 },
        { &b->dgamma2, 64 }, { &b->dbeta2, 64 },
        { &b->rmean,  64 }, { &b->rvar, 64 },
        { &b->smean,  64 }, { &b->sistd, 64 },
        { &b->s1,     (size_t)KFC * CPAD * 4 },
        { &b->s2,     (size_t)KFC * CPAD * 4 },
        { &b->s3,     (size_t)KFC * CPAD * 4 },
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
        { &b->mg,     64 }, { &b->vg, 64 },
        { &b->mb,     64 }, { &b->vb, 64 },
        { &b->lab,    (size_t)nimg * 4 + 64 },
    };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); ++i) {
        vc4_status st = vc4_alloc(ctx, a[i].sz, a[i].p);
        if (st != VC4_OK) {
            printf("alloc %zu (%zu B) failed: %s\n", i, a[i].sz,
                   vc4_strerror(st));
            return st;
        }
    }
    b->P_host = malloc((size_t)NP * 4);
    b->dyP_host = calloc((size_t)NP, 4);
    b->h_host = malloc((size_t)B * KFC * 4);
    b->dh_host = malloc((size_t)B * KFC * 4);
    return VC4_OK;
}

/* deterministic NCLS-class dataset: a Gaussian blob at one of three
 * positions plus noise. */
static void make_dataset(rng_t *r, float *X, int32_t *lab, int n)
{
    static const int cx[NCLS] = { 4, 9, 13 };
    static const int cy[NCLS] = { 5, 12, 7 };
    for (int i = 0; i < n; ++i) {
        int c = i % NCLS;
        lab[i] = c;
        float *im = X + (size_t)i * IMG * IMG;
        for (int y = 0; y < IMG; ++y)
            for (int x = 0; x < IMG; ++x) {
                float dx = (float)(x - cx[c]), dy = (float)(y - cy[c]);
                im[y * IMG + x] = expf(-(dx * dx + dy * dy) / 1.6f)
                                  + rf(r) * 0.10f;
            }
    }
}

static void init_weights(rng_t *r, float *W, float *Wfc, float *gam,
                         float *bet)
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
        Wfc[(size_t)NF * CPAD + n] = 0.0f;   /* unused class columns */
    for (int c = 0; c < COUTPAD; ++c) {
        gam[c] = 1.0f;
        bet[c] = 0.0f;
    }
}

static uint64_t fnv(const void *p, size_t n)
{
    const uint8_t *b = p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* ------------------------------------------------------------------ macros */
#define CK(x) do { st = (x); if (st != VC4_OK) { \
        printf("FAIL line %d: %s <- %s (%s)\n", __LINE__, vc4_strerror(st), \
               #x, vc4_last_error(ctx)); \
        printf("     arena=%zu used=%zu ws_free=%zu\n", \
               vc4_arena_bytes(ctx), vc4_arena_used(ctx), \
               vc4__ws_free(ctx)); \
        rc = 1; goto out; } } while (0)

/* one optimizer step on a tensor of n floats from gradient G */
#define OPT_W(Wv, G, M, V, n) do { \
        if (use_adam) { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &(G), &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(M), &tr.s_b1, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.s_ib1, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &(M), (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(V), &tr.s_b2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s1, &tr.s_ib2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.s2, &tr.s3, &(V), (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(V), &tr.s_eps, &tr.s2, (n))); \
            CK(vc4_rsqrt(ctx, &tr.s2, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(M), &tr.s2, &tr.s3, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &tr.s3, &tr.lrneg, &tr.s2, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(Wv), &tr.s2, &(Wv), (n))); \
        } else { \
            CK(vc4_elementwise(ctx, VC4_OP_MUL, &(G), &tr.lrneg, &tr.s1, (n))); \
            CK(vc4_elementwise(ctx, VC4_OP_ADD, &(Wv), &tr.s1, &(Wv), (n))); \
        } \
    } while (0)

/* forward over a batch of n images [base, base+n) */
#define FORWARD(T, base, n, Wv, Wfcv) do { \
        for (uint32_t _i = 0; _i < (n); ++_i) { \
            vc4_buf _xi = slice((T).X, (size_t)((base) + _i) * NIMG * 4, NIMG * 4); \
            vc4_buf _zi = slice((T).Z, (size_t)_i * NZ * 4, NZ * 4); \
            int _ho, _wo; \
            CK(vc4_conv2d(ctx, &_xi, CINPAD, IMG, IMG, &(Wv), COUTPAD, KH, KH, \
                          NULL, 1, 0, &_zi, &_ho, &_wo)); \
        } \
        CK(vc4_batchnorm_fwd(ctx, &(T).Z, &tr.gamma, &tr.beta, &(T).BN, \
                             &(T).smean, &(T).sistd, &(T).rmean, &(T).rvar, \
                             (n), COUTPAD, HO * HO, 0.1f, 1e-5f, 1)); \
        CK(vc4_relu(ctx, &(T).BN, &(T).A, (size_t)(n) * NZ)); \
        for (uint32_t _i = 0; _i < (n); ++_i) { \
            vc4_buf _ai = slice((T).A, (size_t)_i * NZ * 4, NZ * 4 + 64); \
            vc4_buf _pi = slice((T).P, (size_t)_i * NP * 4, NP * 4); \
            int _ho, _wo; \
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
            CK(vc4_softmax_ce_bwd(ctx, &(T).logits, &_labs, &(T).dlogits, \
                                  &(T).loss, NULL, (n), CPAD, \
                                  1.0f / (float)(n))); \
        } \
    } while (0)

int main(int argc, char **argv)
{
    const char *opt = "sgd";
    const char *savepath = "vc4_cnntrain.weights";
    int steps = 120, reps = 1;
    float lr_opt = 0.0f;
    uint32_t seed = 12345, B = 8;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--opt") && i + 1 < argc) opt = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) B = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr_opt = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) savepath = argv[++i];
    }
    int use_adam = strcmp(opt, "adam") == 0;
    if (!use_adam && strcmp(opt, "sgd") != 0) { printf("bad --opt\n"); return 2; }
    if (B % 8 || B == 0 || NTRAIN % B) {
        printf("bad --batch (multiple of 8, divides %d)\n", NTRAIN);
        return 2;
    }

    int lock = gpu_lock();
    if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n"); gpu_unlock(lock); return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("libvc4infer %s cnn trainer: conv3x3(1->8)-bn-relu-pool-fc(512->3), "
           "opt=%s seed=%u batch=%u arena %.1f MiB\n", VC4INFER_VERSION_STRING,
           opt, seed, B, (double)vc4_arena_bytes(ctx) / (1 << 20));

    bt_t tr;   /* one arena layout: images [0,NTRAIN) train, then test */
    if (bt_alloc(ctx, &tr, NTRAIN + NTEST, B) != VC4_OK)
        return 2;

    rng_t r = { seed };
    float *Xtr = malloc((size_t)NTRAIN * IMG * IMG * 4);
    float *Xte = malloc((size_t)NTEST * IMG * IMG * 4);
    int32_t *Ltr = malloc(NTRAIN * 4), *Lte = malloc(NTEST * 4);
    make_dataset(&r, Xtr, Ltr, NTRAIN);
    make_dataset(&r, Xte, Lte, NTEST);

    float *xp = calloc((size_t)(NTRAIN + NTEST) * NIMG, 4);
    for (int i = 0; i < NTRAIN; ++i)
        memcpy(xp + (size_t)i * NIMG, Xtr + (size_t)i * IMG * IMG, IMG * IMG * 4);
    for (int i = 0; i < NTEST; ++i)
        memcpy(xp + (size_t)(NTRAIN + i) * NIMG,
               Xte + (size_t)i * IMG * IMG, IMG * IMG * 4);
    vc4_upload(ctx, &tr.X, xp, (size_t)(NTRAIN + NTEST) * NIMG * 4);
    free(xp);
    int32_t *lall = malloc((size_t)(NTRAIN + NTEST) * 4);
    memcpy(lall, Ltr, NTRAIN * 4);
    memcpy(lall + NTRAIN, Lte, NTEST * 4);
    vc4_upload(ctx, &tr.lab, lall, (size_t)(NTRAIN + NTEST) * 4);
    free(lall);

    float *w0 = malloc((size_t)NW * 4), *wfc0 = malloc((size_t)KFC * CPAD * 4);
    float *gam0 = malloc(64), *bet0 = malloc(64);
    init_weights(&r, w0, wfc0, gam0, bet0);

    vc4_buf W, Wfc;
    vc4_alloc(ctx, (size_t)NW * 4, &W);
    vc4_alloc(ctx, (size_t)KFC * CPAD * 4, &Wfc);

    const size_t NFW = (size_t)KFC * CPAD;
    const float lr = lr_opt > 0.0f ? lr_opt : (use_adam ? 0.01f : 0.25f);
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    fill_scalar(ctx, &tr.s_b1, NFW, b1);
    fill_scalar(ctx, &tr.s_ib1, NFW, 1.0f - b1);
    fill_scalar(ctx, &tr.s_b2, NFW, b2);
    fill_scalar(ctx, &tr.s_ib2, NFW, 1.0f - b2);
    fill_scalar(ctx, &tr.s_eps, NFW, eps);
    fill_scalar(ctx, &tr.lrneg, NFW, -lr);

    int rc = 0;
    vc4_status st = VC4_OK;
    uint64_t hashes[8] = { 0 };
    double first_loss = 0, last_loss = 0;
    uint64_t t_start = now_ns();

    for (int rep = 0; rep < reps; ++rep) {
        /* fresh weights, zero optimizer state, zero running statistics */
        vc4_upload(ctx, &W, w0, (size_t)NW * 4);
        vc4_upload(ctx, &Wfc, wfc0, (size_t)KFC * CPAD * 4);
        vc4_upload(ctx, &tr.gamma, gam0, 64);
        vc4_upload(ctx, &tr.beta, bet0, 64);
        zero_device(ctx, &tr.rmean, 16);
        zero_device(ctx, &tr.rvar, 16);
        zero_device(ctx, &tr.m1, NFW);
        zero_device(ctx, &tr.v1, NFW);
        zero_device(ctx, &tr.m2, NW);
        zero_device(ctx, &tr.v2, NW);
        zero_device(ctx, &tr.mg, 16);
        zero_device(ctx, &tr.vg, 16);
        zero_device(ctx, &tr.mb, 16);
        zero_device(ctx, &tr.vb, 16);

        float tot;
        FORWARD(tr, 0, B, W, Wfc);
        vc4_download(ctx, &tr.loss, &tot, 4);
        if (rep == 0)
            first_loss = tot;
        printf("rep %d initial: train loss=%.6f\n", rep, (double)tot);

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

            /* pool backward per image (the pool input A is per image), then
             * relu + batchnorm backward over the whole batch: batchnorm's
             * gradient sums over every N*H*W element of the channel, so it
             * must see the batch at once. */
            for (uint32_t i = 0; i < B; ++i) {
                vc4_buf ai = slice(tr.A, (size_t)i * NZ * 4, NZ * 4 + 64);
                vc4_buf dpi = slice(tr.dP, (size_t)i * NP * 4, NP * 4);
                vc4_buf dai = slice(tr.dA, (size_t)i * NZ * 4, NZ * 4);
                memset(tr.dyP_host, 0, (size_t)NP * 4);
                const float *hh = tr.dh_host + (size_t)i * KFC;
                for (int c = 0; c < COUTREAL; ++c)
                    for (int y = 0; y < POOL; ++y)
                        memcpy(tr.dyP_host + (c * POOL + y) * PPITCH,
                               hh + c * POOL * POOL + y * POOL, POOL * 4);
                CK(vc4_upload(ctx, &dpi, tr.dyP_host, (size_t)NP * 4));
                CK(vc4_maxpool2x2_backward(ctx, &ai, &dpi, COUTPAD, HO, HO,
                                           PPITCH, &dai));
            }
            CK(vc4_relu_backward(ctx, &tr.BN, &tr.dA, &tr.dBN, (size_t)B * NZ));
            CK(vc4_batchnorm_bwd(ctx, &tr.Z, &tr.gamma, &tr.dBN, &tr.smean,
                                 &tr.sistd, &tr.dZ, &tr.dgamma, &tr.dbeta, B,
                                 COUTPAD, HO * HO));
            zero_device(ctx, &tr.dWacc, NW);
            for (uint32_t i = 0; i < B; ++i) {
                vc4_buf xi = slice(tr.X, (size_t)(base + i) * NIMG * 4, NIMG * 4);
                vc4_buf dzi = slice(tr.dZ, (size_t)i * NZ * 4, NZ * 4);
                vc4_buf dxi = slice(tr.dX, (size_t)i * NIMG * 4, NIMG * 4);
                CK(vc4_conv2d_backward(ctx, &xi, CINPAD, IMG, IMG, &W, COUTPAD,
                                       KH, KH, &dzi, 0, &tr.dWi, &dxi));
                CK(vc4_elementwise(ctx, VC4_OP_ADD, &tr.dWacc, &tr.dWi,
                                   &tr.dWacc, NW));
            }
            OPT_W(Wfc, tr.dWfc, tr.m1, tr.v1, NFW);
            OPT_W(W, tr.dWacc, tr.m2, tr.v2, NW);
            OPT_W(tr.gamma, tr.dgamma, tr.mg, tr.vg, 16);
            OPT_W(tr.beta, tr.dbeta, tr.mb, tr.vb, 16);

            if (step == 1 || step % 20 == 0 || step == steps) {
                vc4_download(ctx, &tr.loss, &tot, 4);
                printf("rep %d step %4d: loss=%.6f\n", rep, step, (double)tot);
            }
        }
        uint64_t t1 = now_ns();
        last_loss = tot;
        printf("rep %d arena: %.2f MiB of %.2f MiB used after %d steps\n", rep,
               (double)vc4_arena_used(ctx) / (1 << 20),
               (double)vc4_arena_bytes(ctx) / (1 << 20), steps);

        /* final losses over the whole train and test sets */
        double ltr = 0, lte = 0;
        for (uint32_t base = 0; base < NTRAIN; base += B) {
            FORWARD(tr, base, B, W, Wfc);
            vc4_download(ctx, &tr.loss, &tot, 4);
            ltr += tot;
        }
        for (uint32_t base = 0; base < NTEST; base += B) {
            FORWARD(tr, NTRAIN + base, B, W, Wfc);
            vc4_download(ctx, &tr.loss, &tot, 4);
            lte += tot;
        }
        ltr /= NTRAIN;
        lte /= NTEST;

        float *wf = malloc((size_t)NW * 4);
        float *wff = malloc((size_t)KFC * CPAD * 4);
        float gm[16], bt2[16], rm[16], rv[16];
        vc4_download(ctx, &W, wf, (size_t)NW * 4);
        vc4_download(ctx, &Wfc, wff, (size_t)KFC * CPAD * 4);
        vc4_download(ctx, &tr.gamma, gm, 64);
        vc4_download(ctx, &tr.beta, bt2, 64);
        vc4_download(ctx, &tr.rmean, rm, 64);
        vc4_download(ctx, &tr.rvar, rv, 64);
        hashes[0] = fnv(wf, (size_t)NW * 4);
        hashes[1] = fnv(wff, (size_t)KFC * CPAD * 4);
        hashes[2] = fnv(gm, 64);
        hashes[3] = fnv(bt2, 64);
        hashes[4] = fnv(rm, 64);
        hashes[5] = fnv(rv, 64);
        printf("rep %d final: train loss=%.9f test loss=%.9f  %.3f s (%.2f "
               "steps/s)\n", rep, ltr, lte, (double)(t1 - t0) / 1e9,
               steps / ((double)(t1 - t0) / 1e9));
        printf("rep %d weights hash conv=%016llx fc=%016llx gamma=%016llx "
               "beta=%016llx rmean=%016llx rvar=%016llx\n", rep,
               (unsigned long long)hashes[0],
               (unsigned long long)hashes[1],
               (unsigned long long)hashes[2],
               (unsigned long long)hashes[3],
               (unsigned long long)hashes[4],
               (unsigned long long)hashes[5]);

        /* save: conv W, fc W, gamma, beta, running stats */
        char path[512];
        snprintf(path, sizeof(path), "%s%s%d", savepath, reps > 1 ? ".r" : "",
                 reps > 1 ? rep : 0);
        FILE *f = fopen(path, "wb");
        if (!f) {
            printf("FAIL: cannot write %s\n", path);
            rc = 1;
        } else {
            uint32_t hdr = NW;
            uint64_t hdr2 = NFW;
            fwrite(&hdr, 4, 1, f);
            fwrite(wf, 4, NW, f);
            fwrite(&hdr2, 8, 1, f);
            fwrite(wff, 4, NFW, f);
            fwrite(gm, 4, 16, f);
            fwrite(bt2, 4, 16, f);
            fwrite(rm, 4, 16, f);
            fwrite(rv, 4, 16, f);
            fclose(f);
            printf("rep %d saved %s\n", rep, path);
        }
        if (rep == 0) {
            /* remember the first run's file for the byte comparison */
            snprintf(g_save_first, sizeof(g_save_first), "%s", path);
        }
        free(wf);
        free(wff);
    }
    (void)first_loss;
    (void)last_loss;
    (void)hashes;

    if (reps > 1) {
        /* byte-for-byte comparison of the two saved weight files */
        FILE *a = fopen(g_save_first, "rb");
        char path2[512];
        snprintf(path2, sizeof(path2), "%s.r%d", savepath, reps - 1);
        FILE *b = fopen(path2, "rb");
        int same = a && b;
        if (same) {
            int ca, cb;
            do {
                ca = fgetc(a);
                cb = fgetc(b);
                if (ca != cb) { same = 0; break; }
            } while (ca != EOF);
        }
        if (a) fclose(a);
        if (b) fclose(b);
        printf("%s: run 1 and run %d weights are %s\n",
               same ? "PASS" : "FAIL", reps, same ? "bit-identical" : "different");
        if (!same)
            rc = 1;
    }
out:
    if (!rc && last_loss > 0.5) {
        printf("FAIL: loss did not fall below 0.5 (%.6f)\n", last_loss);
        rc = 1;
    }
    (void)now_ns; (void)t_start;
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(rc ? "CNN TRAIN FAILED\n" : "CNN TRAIN PASSED\n");
    return rc;
}
