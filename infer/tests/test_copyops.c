/* test_copyops.c - element verification of the layout copy kernels.
 *
 *   vc4_copy2d*   the affine layout copy: 16-word destination vectors, a
 *                 per-lane source address, per-row / per-block / per-vector
 *                 byte strides, 1/2/4 vectors per DMA store and 1..16 words
 *                 per vector (the short-row tail), plus a lane count that
 *                 zero-pads channels
 *   vc4_fill2d    the constant fill (-inf MaxPool border, zero padding)
 *   vc4_upsample2x_nhwc  nearest 2x on the NHWC plane (Resize)
 *
 * Every case is checked against an independent CPU oracle written from the
 * *semantics* of the layout operation (a channel concat, an NHWC->NCHW
 * transpose, a nearest 2x resize, ...), never from the stride arithmetic the
 * call is built with, so a wrong stride cannot validate itself.  The copies
 * are pure byte moves, so the comparison is exact.
 *
 * The shapes deliberately include the ones that break naive kernels: W not a
 * multiple of 4, odd channel counts, H=1, extents that are not multiples of
 * 16 floats, a source pixel stride that differs from C*4, a 4-byte lane
 * gather, a 3-channel -> 16-channel stem with zero padding and a -inf
 * border.
 */
#include "vc4infer.h"
#include "util.h"

static int g_fail;
static vc4_ctx *g_ctx;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d ", __FILE__, __LINE__);                         \
            printf(__VA_ARGS__);                                               \
            printf(" [%s]\n", vc4_last_error(g_ctx));                           \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

/* ---- arena helpers ------------------------------------------------------ */
typedef struct {
    vc4_buf buf;
    float *host;
    size_t n;
} tensor;

static int mk(vc4_ctx *ctx, tensor *t, size_t nfloats)
{
    t->n = nfloats;
    t->host = calloc(nfloats ? nfloats : 1, sizeof(float));
    if (!t->host)
        return 1;
    if (vc4_alloc(ctx, nfloats * 4 + 64, &t->buf) != VC4_OK) {
        free(t->host);
        t->host = NULL;
        return 1;
    }
    vc4_upload(ctx, &t->buf, t->host, nfloats * 4);
    return 0;
}

static void free_t(tensor *t)
{
    free(t->host);
    t->host = NULL;
}

static void pull(vc4_ctx *ctx, tensor *t)
{
    vc4_download(ctx, &t->buf, t->host, t->n * 4);
}

static void report(const char *name, long bad, size_t n, uint64_t ns)
{
    printf("%s %-48s %6ld/%zu bad  %7.3f ms  %7.1f MB/s\n",
           bad ? "FAIL" : "PASS", name, bad, n, ns / 1e6,
           n ? (double)n * 4.0 / (double)ns * 1e3 : 0.0);
    if (bad)
        ++g_fail;
}

/* `nvec` 16-word vectors per row: full 4-vector chunks, then (if the count is
 * not a multiple of 4) one 4-vector chunk that *ends* at the end of the run --
 * its last vectors overlap the previous chunk with identical data.  This is
 * the runner's convention for extents that are not a multiple of 64 floats. */
static vc4_status copy_run(vc4_ctx *ctx, vc4_buf src, vc4_buf dst,
                           uint32_t rows, uint32_t nvec, uint32_t srow,
                           uint32_t drow, uint32_t svec, uint32_t dvec,
                           uint32_t slane)
{
    vc4_status st = VC4_OK;
    if (nvec < 4)
        return vc4_copy2d_n(ctx, &src, &dst, rows, nvec, srow, drow, svec,
                            dvec < 64 ? 64 : dvec, slane, 16, 1, 16);
    const uint32_t full = (nvec / 4) * 4;
    if (full)
        st = vc4_copy2d(ctx, &src, &dst, rows, full, srow, drow, svec, dvec,
                        slane);
    if (st == VC4_OK && nvec % 4) {
        const uint32_t k = nvec - 4;
        src.off += k * svec;
        src.bytes -= k * svec;
        dst.off += k * dvec;
        dst.bytes -= k * dvec;
        st = vc4_copy2d(ctx, &src, &dst, rows, 4, srow, drow, svec, dvec,
                        slane);
    }
    return st;
}

static uint32_t units_for(uint32_t n)
{
    return n % 4 == 0 ? 4 : (n % 2 == 0 ? 2 : 1);
}

/* ---- 1. contiguous run, extent not a multiple of 16 floats --------------- */
static void case_contig(vc4_ctx *ctx)
{
    const size_t n = 1050;                     /* the head slice: 65.6 vec */
    tensor s, d;
    if (mk(ctx, &s, n) || mk(ctx, &d, n)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < n; ++i)
        s.host[i] = (float)(i * 3 + 1);
    vc4_upload(ctx, &s.buf, s.host, n * 4);
    uint64_t t0 = now_ns();
    vc4_status st = VC4_OK;
    if (n % 16) {
        vc4_buf sb = s.buf, db = d.buf;
        const size_t off = (n / 16) * 16 * 4;
        sb.off += off; sb.bytes -= off;
        db.off += off; db.bytes -= off;
        st = vc4_copy2d_n(ctx, &sb, &db, 1, 1, 0, 0, 64, 64, 4, 16, 1,
                          (uint32_t)(n % 16));
    }
    if (st == VC4_OK)
        st = copy_run(ctx, s.buf, d.buf, 1, (uint32_t)(n / 16), 0, 0, 64, 64,
                      4);
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "contig: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (size_t i = 0; i < n; ++i)
        bad += d.host[i] != s.host[i];
    report("contiguous 1050 floats (65.6 vec)", bad, n, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 2. channel concat: W%4 != 0, pitched source, >16 channels ---------- */
static void case_concat(vc4_ctx *ctx)
{
    /* [1][40][5] NHWC, sources of C={16,32,16} at pitch 8 -> [1][64][5] */
    const int H = 40, W = 5, SP = 8, CT = 64;
    const int CS[3] = {16, 32, 16};
    const int CO[3] = {0, 16, 48};
    tensor src[3], dst;
    for (int i = 0; i < 3; ++i) {
        if (mk(ctx, &src[i], (size_t)H * SP * CS[i])) {
            for (int j = 0; j <= i; ++j) free_t(&src[j]);
            return;
        }
        for (size_t k = 0; k < src[i].n; ++k)
            src[i].host[k] = (float)(1000 * i + k);
        vc4_upload(ctx, &src[i].buf, src[i].host, src[i].n * 4);
    }
    const size_t dsz = (size_t)H * W * CT;
    if (mk(ctx, &dst, dsz)) {
        for (int i = 0; i < 3; ++i) free_t(&src[i]);
        return;
    }
    uint64_t t0 = now_ns();
    vc4_status st = VC4_OK;
    for (int i = 0; i < 3 && st == VC4_OK; ++i) {
        vc4_buf db = dst.buf;
        db.off += (size_t)CO[i] * 4;          /* destination channel base */
        db.bytes -= (size_t)CO[i] * 4;
        st = vc4_copy2d_x(ctx, &src[i].buf, &db, (uint32_t)H, (uint32_t)W,
                          (uint32_t)(SP * CS[i] * 4), (uint32_t)(W * CT * 4),
                          (uint32_t)(CS[i] * 4), (uint32_t)(CT * 4), 4, 16,
                          (uint32_t)(CS[i] / 16), 64, 64,
                          units_for((uint32_t)W), 16);
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "concat: %s", vc4_strerror(st));
    pull(ctx, &dst);
    long bad = 0;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int i = 0; i < 3; ++i)
                for (int c = 0; c < CS[i]; ++c) {
                    float want = src[i].host[((size_t)h * SP + w) * CS[i] + c];
                    float got = dst.host[((size_t)h * W + w) * CT + CO[i] + c];
                    bad += got != want;
                }
    report("concat [40][5] C16+32+16 -> C64, pitch 8->5", bad, dsz, t1 - t0);
    for (int i = 0; i < 3; ++i) free_t(&src[i]);
    free_t(&dst);
}

/* ---- 3. split: pitched source and pitched outputs ---------------------- */
static void case_split(vc4_ctx *ctx)
{
    /* [1][64][3][10] (W=8 data, pitch 10) -> 4 x [1][16][3][10] */
    const int H = 3, W = 8, WP = 10, C = 64;
    tensor s, o[4];
    const size_t ns = (size_t)H * WP * C, no = (size_t)H * WP * 16;
    if (mk(ctx, &s, ns))
        return;
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i + 1);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    for (int i = 0; i < 4; ++i)
        if (mk(ctx, &o[i], no)) {
            for (int j = 0; j <= i; ++j) free_t(&o[j]);
            free_t(&s);
            return;
        }
    uint64_t t0 = now_ns();
    vc4_status st = VC4_OK;
    for (int i = 0; i < 4 && st == VC4_OK; ++i) {
        vc4_buf sb = s.buf;
        sb.off += (size_t)i * 16 * 4;         /* source channel base */
        sb.bytes -= (size_t)i * 16 * 4;
        st = vc4_copy2d_x(ctx, &sb, &o[i].buf, (uint32_t)H, (uint32_t)W,
                          (uint32_t)(WP * C * 4), (uint32_t)(WP * 16 * 4),
                          (uint32_t)(C * 4), 16 * 4u, 4, 16, 1, 64, 64,
                          units_for((uint32_t)W), 16);
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "split: %s", vc4_strerror(st));
    long bad = 0;
    for (int i = 0; i < 4; ++i) {
        pull(ctx, &o[i]);
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < 16; ++c) {
                    float want = s.host[((size_t)h * WP + w) * C + i * 16 + c];
                    bad += o[i].host[((size_t)h * WP + w) * 16 + c] != want;
                }
    }
    report("split pitched 64ch -> 4x16ch (W%4 != 0)", bad, 4 * no, t1 - t0);
    free_t(&s);
    for (int i = 0; i < 4; ++i) free_t(&o[i]);
}

/* ---- 4. NHWC -> NCHW, H*W not a multiple of 16 ------------------------- */
static void case_nhwc_to_nchw(vc4_ctx *ctx, int C, int H, int W, int pitch)
{
    tensor s, d;
    const size_t ns = (size_t)H * pitch * C, nd = (size_t)C * H * W;
    if (mk(ctx, &s, ns) || mk(ctx, &d, nd)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i + 1);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    const int hw = H * W;
    uint64_t t0 = now_ns();
    vc4_status st = VC4_OK;
    if (pitch == W) {
        /* compact: a vector is 16 consecutive pixels of one channel */
        const int hv = hw / 16, rem = hw % 16;
        if (hv)
            st = copy_run(ctx, s.buf, d.buf, (uint32_t)C, (uint32_t)hv, 4,
                          (uint32_t)(hw * 4), (uint32_t)(16 * C * 4), 64,
                          (uint32_t)(C * 4));
        if (st == VC4_OK && rem) {
            vc4_buf sb = s.buf, db = d.buf;
            const size_t so = (size_t)(hw - rem) * C * 4;
            const size_t dofs = (size_t)(hw - rem) * 4;
            sb.off += so; sb.bytes -= so;
            db.off += dofs; db.bytes -= dofs;
            st = vc4_copy2d_n(ctx, &sb, &db, (uint32_t)C, 1, 4,
                              (uint32_t)(hw * 4), (uint32_t)(C * 4), 64,
                              (uint32_t)(C * 4), 16, 1, (uint32_t)rem);
        }
    } else if (W <= 16) {
        /* pitched source: one W-word run per (channel, source row), the lanes
         * are the W columns of that row */
        st = vc4_copy2d_n(ctx, &s.buf, &d.buf, (uint32_t)C, (uint32_t)H, 4,
                          (uint32_t)(hw * 4), (uint32_t)(pitch * C * 4),
                          (uint32_t)(W * 4), (uint32_t)(C * 4), W, 1, W);
    } else {
        for (int w0 = 0; w0 < W && st == VC4_OK; w0 += 16) {
            int n = W - w0 < 16 ? W - w0 : 16;
            vc4_buf sb = s.buf, db = d.buf;
            sb.off += (size_t)w0 * C * 4; sb.bytes -= (size_t)w0 * C * 4;
            db.off += (size_t)w0 * 4;     db.bytes -= (size_t)w0 * 4;
            st = vc4_copy2d_n(ctx, &sb, &db, (uint32_t)C, (uint32_t)H, 4,
                              (uint32_t)(hw * 4), (uint32_t)(pitch * C * 4),
                              (uint32_t)(W * 4), (uint32_t)(C * 4), n, 1, n);
        }
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "nhwc->nchw: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < hw; ++p) {
            float want = s.host[(size_t)(p / W) * pitch * C + (p % W) * C + c];
            bad += d.host[(size_t)c * hw + p] != want;
        }
    char nm[72];
    snprintf(nm, sizeof(nm), "nhwc->nchw C%d H%d W%d pitch%d", C, H, W, pitch);
    report(nm, bad, nd, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 5. channel slice of a pitched plane (4-byte lanes) ---------------- */
static void case_slice(vc4_ctx *ctx)
{
    /* the DFL selection: [16][4][528] pixel-major -> channel 0 -> [4][525] */
    const int C = 16, H = 4, W = 525, WP = 528;
    tensor s, d;
    const size_t ns = (size_t)H * WP * C, nd = (size_t)H * W;
    if (mk(ctx, &s, ns) || mk(ctx, &d, nd)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i % 977);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    const int nv = W / 16, rem = W % 16;
    uint64_t t0 = now_ns();
    vc4_status st = copy_run(ctx, s.buf, d.buf, (uint32_t)H, (uint32_t)nv,
                             (uint32_t)(WP * C * 4), (uint32_t)(W * 4),
                             (uint32_t)(16 * C * 4), 64, (uint32_t)(C * 4));
    if (st == VC4_OK && rem) {
        vc4_buf sb = s.buf, db = d.buf;
        const size_t so = (size_t)nv * 16 * C * 4;
        const size_t dofs = (size_t)nv * 16 * 4;
        sb.off += so; sb.bytes -= so;
        db.off += dofs; db.bytes -= dofs;
        st = vc4_copy2d_n(ctx, &sb, &db, (uint32_t)H, 1,
                          (uint32_t)(WP * C * 4), (uint32_t)(W * 4),
                          (uint32_t)(C * 4), 64, (uint32_t)(C * 4), 16, 1,
                          (uint32_t)rem);
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "slice: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (int p = 0; p < H * W; ++p) {
        const int h = p / W, w = p % W;
        float want = s.host[((size_t)h * WP + w) * C];
        bad += d.host[p] != want;
    }
    report("slice channel 0 of [4][525] (4-byte lanes)", bad, nd, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 6. transpose [0,2,1,3] on 525-wide rows -------------------------- */
static void case_transpose(vc4_ctx *ctx)
{
    const int J = 4, K = 16, W = 525;
    tensor s, d;
    const size_t n = (size_t)J * K * W;
    if (mk(ctx, &s, n) || mk(ctx, &d, n)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < n; ++i)
        s.host[i] = (float)((i * 7) % 1009);
    vc4_upload(ctx, &s.buf, s.host, n * 4);
    const int nv = W / 16, rem = W % 16;
    uint64_t t0 = now_ns();
    vc4_status st = vc4_copy2d_x(ctx, &s.buf, &d.buf, (uint32_t)K,
                                 (uint32_t)nv, (uint32_t)(W * 4),
                                 (uint32_t)(J * W * 4), 64, 64, 4, 16,
                                 (uint32_t)J, (uint32_t)(K * W * 4),
                                 (uint32_t)(W * 4), 4, 16);
    if (st == VC4_OK && rem) {
        vc4_buf sb = s.buf, db = d.buf;
        const size_t sh = (size_t)nv * 16 * 4;
        sb.off += sh; sb.bytes -= sh;
        db.off += sh; db.bytes -= sh;
        st = vc4_copy2d_x(ctx, &sb, &db, (uint32_t)K, 1, (uint32_t)(W * 4),
                          (uint32_t)(J * W * 4), 64, 64, 4, 16, (uint32_t)J,
                          (uint32_t)(K * W * 4), (uint32_t)(W * 4), 1,
                          (uint32_t)rem);
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "transpose: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
            for (int w = 0; w < W; ++w) {
                float want = s.host[((size_t)j * K + k) * W + w];
                bad += d.host[((size_t)k * J + j) * W + w] != want;
            }
    report("transpose [4][16][525] -> [16][4][525]", bad, n, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 7. nearest 2x resize, NHWC, C%16 == 0 ---------------------------- */
static void case_resize2x(vc4_ctx *ctx)
{
    const int C = 128, H = 5, W = 5;
    tensor s, d;
    const size_t ns = (size_t)H * W * C;
    const size_t nd = (size_t)2 * H * 16 * C;      /* opitch = align16(2W) */
    if (mk(ctx, &s, ns) || mk(ctx, &d, nd)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i + 1);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    int ho = 0, wo = 0, op = 0;
    uint64_t t0 = now_ns();
    vc4_status st = vc4_upsample2x_nhwc(ctx, &s.buf, 1, C, H, W, W, &d.buf,
                                        &ho, &wo, &op);
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "resize: %s", vc4_strerror(st));
    CHECK(ho == 2 * H && wo == 2 * W, "resize shape %dx%d", ho, wo);
    pull(ctx, &d);
    long bad = 0;
    for (int y = 0; y < 2 * H; ++y)
        for (int x = 0; x < 2 * W; ++x)
            for (int c = 0; c < C; ++c) {
                float want = s.host[((size_t)(y / 2) * W + x / 2) * C + c];
                bad += d.host[((size_t)y * op + x) * C + c] != want;
            }
    report("resize nearest 2x [5][5][128] -> [10][10][128]", bad, nd, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 8. -inf MaxPool border: fill then interior ----------------------- */
static void case_pool_border(vc4_ctx *ctx)
{
    const int C = 128, H = 5, W = 5, P = 2, HP = H + 2 * P, WP = W + 2 * P;
    const int pitch = 16;                      /* align16(WP) */
    tensor s, d;
    const size_t ns = (size_t)H * pitch * C, nd = (size_t)HP * pitch * C;
    if (mk(ctx, &s, ns) || mk(ctx, &d, nd)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i + 1);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    uint64_t t0 = now_ns();
    vc4_status st = vc4_fill2d(ctx, &d.buf, (uint32_t)(nd / 16 / 4), 4,
                               256, 64, -INFINITY);
    if (st == VC4_OK) {
        vc4_buf db = d.buf;
        db.off += (size_t)(P * pitch + P) * C * 4;
        db.bytes -= (size_t)(P * pitch + P) * C * 4;
        const uint32_t vpr = (uint32_t)(W * C / 16);
        st = vc4_copy2d_x(ctx, &s.buf, &db, (uint32_t)H, vpr,
                          (uint32_t)(pitch * C * 4), (uint32_t)(pitch * C * 4),
                          64, 64, 4, 16, 1, 64, 64, units_for(vpr), 16);
    }
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "pool border: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (int h = 0; h < HP; ++h)
        for (int w = 0; w < WP; ++w)
            for (int c = 0; c < C; ++c) {
                const int sy = h - P, sx = w - P;
                float want = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                 ? s.host[((size_t)sy * pitch + sx) * C + c]
                                 : -INFINITY;
                float got = d.host[((size_t)h * pitch + w) * C + c];
                bad += got != want;
            }
    if (bad) {
        int shown = 0;
        for (int h = 0; h < HP && shown < 8; ++h)
            for (int w = 0; w < WP && shown < 8; ++w)
                for (int c = 0; c < C; ++c) {
                    const int sy = h - P, sx = w - P;
                    float want = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                     ? s.host[((size_t)sy * pitch + sx) * C + c]
                                     : -INFINITY;
                    float got = d.host[((size_t)h * pitch + w) * C + c];
                    if (got != want) {
                        printf("   bad h=%d w=%d c=%d got %g want %g\n",
                               h, w, c, got, want);
                        ++shown;
                        break;
                    }
                }
    }
    report("maxpool -inf border [5][5][128] -> [9][9][128]", bad, nd, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 9. first conv prologue: NCHW 3ch -> NHWC 16ch (zero padded) ------ */
static void case_stem(vc4_ctx *ctx)
{
    const int C = 3, CP = 16, H = 16, W = 16;
    tensor s, d;
    const size_t ns = (size_t)C * H * W, nd = (size_t)H * W * CP;
    if (mk(ctx, &s, ns) || mk(ctx, &d, nd)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < ns; ++i)
        s.host[i] = (float)(i + 1);
    vc4_upload(ctx, &s.buf, s.host, ns * 4);
    uint64_t t0 = now_ns();
    /* one vector = one pixel's 16 channels, so vpr = H*W */
    vc4_status st = vc4_copy2d_n(ctx, &s.buf, &d.buf, 1,
                                 (uint32_t)(H * W), 0, 0, 4, 64,
                                 (uint32_t)(H * W * 4), C, 4, 16);
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "stem: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (int p = 0; p < H * W; ++p)
        for (int c = 0; c < CP; ++c) {
            float want = c < C ? s.host[(size_t)c * H * W + p] : 0.0f;
            bad += d.host[(size_t)p * CP + c] != want;
        }
    report("first conv prologue NCHW 3ch -> NHWC 16ch", bad, nd, t1 - t0);
    free_t(&s);
    free_t(&d);
}

/* ---- 10. throughput on a 1 MiB copy ---------------------------------- */
static void case_bulk(vc4_ctx *ctx, uint32_t units)
{
    const size_t n = 1u << 18;                 /* 262144 floats = 1 MiB */
    tensor s, d;
    if (mk(ctx, &s, n) || mk(ctx, &d, n)) { free_t(&s); free_t(&d); return; }
    for (size_t i = 0; i < n; ++i)
        s.host[i] = (float)i;
    vc4_upload(ctx, &s.buf, s.host, n * 4);
    const uint32_t rows = (uint32_t)(n / 16 / 16), vpr = 16;
    uint64_t t0 = now_ns();
    vc4_status st = vc4_copy2d_x(ctx, &s.buf, &d.buf, rows, vpr, 1024, 1024,
                                 64, 64, 4, 16, 1, 64, 64, units, 16);
    uint64_t t1 = now_ns();
    CHECK(st == VC4_OK, "bulk: %s", vc4_strerror(st));
    pull(ctx, &d);
    long bad = 0;
    for (size_t i = 0; i < n; ++i)
        bad += d.host[i] != s.host[i];
    char nm[64];
    snprintf(nm, sizeof(nm), "bulk 1 MiB contiguous (units=%u)", units);
    report(nm, bad, n, t1 - t0);
    free_t(&s);
    free_t(&d);
}

int main(void)
{
    vc4_ctx *ctx = NULL;
    const int lock = gpu_lock();
    if (lock < 0)
        return 2;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: vc4_init: %s (%s)\n", vc4_strerror(st),
               ctx ? vc4_last_error(ctx) : "-");
        gpu_unlock(lock);
        return 1;
    }
    g_ctx = ctx;
    printf("arena %u MiB\n", (unsigned)(vc4_arena_bytes(ctx) >> 20));
    case_contig(ctx);
    case_concat(ctx);
    case_split(ctx);
    case_nhwc_to_nchw(ctx, 16, 5, 5, 16);       /* 25 words per channel   */
    case_nhwc_to_nchw(ctx, 64, 20, 20, 20);     /* 400 words, pitch == W  */
    case_nhwc_to_nchw(ctx, 80, 10, 10, 16);     /* 100 words, pitched     */
    case_nhwc_to_nchw(ctx, 16, 1, 7, 16);       /* H=1, W%4 != 0          */
    case_slice(ctx);
    case_transpose(ctx);
    case_pool_border(ctx);
    case_stem(ctx);
    case_resize2x(ctx);
    case_bulk(ctx, 4);
    case_bulk(ctx, 1);
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf(g_fail ? "%d FAILURES\n" : "ALL COPY OPS PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
