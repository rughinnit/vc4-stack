/*
 * test_bwd_one.c - run exactly ONE backward entry point in its own process.
 *
 * This is the bring-up tool for the composite backward ops: the per-op
 * gradient checks pass, but a composite op wedges the QPU, and a wedge takes
 * the whole board with it.  One op per process with a hard timeout turns that
 * into a bisect: the first op whose process times out is the one that hangs.
 *
 *   test_bwd_one <op>
 *
 * Exit status: 0 ok (with the error against an fp64 reference on stdout),
 *              1 wrong numbers, 2 usage/init, 3 the entry point refused.
 *              A timeout (124) means the launch hung the QPU.
 *
 * The input is fixed and deterministic so a rerun is comparable.
 */
#include "vc4infer.h"
#include "util.h"

#include <math.h>

#define N 64
#define R 8
#define C 8

static vc4_ctx *ctx;
static int errors;

static vc4_buf bufs(size_t n, const float *h)
{
    vc4_buf b;
    if (vc4_alloc(ctx, (n + 16) * 4 + 64, &b) != VC4_OK) {
        printf("alloc failed\n");
        exit(2);
    }
    if (h)
        vc4_upload(ctx, &b, h, n * 4);
    return b;
}

static void report(const char *nm, const vc4_buf *b, size_t n, const double *ref)
{
    float *y = malloc(n * 4);
    vc4_download(ctx, b, y, n * 4);
    double worst = 0, worst_rel = 0;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = fabs((double)y[i] - ref[i]);
        double d = fabs(ref[i]) > 1e-3 ? fabs(ref[i]) : 1e-3;
        if (e > worst) worst = e;
        if (e / d > worst_rel) worst_rel = e / d;
        if (e / d > 2e-3) bad++;
    }
    printf("  %-18s n=%zu max_abs=%.3g max_rel=%.3g bad=%zu\n", nm, n, worst,
           worst_rel, bad);
    if (bad) errors++;
    free(y);
}

static void chk(vc4_status st, const char *what)
{
    if (st == VC4_ERR_UNSUPPORTED) {
        printf("REFUSED %s: %s\n", what, vc4_last_error(ctx));
        vc4_shutdown(ctx);
        exit(3);
    }
    if (st != VC4_OK) {
        printf("FAIL %s: %s (%s)\n", what, vc4_strerror(st),
               vc4_last_error(ctx));
        vc4_shutdown(ctx);
        exit(1);
    }
}

static void fill(float *x, size_t n, float lo, float span)
{
    for (size_t i = 0; i < n; ++i) {
        int32_t v = (int32_t)((i * 2654435761u) >> 9 & 0x7FFFFFu) - (1 << 22);
        x[i] = lo + span * ((float)v / 4194304.0f);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: test_bwd_one <op>\n");
        return 2;
    }
    const char *op = argv[1];
    int lock = gpu_lock();
    if (lock < 0)
        return 2;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("init failed\n");
        gpu_unlock(lock);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("%s (arena %.1f MiB)\n", op, (double)vc4_arena_bytes(ctx) / (1 << 20));

    static float x[N * 8], g[N * 8];
    static double ref[N * 8];
    vc4_status st = VC4_OK;

    if (!strcmp(op, "softmax_bwd")) {
        /* y = softmax of a row-major [R][C]; dx = y*(g - sum g*y) */
        float lg[R * C];
        fill(lg, R * C, -1.0f, 2.0f);
        vc4_buf bl = bufs(R * C, lg), bp = bufs(R * C, NULL);
        vc4_buf bg = bufs(R * C, NULL), bd = bufs(R * C, NULL);
        chk(vc4_softmax(ctx, &bl, R, C, &bp), "softmax fwd");
        float p[R * C];
        vc4_download(ctx, &bp, p, R * C * 4);
        fill(g, R * C, -1.0f, 2.0f);
        vc4_upload(ctx, &bg, g, R * C * 4);
        chk(vc4_softmax_bwd(ctx, &bp, &bg, &bd, R, C), "softmax_bwd");
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int c = 0; c < C; ++c) s += (double)g[r * C + c] * p[r * C + c];
            for (int c = 0; c < C; ++c)
                ref[r * C + c] = p[r * C + c] * ((double)g[r * C + c] - s);
        }
        report(op, &bd, R * C, ref);
    } else if (!strcmp(op, "softmax_ce_bwd")) {
        const int CC = 16;                  /* the fused path is 16 classes */
        float lg[R * CC];
        int32_t lab[R];
        fill(lg, R * CC, -1.0f, 2.0f);
        for (int r = 0; r < R; ++r) lab[r] = r % CC;
        vc4_buf bl = bufs(R * CC, lg), bb = bufs(R, (const float *)NULL);
        vc4_upload(ctx, &bb, lab, R * 4);
        vc4_buf bd = bufs(R * CC, NULL), bl2 = bufs(1, NULL);
        float zero = 0;
        vc4_upload(ctx, &bl2, &zero, 4);
        chk(vc4_softmax_ce_bwd(ctx, &bl, &bb, &bd, &bl2, NULL, R, CC,
                              1.0f / (float)R), "softmax_ce_bwd");
        for (int r = 0; r < R; ++r) {
            double m = -1e30, sm = 0;
            for (int c = 0; c < CC; ++c)
                if (lg[r * CC + c] > m) m = lg[r * CC + c];
            for (int c = 0; c < CC; ++c) sm += exp((double)lg[r * CC + c] - m);
            for (int c = 0; c < CC; ++c) {
                double p = exp((double)lg[r * CC + c] - m) / sm;
                ref[r * CC + c] = (p - (c == lab[r] ? 1.0 : 0.0)) / (double)R;
            }
        }
        report(op, &bd, R * CC, ref);
        /* and the loss scalar */
        float ls;
        vc4_download(ctx, &bl2, &ls, 4);
        double lref = 0;
        for (int r = 0; r < R; ++r) {
            double m = -1e30, sm = 0;
            for (int c = 0; c < CC; ++c)
                if (lg[r * CC + c] > m) m = lg[r * CC + c];
            for (int c = 0; c < CC; ++c) sm += exp((double)lg[r * CC + c] - m);
            lref += -(lg[r * CC + lab[r]] - m) + log(sm);
        }
        lref /= R;
        double le = fabs((double)ls - lref) / (fabs(lref) > 1e-3 ? fabs(lref)
                                                                : 1e-3);
        printf("  %-18s got=%.9f ref=%.9f rel=%.2e\n", "ce loss", (double)ls,
               lref, le);
        if (le > 2e-3) errors++;
    } else if (!strcmp(op, "batchnorm_bwd")) {
        const int Nn = 4, Cc = 4, HW = 4;
        const size_t n = (size_t)Nn * Cc * HW;
        float xb[N], gm[Cc], bt[Cc], dy[N];
        fill(xb, n, -1.0f, 2.0f);
        for (int c = 0; c < Cc; ++c) { gm[c] = 1.0f + 0.1f * c; bt[c] = 0; }
        fill(dy, n, -1.0f, 2.0f);
        vc4_buf bx = bufs(n, xb), bg = bufs(Cc, gm), bb = bufs(Cc, bt);
        vc4_buf by = bufs(n, NULL), bm = bufs(Cc, NULL), bv = bufs(Cc, NULL);
        vc4_buf bdy = bufs(n, dy), bdx = bufs(n, NULL);
        vc4_buf bdg = bufs(Cc, NULL), bdb = bufs(Cc, NULL);
        chk(vc4_batchnorm_fwd(ctx, &bx, &bg, &bb, &by, &bm, &bv, NULL, NULL,
                              Nn, Cc, HW, 0.1f, 1e-5f, 0), "batchnorm_fwd");
        chk(vc4_batchnorm_bwd(ctx, &bx, &bg, &bdy, &bm, &bv, &bdx, &bdg, &bdb,
                              Nn, Cc, HW), "batchnorm_bwd");
        float mean[Cc], istd[Cc];
        vc4_download(ctx, &bm, mean, Cc * 4);
        vc4_download(ctx, &bv, istd, Cc * 4);
        const double M = (double)Nn * HW;
        for (int c = 0; c < Cc; ++c) {
            double sd = 0, sg = 0;
            for (int nn = 0; nn < Nn; ++nn)
                for (int i = 0; i < HW; ++i) {
                    size_t j = ((size_t)nn * Cc + c) * HW + i;
                    double xh = ((double)xb[j] - mean[c]) * istd[c];
                    sd += dy[j];
                    sg += dy[j] * xh;
                }
            for (int nn = 0; nn < Nn; ++nn)
                for (int i = 0; i < HW; ++i) {
                    size_t j = ((size_t)nn * Cc + c) * HW + i;
                    double xh = ((double)xb[j] - mean[c]) * istd[c];
                    ref[j] = ((double)gm[c] * istd[c] / M) *
                             (M * dy[j] - sd - xh * sg);
                }
            ref[n + c] = sg;              /* dgamma */
            ref[n + Cc + c] = sd;         /* dbeta */
        }
        report("batchnorm dx", &bdx, n, ref);
        report("batchnorm dgamma", &bdg, Cc, ref + n);
        report("batchnorm dbeta", &bdb, Cc, ref + n + Cc);
    } else if (!strcmp(op, "layernorm_bwd")) {
        float xl[R * C], gm[C], bt[C], dy[R * C];
        fill(xl, R * C, -1.0f, 2.0f);
        fill(dy, R * C, -1.0f, 2.0f);
        for (int c = 0; c < C; ++c) { gm[c] = 1.0f + 0.05f * c; bt[c] = 0; }
        vc4_buf bx = bufs(R * C, xl), bg = bufs(C, gm), bb = bufs(C, bt);
        vc4_buf by = bufs(R * C, NULL), bm = bufs(R, NULL), bv = bufs(R, NULL);
        vc4_buf bdy = bufs(R * C, dy), bdx = bufs(R * C, NULL);
        vc4_buf bdg = bufs(C, NULL), bdb = bufs(C, NULL);
        chk(vc4_layernorm_fwd(ctx, &bx, &bg, &bb, &by, &bm, &bv, R, C, 1e-5f),
            "layernorm_fwd");
        chk(vc4_layernorm_bwd(ctx, &bx, &bg, &bdy, &bm, &bv, &bdx, &bdg, &bdb,
                              R, C), "layernorm_bwd");
        float mean[R], istd[R];
        vc4_download(ctx, &bm, mean, R * 4);
        vc4_download(ctx, &bv, istd, R * 4);
        for (int r = 0; r < R; ++r) {
            double a = 0, b = 0;
            for (int c = 0; c < C; ++c) {
                double xh = ((double)xl[r * C + c] - mean[r]) * istd[r];
                double gx = dy[r * C + c] * gm[c];
                a += gx;
                b += gx * xh;
            }
            a /= C;
            b /= C;
            for (int c = 0; c < C; ++c) {
                double xh = ((double)xl[r * C + c] - mean[r]) * istd[r];
                double gx = dy[r * C + c] * gm[c];
                ref[r * C + c] = istd[r] * (gx - a - xh * b);
            }
        }
        report("layernorm dx", &bdx, R * C, ref);
        /* and the forward, so a mismatch localises to fwd or bwd */
        float yf[R * C];
        vc4_download(ctx, &by, yf, R * C * 4);
        for (int r = 0; r < R; ++r) {
            double mu = 0, var = 0;
            for (int c = 0; c < C; ++c) mu += xl[r * C + c];
            mu /= C;
            for (int c = 0; c < C; ++c) var += (xl[r * C + c] - mu) *
                                             (xl[r * C + c] - mu);
            var /= C;
            double is = 1.0 / sqrt(var + 1e-5);
            for (int c = 0; c < C; ++c)
                ref[r * C + c] = (xl[r * C + c] - mu) * is * gm[c] + bt[c];
        }
        report("layernorm fwd", &by, R * C, ref);
    } else if (!strcmp(op, "avgpool_bwd") || !strcmp(op, "adaptive_avgpool_bwd")
               || !strcmp(op, "global_avgpool_bwd")
               || !strcmp(op, "maxpooln_bwd")
               || !strcmp(op, "resize_nearest_bwd")
               || !strcmp(op, "resize_bilinear_bwd")) {
        /* one channel, 6x6 -> pool/resize, dy = 1: the gradient is the
         * indicator sum, which is a geometry check independent of arithmetic */
        const int H = 6, W = 6, Cc = 1;
        float xp[H * W];
        fill(xp, H * W, -1.0f, 2.0f);
        vc4_buf bx = bufs(H * W, xp);
        vc4_buf bdx = bufs(H * W, NULL);
        if (!strcmp(op, "global_avgpool_bwd")) {
            float dy1 = 1.0f;
            vc4_buf bdy = bufs(1, &dy1);
            chk(vc4_global_avgpool_bwd(ctx, &bdy, &bdx, Cc, H, W), op);
            for (int i = 0; i < H * W; ++i) ref[i] = 1.0 / (H * W);
            report(op, &bdx, H * W, ref);
        } else if (!strcmp(op, "avgpool_bwd")) {
            const int Nw = 2, S = 2;
            const int Ho = (H - Nw) / S + 1, Wo = (W - Nw) / S + 1;
            float dy1[9 * 4];
            for (int i = 0; i < Ho * Wo; ++i) dy1[i] = 1.0f;
            vc4_buf bdy = bufs(Ho * Wo, dy1);
            chk(vc4_avgpool_bwd(ctx, &bdy, &bdx, Cc, H, W, Nw, S, 0, 0), op);
            for (int i = 0; i < H * W; ++i) ref[i] = 1.0 / (Nw * Nw);
            report(op, &bdx, H * W, ref);
        } else if (!strcmp(op, "adaptive_avgpool_bwd")) {
            const int oH = 3, oW = 3;
            float dy1[16];
            for (int i = 0; i < oH * oW; ++i) dy1[i] = 1.0f;
            vc4_buf bdy = bufs(oH * oW, dy1);
            chk(vc4_adaptive_avgpool_bwd(ctx, &bdy, &bdx, Cc, H, W, oH, oW),
                op);
            for (int i = 0; i < H * W; ++i) ref[i] = 1.0 / 4.0;
            report(op, &bdx, H * W, ref);
        } else if (!strcmp(op, "maxpooln_bwd")) {
            const int Nw = 2, S = 2;
            const int Ho = (H - Nw) / S + 1, Wo = (W - Nw) / S + 1;
            const int dp = (Wo + 15) & ~15, dr = (Ho + 3) & ~3;
            float dyp[1 * 8 * 16];
            memset(dyp, 0, sizeof(dyp));
            for (int y = 0; y < Ho; ++y)
                for (int xx = 0; xx < Wo; ++xx)
                    dyp[y * dp + xx] = 1.0f;
            vc4_buf bdy = bufs(dr * dp, dyp);
            chk(vc4_maxpooln_bwd(ctx, &bx, &bdy, &bdx, Cc, H, W, Nw, S, 0), op);
            for (int y = 0; y < H; ++y)
                for (int xx = 0; xx < W; ++xx) {
                    /* the max of the 2x2 window is unique here (distinct-ish
                     * values); the gradient goes to the argmax, so the
                     * reference is the count of positions that are a window
                     * maximum */
                    double m = -1e30;
                    for (int ky = 0; ky < Nw; ++ky)
                        for (int kx = 0; kx < Nw; ++kx) {
                            int yy = (y / S) * S + ky, x2 = (xx / S) * S + kx;
                            if (yy < H && x2 < W && xp[yy * W + x2] > m)
                                m = xp[yy * W + x2];
                        }
                    ref[y * W + xx] = (xp[y * W + xx] == m) ? 1.0 : 0.0;
                }
            report(op, &bdx, H * W, ref);
        } else if (!strcmp(op, "resize_nearest_bwd")) {
            float dy1[16];
            for (int i = 0; i < 9; ++i) dy1[i] = 1.0f;
            vc4_buf bdy = bufs(9, dy1);
            chk(vc4_resize_nearest_bwd(ctx, &bdy, &bdx, Cc, H, W, 3, 3), op);
            for (int i = 0; i < H * W; ++i) ref[i] = 0.0;
            for (int oy = 0; oy < 3; ++oy)
                for (int ox = 0; ox < 3; ++ox) {
                    int iy = (int)((int64_t)oy * H / 3);
                    int ix = (int)((int64_t)ox * W / 3);
                    ref[iy * W + ix] += 1.0;
                }
            report(op, &bdx, H * W, ref);
        } else {
            float dy1[256];
            for (int i = 0; i < 144; ++i) dy1[i] = 1.0f;
            vc4_buf bdy = bufs(144, dy1);
            chk(vc4_resize_bilinear_bwd(ctx, &bdy, &bdx, Cc, H, W, 12, 12), op);
            for (int i = 0; i < H * W; ++i) ref[i] = 0.0;
            for (int oy = 0; oy < 12; ++oy) {
                double sy = ((double)oy + 0.5) * H / 12 - 0.5;
                int i0 = (int)floor(sy);
                double fy = sy - i0;
                if (i0 < 0) { i0 = 0; fy = 0; }
                if (i0 >= H - 1) { i0 = H - 1; fy = 0; }
                int i1 = i0 + 1 < H ? i0 + 1 : H - 1;
                for (int ox = 0; ox < 12; ++ox) {
                    double sx = ((double)ox + 0.5) * W / 12 - 0.5;
                    int j0 = (int)floor(sx);
                    double fx = sx - j0;
                    if (j0 < 0) { j0 = 0; fx = 0; }
                    if (j0 >= W - 1) { j0 = W - 1; fx = 0; }
                    int j1 = j0 + 1 < W ? j0 + 1 : W - 1;
                    ref[i0 * W + j0] += (1 - fy) * (1 - fx);
                    ref[i0 * W + j1] += (1 - fy) * fx;
                    ref[i1 * W + j0] += fy * (1 - fx);
                    ref[i1 * W + j1] += fy * fx;
                }
            }
            report(op, &bdx, H * W, ref);
        }
    } else if (!strcmp(op, "binop_bwd")) {
        /* (4,5) * (4,1) with broadcasting */
        float xg[4 * 5 + 4], dy[4 * 5];
        fill(xg, 4 * 5 + 4, -1.0f, 2.0f);
        fill(dy, 4 * 5, -1.0f, 2.0f);
        int od[2] = { 4, 5 }, xd[2] = { 4, 5 }, yd[2] = { 4, 1 };
        vc4_buf bx = bufs(20, xg), by = bufs(4, xg + 20), bg = bufs(20, dy);
        vc4_buf bdx = bufs(20, NULL), bdy = bufs(4, NULL);
        chk(vc4_binop_bwd(ctx, VC4_BIN_MUL, &bx, &by, &bg, od, xd, yd, 2, &bdx,
                          &bdy), op);
        for (int i = 0; i < 20; ++i) ref[i] = dy[i] * (double)xg[20 + i / 5];
        for (int i = 0; i < 4; ++i) {
            double s = 0;
            for (int j = 0; j < 5; ++j) s += dy[i * 5 + j] * xg[i * 5 + j];
            ref[20 + i] = s;
        }
        report("binop dx", &bdx, 20, ref);
        report("binop dy", &bdy, 4, ref + 20);
    } else if (!strcmp(op, "cat_bwd") || !strcmp(op, "split_bwd")) {
        const int outer = 2, inner = 3, nsrc = 2;
        int ad[2] = { 2, 4 };
        const size_t tot = (size_t)outer * 6 * inner;
        float dyt[tot];
        fill(dyt, tot, -1.0f, 2.0f);
        vc4_buf bdy = bufs(tot, dyt);
        if (!strcmp(op, "cat_bwd")) {
            vc4_buf p0 = bufs((size_t)outer * 2 * inner, NULL);
            vc4_buf p1 = bufs((size_t)outer * 4 * inner, NULL);
            vc4_buf *pp[2] = { &p0, &p1 };
            chk(vc4_cat_bwd(ctx, &bdy, pp, ad, nsrc, outer, inner), op);
            for (int o = 0; o < outer; ++o)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < inner; ++k)
                        ref[(size_t)(o * 2 + j) * inner + k] =
                            dyt[((size_t)o * 6 + j) * inner + k];
            report("cat part0", &p0, (size_t)outer * 2 * inner, ref);
            for (int o = 0; o < outer; ++o)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < inner; ++k)
                        ref[(size_t)(o * 4 + j) * inner + k] =
                            dyt[((size_t)o * 6 + 2 + j) * inner + k];
            report("cat part1", &p1, (size_t)outer * 4 * inner, ref);
        } else {
            vc4_buf p0 = bufs((size_t)outer * 2 * inner, dyt);
            float d1[(size_t)2 * 4 * 3];
            fill(d1, (size_t)outer * 4 * inner, -1.0f, 2.0f);
            vc4_buf p1 = bufs((size_t)outer * 4 * inner, d1);
            const vc4_buf *pp[2] = { &p0, &p1 };
            vc4_buf bdx = bufs(tot, NULL);
            chk(vc4_split_bwd(ctx, pp, &bdx, ad, nsrc, outer, inner), op);
            for (int o = 0; o < outer; ++o) {
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < inner; ++k)
                        ref[((size_t)o * 6 + j) * inner + k] =
                            dyt[((size_t)o * 2 + j) * inner + k];
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < inner; ++k)
                        ref[((size_t)o * 6 + 2 + j) * inner + k] =
                            d1[((size_t)o * 4 + j) * inner + k];
            }
            report(op, &bdx, tot, ref);
        }
    } else if (!strcmp(op, "pad_bwd")) {
        const int Nn = 1, Cc = 1, H = 4, W = 5, pt = 1, pl = 2;
        const int Hp = H + 2 * pt, Wp = W + 2 * pl;
        float dy[(4 + 2) * (5 + 4)];
        fill(dy, (size_t)Hp * Wp, -1.0f, 2.0f);
        vc4_buf bdy = bufs((size_t)Hp * Wp, dy), bdx = bufs(H * W, NULL);
        chk(vc4_pad_bwd(ctx, &bdy, &bdx, Nn, Cc, H, W, pt, pl), op);
        for (int y = 0; y < H; ++y)
            for (int xx = 0; xx < W; ++xx)
                ref[y * W + xx] = dy[(size_t)(y + pt) * Wp + xx + pl];
        report(op, &bdx, H * W, ref);
    } else if (!strcmp(op, "transpose_bwd")) {
        float dy[R * C];
        fill(dy, R * C, -1.0f, 2.0f);
        vc4_buf bdy = bufs(R * C, dy), bdx = bufs(R * C, NULL);
        chk(vc4_transpose_bwd(ctx, &bdy, &bdx, R, C), op);
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < C; ++j)
                ref[(size_t)i * C + j] = dy[(size_t)j * R + i];
        report(op, &bdx, R * C, ref);
    } else if (!strcmp(op, "reduce_bwd")) {
        int dims[3] = { 2, 4, 3 };
        float gy[2 * 3];
        fill(gy, 6, -1.0f, 2.0f);
        vc4_buf bg = bufs(6, gy), bdx = bufs(24, NULL);
        chk(vc4_reduce_bwd(ctx, &bg, &bdx, dims, 3, 1, 0), op);
        for (int o = 0; o < 2; ++o)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 3; ++k)
                    ref[(size_t)(o * 4 + j) * 3 + k] = gy[o * 3 + k];
        report(op, &bdx, 24, ref);
    } else if (!strcmp(op, "gather_bwd") || !strcmp(op, "scatter_bwd")) {
        int32_t idx[6] = { 0, 3, 3, 1, 2, 0 };
        float gf[6];
        fill(gf, 6, -1.0f, 2.0f);
        if (!strcmp(op, "gather_bwd")) {
            vc4_buf bg = bufs(6, gf), bdx = bufs(4, NULL);
            chk(vc4_gather_bwd(ctx, &bg, &bdx, idx, 6, 4), op);
            for (int i = 0; i < 4; ++i) ref[i] = 0;
            for (int i = 0; i < 6; ++i) ref[idx[i]] += gf[i];
            report(op, &bdx, 4, ref);
        } else {
            vc4_buf bg = bufs(4, gf), bdx = bufs(6, NULL);
            chk(vc4_scatter_bwd(ctx, &bg, &bdx, idx, 6, 4), op);
            for (int i = 0; i < 6; ++i) ref[i] = gf[idx[i]];
            report(op, &bdx, 6, ref);
        }
    } else if (!strcmp(op, "bwop_run")) {
        fill(x, N, -1.0f, 2.0f);
        fill(g, N, -1.0f, 2.0f);
        vc4_buf bx = bufs(N, x), bg = bufs(N, g), bo = bufs(N, NULL);
        chk(vc4_bwop_run(ctx, VC4_BW_SIGMOID, &bx, &bg, &bo, N, 0, 0),
            "bwop sigmoid");
        for (int i = 0; i < N; ++i) {
            double s = 1.0 / (1.0 + exp(-(double)x[i]));
            ref[i] = (double)g[i] * s * (1 - s);
        }
        report(op, &bo, N, ref);
    } else if (!strcmp(op, "gate")) {
        /* Asserted without launching anything.  vc4_gather is element-verified
         * now, so it must reach its own buffer-size check (a 1000-entry table
         * does not fit in 64 B) instead of the old "not verified" gate.
         * vc4_blit keeps its 64 B granularity refusal. */
        {
            vc4_buf dummy = { 0x1000, 64 }, dout = { 0x2000, 64 };
            vc4_status g = vc4_gather(ctx, &dummy, &dout, 1000, 0.0f);
            if (g != VC4_ERR_INVAL ||
                strstr(vc4_last_error(ctx), "not verified on this board")) {
                printf("FAIL vc4_gather: st=%d %s (expected the size check)\n",
                       g, vc4_last_error(ctx));
                errors++;
            } else {
                printf("  %-11s open, refuses a short table\n", "vc4_gather");
            }
            g = vc4_blit(ctx, &dummy, &dout, 100);
            if (g != VC4_ERR_UNSUPPORTED ||
                !strstr(vc4_last_error(ctx), "multiple of 64") ||
                strstr(vc4_last_error(ctx), "not verified on this board")) {
                printf("FAIL vc4_blit 100 B: st=%d %s (expected the 64 B "
                       "granularity refusal)\n", g, vc4_last_error(ctx));
                errors++;
            } else {
                printf("  %-11s open, refuses 100 B\n", "vc4_blit");
            }
        }
    } else {
        printf("unknown op %s\n", op);
        vc4_shutdown(ctx);
        gpu_unlock(lock);
        return 2;
    }

    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf("%s %s\n", errors ? "WRONG" : "ok", op);
    return errors ? 1 : 0;
}
