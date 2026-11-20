/* test_nhwc.c - NHWC data-plane ops vs a float64 CPU oracle.
 *
 * Every reported element of every run is compared.  The CPU code is the
 * verifier only (it is compiled into the test binary, never the library).
 *
 *   conv (stride 1 and 2, 1x1, 3x3, bias, batch 2)
 *   depthwise 3x3 pad 1
 *   maxpool 5x5 stride 1 pad 2 (SPPF shape)
 *   channel concat, nearest 2x upsample
 *   NCHW -> NHWC layout conversion (C > 16) feeding a grouped conv, which
 *   is the path the ONNX runner takes
 *
 * usage: test_nhwc [reps]
 */
#include "vc4infer.h"
#include "util.h"

static const float TOL_CONV = 3e-4f;   /* fp32 k=576 truncating-add floor */

/* reference NHWC conv: in [H][W][Cin] (pitch), out [ho][Wg][Cout] */
static void ref_conv_nhwc(const float *in, int Cin, int H, int W, int pitch,
                          const float *wt, int Cout, int KH, int KW,
                          const float *bias, int stride, int pad,
                          double *out, int Wg, int ho, int wo, int nimg)
{
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int co = 0; co < Cout; ++co) {
                    double acc = bias ? (double)bias[co] : 0.0;
                    if (x < wo) {
                        for (int ci = 0; ci < Cin; ++ci)
                            for (int ky = 0; ky < KH; ++ky)
                                for (int kx = 0; kx < KW; ++kx) {
                                    int sy = y * stride + ky - pad;
                                    int sx = x * stride + kx - pad;
                                    if (sy < 0 || sy >= H || sx < 0 || sx >= W)
                                        continue;
                                    acc += (double)in[(((size_t)n * H + sy) * pitch + sx) *
                                                      Cin + ci] *
                                           (double)wt[(((size_t)co * Cin + ci) * KH + ky) *
                                                     KW + kx];
                                }
                    } else {
                        acc = 0.0;
                    }
                    out[(((size_t)n * ho + y) * Wg + x) * Cout + co] = acc;
                }
}

static void ref_depthwise(const float *in, int C, int H, int W, int pitch,
                          const float *wt, double *out, int Wg, int nimg)
{
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int c = 0; c < C; ++c) {
                    double acc = 0.0;
                    if (x < W)
                        for (int ky = 0; ky < 3; ++ky)
                            for (int kx = 0; kx < 3; ++kx) {
                                int sy = y + ky - 1, sx = x + kx - 1;
                                if (sy < 0 || sy >= H || sx < 0 || sx >= W)
                                    continue;
                                acc += (double)in[(((size_t)n * H + sy) * pitch + sx) *
                                                  C + c] *
                                       (double)wt[((size_t)c * 9) + ky * 3 + kx];
                            }
                    out[(((size_t)n * H + y) * Wg + x) * C + c] = acc;
                }
}

static void ref_pool(const float *in, int C, int H, int W, int pitch, int k,
                     int pad, double *out, int Wg, int ho, int wo, int nimg)
{
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int c = 0; c < C; ++c) {
                    double m = -1e300;
                    if (x < wo) {
                        for (int ky = 0; ky < k; ++ky)
                            for (int kx = 0; kx < k; ++kx) {
                                int sy = y + ky - pad, sx = x + kx - pad;
                                double v;
                                if (sy < 0 || sy >= H || sx < 0 || sx >= W)
                                    v = 0.0;          /* zero padding, as torch */
                                else
                                    v = in[(((size_t)n * H + sy) * pitch + sx) * C + c];
                                if (v > m)
                                    m = v;
                            }
                    } else {
                        m = 0.0;
                    }
                    out[(((size_t)n * ho + y) * Wg + x) * C + c] = m;
                }
}


/* compare only the logical columns [0,wo); columns [wo,Wg) are kernel junk */
static void check_plane(check_t *ck, const float *got, const double *ref,
                        int nimg, int rows, int Wg, int C, int wo, double tol)
{
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int c = 0; c < C; ++c) {
                    size_t i = (((size_t)n * rows + y) * Wg + x) * C + c;
                    if (x < wo)
                        check_elem(ck, got[i], ref[i], tol, i < 8);
                }
}

/* ------------------------------------------------------------------ helpers */
static vc4_buf alloc_up(vc4_ctx *ctx, const float *src, size_t n, int *ok)
{
    vc4_buf b = { 0, 0 };
    if (vc4_alloc(ctx, n * 4, &b) != VC4_OK) {
        *ok = 0;
        return b;
    }
    if (src)
        vc4_upload(ctx, &b, src, n * 4);
    return b;
}

static void wpanel_alloc(vc4_ctx *ctx, int Cin, int Cout, int KH, int KW,
                         int bias, vc4_buf *wp, int *ok)
{
    size_t bytes = vc4_conv_nhwc_wpanel_bytes(Cin, Cout, KH, KW, bias);
    if (!bytes || vc4_alloc(ctx, bytes, wp) != VC4_OK)
        *ok = 0;
}

/* --------------------------------------------------------------- conv test */
static int one_conv(vc4_ctx *ctx, int Cin, int H, int W, int Cout, int KH,
                    int KW, int stride, int pad, int bias, int nimg, int reps)
{
    const int ho = (H + 2 * pad - KH) / stride + 1;
    const int wo = (W + 2 * pad - KW) / stride + 1;
    const int Wg = (wo + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * Cin;
    const size_t nwt = (size_t)Cout * Cin * KH * KW;
    const size_t nout = (size_t)nimg * ho * Wg * Cout;

    float *in = malloc(nin * 4), *wt = malloc(nwt * 4), *bi = malloc((size_t)Cout * 4);
    float *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !wt || !bi || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand() * 0.15f;
    for (int i = 0; i < Cout; ++i) bi[i] = frand();
    memset(out, 0xCD, nout * 4);

    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bwt = alloc_up(ctx, wt, nwt, &ok);
    vc4_buf bbi = alloc_up(ctx, bi, (size_t)Cout, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    vc4_buf wp;
    wpanel_alloc(ctx, Cin, Cout, KH, KW, bias, &wp, &ok);
    if (!ok)
        return 1;
    if (vc4_conv_nhwc_pack_w(ctx, &bwt, Cin, Cout, KH, KW,
                             bias ? &bbi : NULL, &wp) != VC4_OK) {
        printf("FAIL: pack_w %d->%d %dx%d: %s\n", Cin, Cout, KH, KW,
               vc4_last_error(ctx));
        return 1;
    }
    memset(ref, 0, nout * sizeof(double));
    ref_conv_nhwc(in, Cin, H, W, W, wt, Cout, KH, KW, bias ? bi : NULL, stride,
                  pad, ref, Wg, ho, wo, nimg);

    int clean = 0;
    double best_ms = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);   /* 0xCD poison */
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_conv_nhwc(ctx, &bin, nimg, Cin, H, W, W, &wp, Cout,
                                      KH, KW, stride, pad, &bout, &gho, &gwo, &gp);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: conv_nhwc(%d,%d,%d / %d %dx%d s%d p%d) -> %s (%s)\n",
                   Cin, H, W, Cout, KH, KW, stride, pad, vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        if (gho != ho || gwo != wo || gp != Wg) {
            printf("FAIL: conv geometry %dx%d p%d, expected %dx%d p%d\n",
                   gho, gwo, gp, ho, wo, Wg);
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, ho, Wg, Cout, wo, TOL_CONV);
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall_ms = (double)(t1 - t0) / 1e6;
        double flops = 2.0 * nimg * Cout * ho * wo * Cin * KH * KW;
        if (rep == 0 || wall_ms < best_ms)
            best_ms = wall_ms;
        printf("  conv %d->%d %dx%d k%d s%d p%d n%d rep%d: mism=%ld/%zu "
               "maxabs=%.3g maxrel=%.3g | gpu %.2f ms %.2f GFLOP/s wall %.2f ms"
               " launches %llu\n",
               Cin, Cout, H, W, KH, stride, pad, nimg, rep, ck.mismatches, nout,
               ck.max_abs, ck.max_rel, gpu_ms, gpu_ms > 0 ? flops / gpu_ms / 1e6 : 0,
               wall_ms, (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s conv %d->%d %dx%d k%d s%d p%d n%d: %d/%d clean, best wall %.2f ms\n",
           clean == reps ? "PASS" : "FAIL", Cin, Cout, H, W, KH, stride, pad,
           nimg, clean, reps, best_ms);
    return clean == reps ? 0 : 1;
}

/* ---------------------------------------------- dilated / grouped conv test */
/* Same oracle as ref_conv_nhwc with an explicit dilation. */
static void ref_conv_dil(const float *in, int Cin, int H, int W, int pitch,
                         const float *wt, int Cout, int KH, int KW,
                         const float *bias, int stride, int pad, int dil,
                         double *out, int Wg, int ho, int wo, int nimg)
{
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int co = 0; co < Cout; ++co) {
                    double acc = bias ? (double)bias[co] : 0.0;
                    if (x < wo) {
                        for (int ci = 0; ci < Cin; ++ci)
                            for (int ky = 0; ky < KH; ++ky)
                                for (int kx = 0; kx < KW; ++kx) {
                                    int sy = y * stride + ky * dil - pad;
                                    int sx = x * stride + kx * dil - pad;
                                    if (sy < 0 || sy >= H || sx < 0 || sx >= W)
                                        continue;
                                    acc += (double)in[(((size_t)n * H + sy) * pitch + sx) *
                                                      Cin + ci] *
                                           (double)wt[(((size_t)co * Cin + ci) * KH + ky) *
                                                     KW + kx];
                                }
                    } else {
                        acc = 0.0;
                    }
                    out[(((size_t)n * ho + y) * Wg + x) * Cout + co] = acc;
                }
}

static int one_conv_dil(vc4_ctx *ctx, int Cin, int H, int W, int Cout, int KH,
                        int KW, int stride, int pad, int dil, int nimg,
                        int reps)
{
    const int ho = (H + 2 * pad - dil * (KH - 1) - 1) / stride + 1;
    const int wo = (W + 2 * pad - dil * (KW - 1) - 1) / stride + 1;
    const int Wg = (wo + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * Cin;
    const size_t nwt = (size_t)Cout * Cin * KH * KW;
    const size_t nout = (size_t)nimg * ho * Wg * Cout;

    float *in = malloc(nin * 4), *wt = malloc(nwt * 4);
    float *bi = malloc((size_t)Cout * 4), *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !wt || !bi || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand() * 0.15f;
    for (int i = 0; i < Cout; ++i) bi[i] = frand();

    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bwt = alloc_up(ctx, wt, nwt, &ok);
    vc4_buf bbi = alloc_up(ctx, bi, (size_t)Cout, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    vc4_buf wp;
    wpanel_alloc(ctx, Cin, Cout, KH, KW, 1, &wp, &ok);
    if (!ok)
        return 1;
    if (vc4_conv_nhwc_pack_w(ctx, &bwt, Cin, Cout, KH, KW, &bbi, &wp) !=
        VC4_OK) {
        printf("FAIL: pack_w dil: %s\n", vc4_last_error(ctx));
        return 1;
    }
    ref_conv_dil(in, Cin, H, W, W, wt, Cout, KH, KW, bi, stride, pad, dil, ref,
                 Wg, ho, wo, nimg);

    int clean = 0;
    double best_ms = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_conv_nhwc_dil(ctx, &bin, nimg, Cin, H, W, W, &wp,
                                          Cout, KH, KW, stride, pad, dil, &bout,
                                          &gho, &gwo, &gp);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: conv_dil(%d,%d,%d / %d k%d s%d p%d d%d) -> %s (%s)\n",
                   Cin, H, W, Cout, KH, stride, pad, dil, vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        if (gho != ho || gwo != wo || gp != Wg) {
            printf("FAIL: conv_dil geometry %dx%d p%d, expected %dx%d p%d\n",
                   gho, gwo, gp, ho, wo, Wg);
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, ho, Wg, Cout, wo, TOL_CONV);
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall_ms = (double)(t1 - t0) / 1e6;
        double flops = 2.0 * nimg * Cout * ho * wo * Cin * KH * KW;
        if (rep == 0 || wall_ms < best_ms)
            best_ms = wall_ms;
        printf("  convd %d->%d %dx%d k%d s%d p%d d%d n%d rep%d: mism=%ld/%zu "
               "maxabs=%.3g maxrel=%.3g | gpu %.2f ms %.2f GFLOP/s wall %.2f ms"
               " launches %llu\n",
               Cin, Cout, H, W, KH, stride, pad, dil, nimg, rep, ck.mismatches,
               nout, ck.max_abs, ck.max_rel, gpu_ms,
               gpu_ms > 0 ? flops / gpu_ms / 1e6 : 0, wall_ms,
               (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s convd %d->%d %dx%d k%d s%d p%d d%d n%d: %d/%d clean, best "
           "wall %.2f ms\n", clean == reps ? "PASS" : "FAIL", Cin, Cout, H, W,
           KH, stride, pad, dil, nimg, clean, reps, best_ms);
    return clean == reps ? 0 : 1;
}

static void ref_group(const float *in, int Cin, int H, int W, int pitch,
                      const float *wt, int Cout, int groups, int KH, int KW,
                      const float *bias, int stride, int pad, double *out,
                      int Wg, int ho, int wo, int nimg)
{
    const int cg = Cin / groups, og = Cout / groups;
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < ho; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int co = 0; co < Cout; ++co) {
                    const int g = co / og, cl = co % og;
                    double acc = bias ? (double)bias[co] : 0.0;
                    if (x < wo) {
                        for (int ci = 0; ci < cg; ++ci)
                            for (int ky = 0; ky < KH; ++ky)
                                for (int kx = 0; kx < KW; ++kx) {
                                    int sy = y * stride + ky - pad;
                                    int sx = x * stride + kx - pad;
                                    if (sy < 0 || sy >= H || sx < 0 || sx >= W)
                                        continue;
                                    acc += (double)in[(((size_t)n * H + sy) * pitch + sx) *
                                                      Cin + g * cg + ci] *
                                           (double)wt[(((size_t)co * cg + ci) * KH + ky) *
                                                     KW + kx];
                                }
                    } else {
                        acc = 0.0;
                    }
                    out[(((size_t)n * ho + y) * Wg + x) * Cout + co] = acc;
                }
}

static int one_group(vc4_ctx *ctx, int Cin, int Cout, int groups, int H, int W,
                     int KH, int KW, int stride, int pad, int nimg, int reps)
{
    const int cg = Cin / groups, og = Cout / groups;
    const int ho = (H + 2 * pad - KH) / stride + 1;
    const int wo = (W + 2 * pad - KW) / stride + 1;
    const int Wg = (wo + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * Cin;
    const size_t nwt = (size_t)Cout * cg * KH * KW;
    const size_t nout = (size_t)nimg * ho * Wg * Cout;
    const size_t nwg = (size_t)og * cg * KH * KW;

    float *in = malloc(nin * 4), *wt = malloc(nwt * 4);
    float *bi = malloc((size_t)Cout * 4), *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !wt || !bi || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand() * 0.15f;
    for (int i = 0; i < Cout; ++i) bi[i] = frand();

    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    vc4_buf *bwt = calloc((size_t)groups, sizeof(*bwt));
    vc4_buf *bwp = calloc((size_t)groups, sizeof(*bwp));
    vc4_buf *bbi = calloc((size_t)groups, sizeof(*bbi));
    const vc4_buf **wpl = calloc((size_t)groups, sizeof(*wpl));
    for (int g = 0; g < groups && ok; ++g) {
        bwt[g] = alloc_up(ctx, wt + (size_t)g * nwg, nwg, &ok);
        bbi[g] = alloc_up(ctx, bi + (size_t)g * og, (size_t)og, &ok);
        wpanel_alloc(ctx, cg, og, KH, KW, 1, &bwp[g], &ok);
        wpl[g] = &bwp[g];
        if (ok && vc4_conv_nhwc_pack_w(ctx, &bwt[g], cg, og, KH, KW, &bbi[g],
                                       &bwp[g]) != VC4_OK) {
            printf("FAIL: group pack_w: %s\n", vc4_last_error(ctx));
            return 1;
        }
    }
    if (!ok)
        return 1;
    ref_group(in, Cin, H, W, W, wt, Cout, groups, KH, KW, bi, stride, pad, ref,
              Wg, ho, wo, nimg);

    int clean = 0;
    double best_ms = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_conv_nhwc_group(ctx, &bin, nimg, Cin, H, W, W, wpl,
                                            Cout, groups, KH, KW, stride, pad, 1,
                                            &bout, &gho, &gwo, &gp);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: conv_group(%d->%d g%d %dx%d) -> %s (%s)\n", Cin, Cout,
                   groups, H, W, vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        if (gho != ho || gwo != wo || gp != Wg) {
            printf("FAIL: conv_group geometry %dx%d p%d, expected %dx%d p%d\n",
                   gho, gwo, gp, ho, wo, Wg);
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, ho, Wg, Cout, wo, TOL_CONV);
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall_ms = (double)(t1 - t0) / 1e6;
        double flops = 2.0 * nimg * Cout * ho * wo * cg * KH * KW;
        if (rep == 0 || wall_ms < best_ms)
            best_ms = wall_ms;
        printf("  convg %d->%d g%d %dx%d k%d s%d p%d n%d rep%d: mism=%ld/%zu "
               "maxabs=%.3g maxrel=%.3g | gpu %.2f ms %.2f GFLOP/s wall %.2f ms "
               "launches %llu\n",
               Cin, Cout, groups, H, W, KH, stride, pad, nimg, rep, ck.mismatches,
               nout, ck.max_abs, ck.max_rel, gpu_ms,
               gpu_ms > 0 ? flops / gpu_ms / 1e6 : 0, wall_ms,
               (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s convg %d->%d g%d %dx%d k%d s%d p%d n%d: %d/%d clean, best "
           "wall %.2f ms\n", clean == reps ? "PASS" : "FAIL", Cin, Cout, groups,
           H, W, KH, stride, pad, nimg, clean, reps, best_ms);
    return clean == reps ? 0 : 1;
}

/* ------------------------------------- NCHW -> NHWC layout, then grouped conv */
/*
 * The ONNX runner holds activations in NCHW and converts them on the QPUs
 * before an NHWC conv: 16 lanes are 16 channels, and one copy2d_x call moves
 * one 16-channel block of every pixel (an NCHW plane of H*W floats is the lane
 * stride).  Cin > 16 therefore needs one call per channel block -- copy2d_x
 * rejects nlane > 16, which is exactly how the 32->32 g2 Conv of the ONNX sweep
 * failed ("copy2d_x: invalid argument").  one_group() above starts from an NHWC
 * buffer built on the ARM, so it never exercised the conversion.
 */
static int one_group_nchw(vc4_ctx *ctx, int Cin, int Cout, int groups, int H,
                          int W, int KH, int KW, int stride, int pad, int nimg,
                          int reps)
{
    const int cg = Cin / groups, og = Cout / groups;
    const int ho = (H + 2 * pad - KH) / stride + 1;
    const int wo = (W + 2 * pad - KW) / stride + 1;
    const int Wg = (wo + 15) & ~15;
    const int cp = Cin;
    const size_t nin = (size_t)nimg * Cin * H * W;        /* NCHW */
    const size_t nn = (size_t)nimg * H * W * cp;          /* NHWC */
    const size_t nwt = (size_t)Cout * cg * KH * KW;
    const size_t nout = (size_t)nimg * ho * Wg * Cout;
    const size_t nwg = (size_t)og * cg * KH * KW;
    const uint32_t units = (uint32_t)(W % 4 == 0 ? 4 : (W % 2 == 0 ? 2 : 1));

    float *nchw = malloc(nin * 4), *nhwc = malloc(nn * 4);
    float *wt = malloc(nwt * 4), *bi = malloc((size_t)Cout * 4);
    float *out = malloc(nout * 4), *got = malloc(nn * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!nchw || !nhwc || !wt || !bi || !out || !got || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) nchw[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand() * 0.15f;
    for (int i = 0; i < Cout; ++i) bi[i] = frand();
    for (int n = 0; n < nimg; ++n)
        for (int c = 0; c < Cin; ++c)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    nhwc[(((size_t)n * H + y) * W + x) * cp + c] =
                        nchw[((size_t)n * Cin + c) * H * W + (size_t)y * W + x];

    vc4_buf bn = alloc_up(ctx, nchw, nin, &ok);
    vc4_buf bt = alloc_up(ctx, NULL, nn, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    vc4_buf *bwt = calloc((size_t)groups, sizeof(*bwt));
    vc4_buf *bwp = calloc((size_t)groups, sizeof(*bwp));
    vc4_buf *bbi = calloc((size_t)groups, sizeof(*bbi));
    const vc4_buf **wpl = calloc((size_t)groups, sizeof(*wpl));
    for (int g = 0; g < groups && ok; ++g) {
        bwt[g] = alloc_up(ctx, wt + (size_t)g * nwg, nwg, &ok);
        bbi[g] = alloc_up(ctx, bi + (size_t)g * og, (size_t)og, &ok);
        wpanel_alloc(ctx, cg, og, KH, KW, 1, &bwp[g], &ok);
        wpl[g] = &bwp[g];
        if (ok && vc4_conv_nhwc_pack_w(ctx, &bwt[g], cg, og, KH, KW, &bbi[g],
                                       &bwp[g]) != VC4_OK) {
            printf("FAIL: group_nchw pack_w: %s\n", vc4_last_error(ctx));
            return 1;
        }
    }
    if (!ok)
        return 1;
    /* pitch is in pixels, the channel block is not part of it */
    ref_group(nhwc, Cin, H, W, W, wt, Cout, groups, KH, KW, bi, stride, pad,
              ref, Wg, ho, wo, nimg);

    int clean = 0;
    double best_ms = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        uint64_t t0 = now_ns();
        /* ---- the NCHW -> NHWC conversion, one call per 16-channel block */
        for (int n = 0; n < nimg; ++n)
            for (int k0 = 0; k0 < Cin; k0 += 16) {
                const int nlan = Cin - k0 < 16 ? Cin - k0 : 16;
                const uint32_t so =
                    (uint32_t)(((size_t)n * Cin + k0) * H * W * 4);
                const uint32_t dof =
                    (uint32_t)((size_t)n * H * W * cp * 4 + (size_t)k0 * 4);
                vc4_buf s = bn, d = bt;
                s.off = bn.off + so;
                s.bytes = bn.bytes - so;
                d.off = bt.off + dof;
                d.bytes = bt.bytes - dof;
                vc4_status cst = vc4_copy2d_x(
                    ctx, &s, &d, (uint32_t)H, (uint32_t)W, (uint32_t)(W * 4),
                    (uint32_t)(W * cp * 4), 4, (uint32_t)(cp * 4),
                    (uint32_t)(H * W * 4), (uint32_t)nlan, 1, 64, 64, units,
                    16);
                if (cst != VC4_OK) {
                    printf("FAIL: nchw->nhwc block %d C%d %dx%d -> %s (%s)\n",
                           k0 / 16, Cin, H, W, vc4_strerror(cst),
                           vc4_last_error(ctx));
                    return 1;
                }
            }
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_conv_nhwc_group(ctx, &bt, nimg, Cin, H, W, W, wpl,
                                            Cout, groups, KH, KW, stride, pad, 1,
                                            &bout, &gho, &gwo, &gp);
        uint64_t t1 = now_ns();
        if (st != VC4_OK) {
            printf("FAIL: conv_group_nchw(%d->%d g%d %dx%d) -> %s (%s)\n", Cin,
                   Cout, groups, H, W, vc4_strerror(st), vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        /* the conversion is a byte copy: it must be exact */
        vc4_download(ctx, &bt, got, nn * 4);
        long lay = 0;
        for (size_t i = 0; i < nn; ++i)
            if (got[i] != nhwc[i])
                ++lay;
        if (lay) {
            printf("FAIL: nchw->nhwc C%d %dx%d: %ld/%zu words differ\n", Cin, H,
                   W, lay, nn);
            return 1;
        }
        if (gho != ho || gwo != wo || gp != Wg) {
            printf("FAIL: conv_group_nchw geometry %dx%d p%d, expected %dx%d "
                   "p%d\n", gho, gwo, gp, ho, wo, Wg);
            return 1;
        }
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, ho, Wg, Cout, wo, TOL_CONV);
        double gpu_ms = (double)((s1.pack_ns - s0.pack_ns) +
                                 (s1.kernel_ns - s0.kernel_ns)) / 1e6;
        double wall_ms = (double)(t1 - t0) / 1e6;
        double flops = 2.0 * nimg * Cout * ho * wo * cg * KH * KW;
        if (rep == 0 || wall_ms < best_ms)
            best_ms = wall_ms;
        printf("  convn %d->%d g%d %dx%d k%d s%d p%d n%d rep%d: mism=%ld/%zu "
               "maxabs=%.3g maxrel=%.3g | gpu %.2f ms %.2f GFLOP/s wall %.2f ms "
               "launches %llu\n",
               Cin, Cout, groups, H, W, KH, stride, pad, nimg, rep,
               ck.mismatches, nout, ck.max_abs, ck.max_rel, gpu_ms,
               gpu_ms > 0 ? flops / gpu_ms / 1e6 : 0, wall_ms,
               (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s convn %d->%d g%d %dx%d k%d s%d p%d n%d: %d/%d clean, best "
           "wall %.2f ms\n", clean == reps ? "PASS" : "FAIL", Cin, Cout, groups,
           H, W, KH, stride, pad, nimg, clean, reps, best_ms);
    return clean == reps ? 0 : 1;
}

/* ---------------------------------------------------------- depthwise test */
static int one_depthwise(vc4_ctx *ctx, int C, int H, int W, int nimg, int reps)
{
    const int Wg = (W + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * C;
    const size_t nwt = (size_t)C * 9;
    const size_t nout = (size_t)nimg * H * Wg * C;
    float *in = malloc(nin * 4), *wt = malloc(nwt * 4), *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !wt || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    for (size_t i = 0; i < nwt; ++i) wt[i] = frand();
    memset(out, 0xCD, nout * 4);

    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bwt = alloc_up(ctx, wt, nwt, &ok);
    vc4_buf bw = alloc_up(ctx, NULL, vc4_depthwise_wpanel_bytes(C)/4, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    if (!ok)
        return 1;
    if (vc4_depthwise_pack_w(ctx, &bwt, C, &bw) != VC4_OK) {
        printf("FAIL: depthwise_pack_w: %s\n", vc4_last_error(ctx));
        return 1;
    }
    ref_depthwise(in, C, H, W, W, wt, ref, Wg, nimg);

    int clean = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_depthwise_nhwc(ctx, &bin, nimg, C, H, W, W, &bw,
                                           &bout, &gho, &gwo, &gp);
        if (st != VC4_OK) {
            printf("FAIL: depthwise -> %s (%s)\n", vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, H, Wg, C, W, TOL_CONV);
        printf("  depthwise c%d %dx%d n%d rep%d: mism=%ld/%zu maxabs=%.3g "
               "maxrel=%.3g launches %llu\n", C, H, W, nimg, rep, ck.mismatches,
               nout, ck.max_abs, ck.max_rel,
               (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s depthwise c%d %dx%d n%d: %d/%d clean\n",
           clean == reps ? "PASS" : "FAIL", C, H, W, nimg, clean, reps);
    return clean == reps ? 0 : 1;
}

/* --------------------------------------------------------------- pool test */
static int one_pool(vc4_ctx *ctx, int C, int H, int W, int k, int pad,
                    int nimg, int reps)
{
    const int ho = H + 2 * pad - k + 1, wo = W + 2 * pad - k + 1;
    const int Wg = (wo + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * C;
    const size_t nout = (size_t)nimg * ho * Wg * C;
    float *in = malloc(nin * 4), *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    memset(out, 0xCD, nout * 4);
    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    if (!ok)
        return 1;
    ref_pool(in, C, H, W, W, k, pad, ref, Wg, ho, wo, nimg);
    int clean = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_maxpool_nhwc(ctx, &bin, nimg, C, H, W, W, k, pad,
                                         &bout, &gho, &gwo, &gp);
        if (st != VC4_OK) {
            printf("FAIL: pool -> %s (%s)\n", vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, ho, Wg, C, wo, 1e-6);
        printf("  pool %dx%d p%d c%d n%d rep%d: mism=%ld/%zu maxrel=%.3g "
               "launches %llu\n", k, k, pad, C, nimg, rep, ck.mismatches, nout,
               ck.max_rel, (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s pool %dx%d p%d c%d n%d: %d/%d clean\n",
           clean == reps ? "PASS" : "FAIL", k, k, pad, C, nimg, clean, reps);
    return clean == reps ? 0 : 1;
}

/* ------------------------------------------------------------- concat test */
static int one_concat(vc4_ctx *ctx, int nimg, int M, const int *cs, int nsrc,
                      int reps)
{
    int ctot = 0;
    for (int i = 0; i < nsrc; ++i)
        ctot += cs[i];
    float *src[8];
    vc4_buf b[8];
    int ok = 1;
    for (int i = 0; i < nsrc; ++i) {
        src[i] = malloc((size_t)nimg * M * cs[i] * 4);
        if (!src[i])
            return 1;
        for (size_t j = 0; j < (size_t)nimg * M * cs[i]; ++j)
            src[i][j] = frand();
        b[i] = alloc_up(ctx, src[i], (size_t)nimg * M * cs[i], &ok);
    }
    const size_t nout = (size_t)nimg * M * ctot;
    float *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    if (!out || !ref || !ok)
        return 1;
    memset(out, 0xCD, nout * 4);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    if (!ok)
        return 1;
    for (int n = 0; n < nimg; ++n)
        for (int p = 0; p < M; ++p) {
            size_t o = ((size_t)n * M + p) * ctot;
            int co = 0;
            for (int i = 0; i < nsrc; ++i) {
                for (int c = 0; c < cs[i]; ++c)
                    ref[o + co + c] = src[i][((size_t)n * M + p) * cs[i] + c];
                co += cs[i];
            }
        }
    int clean = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        const vc4_buf *bp[8];
        for (int i = 0; i < nsrc; ++i)
            bp[i] = &b[i];
        vc4_status st = vc4_concat_nhwc(ctx, bp, cs, nsrc, nimg, (size_t)M,
                                        &bout, ctot);
        if (st != VC4_OK) {
            printf("FAIL: concat -> %s (%s)\n", vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        for (size_t i = 0; i < nout; ++i)
            check_elem(&ck, out[i], ref[i], 1e-7, i < 8);
        printf("  concat ->%d M=%d n%d rep%d: mism=%ld/%zu maxrel=%.3g "
               "launches %llu\n", ctot, M, nimg, rep, ck.mismatches, nout,
               ck.max_rel, (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s concat ->%d M=%d n%d: %d/%d clean\n",
           clean == reps ? "PASS" : "FAIL", ctot, M, nimg, clean, reps);
    return clean == reps ? 0 : 1;
}

/* ----------------------------------------------------------- upsample test */
static int one_up2(vc4_ctx *ctx, int C, int H, int W, int nimg, int reps)
{
    const int Wg = (2 * W + 15) & ~15;
    const size_t nin = (size_t)nimg * H * W * C;
    const size_t nout = (size_t)nimg * 2 * H * Wg * C;
    float *in = malloc(nin * 4), *out = malloc(nout * 4);
    double *ref = malloc(nout * sizeof(double));
    int ok = 1;
    if (!in || !out || !ref)
        return 1;
    for (size_t i = 0; i < nin; ++i) in[i] = frand();
    memset(out, 0xCD, nout * 4);
    vc4_buf bin = alloc_up(ctx, in, nin, &ok);
    vc4_buf bout = alloc_up(ctx, out, nout, &ok);
    if (!ok)
        return 1;
    for (int n = 0; n < nimg; ++n)
        for (int y = 0; y < 2 * H; ++y)
            for (int x = 0; x < Wg; ++x)
                for (int c = 0; c < C; ++c) {
                    double v = 0.0;
                    if (x < 2 * W)
                        v = in[((size_t)n * H + y / 2) * W * C + (x / 2) * C + c];
                    ref[(((size_t)n * 2 * H + y) * Wg + x) * C + c] = v;
                }
    int clean = 0;
    for (int rep = 0; rep < reps; ++rep) {
        vc4_upload(ctx, &bout, out, nout * 4);
        vc4_stats s0, s1;
        vc4_get_stats(ctx, &s0);
        int gho = 0, gwo = 0, gp = 0;
        vc4_status st = vc4_upsample2x_nhwc(ctx, &bin, nimg, C, H, W, W, &bout,
                                            &gho, &gwo, &gp);
        if (st != VC4_OK) {
            printf("FAIL: up2 -> %s (%s)\n", vc4_strerror(st),
                   vc4_last_error(ctx));
            return 1;
        }
        vc4_get_stats(ctx, &s1);
        vc4_download(ctx, &bout, out, nout * 4);
        check_t ck = { 0, 0, 0, 1 };
        check_plane(&ck, out, ref, nimg, 2 * H, Wg, C, 2 * W, 1e-7);
        printf("  up2 c%d %dx%d n%d rep%d: mism=%ld/%zu launches %llu\n",
               C, H, W, nimg, rep, ck.mismatches, nout,
               (unsigned long long)(s1.launches - s0.launches));
        if (ck.mismatches == 0)
            ++clean;
    }
    printf("%s up2 c%d %dx%d n%d: %d/%d clean\n",
           clean == reps ? "PASS" : "FAIL", C, H, W, nimg, clean, reps);
    return clean == reps ? 0 : 1;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    int lock = gpu_lock();
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) {
        printf("FAIL: vc4_init\n");
        return 1;
    }
    printf("libvc4infer %s NHWC ops  arena %.1f MiB  reps=%d\n",
           VC4INFER_VERSION_STRING, vc4_arena_bytes(ctx) / 1048576.0, reps);

    int bad = 0;
    /* 1. the pitch blocker: 64x64x64 (16 KB NCHW plane) in channel-tiled NHWC */
    bad |= one_conv(ctx, 64, 64, 64, 16, 3, 3, 1, 1, 1, 1, reps);
    bad |= one_conv(ctx, 64, 64, 64, 32, 3, 3, 1, 1, 1, 2, reps);
    /* 2. stride 2 (decimation by source pitch) and 1x1 */
    bad |= one_conv(ctx, 32, 32, 32, 32, 3, 3, 2, 1, 1, 1, reps);
    bad |= one_conv(ctx, 32, 32, 32, 16, 3, 3, 2, 1, 0, 1, reps);
    bad |= one_conv(ctx, 32, 24, 40, 16, 1, 1, 1, 0, 1, 1, reps);
    bad |= one_conv(ctx, 16, 16, 16, 16, 3, 3, 1, 0, 1, 1, reps);
    /* dilation: 3x3 d2 (effective 5x5), 3x3 d2 stride 2, 5x5 d1 */
    bad |= one_conv_dil(ctx, 32, 32, 32, 32, 3, 3, 1, 2, 2, 1, reps);
    bad |= one_conv_dil(ctx, 32, 32, 32, 16, 3, 3, 2, 2, 2, 1, reps);
    bad |= one_conv_dil(ctx, 16, 20, 20, 16, 5, 5, 1, 2, 1, 1, reps);
    bad |= one_conv_dil(ctx, 32, 20, 20, 32, 3, 3, 1, 1, 3, 2, reps);
    /* grouped conv: per-group Cin/Cout multiples of 16 */
    bad |= one_group(ctx, 64, 64, 2, 16, 16, 3, 3, 1, 1, 1, reps);
    bad |= one_group(ctx, 64, 64, 4, 16, 16, 3, 3, 2, 1, 2, reps);
    bad |= one_group(ctx, 96, 96, 3, 16, 16, 3, 3, 1, 1, 2, reps);
    /* the ONNX runner path: NCHW input, converted to NHWC on the QPUs first
     * (Cin > 16 exercises the per-channel-block copy2d_x loop) */
    bad |= one_group_nchw(ctx, 32, 32, 2, 8, 8, 3, 3, 1, 1, 1, reps);
    bad |= one_group_nchw(ctx, 48, 48, 3, 8, 8, 3, 3, 1, 1, 1, reps);
    /* 3. depthwise, SPPF-style pool, concat, upsample */
    bad |= one_depthwise(ctx, 32, 24, 24, 1, reps);
    bad |= one_depthwise(ctx, 64, 16, 16, 2, reps);
    bad |= one_pool(ctx, 32, 20, 20, 5, 2, 1, reps);
    bad |= one_pool(ctx, 32, 20, 20, 5, 2, 2, reps);
    bad |= one_up2(ctx, 32, 8, 8, 1, reps);
    bad |= one_up2(ctx, 64, 8, 8, 2, reps);
    {
        int cs1[3] = { 16, 32, 48 };
        int cs2[2] = { 16, 32 };
        bad |= one_concat(ctx, 1, 64, cs1, 3, reps);
        bad |= one_concat(ctx, 2, 64, cs2, 2, reps);
    }
    vc4_shutdown(ctx);
    gpu_unlock(lock);
    printf("%s\n", bad ? "NHWC OPS FAILED" : "ALL NHWC OPS PASSED");
    return bad ? 1 : 0;
}
