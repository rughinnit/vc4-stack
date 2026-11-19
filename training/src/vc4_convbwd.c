/*
 * vc4_convbwd.c - convolution backward (dX, dW), col2im, and max pooling.
 *
 * All the arithmetic is QPU kernels:
 *
 *   im2col   (row-major form)  A[p][k], p = output pixel, k = tap*Cin + ci
 *   col2im                     dXpad[ci][d] = sum_tap dA[d - tap][tap*Cin+ci]
 *   gemm3 / pack3              dA = dY * W^T,  dW = dY^T * A   (asm3 GEMM)
 *   pool                       max of the 2x2 group, and the argmax-mask
 *                              backward that routes dy to the max element
 *
 * The ARM only moves bytes here too: padded-image fills, the dY -> dYcol
 * (NCHW -> pixel-major) gather, the tap-major weight permutation, the dW
 * un-permutation and the NCHW crop of the padded dX.  None of those touch a
 * multiply-accumulate.
 *
 * Shapes: stride 1, KH == KW in {1,3}, Cin % 16 == 0, Cout % 16 == 0 for the
 * native path (the trainer pads 1 -> 16 channels and 8 -> 16 filters on the
 * host, which is a zero fill plus a copy).
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================================================================
 * padded NCHW image in scratch: zero fill + one row copy per row
 * (layout only -- the same byte work vc4_conv2d does)
 * ================================================================== */
static float *pad_image(vc4_ctx *ctx, const vc4_buf *in, int Cin, int H, int W,
                        int pad, int *hp_out, int *wp_out, uint32_t *bus_out)
{
    const int hp = H + 2 * pad;
    const int wp = (int)vc4__align16((uint32_t)(W + 2 * pad));
    float *p = vc4__scratch_alloc(ctx, (size_t)Cin * hp * wp * 4 + 64, NULL);
    if (!p)
        return NULL;
    memset((uint8_t *)p, 0, (size_t)Cin * hp * wp * 4 + 64);
    for (int c = 0; c < Cin; ++c)
        for (int y = 0; y < H; ++y)
            memcpy((uint8_t *)p + ((size_t)c * hp * wp +
                                   (size_t)(y + pad) * wp + pad) * 4,
                   ctx->arena + in->off + ((size_t)c * H + (size_t)y) * W * 4,
                   (size_t)W * 4);
    *hp_out = hp;
    *wp_out = wp;
    *bus_out = ctx->gpu_bus + (uint32_t)((uint8_t *)p - ctx->arena);
    return p;
}

/* ==================================================================
 * im2col, row-major output [M][K]
 * ================================================================== */
static vc4_status im2col_row_run(vc4_ctx *ctx, uint32_t pad_bus, int plane_bytes,
                                 int Wp, int Cin, int K, int KH, int KW,
                                 uint32_t M, uint32_t dst_bus)
{
    const uint32_t tr = M / 16u;
    int pnq = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
    const uint32_t per = tr / (uint32_t)pnq;

    char ncin[16], nk_[16], nkh[8], nkw[8], name[72];
    char *argv[] = { (char *)"im2col", (char *)"--cin", ncin,
                     (char *)"--k", nk_, (char *)"--kh", nkh,
                     (char *)"--kw", nkw, (char *)"--vout", (char *)"r", NULL };
    snprintf(ncin, sizeof(ncin), "%d", Cin);
    snprintf(nk_, sizeof(nk_), "%d", K);
    snprintf(nkh, sizeof(nkh), "%d", KH);
    snprintf(nkw, sizeof(nkw), "%d", KW);
    snprintf(name, sizeof(name), "im2colr_%d_%d_%d_%d", Cin, K, KH, KW);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, 4, &k);
    if (st != VC4_OK)
        return st;
    for (int i = 0; i < pnq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = (uint32_t)plane_bytes;
        u[1] = (uint32_t)i * per * 64u;
        u[2] = pad_bus;
        u[3] = 64u;
        u[4] = dst_bus;
        u[5] = per;
        u[6] = (uint32_t)i;
        u[7] = 64u;                    /* one 16-k block = 64 B in a row    */
        u[8] = 64u * (uint32_t)K;      /* 16 pixels x K floats              */
        u[9] = (uint32_t)Wp * 4u;
        u[10] = (uint32_t)i * per * 64u * (uint32_t)K;
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, pnq, &ns);
    ctx->st.pack_ns += ns;
    return st;
}

/* ==================================================================
 * col2im: row-major dA [M][K] -> padded dXpad [Cin][Hp][Wp]
 * ================================================================== */
static vc4_status col2im_run(vc4_ctx *ctx, uint32_t dA_bus, int Cin, int K,
                             int KH, int KW, int HO, int WO,
                             uint32_t dx_bus, int *Hp_out, int *Wp_out)
{
    const int Hp = HO + KH - 1;
    const int Wp = (int)vc4__align16((uint32_t)(WO + KW - 1));
    const uint32_t M = (uint32_t)HO * (uint32_t)Wp;
    const uint32_t K64 = 64u * (uint32_t)K;
    const uint32_t plane = (uint32_t)Hp * (uint32_t)Wp * 4u;
    const uint32_t wp16 = (uint32_t)Wp / 16u;
    const uint32_t nsrc = M / 16u;                 /* real dA row blocks     */
    const uint32_t nrb = (uint32_t)Hp * wp16;      /* destination row blocks */
    const uint32_t CB = (uint32_t)Cin / 16u;
    /* Guard blocks: the gather reads source pixels outside [0, M) at the
       padded-image border (kx/ky underflow below, bottom/right rows above). */
    const uint32_t G = (uint32_t)(KH - 1) * wp16 + 2u;
    const uint32_t T = G;

    vc4_buf pb;
    uint8_t *panel = vc4__scratch_alloc(ctx, (size_t)(G + nsrc + T) * K64, &pb);
    if (!panel) {
        vc4__set_err(ctx, "col2im: no room for a %u-block panel", G + nsrc + T);
        return VC4_ERR_NOMEM;
    }
    memset(panel, 0, (size_t)(G + nsrc + T) * K64);
    const uint32_t panel_real = pb.off + ctx->gpu_bus + G * K64;

    vc4_status st = vc4__pack3_run(ctx, dA_bus, (uint32_t)K, K64,
                                   panel_real, K64, nsrc);
    if (st != VC4_OK)
        return st;

    char name[64];
    snprintf(name, sizeof(name), "col2im_%d_%d_%d_%d_%d_%d",
             Cin, K, KH, KW, Wp, Hp);
    char ncin[16], nk_[16], nkh[8], nkw[8], nwp[16], nhp[16];
    char *argv[] = { (char *)"col2im", (char *)"--cin", ncin, (char *)"--k", nk_,
                     (char *)"--kh", nkh, (char *)"--kw", nkw,
                     (char *)"--wp", nwp, (char *)"--hp", nhp, NULL };
    snprintf(ncin, sizeof(ncin), "%d", Cin);
    snprintf(nk_, sizeof(nk_), "%d", K);
    snprintf(nkh, sizeof(nkh), "%d", KH);
    snprintf(nkw, sizeof(nkw), "%d", KW);
    snprintf(nwp, sizeof(nwp), "%d", Wp);
    snprintf(nhp, sizeof(nhp), "%d", Hp);

    vc4_kernel *k = NULL;
    st = vc4__kernel(ctx, name, argv, VC4_MAX_QPU, &k);
    if (st != VC4_OK)
        return st;

    /* one task (rb, cb) per QPU, ceil(tasks/12) launches -- no loop/branch in
       the kernel, the same shape the softmax kernel adopted after a counter
       loop around a long body hung the V3D */
    const uint32_t tasks = nrb * CB;
    for (uint32_t base = 0; base < tasks && st == VC4_OK; base += VC4_MAX_QPU) {
        uint32_t left = tasks - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        for (int i = 0; i < nq; ++i) {
            uint32_t t = base + (uint32_t)i;
            uint32_t rb = t / CB, cb = t % CB;
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = panel_real + rb * K64 + cb * 16u * 64u;
            u[1] = dx_bus + cb * 16u * plane + rb * 64u;
            u[2] = (uint32_t)i;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 16u * 16u * (uint64_t)(KH * KW);
    }
    *Hp_out = Hp;
    *Wp_out = Wp;
    return st;
}

/* ==================================================================
 * public: im2col (row-major)
 * ================================================================== */
vc4_status vc4_im2col(vc4_ctx *ctx, const vc4_buf *in, int Cin, int H, int W,
                      int KH, int KW, int pad, vc4_buf *out, int *rows)
{
    if (!ctx || !in || !out || !rows)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || H <= 0 || W <= 0 || KH <= 0 || KW <= 0)
        return VC4_ERR_INVAL;
    if (!((KH == 3 && KW == 3) || (KH == 1 && KW == 1)) || pad < 0)
        return VC4_ERR_UNSUPPORTED;
    if (KH == 1 && pad != 0)
        return VC4_ERR_UNSUPPORTED;
    const int ho = H + 2 * pad - KH + 1;
    const int wo = W + 2 * pad - KW + 1;
    if (ho <= 0 || wo <= 0)
        return VC4_ERR_INVAL;
    const int hp = H + 2 * pad;
    const int wp = (int)vc4__align16((uint32_t)(W + 2 * pad));
    if ((size_t)hp * wp * 4u > 8191u)
        return VC4_ERR_UNSUPPORTED;
    const uint32_t M = (uint32_t)ho * (uint32_t)wp;
    const int K = Cin * KH * KW;
    if (in->bytes < (size_t)Cin * H * W * 4 ||
        out->bytes < (size_t)M * K * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    int php, pwp;
    uint32_t pbus;
    if (!pad_image(ctx, in, Cin, H, W, pad, &php, &pwp, &pbus)) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    vc4_status st = im2col_row_run(ctx, pbus, php * pwp * 4, pwp, Cin, K,
                                   KH, KW, M, vc4_bus_addr(ctx, out));
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK)
        *rows = (int)M;
    return st;
}

/* ==================================================================
 * public: col2im
 * ================================================================== */
vc4_status vc4_col2im(vc4_ctx *ctx, const vc4_buf *dA, int Cin, int K,
                      int KH, int KW, int HO, int WO, vc4_buf *dXpad,
                      int *Hp, int *Wp)
{
    if (!ctx || !dA || !dXpad || !Hp || !Wp)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || KH <= 0 || KW <= 0 || HO <= 0 || WO <= 0)
        return VC4_ERR_INVAL;
    if (Cin % 16 || K != Cin * KH * KW ||
        !((KH == 3 && KW == 3) || (KH == 1 && KW == 1)))
        return VC4_ERR_UNSUPPORTED;
    const int hp = HO + KH - 1;
    const int wp = (int)vc4__align16((uint32_t)(WO + KW - 1));
    const uint32_t M = (uint32_t)HO * (uint32_t)wp;
    if (dA->bytes < (size_t)M * K * 4 ||
        dXpad->bytes < (size_t)Cin * hp * wp * 4 + 64)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    memset(ctx->arena + dXpad->off, 0, (size_t)Cin * hp * wp * 4 + 64);
    vc4__mark_dirty(ctx, dXpad->off, (size_t)Cin * hp * wp * 4 + 64);
    int hp2, wp2;
    vc4_status st = col2im_run(ctx, vc4_bus_addr(ctx, dA), Cin, K, KH, KW,
                               HO, WO, vc4_bus_addr(ctx, dXpad), &hp2, &wp2);
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) {
        *Hp = hp2;
        *Wp = wp2;
    }
    return st;
}

/* ==================================================================
 * public: conv2d backward
 * ================================================================== */
vc4_status vc4_conv2d_backward(vc4_ctx *ctx,
                               const vc4_buf *in, int Cin, int H, int W,
                               const vc4_buf *wt, int Cout, int KH, int KW,
                               const vc4_buf *dy,
                               int pad,
                               vc4_buf *dW, vc4_buf *dX)
{
    if (!ctx || !in || !wt || !dy || !dW || !dX)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0)
        return VC4_ERR_INVAL;
    if (Cin % 16 || Cout % 16)
        return VC4_ERR_UNSUPPORTED;
    if (!((KH == 3 && KW == 3) || (KH == 1 && KW == 1)) || pad < 0)
        return VC4_ERR_UNSUPPORTED;
    if (KH == 1 && pad != 0)
        return VC4_ERR_UNSUPPORTED;
    const int ho = H + 2 * pad - KH + 1;
    const int wo = W + 2 * pad - KW + 1;
    const int hp = H + 2 * pad;
    const int wp = (int)vc4__align16((uint32_t)(W + 2 * pad));
    if (ho <= 0 || wo <= 0)
        return VC4_ERR_INVAL;
    if ((size_t)hp * wp * 4u > 8191u)
        return VC4_ERR_UNSUPPORTED;
    const uint32_t M = (uint32_t)ho * (uint32_t)wp;
    const int K = Cin * KH * KW;
    if (in->bytes < (size_t)Cin * H * W * 4 ||
        wt->bytes < (size_t)Cout * K * 4 ||
        dy->bytes < (size_t)Cout * ho * wo * 4 ||
        dW->bytes < (size_t)Cout * K * 4 ||
        dX->bytes < (size_t)Cin * H * W * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    vc4_status st = VC4_OK;
    int php, pwp;
    uint32_t pbus;

    /* ---- padded input image and the row-major im2col matrix ---------- */
    if (!pad_image(ctx, in, Cin, H, W, pad, &php, &pwp, &pbus))
        { st = VC4_ERR_NOMEM; goto out; }
    vc4_buf acol;
    float *Acol = vc4__scratch_alloc(ctx, (size_t)M * K * 4, &acol);
    if (!Acol) { st = VC4_ERR_NOMEM; goto out; }
    st = im2col_row_run(ctx, pbus, php * pwp * 4, pwp, Cin, K, KH, KW, M,
                        ctx->gpu_bus + acol.off);
    if (st != VC4_OK) goto out;

    /* ---- dY -> dYcol [M][Cout]  (NCHW to pixel-major; zero pad cols) -- */
    vc4_buf dyc;
    float *DY = vc4__scratch_alloc(ctx, (size_t)M * Cout * 4, &dyc);
    if (!DY) { st = VC4_ERR_NOMEM; goto out; }
    memset(DY, 0, (size_t)M * Cout * 4);
    {
        const float *src = (const float *)(ctx->arena + dy->off);
        for (int co = 0; co < Cout; ++co)
            for (int y = 0; y < ho; ++y)
                for (int x = 0; x < wo; ++x)
                    DY[((size_t)y * wp + (size_t)x) * Cout + co] =
                        src[((size_t)co * ho + (size_t)y) * wo + x];
    }

    /* ---- weights -> tap-major Wp [Cout][K] -> WpT [K][Cout] ---------- */
    vc4_buf wpb, wptb;
    float *Wp = vc4__scratch_alloc(ctx, (size_t)Cout * K * 4, &wpb);
    float *WpT = vc4__scratch_alloc(ctx, (size_t)K * Cout * 4, &wptb);
    if (!Wp || !WpT) { st = VC4_ERR_NOMEM; goto out; }
    {
        const float *src = (const float *)(ctx->arena + wt->off);
        const int taps = KH * KW;
        for (int co = 0; co < Cout; ++co)
            for (int ci = 0; ci < Cin; ++ci)
                for (int t = 0; t < taps; ++t)
                    Wp[(size_t)co * K + (size_t)t * Cin + ci] =
                        src[((size_t)co * Cin + ci) * taps + t];
    }
    st = vc4__tpose_run(ctx, ctx->gpu_bus + wpb.off, (uint32_t)Cout,
                        (uint32_t)K, ctx->gpu_bus + wptb.off);
    if (st != VC4_OK) goto out;

    /* ---- dA = dYcol * WpT^T   (pack + gemm3 bmode=1) ----------------- */
    vc4_buf dyc_p, wpt_p, dab;
    float *DYP = vc4__scratch_alloc(ctx, (size_t)M * Cout * 4, &dyc_p);
    float *WPTP = vc4__scratch_alloc(ctx, (size_t)K * Cout * 4, &wpt_p);
    float *dA = vc4__scratch_alloc(ctx, (size_t)M * K * 4, &dab);
    if (!DYP || !WPTP || !dA) { st = VC4_ERR_NOMEM; goto out; }
    st = vc4__pack3_run(ctx, ctx->gpu_bus + dyc.off, (uint32_t)Cout,
                        64u * (uint32_t)Cout, ctx->gpu_bus + dyc_p.off,
                        64u * (uint32_t)Cout, M / 16u);
    if (st == VC4_OK)
        st = vc4__pack3_run(ctx, ctx->gpu_bus + wptb.off, (uint32_t)Cout,
                            64u * (uint32_t)Cout, ctx->gpu_bus + wpt_p.off,
                            64u * (uint32_t)Cout, (uint32_t)K / 16u);
    if (st == VC4_OK)
        st = vc4__gemm3_run(ctx, M, (uint32_t)K, (uint32_t)Cout, 1,
                            ctx->gpu_bus + wpt_p.off, ctx->gpu_bus + dab.off,
                            ctx->gpu_bus + dyc_p.off);
    if (st != VC4_OK) goto out;

    /* ---- dXpad = col2im(dA), then crop to NCHW dX -------------------- */
    vc4_buf dxp;
    float *DXp = vc4__scratch_alloc(ctx, (size_t)Cin * hp * wp * 4 + 64, &dxp);
    if (!DXp) { st = VC4_ERR_NOMEM; goto out; }
    memset(DXp, 0, (size_t)Cin * hp * wp * 4 + 64);
    int hp2, wp2;
    st = col2im_run(ctx, ctx->gpu_bus + dab.off, Cin, K, KH, KW, ho, wo,
                    ctx->gpu_bus + dxp.off, &hp2, &wp2);
    if (st != VC4_OK) goto out;
    vc4__l2_flush(ctx);
    {
        float *dst = (float *)(ctx->arena + dX->off);
        for (int c = 0; c < Cin; ++c)
            for (int y = 0; y < H; ++y)
                memcpy(dst + ((size_t)c * H + (size_t)y) * W,
                       DXp + ((size_t)c * hp + (size_t)(y + pad)) * wp + pad,
                       (size_t)W * 4);
        vc4__mark_dirty(ctx, dX->off, (size_t)Cin * H * W * 4);
    }

    /* ---- dWp = dY^T * Acol   (transpose + pack + gemm3 bmode=0) ------ */
    vc4_buf dytb, dyt_p, dwpb;
    float *DYt = vc4__scratch_alloc(ctx, (size_t)Cout * M * 4, &dytb);
    float *DYTP = vc4__scratch_alloc(ctx, (size_t)Cout * M * 4, &dyt_p);
    float *dWp = vc4__scratch_alloc(ctx, (size_t)Cout * K * 4, &dwpb);
    if (!DYt || !DYTP || !dWp) { st = VC4_ERR_NOMEM; goto out; }
    st = vc4__tpose_run(ctx, ctx->gpu_bus + dyc.off, M, (uint32_t)Cout,
                        ctx->gpu_bus + dytb.off);
    if (st == VC4_OK)
        st = vc4__pack3_run(ctx, ctx->gpu_bus + dytb.off, M, 64u * M,
                            ctx->gpu_bus + dyt_p.off, 64u * M, Cout / 16u);
    if (st == VC4_OK)
        st = vc4__gemm3_run(ctx, (uint32_t)Cout, (uint32_t)K, M, 0,
                            ctx->gpu_bus + acol.off, ctx->gpu_bus + dwpb.off,
                            ctx->gpu_bus + dyt_p.off);
    if (st != VC4_OK) goto out;
    vc4__l2_flush(ctx);
    {
        float *dst = (float *)(ctx->arena + dW->off);
        const int taps = KH * KW;
        for (int co = 0; co < Cout; ++co)
            for (int ci = 0; ci < Cin; ++ci)
                for (int t = 0; t < taps; ++t)
                    dst[((size_t)co * Cin + ci) * taps + t] =
                        dWp[(size_t)co * K + (size_t)t * Cin + ci];
        vc4__mark_dirty(ctx, dW->off, (size_t)Cout * K * 4);
    }

out:
    vc4__scratch_reset(ctx);
    return st;
}

/* ==================================================================
 * 2x2 stride-2 max pool
 *
 * One QPU = one 64 B row task, enumerated by the host; no loop and no branch
 * in the kernel.  Forward tasks are (channel, output row, 16-wide chunk),
 * backward tasks are (channel, input row, 16-wide chunk).
 * ================================================================== */
vc4_status vc4_maxpool2x2(vc4_ctx *ctx, const vc4_buf *in, int C, int H, int W,
                          int out_pitch, vc4_buf *out, int *Ho, int *Wo)
{
    if (!ctx || !in || !out || !Ho || !Wo)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (C <= 0 || H <= 0 || W <= 0 || H % 8 || W % 2 || W % 16)
        return VC4_ERR_UNSUPPORTED;          /* 4 output rows per task     */
    const int ho = H / 2, wo = W / 2;
    if (out_pitch < wo || out_pitch % 16)
        return VC4_ERR_UNSUPPORTED;
    if (in->bytes < (size_t)C * H * W * 4 + 64 ||
        out->bytes < (size_t)C * ho * out_pitch * 4)
        return VC4_ERR_INVAL;

    char name[72], nh[16], nw[16], no[16], nc[16];
    char *argv[] = { (char *)"pool", (char *)"--h", nh, (char *)"--w", nw,
                     (char *)"--op", no, (char *)"--c", nc,
                     (char *)"--mode", (char *)"fwd", NULL };
    snprintf(nh, sizeof(nh), "%d", H);
    snprintf(nw, sizeof(nw), "%d", W);
    snprintf(no, sizeof(no), "%d", out_pitch);
    snprintf(nc, sizeof(nc), "%d", C);
    snprintf(name, sizeof(name), "pool_%d_%d_%d_c%d_fwd", H, W, out_pitch, C);

    vc4__scratch_reset(ctx);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, VC4_MAX_QPU, &k);
    if (st != VC4_OK) { vc4__scratch_reset(ctx); return st; }
    const int ch = out_pitch / 16;
    const uint32_t ng = (uint32_t)ho / 4u;          /* 4 output rows / task */
    const uint32_t tasks = (uint32_t)C * ng * (uint32_t)ch;
    const uint32_t in_bus = vc4_bus_addr(ctx, in);
    const uint32_t out_bus = vc4_bus_addr(ctx, out);
    for (uint32_t base = 0; base < tasks && st == VC4_OK;
         base += VC4_MAX_QPU) {
        uint32_t left = tasks - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        for (int i = 0; i < nq; ++i) {
            uint32_t t = base + (uint32_t)i;
            uint32_t chunk = t % (uint32_t)ch;
            uint32_t ry = (t / (uint32_t)ch) % ng;
            uint32_t c = t / ((uint32_t)ch * ng);
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = in_bus + c * (uint32_t)H * (uint32_t)W * 4u +
                   8u * ry * (uint32_t)W * 4u + chunk * 128u;
            u[1] = out_bus + c * (uint32_t)ho * (uint32_t)out_pitch * 4u +
                   4u * ry * (uint32_t)out_pitch * 4u + chunk * 64u;
            u[2] = (uint32_t)i;
            u[3] = chunk;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 16u * 3u;
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) { *Ho = ho; *Wo = wo; }
    return st;
}

/* ------------------------------------------------------------------
 * Generic NxN / stride-S max pool.
 *
 * The border is materialised on the host as -inf (byte fill + row copy), so
 * the QPU kernel is a straight-line 4-output-row x 16-column task with no
 * bounds checks, no loop and no branch.  The output row pitch is
 * align16(wo) and the output has align4(ho) rows, because the verified
 * 4-row VPM->DMA store writes whole rows.  Callers that need a compact
 * [C][ho][wo] tensor copy the valid lanes (byte gather, no arithmetic).
 * ------------------------------------------------------------------ */
vc4_status vc4_maxpooln(vc4_ctx *ctx, const vc4_buf *in, int C, int H, int W,
                        int N, int stride, int pad,
                        vc4_buf *out, int *Ho, int *Wo)
{
    if (!ctx || !in || !out || !Ho || !Wo)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (C <= 0 || H <= 0 || W <= 0 || N <= 0 || stride <= 0 || pad < 0)
        return VC4_ERR_INVAL;
    const int ho = (H + 2 * pad - N) / stride + 1;
    const int wo = (W + 2 * pad - N) / stride + 1;
    if (ho <= 0 || wo <= 0)
        return VC4_ERR_UNSUPPORTED;
    const int ho_pad = (ho + 3) & ~3;
    const int op = (wo + 15) & ~15;
    /* the kernel reads the full padded pitch, including the lanes past wo */
    int hp = H + 2 * pad;
    int wp = W + 2 * pad;
    if (hp < (ho_pad - 1) * stride + N)
        hp = (ho_pad - 1) * stride + N;
    if (wp < (op - 1) * stride + N)
        wp = (op - 1) * stride + N;
    if (in->bytes < (size_t)C * H * W * 4 ||
        out->bytes < (size_t)C * ho_pad * op * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    vc4_buf pb;
    float *pimg = vc4__scratch_alloc(ctx, (size_t)C * hp * wp * 4 + 64, &pb);
    if (!pimg) {
        vc4__set_err(ctx, "maxpooln: no room for a %dx%dx%d padded image",
                     C, hp, wp);
        return VC4_ERR_NOMEM;
    }
    /* -inf border: a window that includes padding must never win over a real
       value (zeros would change the max of an all-negative window) */
    for (size_t i = 0; i < (size_t)C * hp * wp + 16; ++i)
        ((uint32_t *)pimg)[i] = 0xFF800000u;
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            memcpy((uint8_t *)pimg +
                   ((size_t)c * hp * wp + (size_t)(y + pad) * wp + pad) * 4,
                   ctx->arena + in->off + ((size_t)c * H + (size_t)y) * W * 4,
                   (size_t)W * 4);
    vc4__mark_dirty(ctx, pb.off, (size_t)C * hp * wp * 4);
    const uint32_t pbus = ctx->gpu_bus + pb.off;

    char name[72], nn[16], ns[16], nw[16], no[16];
    char *argv[] = { (char *)"pooln", (char *)"--n", nn, (char *)"--s", ns,
                     (char *)"--wp", nw, (char *)"--op", no, NULL };
    snprintf(nn, sizeof(nn), "%d", N);
    snprintf(ns, sizeof(ns), "%d", stride);
    snprintf(nw, sizeof(nw), "%d", wp);
    snprintf(no, sizeof(no), "%d", op);
    snprintf(name, sizeof(name), "pooln_%d_%d_%d_%d", N, stride, wp, op);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, VC4_MAX_QPU, &k);
    if (st != VC4_OK) { vc4__scratch_reset(ctx); return st; }
    const uint32_t nch = (uint32_t)op / 16u;
    const uint32_t ng = (uint32_t)ho_pad / 4u;
    const uint32_t tasks = (uint32_t)C * ng * nch;
    const uint32_t out_bus = vc4_bus_addr(ctx, out);
    for (uint32_t base = 0; base < tasks && st == VC4_OK;
         base += VC4_MAX_QPU) {
        uint32_t left = tasks - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        for (int i = 0; i < nq; ++i) {
            uint32_t t = base + (uint32_t)i;
            uint32_t chunk = t % nch;
            uint32_t ry = (t / nch) % ng;
            uint32_t c = t / (nch * ng);
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = pbus + c * (uint32_t)hp * (uint32_t)wp * 4u +
                   ry * 4u * (uint32_t)stride * (uint32_t)wp * 4u +
                   chunk * (uint32_t)stride * 64u;
            u[1] = out_bus + c * (uint32_t)ho_pad * (uint32_t)op * 4u +
                   4u * ry * (uint32_t)op * 4u + chunk * 64u;
            u[2] = (uint32_t)i;
            u[3] = chunk;
        }
        uint64_t ns2 = 0;
        st = vc4__launch(ctx, k, nq, &ns2);
        ctx->st.kernel_ns += ns2;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 16u * 4u * (uint64_t)(N * N);
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) { *Ho = ho; *Wo = wo; }
    return st;
}

vc4_status vc4_maxpool2x2_backward(vc4_ctx *ctx, const vc4_buf *in,
                                   const vc4_buf *dy, int C, int H, int W,
                                   int dy_pitch, vc4_buf *dx)
{
    if (!ctx || !in || !dy || !dx)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (C <= 0 || H <= 0 || W <= 0 || H % 4 || W % 2 || W % 16)
        return VC4_ERR_UNSUPPORTED;          /* 4 input rows per task      */
    const int ho = H / 2, wo = W / 2;
    if (dy_pitch < wo || dy_pitch % 16)
        return VC4_ERR_UNSUPPORTED;
    if (in->bytes < (size_t)C * H * W * 4 + 64 ||
        dy->bytes < (size_t)C * ho * dy_pitch * 4 ||
        dx->bytes < (size_t)C * H * W * 4)
        return VC4_ERR_INVAL;

    char name[72], nh[16], nw[16], no[16], nc[16];
    char *argv[] = { (char *)"pool", (char *)"--h", nh, (char *)"--w", nw,
                     (char *)"--op", no, (char *)"--c", nc,
                     (char *)"--mode", (char *)"bwd", NULL };
    snprintf(nh, sizeof(nh), "%d", H);
    snprintf(nw, sizeof(nw), "%d", W);
    snprintf(no, sizeof(no), "%d", dy_pitch);
    snprintf(nc, sizeof(nc), "%d", C);
    snprintf(name, sizeof(name), "pool_%d_%d_%d_c%d_bwd", H, W, dy_pitch, C);

    vc4__scratch_reset(ctx);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, VC4_MAX_QPU, &k);
    if (st != VC4_OK) { vc4__scratch_reset(ctx); return st; }
    const int nch = W / 16;
    const uint32_t ng = (uint32_t)H / 4u;           /* 4 input rows / task */
    const uint32_t tasks = (uint32_t)C * ng * (uint32_t)nch;
    const uint32_t in_bus = vc4_bus_addr(ctx, in);
    const uint32_t dx_bus = vc4_bus_addr(ctx, dx);
    const uint32_t dy_bus = vc4_bus_addr(ctx, dy);
    const uint32_t plane = (uint32_t)H * (uint32_t)W * 4u;
    const uint32_t dyplane = (uint32_t)ho * (uint32_t)dy_pitch * 4u;
    for (uint32_t base = 0; base < tasks && st == VC4_OK;
         base += VC4_MAX_QPU) {
        uint32_t left = tasks - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        for (int i = 0; i < nq; ++i) {
            uint32_t t = base + (uint32_t)i;
            uint32_t chunk = t % (uint32_t)nch;
            uint32_t r4 = (t / (uint32_t)nch) % ng;
            uint32_t c = t / ((uint32_t)nch * ng);
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = in_bus + c * plane + 4u * r4 * (uint32_t)W * 4u +
                   chunk * 64u;
            u[1] = u[0];
            u[2] = (uint32_t)i;
            u[3] = dy_bus + c * dyplane + 2u * r4 * (uint32_t)dy_pitch * 4u +
                   chunk * 32u;
            u[4] = chunk;
            u[5] = dx_bus + c * plane + 4u * r4 * (uint32_t)W * 4u +
                   chunk * 64u;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 16u * 2u;
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    return st;
}
