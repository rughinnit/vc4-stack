/*
 * vc4_nhwc.c - the NHWC data plane used by the YOLO path.
 *
 * Why NHWC: the NCHW conv path's im2col loads 16 channels at a pitch equal to
 * the whole padded plane, and the VDR pitch field is 13 bits (8191 B), so a
 * plane larger than 2047 floats cannot be read.  In NHWC a VDR row is one
 * pixel's 16-channel block at a pitch of stride*Cin*4 bytes, so any feature map
 * up to 2047 channels works (a 160x160x64 map is 6.4 MB) and the GEMM's
 * row-major C[M][N] (M = pixels, N = channels) is already the output layout.
 *
 * Every op here is a single QPU launch per tile pass (conv = im2col + GEMM),
 * the task ranges are enumerated on the ARM (a row range per QPU, plus the
 * image and channel-block index), so one launch covers a whole batch.  The
 * only ARM work is the zero-padded NHWC copy (a byte copy, no arithmetic) and
 * the bias rows of the im2col panel (a 1.0 fill, no arithmetic).
 *
 * Weights are packed once: vc4_conv_nhwc_pack_w builds the k-major B panel
 * (tap-major k order: k = tap*Cin + ci), vc4_depthwise_pack_w permutes the
 * 3x3 per-channel weights to WP[tap][C].
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NHWC_SLACK (256u * 1024u)     /* over-read guard for the DMA windows */
#define NHWC_MAX_QPU 12
#define NHWC_IM2COL_QPU 4             /* 16 VPM rows per QPU, 64 rows total   */

static uint32_t al16(uint32_t v) { return (v + 15u) & ~15u; }

/* one QPU task = (image, channel block, contiguous row range).  When there are
 * more (image, block) units than QPUs, splits = 1 and the caller launches in
 * batches of maxq units. */
static void nhwc_split(int units, int maxq, int rows, int *splits, int *ubatch)
{
    int sp = units <= maxq ? maxq / units : 1;
    if (sp < 1) sp = 1;
    if (sp > rows) sp = rows;
    *splits = sp;
    *ubatch = units <= maxq ? units : maxq;
}

/* one zero-padded NHWC image set, shared by every group of a grouped conv */
typedef struct {
    uint32_t bus;         /* GPU bus of image 0 */
    size_t   img_bytes;   /* bytes per image */
} nhwc_pad;

static int nhwc_pad_image(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                          int H, int W, int in_pitch, int pad, nhwc_pad *pd)
{
    const int hp = H + 2 * pad, wp = W + 2 * pad;
    const size_t per = (size_t)hp * wp * C * 4u;
    float *p = vc4__scratch_alloc(ctx, per * (size_t)nimg + NHWC_SLACK, NULL);
    if (!p)
        return 0;
    memset((uint8_t *)p, 0, per * (size_t)nimg + NHWC_SLACK);
    if (nimg)
        vc4__invalidate_off(ctx, in->off, (size_t)nimg * H * in_pitch * C * 4u);
    for (int n = 0; n < nimg; ++n) {
        const uint8_t *src = ctx->arena + in->off +
                             (size_t)n * H * in_pitch * C * 4u;
        uint8_t *dst = (uint8_t *)p + (size_t)n * per;
        for (int y = 0; y < H; ++y)
            memcpy(dst + ((size_t)(y + pad) * wp + pad) * C * 4u,
                   src + (size_t)y * in_pitch * C * 4u, (size_t)W * C * 4u);
    }
    pd->bus = ctx->gpu_bus + (uint32_t)((uint8_t *)p - ctx->arena);
    pd->img_bytes = per;
    return 1;
}

/* --------------------------------------------------------------- conv 2d */
size_t vc4_conv_nhwc_wpanel_bytes(int Cin, int Cout, int KH, int KW, int bias)
{
    if (Cin <= 0 || Cout <= 0 || KH <= 0 || KW <= 0 || Cin % 16 || Cout % 16)
        return 0;
    uint32_t K = (uint32_t)Cin * (uint32_t)(KH * KW);
    return (size_t)Cout * (al16(K) + (bias ? 16u : 0u)) * 4u;
}

vc4_status vc4_conv_nhwc_pack_w(vc4_ctx *ctx, const vc4_buf *wt, int Cin,
                                int Cout, int KH, int KW, const vc4_buf *bias,
                                vc4_buf *wpanel)
{
    if (!ctx || !wt || !wpanel)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || Cout <= 0 || KH <= 0 || KW <= 0 || Cin % 16 || Cout % 16)
        return VC4_ERR_UNSUPPORTED;
    const uint32_t K = (uint32_t)Cin * (uint32_t)(KH * KW);
    const uint32_t Kp = al16(K);
    const uint32_t nk = Kp + (bias ? 16u : 0u);
    if (wt->bytes < (size_t)Cout * K * 4 || wpanel->bytes < (size_t)Cout * nk * 4)
        return VC4_ERR_INVAL;
    if (bias && bias->bytes < (size_t)Cout * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    float *wr = vc4__scratch_alloc(ctx, (size_t)Cout * K * 4, NULL);
    if (!wr) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    /* wt is [Cout][Cin][KH][KW] (ci-major); the panel k order is tap-major */
    {
        vc4__invalidate_off(ctx, wt->off, (size_t)Cout * K * 4);
        const float *src = (const float *)(ctx->arena + wt->off);
        const int taps = KH * KW;
        for (int co = 0; co < Cout; ++co)
            for (int ci = 0; ci < Cin; ++ci)
                for (int t = 0; t < taps; ++t)
                    wr[(size_t)co * K + (size_t)t * Cin + ci] =
                        src[(size_t)co * K + (size_t)ci * taps + t];
    }
    vc4__invalidate_off(ctx, wpanel->off, (size_t)Cout * nk * 4);
    memset(ctx->arena + wpanel->off, 0, (size_t)Cout * nk * 4);
    /* wpanel is a caller-owned (persistent) buffer, and the CPU has just
     * written it.  Through vc4d the range has to be flushed to DRAM before
     * the QPUs read it; the launch path tracks only the per-op scratch, so
     * every CPU write into a persistent buffer has to mark itself. */
    vc4__mark_dirty(ctx, wpanel->off, (size_t)Cout * nk * 4);
    vc4_status st = vc4__pack3_run(ctx, ctx->gpu_bus + (uint32_t)
                                  ((uint8_t *)wr - ctx->arena), K, 64u * K,
                                  vc4_bus_addr(ctx, wpanel), 64u * nk,
                                  (uint32_t)Cout / 16u);
    if (st == VC4_OK && bias) {
        vc4__invalidate_off(ctx, bias->off, (size_t)Cout * 4);
        const float *bsrc = (const float *)(ctx->arena + bias->off);
        float *wp = (float *)(ctx->arena + wpanel->off);
        for (uint32_t cb = 0; cb < (uint32_t)Cout / 16u; ++cb)
            memcpy(wp + ((size_t)cb * nk + Kp) * 16u,
                   bsrc + (size_t)cb * 16u, 16u * 4u);
        /* The bias row sits past the k range the pack kernel writes, so it is
         * an ordinary CPU write into a persistent buffer and must be flushed
         * too -- this one happens after the pack launch. */
        vc4__mark_dirty(ctx, wpanel->off, (size_t)Cout * nk * 4);
    }

    vc4__scratch_reset(ctx);
    return st;
}

/*
 * One im2col + GEMM pass.
 *
 * `Cin` channels of a pixel that stores `Cin_pitch` channels are read,
 * starting at byte `cin_off` (a grouped conv selects one channel block).
 * The VDR pitch and the 16-pixel block step use Cin_pitch, the tap step and
 * the channel-block count use Cin.  The padded image and the activation
 * panel are owned by the caller, so a grouped conv pads once and reuses the
 * panel for every group.
 */
static vc4_status nhwc_conv_pass(vc4_ctx *ctx, const nhwc_pad *pd, int nimg,
                                 int cin_off, int Cin, int Cin_pitch, int W,
                                 int KH, int KW, int stride, int pad,
                                 int dilation, const vc4_buf *wpanel,
                                 int Cout, int cstride, int cout_off,
                                 uint32_t out_bus, int ho, int Wg,
                                 const vc4_buf *pb, int act)
{
    const int nx = Wg / 16;
    const uint32_t M = (uint32_t)ho * (uint32_t)Wg;
    const uint32_t nk = (uint32_t)(wpanel->bytes / ((size_t)Cout * 4u));

    char ncin[16], nkh[8], nkw[8], nst[8], ndl[8], name[96];
    char *argv[] = { (char *)"im2col_nhwc", (char *)"--cin", ncin,
                     (char *)"--kh", nkh, (char *)"--kw", nkw,
                     (char *)"--stride", nst, (char *)"--dil", ndl, NULL };
    snprintf(ncin, sizeof(ncin), "%d", Cin);
    snprintf(nkh, sizeof(nkh), "%d", KH);
    snprintf(nkw, sizeof(nkw), "%d", KW);
    snprintf(nst, sizeof(nst), "%d", stride);
    snprintf(ndl, sizeof(ndl), "%d", dilation);
    snprintf(name, sizeof(name), "im2col_nhwc2_%d_%d_%d_s%d_d%d", Cin, KH, KW,
             stride, dilation);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, NHWC_IM2COL_QPU, &k);
    if (st != VC4_OK)
        return st;
    const uint32_t row_pitch = (uint32_t)(W + 2 * pad) * Cin_pitch * 4u;
    const uint32_t blk_step = 16u * (uint32_t)stride * Cin_pitch * 4u;
    const uint32_t dst_step = 64u * nk;
    const int chunk = nimg > NHWC_IM2COL_QPU ? NHWC_IM2COL_QPU : nimg;
    for (int base = 0; base < nimg && st == VC4_OK; base += chunk) {
        int nb = nimg - base;
        if (nb > chunk)
            nb = chunk;
        int splits = NHWC_IM2COL_QPU / nb;
        if (splits < 1)
            splits = 1;
        if (splits > ho)
            splits = ho;
        int nq = nb * splits;
        int q = 0;
        for (int n = 0; n < nb; ++n) {
            int per = ho / splits, rem = ho % splits;
            for (int s = 0; s < splits; ++s, ++q) {
                int row0 = s * per + (s < rem ? s : rem);
                int cnt = per + (s < rem ? 1 : 0);
                uint32_t *u = vc4__uni(ctx, k, q);
                u[0] = pd->bus + (uint32_t)((size_t)(base + n) * pd->img_bytes)
                       + (uint32_t)cin_off * 4u;
                u[1] = ctx->gpu_bus + pb->off +
                       (uint32_t)((size_t)(base + n) * M * nk * 4u);
                u[2] = blk_step;
                u[3] = dst_step;
                u[4] = (uint32_t)cnt;
                u[5] = (uint32_t)(row0 * stride) * row_pitch;
                u[6] = (uint32_t)row0 * (uint32_t)nx * dst_step;
                u[7] = (uint32_t)stride * Cin_pitch * 4u;
                u[8] = (uint32_t)q;
                u[9] = row_pitch;
                u[10] = (uint32_t)nx;
                u[11] = (uint32_t)stride * row_pitch;
                u[12] = (uint32_t)Cin_pitch * 4u * (uint32_t)dilation;
            }
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.pack_ns += ns;
    }
    if (st != VC4_OK)
        return st;

    /* GEMM: N = Cout, K = nk (bias folded), C row stride = cstride and the
     * output channel block starts at cout_off. */
    return vc4__gemm3_run_c(ctx, (uint32_t)nimg * M, (uint32_t)Cout, nk, 1,
                            vc4_bus_addr(ctx, wpanel), out_bus + cout_off * 4u,
                            ctx->gpu_bus + pb->off, (uint32_t)cstride, act);
}

static vc4_status conv_nhwc_common(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                                   int Cin, int H, int W, int in_pitch,
                                   const vc4_buf *const *wpanel, int Cout,
                                   int groups, int KH, int KW, int stride,
                                   int pad, int dilation, vc4_act act,
                                   vc4_buf *out, int *ho_out, int *wo_out,
                                   int *opitch_out)
{
    if (!ctx || !in || !wpanel || !wpanel[0] || !out || !ho_out || !wo_out ||
        !opitch_out || nimg < 1 || groups < 1)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || H <= 0 || W <= 0 || in_pitch < W || Cout <= 0) {
        vc4__set_err(ctx, "conv_nhwc: bad dims in=%dx%dx%d p%d out=%d", Cin, H,
                     W, in_pitch, Cout);
        return VC4_ERR_INVAL;
    }
    if (Cin % groups || Cout % groups) {
        vc4__set_err(ctx, "conv_nhwc: groups %d does not divide %d/%d",
                     groups, Cin, Cout);
        return VC4_ERR_UNSUPPORTED;
    }
    const int Cin_g = Cin / groups, Cout_g = Cout / groups;
    if (Cin_g % 16 || Cout_g % 16) {
        vc4__set_err(ctx, "conv_nhwc: per-group Cin/Cout must be multiples "
                     "of 16 (%d/%d)", Cin_g, Cout_g);
        return VC4_ERR_UNSUPPORTED;
    }
    if (!((KH == 1 || KH == 3 || KH == 5) &&
          (KW == 1 || KW == 3 || KW == 5))) {
        vc4__set_err(ctx, "conv_nhwc: only 1/3/5-wide kernels (%dx%d)", KH, KW);
        return VC4_ERR_UNSUPPORTED;
    }
    if ((stride != 1 && stride != 2) || pad < 0 ||
        dilation < 1 || dilation > 8) {
        vc4__set_err(ctx, "conv_nhwc: stride %d pad %d dil %d", stride, pad,
                     dilation);
        return VC4_ERR_UNSUPPORTED;
    }
    const int effh = (KH - 1) * dilation + 1;
    const int effw = (KW - 1) * dilation + 1;
    const int wo = (W + 2 * pad - effw) / stride + 1;
    const int ho = (H + 2 * pad - effh) / stride + 1;
    if (wo <= 0 || ho <= 0)
        return VC4_ERR_INVAL;
    const int Wg = (int)al16((uint32_t)wo);
    const uint32_t M = (uint32_t)ho * (uint32_t)Wg;
    const uint32_t K = (uint32_t)Cin_g * (uint32_t)(KH * KW);
    const uint32_t Kp = al16(K);
    if (Kp != K) {
        vc4__set_err(ctx, "conv_nhwc: per-group K=%u not 16-aligned", K);
        return VC4_ERR_UNSUPPORTED;
    }
    if ((uint32_t)stride * (uint32_t)Cin * 4u > 8191u) {
        vc4__set_err(ctx, "conv_nhwc: VDR pitch %u B > 8191 B (stride %d, "
                     "Cin %d)", (uint32_t)stride * Cin * 4u, stride, Cin);
        return VC4_ERR_UNSUPPORTED;
    }
    for (int g = 0; g < groups; ++g) {
        const uint32_t nk = (uint32_t)(wpanel[g]->bytes /
                                       ((size_t)Cout_g * 4u));
        if (nk != Kp && nk != Kp + 16u) {
            vc4__set_err(ctx, "conv_nhwc: group %d wpanel nk=%u (Kp=%u)", g,
                         nk, Kp);
            return VC4_ERR_INVAL;
        }
    }
    const uint32_t nk = (uint32_t)(wpanel[0]->bytes / ((size_t)Cout_g * 4u));
    if (in->bytes < (size_t)nimg * H * in_pitch * Cin * 4) {
        vc4__set_err(ctx, "conv_nhwc: input %zu B < %zu B", in->bytes,
                     (size_t)nimg * H * in_pitch * Cin * 4);
        return VC4_ERR_INVAL;
    }
    if (out->bytes < (size_t)nimg * M * Cout * 4) {
        vc4__set_err(ctx, "conv_nhwc: output %zu B < %zu B (M=%u nk=%u)",
                     out->bytes, (size_t)nimg * M * Cout * 4, M, nk);
        return VC4_ERR_INVAL;
    }

    /* 1. zero-padded NHWC copy of the whole input (bytes only) */
    vc4__scratch_reset(ctx);
    nhwc_pad pd;
    if (!nhwc_pad_image(ctx, in, nimg, Cin, H, W, in_pitch, pad, &pd)) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }

    /* 2. activation panel, reused by every group */
    vc4_buf pb;
    size_t panel_bytes = (size_t)nimg * M * nk * 4u;
    float *panel = vc4__scratch_alloc(ctx, panel_bytes, &pb);
    if (!panel) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    if (nk > Kp) {
        const uint32_t blocks = (uint32_t)nimg * (M / 16u);
        for (uint32_t rb = 0; rb < blocks; ++rb) {
            float *blk = panel + ((size_t)rb * nk + Kp) * 16u;
            for (int l = 0; l < 16; ++l)
                blk[l] = 1.0f;
        }
    }

    vc4_status st = VC4_OK;
    for (int g = 0; g < groups && st == VC4_OK; ++g)
        st = nhwc_conv_pass(ctx, &pd, nimg, g * Cin_g, Cin_g, Cin, W, KH, KW,
                            stride, pad, dilation, wpanel[g], Cout_g, Cout,
                            g * Cout_g, vc4_bus_addr(ctx, out), ho, Wg, &pb,
                            act);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) {
        *ho_out = ho;
        *wo_out = wo;
        *opitch_out = Wg;
    }
    return st;
}

vc4_status vc4_conv_nhwc_dil_act(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                                 int Cin, int H, int W, int in_pitch,
                                 const vc4_buf *wpanel, int Cout, int KH,
                                 int KW, int stride, int pad, int dilation,
                                 vc4_act act, vc4_buf *out, int *ho_out,
                                 int *wo_out, int *opitch_out)
{
    return conv_nhwc_common(ctx, in, nimg, Cin, H, W, in_pitch, &wpanel, Cout,
                            1, KH, KW, stride, pad, dilation, act, out,
                            ho_out, wo_out, opitch_out);
}

vc4_status vc4_conv_nhwc_dil(vc4_ctx *ctx, const vc4_buf *in, int nimg, int Cin,
                             int H, int W, int in_pitch,
                             const vc4_buf *wpanel, int Cout, int KH, int KW,
                             int stride, int pad, int dilation, vc4_buf *out,
                             int *ho_out, int *wo_out, int *opitch_out)
{
    return vc4_conv_nhwc_dil_act(ctx, in, nimg, Cin, H, W, in_pitch, wpanel,
                                 Cout, KH, KW, stride, pad, dilation,
                                 VC4_ACT_NONE, out, ho_out, wo_out,
                                 opitch_out);
}

vc4_status vc4_conv_nhwc_act(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                             int Cin, int H, int W, int in_pitch,
                             const vc4_buf *wpanel, int Cout, int KH, int KW,
                             int stride, int pad, vc4_act act, vc4_buf *out,
                             int *ho_out, int *wo_out, int *opitch_out)
{
    return vc4_conv_nhwc_dil_act(ctx, in, nimg, Cin, H, W, in_pitch, wpanel,
                                 Cout, KH, KW, stride, pad, 1, act, out,
                                 ho_out, wo_out, opitch_out);
}

vc4_status vc4_conv_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int Cin,
                         int H, int W, int in_pitch, const vc4_buf *wpanel,
                         int Cout, int KH, int KW, int stride, int pad,
                         vc4_buf *out, int *ho_out, int *wo_out,
                         int *opitch_out)
{
    return vc4_conv_nhwc_act(ctx, in, nimg, Cin, H, W, in_pitch, wpanel, Cout,
                             KH, KW, stride, pad, VC4_ACT_NONE, out, ho_out,
                             wo_out, opitch_out);
}

vc4_status vc4_conv_nhwc_group_act(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                                   int Cin, int H, int W, int in_pitch,
                                   const vc4_buf *const *wpanel, int Cout,
                                   int groups, int KH, int KW, int stride,
                                   int pad, int dilation, vc4_act act,
                                   vc4_buf *out, int *ho_out, int *wo_out,
                                   int *opitch_out)
{
    return conv_nhwc_common(ctx, in, nimg, Cin, H, W, in_pitch, wpanel, Cout,
                            groups, KH, KW, stride, pad, dilation, act, out,
                            ho_out, wo_out, opitch_out);
}

vc4_status vc4_conv_nhwc_group(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                               int Cin, int H, int W, int in_pitch,
                               const vc4_buf *const *wpanel, int Cout,
                               int groups, int KH, int KW, int stride, int pad,
                               int dilation, vc4_buf *out, int *ho_out,
                               int *wo_out, int *opitch_out)
{
    return vc4_conv_nhwc_group_act(ctx, in, nimg, Cin, H, W, in_pitch, wpanel,
                                   Cout, groups, KH, KW, stride, pad,
                                   dilation, VC4_ACT_NONE, out, ho_out,
                                   wo_out, opitch_out);
}
/* ------------------------------------------------------------- depthwise */
/*
 * A 3x3 depthwise conv is a 3x3 conv whose weight matrix is block diagonal:
 * B[(tap*C + ci)][co] = w[tap][co] when ci == co, else 0.  Building that panel
 * once turns the depthwise into the already-verified im2col + GEMM path, so it
 * needs no new kernel (it costs C/16 times the MACs of a true depthwise, which
 * at the maps this engine targets is a few tens of ms).
 */
size_t vc4_depthwise_wpanel_bytes(int C)
{
    if (C <= 0 || C % 16)
        return 0;
    return (size_t)C * (size_t)al16((uint32_t)(9 * C)) * 4u;
}

vc4_status vc4_depthwise_pack_w(vc4_ctx *ctx, const vc4_buf *wt, int C,
                                vc4_buf *wp)
{
    if (!ctx || !wt || !wp)
        return VC4_ERR_INVAL;
    if (C <= 0 || C % 16)
        return VC4_ERR_UNSUPPORTED;
    const uint32_t K = (uint32_t)(9 * C);
    if (wt->bytes < (size_t)C * 9u * 4u ||
        wp->bytes < (size_t)C * al16(K) * 4u)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;

    vc4__scratch_reset(ctx);
    float *w = vc4__scratch_alloc(ctx, (size_t)C * K * 4u, NULL);
    if (!w) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    memset(w, 0, (size_t)C * K * 4u);
    vc4__invalidate_off(ctx, wt->off, (size_t)C * al16(K) * 4u);
    const float *src = (const float *)(ctx->arena + wt->off);
    for (int co = 0; co < C; ++co)
        for (int t = 0; t < 9; ++t)
            w[(size_t)co * K + (size_t)t * C + co] = src[(size_t)co * 9 + t];
    memset(ctx->arena + wp->off, 0, (size_t)C * al16(K) * 4u);
    vc4__mark_dirty(ctx, wp->off, (size_t)C * al16(K) * 4u);
    vc4_status st = vc4__pack3_run(ctx, ctx->gpu_bus + (uint32_t)
                                   ((uint8_t *)w - ctx->arena), K, 64u * K,
                                   vc4_bus_addr(ctx, wp), 64u * al16(K),
                                   (uint32_t)C / 16u);
    vc4__scratch_reset(ctx);
    return st;
}

vc4_status vc4_depthwise_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                              int C, int H, int W, int in_pitch,
                              const vc4_buf *wp, vc4_buf *out, int *ho_out,
                              int *wo_out, int *opitch_out)
{
    if (!ctx || !in || !wp || !out || !ho_out || !wo_out || !opitch_out ||
        nimg < 1)
        return VC4_ERR_INVAL;
    if (C <= 0 || C % 16 || H <= 0 || W <= 0 || in_pitch < W)
        return VC4_ERR_UNSUPPORTED;
    if (wp->bytes < (size_t)C * al16((uint32_t)(9 * C)) * 4u)
        return VC4_ERR_INVAL;
    return vc4_conv_nhwc(ctx, in, nimg, C, H, W, in_pitch, wp, C, 3, 3, 1, 1,
                         out, ho_out, wo_out, opitch_out);
}

/* ------------------------------------------------------------------- pool */
vc4_status vc4_maxpool_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                            int H, int W, int in_pitch, int ksize, int pad,
                            vc4_buf *out, int *ho_out, int *wo_out,
                            int *opitch_out)
{
    if (!ctx || !in || !out || !ho_out || !wo_out || !opitch_out || nimg < 1)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (C <= 0 || C % 16 || H <= 0 || W <= 0 || in_pitch < W)
        return VC4_ERR_UNSUPPORTED;
    if (!(ksize == 3 || ksize == 5) || pad < 0 || pad >= ksize)
        return VC4_ERR_UNSUPPORTED;
    const int wo = W + 2 * pad - ksize + 1;
    const int ho = H + 2 * pad - ksize + 1;
    if (wo <= 0 || ho <= 0)
        return VC4_ERR_INVAL;
    const int Wg = (int)al16((uint32_t)wo);
    if (out->bytes < (size_t)nimg * ho * Wg * C * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    nhwc_pad pd;
    if (!nhwc_pad_image(ctx, in, nimg, C, H, W, in_pitch, pad, &pd)) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }

    char ncin[16], nk[8], name[48];
    char *argv[] = { (char *)"pool_nhwc", (char *)"--cin", ncin,
                     (char *)"--k", nk, NULL };
    snprintf(ncin, sizeof(ncin), "%d", C);
    snprintf(nk, sizeof(nk), "%d", ksize);
    snprintf(name, sizeof(name), "pool_nhwc_%d_k%d", C, ksize);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, NHWC_MAX_QPU, &k);
    if (st != VC4_OK) {
        vc4__scratch_reset(ctx);
        return st;
    }
    const int ncb = C / 16;
    const uint32_t srow = (uint32_t)(W + 2 * pad) * C * 4u;
    const uint32_t drow = (uint32_t)Wg * C * 4u;
    const uint32_t dst_img = (uint32_t)ho * drow;
    const int units = nimg * ncb;
    int splits, ubatch;
    nhwc_split(units, NHWC_MAX_QPU, ho, &splits, &ubatch);
    for (int base = 0; base < units && st == VC4_OK; base += ubatch) {
        int nbat = units - base < ubatch ? units - base : ubatch;
        int per = ho / splits, rem = ho % splits, q = 0;
        for (int t = 0; t < nbat; ++t) {
            int id = base + t, n = id / ncb, cb = id % ncb;
            for (int sp = 0; sp < splits; ++sp) {
                int row0 = sp * per + (sp < rem ? sp : rem);
                int cnt = per + (sp < rem ? 1 : 0);
                uint32_t *u = vc4__uni(ctx, k, q);
                u[0] = pd.bus + (uint32_t)((size_t)n * pd.img_bytes) +
                       (uint32_t)cb * 64u;
                u[1] = vc4_bus_addr(ctx, out) + (uint32_t)((size_t)n * dst_img);
                u[2] = (uint32_t)cnt;
                u[3] = (uint32_t)row0 * srow;
                u[4] = (uint32_t)row0 * drow;
                u[5] = (uint32_t)q;
                u[6] = (uint32_t)C * 4u;
                u[7] = srow;
                u[8] = drow;
                u[9] = (uint32_t)(Wg / 4);
                u[10] = (uint32_t)cb * 64u;
                ++q;
            }
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, q, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)q * 16u * (uint64_t)(ksize * ksize - 1) * 4u;
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) {
        *ho_out = ho;
        *wo_out = wo;
        *opitch_out = Wg;
    }
    return st;
}

/* --------------------------------------------------------------- upsample */
vc4_status vc4_upsample2x_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                               int H, int W, int in_pitch, vc4_buf *out,
                               int *ho_out, int *wo_out, int *opitch_out)
{
    if (!ctx || !in || !out || !ho_out || !wo_out || !opitch_out || nimg < 1)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (C <= 0 || C % 16 || H <= 0 || W <= 0 || in_pitch < W)
        return VC4_ERR_UNSUPPORTED;
    const int ho = 2 * H, wo = 2 * W;
    const int Wg = (int)al16((uint32_t)wo);
    if (out->bytes < (size_t)nimg * ho * Wg * C * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    char ncin[16], name[48];
    char *argv[] = { (char *)"up2", (char *)"--cin", ncin, NULL };
    snprintf(ncin, sizeof(ncin), "%d", C);
    snprintf(name, sizeof(name), "up2_%d", C);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, NHWC_MAX_QPU, &k);
    if (st != VC4_OK) {
        vc4__scratch_reset(ctx);
        return st;
    }
    /* the kernel loops over all channel blocks itself */
    const uint32_t srow = (uint32_t)in_pitch * C * 4u;
    const uint32_t drow = (uint32_t)Wg * C * 4u;
    const uint32_t src_img = (uint32_t)H * srow;
    const uint32_t dst_img = (uint32_t)ho * drow;
    const int units = nimg;
    int splits, ubatch;
    nhwc_split(units, NHWC_MAX_QPU, H, &splits, &ubatch);
    for (int base = 0; base < units && st == VC4_OK; base += ubatch) {
        int nbat = units - base < ubatch ? units - base : ubatch;
        int per = H / splits, rem = H % splits, q = 0;
        for (int t = 0; t < nbat; ++t) {
            int n = base + t;
            for (int sp = 0; sp < splits; ++sp) {
                int pair0 = sp * per + (sp < rem ? sp : rem);
                int cnt = per + (sp < rem ? 1 : 0);
                uint32_t *u = vc4__uni(ctx, k, q);
                u[0] = vc4_bus_addr(ctx, in) + (uint32_t)((size_t)n * src_img);
                u[1] = vc4_bus_addr(ctx, out) + (uint32_t)((size_t)n * dst_img);
                u[2] = (uint32_t)cnt;
                u[3] = (uint32_t)pair0 * srow;
                u[4] = (uint32_t)(2 * pair0) * drow;
                u[5] = (uint32_t)q;
                u[6] = (uint32_t)C * 4u;
                u[7] = srow;
                u[8] = drow;
                u[9] = (uint32_t)(Wg / 4);
                ++q;
            }
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, q, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)q * 16u;
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    if (st == VC4_OK) {
        *ho_out = ho;
        *wo_out = wo;
        *opitch_out = Wg;
    }
    return st;
}

/* ----------------------------------------------------------------- concat */
vc4_status vc4_concat_nhwc(vc4_ctx *ctx, const vc4_buf *const *src,
                           const int *Csrc, int nsrc, int nimg,
                           size_t pixels_per_img, vc4_buf *out, int Cdst)
{
    if (!ctx || !src || !Csrc || nsrc < 1 || !out || nimg < 1 || !pixels_per_img)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (pixels_per_img % 4)
        return VC4_ERR_INVAL;
    int ctot = 0;
    for (int i = 0; i < nsrc; ++i) {
        if (Csrc[i] <= 0 || Csrc[i] % 16)
            return VC4_ERR_UNSUPPORTED;
        ctot += Csrc[i];
    }
    if (ctot != Cdst || Cdst % 16)
        return VC4_ERR_INVAL;
    if (out->bytes < (size_t)nimg * pixels_per_img * Cdst * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    const uint32_t groups = (uint32_t)(pixels_per_img / 4u);
    int coff = 0;
    vc4_status st = VC4_OK;
    for (int i = 0; i < nsrc && st == VC4_OK; ++i) {
        char ncin[16], ncout[16], name[48];
        char *argv[] = { (char *)"copy", (char *)"--cin", ncin,
                         (char *)"--cout", ncout, NULL };
        snprintf(ncin, sizeof(ncin), "%d", Csrc[i]);
        snprintf(ncout, sizeof(ncout), "%d", Cdst);
        snprintf(name, sizeof(name), "copy_%d_%d", Csrc[i], Cdst);
        vc4_kernel *k = NULL;
        st = vc4__kernel(ctx, name, argv, NHWC_MAX_QPU, &k);
        if (st != VC4_OK)
            break;
        const size_t simg = pixels_per_img * Csrc[i] * 4u;
        const size_t dimg = pixels_per_img * Cdst * 4u;
        const uint32_t sg_step = 4u * Csrc[i] * 4u;
        const uint32_t dg_step = 4u * Cdst * 4u;
        for (int n = 0; n < nimg && st == VC4_OK; ++n) {
            const uint32_t per = (groups + (uint32_t)NHWC_MAX_QPU - 1u) /
                                 (uint32_t)NHWC_MAX_QPU;
            int q = 0;
            for (int t = 0; t < NHWC_MAX_QPU; ++t) {
                uint32_t g0 = (uint32_t)t * per;
                if (g0 >= groups)
                    break;
                uint32_t gc = groups - g0 < per ? groups - g0 : per;
                uint32_t *u = vc4__uni(ctx, k, q);
                u[0] = vc4_bus_addr(ctx, src[i]) +
                       (uint32_t)((size_t)n * simg) + g0 * sg_step;
                u[1] = vc4_bus_addr(ctx, out) +
                       (uint32_t)((size_t)n * dimg) + g0 * dg_step;
                u[2] = (uint32_t)Csrc[i] * 4u;
                u[3] = (uint32_t)Cdst * 4u;
                u[4] = gc;
                u[5] = (uint32_t)q;
                u[6] = 0;                   /* source starts at channel 0 */
                u[7] = (uint32_t)coff * 4u; /* destination channel offset */
                ++q;
            }
            uint64_t ns = 0;
            st = vc4__launch(ctx, k, q, &ns);
            ctx->st.kernel_ns += ns;
        }
        coff += Csrc[i];
    }
    vc4__l2_flush(ctx);
    vc4__scratch_reset(ctx);
    return st;
}
