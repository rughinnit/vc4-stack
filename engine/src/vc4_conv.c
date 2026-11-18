/*
 * vc4_conv.c - conv2d as im2col on the QPU + the verified GEMM.
 *
 * Path (everything on the GPU except byte shuffling):
 *   1. the padded NCHW image [Cin][H+2p][Wp] is materialised in the arena
 *      (zero fill + one memcpy per row: layout only, no arithmetic);
 *   2. a QPU kernel builds the packed A panel straight from that image.  For
 *      one row block (16 output pixels, contiguous in the padded image) and
 *      one filter tap it issues a VDR of 16 channels x 16 pixels followed by a
 *      vertical VDW, so the panel appears transposed with no im2col matrix and
 *      no ARM arithmetic;
 *   3. the weights [Cout][Cin*KH*KW] are transposed into a k-major B panel by
 *      the same pack kernel;
 *   4. the verified GEMM runs with bmode=1 (B from the packed panel).
 *
 * bias is folded into the GEMM as one extra k term (A row = 1.0,
 * B row = bias), so the bias add is on the GPU too.
 *
 * The final store to NCHW [Cout][HO][WO] is a strided copy on the ARM: it
 * moves bytes, it does not compute.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

vc4_status vc4_conv2d(vc4_ctx *ctx,
                      const vc4_buf *in, int Cin, int H, int W,
                      const vc4_buf *wt, int Cout, int KH, int KW,
                      const vc4_buf *bias,
                      int stride, int pad,
                      vc4_buf *out, int *HO, int *WO)
{
    if (!ctx || !in || !wt || !out || !HO || !WO)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 || KW <= 0)
        return VC4_ERR_INVAL;
    if (stride != 1)
        return VC4_ERR_UNSUPPORTED;          /* registry: conv2d/stride2        */
    if (!((KH == 3 && KW == 3) || (KH == 1 && KW == 1)))
        return VC4_ERR_UNSUPPORTED;          /* registry: only 3x3 and 1x1      */
    if (pad < 0 || (KH == 1 && pad != 0))
        return VC4_ERR_UNSUPPORTED;
    if (Cin % 16 || Cout % 16)
        return VC4_ERR_UNSUPPORTED;          /* panel tiles are 16 wide         */

    const int ho = (H + 2 * pad - KH) / stride + 1;
    const int wo = (W + 2 * pad - KW) / stride + 1;
    if (ho <= 0 || wo <= 0)
        return VC4_ERR_INVAL;
    const int hp = H + 2 * pad;
    const int wp = (int)vc4__align16((uint32_t)(W + 2 * pad));
    if ((size_t)hp * (size_t)wp * 4u > 8191u)
        return VC4_ERR_UNSUPPORTED;          /* 13-bit VDR pitch field          */

    const uint32_t K = (uint32_t)Cin * (uint32_t)(KH * KW);
    const uint32_t Kp = vc4__align16(K);
    const uint32_t nk = Kp + (bias ? 16u : 0u);
    const uint32_t M = (uint32_t)ho * (uint32_t)wp;   /* padded output rows     */
    const size_t in_need = (size_t)Cin * H * W * 4;
    const size_t wt_need = (size_t)Cout * K * 4;
    const size_t out_need = (size_t)Cout * ho * wo * 4;
    if (in->bytes < in_need || wt->bytes < wt_need || out->bytes < out_need)
        return VC4_ERR_INVAL;
    if ((size_t)M * nk * 4 > (64u << 20))
        return VC4_ERR_NOMEM;

    vc4__scratch_reset(ctx);

    /* ---- 1. padded image ------------------------------------------------ */
    /* +16 floats of slack: the last row block of the last channel can read up
       to KH-1 pixels past the end of the plane (the values are discarded) */
    float *pad_img = vc4__scratch_alloc(ctx, (size_t)Cin * hp * wp * 4 + 64, NULL);
    if (!pad_img) {
        vc4__set_err(ctx, "conv2d: no room for a %dx%dx%d padded image", Cin, hp, wp);
        return VC4_ERR_NOMEM;
    }
    memset((uint8_t *)pad_img, 0, (size_t)Cin * hp * wp * 4 + 64);
    for (int c = 0; c < Cin; ++c) {
        for (int y = 0; y < H; ++y) {
            memcpy((uint8_t *)pad_img + ((size_t)c * hp * wp + (size_t)(y + pad) * wp + pad) * 4,
                   ctx->arena + in->off + ((size_t)c * H + (size_t)y) * W * 4,
                   (size_t)W * 4);
        }
    }
    const uint32_t pad_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)pad_img - ctx->arena);

    /* ---- 2. im2col panel ------------------------------------------------- */
    vc4_buf pb;
    float *panel = vc4__scratch_alloc(ctx, (size_t)M * nk * 4, &pb);
    if (!panel) {
        vc4__set_err(ctx, "conv2d: no room for a %ux%u panel", M, nk);
        return VC4_ERR_NOMEM;
    }
    memset((uint8_t *)panel, 0, (size_t)M * nk * 4);

    {
        char ncin[16], nk_[16], nkh[8], nkw[8], name[64];
        char *argv[] = { (char *)"im2col", (char *)"--cin", ncin, (char *)"--k", nk_,
                         (char *)"--kh", nkh, (char *)"--kw", nkw, NULL };
        snprintf(ncin, sizeof(ncin), "%d", Cin);
        snprintf(nk_, sizeof(nk_), "%u", K);
        snprintf(nkh, sizeof(nkh), "%d", KH);
        snprintf(nkw, sizeof(nkw), "%d", KW);
        snprintf(name, sizeof(name), "im2col_%d_%u_%d_%d", Cin, K, KH, KW);

        vc4_kernel *k = NULL;
        vc4_status st = vc4__kernel(ctx, name, argv, 4, &k);
        if (st != VC4_OK) {
            vc4__scratch_reset(ctx);
            return st;
        }
        const uint32_t tr = M / 16u;
        int pnq = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
        const uint32_t per = tr / (uint32_t)pnq;
        for (int i = 0; i < pnq; ++i) {
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = (uint32_t)hp * wp * 4u;          /* plane byte pitch          */
            u[1] = (uint32_t)i * per * 64u;         /* src row-block offset      */
            u[2] = pad_bus;
            u[3] = 64u;                             /* src row-block stride      */
            u[4] = ctx->gpu_bus + pb.off;
            u[5] = per;
            u[6] = (uint32_t)i;
            u[7] = 1024;                            /* dst k-block stride        */
            u[8] = 64u * nk;                        /* dst row-block stride      */
            u[9] = (uint32_t)wp * 4u;               /* Wp*4                      */
            u[10] = (uint32_t)i * per * 64u * nk;   /* dst row-block offset      */
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, pnq, &ns);
        ctx->st.pack_ns += ns;
        if (st != VC4_OK) {
            vc4__scratch_reset(ctx);
            return st;
        }
    }

    /* ---- bias as an extra k term: A row 1.0, B row = bias ---------------- */
    if (bias) {
        if (bias->bytes < (size_t)Cout * 4) {
            vc4__scratch_reset(ctx);
            return VC4_ERR_INVAL;
        }
        /* A: every row block gets 1.0 in lane 0 of k = Kp, zeros elsewhere */
        for (uint32_t rb = 0; rb < M / 16u; ++rb) {
            float *blk = panel + ((size_t)rb * nk + Kp) * 16u;
            for (int l = 0; l < 16; ++l)
                blk[l] = 1.0f;
        }
    }

    /* ---- 3. weights -> k-major B panel ---------------------------------- */
    vc4_buf wb;
    float *wpanel = vc4__scratch_alloc(ctx, (size_t)Cout * nk * 4, &wb);
    if (!wpanel) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    memset((uint8_t *)wpanel, 0, (size_t)Cout * nk * 4);
    {
        /* The im2col kernel writes k blocks in tap-major order
           (k = tap*Cin + ci), while wt is [Cout][Cin][KH][KW] (k = ci*taps +
           tap).  Permute the weights into the panel's k order first; this is a
           byte copy, not arithmetic. */
        uint32_t wt_bus = vc4_bus_addr(ctx, wt);
        if (KH * KW > 1) {
            float *wr = vc4__scratch_alloc(ctx, (size_t)Cout * K * 4, NULL);
            if (!wr) { vc4__scratch_reset(ctx); return VC4_ERR_NOMEM; }
            const float *src = (const float *)(ctx->arena + wt->off);
            const int taps = KH * KW;
            for (int co = 0; co < Cout; ++co)
                for (int ci = 0; ci < Cin; ++ci)
                    for (int t = 0; t < taps; ++t)
                        wr[(size_t)co * K + (size_t)t * Cin + ci] =
                            src[(size_t)co * K + (size_t)ci * taps + t];
            wt_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)wr - ctx->arena);
        }
        /* one column block of the k-major B panel is nk*16 floats = 64*nk
           (nk includes the folded bias k-block) */
        vc4_status st = vc4__pack_run(ctx, wt_bus, K, 64u * K,
                                      ctx->gpu_bus + wb.off, 64u * nk,
                                      (uint32_t)Cout / 16u);
        (void)Kp;
        if (st != VC4_OK) {
            vc4__scratch_reset(ctx);
            return st;
        }
        if (bias) {
            const float *bsrc = (const float *)(ctx->arena + bias->off);
            for (uint32_t cb = 0; cb < (uint32_t)Cout / 16u; ++cb) {
                float *brow = wpanel + ((size_t)cb * nk + Kp) * 16u;
                memcpy(brow, bsrc + (size_t)cb * 16u, 16u * 4u);
            }
        }
    }

    /* ---- 4. GEMM --------------------------------------------------------- */
    vc4_buf cb;
    float *cmat = vc4__scratch_alloc(ctx, (size_t)M * Cout * 4, &cb);
    if (!cmat) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    vc4_status st = vc4__gemm_run(ctx, M, (uint32_t)Cout, nk, 1,
                                  ctx->gpu_bus + wb.off, ctx->gpu_bus + cb.off,
                                  ctx->gpu_bus + pb.off);
    if (st != VC4_OK) {
        vc4__scratch_reset(ctx);
        return st;
    }

    /* ---- 5. NCHW store (byte gather, no arithmetic) ---------------------- */
    vc4__l2_flush(ctx);
    for (int co = 0; co < Cout; ++co) {
        for (int y = 0; y < ho; ++y) {
            const float *src = cmat + ((size_t)y * wp) * Cout + co;
            float *dst = (float *)(ctx->arena + out->off) +
                         ((size_t)co * ho + y) * wo;
            for (int x = 0; x < wo; ++x)
                dst[x] = src[(size_t)x * Cout];
        }
    }
    *HO = ho;
    *WO = wo;
    vc4__scratch_reset(ctx);
    return VC4_OK;
}
