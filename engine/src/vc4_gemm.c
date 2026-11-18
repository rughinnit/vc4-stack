/*
 * vc4_gemm.c - vc4_matmul on the verified hand-written QPU GEMM.
 *
 * Two launches per matmul, in one process and one arena:
 *   1. tools/kgen.py's "pack" kernel: a QPU panel builder that transposes A
 *      into the k-major packed panel  PANEL[(rb*K + k)*16 + r] = A[rb*16+r][k]
 *      with a VDR load and a vertical VDW (no ALU data movement).
 *   2. the GEMM itself: 16x16 C tile per QPU request, 16 lanes = 16 columns,
 *      A read from the panel (one 64 B line per k), B read row-major, 39-word
 *      k loop, VPM->DMA store.
 *
 * Shapes are rounded up to a multiple of 16.  Padding is *byte* work (zero fill
 * and a row copy on the ARM); no multiply-accumulate ever runs on the CPU.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pack_qpus(uint32_t tr)
{
    if (tr % 4 == 0)
        return 4;
    if (tr % 2 == 0)
        return 2;
    return 1;
}

/* Build a k-major panel from a [rowblocks*16][rowlen] row-major source.
 * Used for the A panel (rowlen = K, dst stride 64*K) and for the weight
 * transpose into a k-major B panel (dst stride 64). */
vc4_status vc4__pack_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen,
                         uint32_t src_rb_stride, uint32_t dst_bus,
                         uint32_t dst_rb_stride, uint32_t rowblocks)
{
    if (rowlen % 16 || rowblocks < 1)
        return VC4_ERR_INVAL;
    const uint32_t tr = rowblocks;
    const int pnq = pack_qpus(tr);
    const uint32_t per = tr / (uint32_t)pnq;
    const uint32_t K = rowlen;
    const uint32_t kbytes = K * 4u;

    char nkb[16];
    char pname[48], *pargv[] = { (char *)"pack", (char *)"--kblocks", nkb, NULL };
    snprintf(nkb, sizeof(nkb), "%u", K / 16u);
    snprintf(pname, sizeof(pname), "pack_%u", K);
    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, pname, pargv, 4, &k);
    if (st != VC4_OK)
        return st;

    for (int i = 0; i < pnq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = kbytes / 4u;                       /* source row length        */
        u[1] = (uint32_t)i * per * src_rb_stride; /* rb0 source byte offset   */
        u[2] = src_bus;
        u[3] = src_rb_stride;
        u[4] = dst_bus;
        u[5] = per;                               /* row blocks for this QPU  */
        u[6] = (uint32_t)i;                       /* VPM row base = 16*q      */
        u[7] = 1024;                              /* dst k-block stride       */
        u[8] = dst_rb_stride;
        u[9] = (uint32_t)i * per * dst_rb_stride;
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, pnq, &ns);
    ctx->st.pack_ns += ns;
    return st;
}

vc4_status vc4__gemm_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                         int bmode, uint32_t b_bus, uint32_t c_bus,
                         uint32_t panel_bus)
{
    if (M % 16 || N % 16 || K % 16)
        return VC4_ERR_INVAL;
    const uint32_t tr = M / 16;
    int nq = tr < VC4_MAX_QPU ? (int)tr : VC4_MAX_QPU;

    char name[80], nm[16], nn[16], nk[16], nb[8], nkpu[8];
    char *argv[] = { (char *)"gemm", (char *)"--M", nm, (char *)"--N", nn,
                     (char *)"--K", nk, (char *)"--nqpu", nkpu,
                     (char *)"--bmode", nb, NULL };
    snprintf(nm, sizeof(nm), "%u", M);
    snprintf(nn, sizeof(nn), "%u", N);
    snprintf(nk, sizeof(nk), "%u", K);
    snprintf(nkpu, sizeof(nkpu), "%d", nq);
    snprintf(nb, sizeof(nb), "%d", bmode ? 1 : 0);
    snprintf(name, sizeof(name), "gemm_%u_%u_%u_b%d_q%d", M, N, K, bmode ? 1 : 0, nq);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, nq, &k);
    if (st != VC4_OK)
        return st;

    const uint32_t a_rb = 64u * K;
    const uint32_t c_rb = 16u * N * 4u;
    for (int i = 0; i < nq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = panel_bus;
        u[1] = c_bus;
        u[2] = b_bus;
        u[3] = (uint32_t)i * a_rb;
        u[4] = (uint32_t)nq * a_rb;
        u[5] = (uint32_t)i * c_rb;
        u[6] = (uint32_t)nq * c_rb;
        u[7] = (uint32_t)i;       /* first row block this QPU owns */
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, nq, &ns);
    ctx->st.kernel_ns += ns;
    if (st == VC4_OK)
        ctx->st.flops += 2ull * M * N * K;
    return st;
}

/* zero-fill helper for padded scratch copies */
static void zero(uint8_t *p, size_t bytes) { memset(p, 0, bytes); }

vc4_status vc4_matmul(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                      vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                      vc4_layout layout)
{
    if (!ctx || !A || !B || !C || !M || !N || !K)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (layout != VC4_LAYOUT_ROW_MAJOR)
        return VC4_ERR_UNSUPPORTED;
    if (A->bytes < (size_t)M * K * 4 || B->bytes < (size_t)N * K * 4 ||
        C->bytes < (size_t)M * N * 4) {
        vc4__set_err(ctx, "matmul: buffer too small for %ux%ux%u", M, N, K);
        return VC4_ERR_INVAL;
    }

    const uint32_t Mp = vc4__align16(M), Np = vc4__align16(N), Kp = vc4__align16(K);
    const uint32_t a_bus = vc4_bus_addr(ctx, A);
    const uint32_t b_bus = vc4_bus_addr(ctx, B);
    const uint32_t c_bus = vc4_bus_addr(ctx, C);

    vc4__scratch_reset(ctx);

    /* A panel (always scratch: it is the coalesced copy the GEMM reads) */
    vc4_buf pb;
    uint8_t *panel = vc4__scratch_alloc(ctx, (size_t)Mp * Kp * 4, &pb);
    if (!panel) {
        vc4__set_err(ctx, "matmul: no room for a %ux%u panel", Mp, Kp);
        return VC4_ERR_NOMEM;
    }

    /* A: the packer reads rows of K floats with a constant pitch, and M/16 row
     * blocks of 16 rows, so it needs M and K 16-aligned. */
    uint32_t a_src_bus = a_bus;
    int a_padded = 0;
    if ((M % 16) || (K % 16)) {
        float *ap = vc4__scratch_alloc(ctx, (size_t)Mp * Kp * 4, NULL);
        if (!ap)
            return VC4_ERR_NOMEM;
        zero((uint8_t *)ap, (size_t)Mp * Kp * 4);
        for (uint32_t r = 0; r < M; ++r)
            memcpy((uint8_t *)ap + (size_t)r * Kp * 4,
                   ctx->arena + A->off + (size_t)r * K * 4, (size_t)K * 4);
        a_src_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)ap - ctx->arena);
        a_padded = 1;
    }
    (void)a_padded;

    /* B: read row-major with stride N*4; needs K and N 16-aligned */
    uint32_t b_src_bus = b_bus;
    if ((K % 16) || (N % 16)) {
        float *bp = vc4__scratch_alloc(ctx, (size_t)Kp * Np * 4, NULL);
        if (!bp)
            return VC4_ERR_NOMEM;
        zero((uint8_t *)bp, (size_t)Kp * Np * 4);
        for (uint32_t r = 0; r < K; ++r)
            memcpy((uint8_t *)bp + (size_t)r * Np * 4,
                   ctx->arena + B->off + (size_t)r * N * 4, (size_t)N * 4);
        b_src_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)bp - ctx->arena);
    }

    /* C: the GEMM writes full 16x16 tiles */
    uint32_t c_dst_bus = c_bus;
    float *c_tmp = NULL;
    if ((M % 16) || (N % 16)) {
        c_tmp = vc4__scratch_alloc(ctx, (size_t)Mp * Np * 4, NULL);
        if (!c_tmp)
            return VC4_ERR_NOMEM;
        memset((uint8_t *)c_tmp, 0, (size_t)Mp * Np * 4);
        c_dst_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)c_tmp - ctx->arena);
    }

    vc4_status st = vc4__pack_run(ctx, a_src_bus, Kp, 64u * Kp,
                                  ctx->gpu_bus + pb.off, 64u * Kp, Mp / 16);
    if (st != VC4_OK) {
        vc4__scratch_reset(ctx);
        return st;
    }
    st = vc4__gemm_run(ctx, Mp, Np, Kp, 0, b_src_bus, c_dst_bus,
                       ctx->gpu_bus + pb.off);
    if (st == VC4_OK && c_tmp) {
        vc4__l2_flush(ctx);
        for (uint32_t r = 0; r < M; ++r)
            memcpy(ctx->arena + C->off + (size_t)r * N * 4,
                   (uint8_t *)c_tmp + (size_t)r * Np * 4, (size_t)N * 4);
    }
    vc4__scratch_reset(ctx);
    return st;
}
