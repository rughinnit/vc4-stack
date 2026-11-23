/*
 * vc4_gemm.c - vc4_matmul on the asm10 hand-written QPU GEMM.
 *
 * Two launches per matmul, in one process and one arena:
 *   1. tools/kgen.py's "pack" kernel (asm10/pack10.py): the panel builder that
 *      transposes A into the 8-row k-major panel
 *      PANEL[(rb*K + k)*8 + r] = A[rb*8+r][k]
 *      with VDR loads and 8x8-word VDMs (no ALU data movement), assigned
 *      grid-stride to 4 QPUs.
 *   2. the asm10 GEMM (asm10/gemm6.py): 8x64 C tile per request, three B
 *      vector sets and a 6-phase (3 k pair) k loop, 16 lanes = 16 of the 64
 *      columns, A broadcast with the hardware rotate+replicate, VPM->DMA
 *      store of 4 column chunks x 2 row halves.
 *
 * Shapes are rounded up to 8 rows / 64 columns / 16 k.  The rounding is
 * *byte* work (zero fill and a row copy on the ARM); no multiply-accumulate
 * ever runs on the CPU.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build the 8-row k-major A panel from a [rowblocks*8][rowlen] row-major
 * source whose row pitch is rowlen floats and whose 8-row block stride is
 * 32*rowlen bytes (true for a contiguous Mp x Kp padded copy).
 *
 * The packer works on 8-row "work units" (32*K bytes of source and of panel)
 * assigned grid-stride: QPU i takes units i, i+pnq, i+2*pnq, ...  asm10
 * measured that this alone is worth 17 % over contiguous slabs (the four
 * concurrent read streams stay 8 KB apart instead of 64 KB apart).
 */
vc4_status vc4__pack_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen,
                         uint32_t dst_bus, uint32_t rowblocks)
{
    return vc4__pack_run_q(ctx, src_bus, rowlen, dst_bus, rowblocks, 0);
}

vc4_status vc4__pack_run_q(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen,
                           uint32_t dst_bus, uint32_t rowblocks, int q)
{
    if (rowlen % 16 || rowblocks < 1)
        return VC4_ERR_INVAL;
    const uint32_t K = rowlen;
    const uint32_t unitb = 32u * K;          /* bytes per 8-row work unit */
    const uint32_t units = rowblocks;
    /* the packer is a plain grid-stride loop with no in-kernel guard, so its
     * QPU count must divide the work exactly.  An explicit q that does not
     * divide is refused; the default walks down to a divisor, because the
     * old default of 4 silently left a partial unit for odd block counts. */
    int pnq = q > 0 ? q : (units < 4u ? (int)units : 4);
    if (q <= 0)
        while (pnq > 1 && units % (uint32_t)pnq)
            --pnq;
    if (pnq < 1 || pnq > 4 || (uint32_t)pnq > units || units % (uint32_t)pnq)
        return VC4_ERR_INVAL;

    char nk[16];
    char pname[48], *pargv[] = { (char *)"pack", (char *)"--k", nk, NULL };
    snprintf(nk, sizeof(nk), "%u", K);
    snprintf(pname, sizeof(pname), "pack10_%u", K);
    vc4_kernel *k = NULL;
    /* the blob is independent of nqpu; register it for the max of 4 so a
       different M with the same K can reuse it (the cache key is K only) */
    vc4_status st = vc4__kernel(ctx, pname, pargv, 4, &k);
    if (st != VC4_OK)
        return st;

    for (int i = 0; i < pnq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = K;                                        /* row length (floats) */
        u[1] = (uint32_t)i * unitb;                      /* first unit offset   */
        u[2] = src_bus;
        u[3] = unitb;                                    /* unit byte stride    */
        u[4] = dst_bus;
        u[5] = (units - (uint32_t)i + (uint32_t)pnq - 1u) / (uint32_t)pnq;
        u[6] = (uint32_t)i;                              /* VPM row base = 8*q  */
        u[7] = unitb * (uint32_t)pnq;                    /* unit byte step      */
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, pnq, &ns);
    ctx->st.pack_ns += ns;
    return st;
}

vc4_status vc4__gemm_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t b_bus, uint32_t c_bus, uint32_t panel_bus)
{
    return vc4__gemm_run_q(ctx, M, N, K, b_bus, c_bus, panel_bus, 0,
                           VC4_ACT_NONE, 0);
}

/* The activation's part of a blob name (empty for none).  A kernel with an
 * epilogue is a different program, so it must not share the cache entry of
 * the plain one. */
static void act_suffix(char *out, size_t n, int act)
{
    if (act == VC4_ACT_NONE) {
        out[0] = '\0';
        return;
    }
    snprintf(out, n, "_a%d", act);
}

/* A row-major C row pitch of N*4 bytes needs (N-16)*4 in the VDW store's
 * 16-bit memory-stride field (vpmwords.vdw_stride), which only fits while
 * N <= 16384.  Wider results are written as one dense 8x64 tile per (row
 * block, column block) and unpacked here: the tile pitch is 64 floats, so
 * the stride is 192 and the field cannot overflow. */
int vc4__matmul_compact(uint32_t N) { return N > 16384u; }

/* The asm10 GEMM's blob name.  Shared with the autotuner, which names the
 * kernel its recorded parameters select. */
void vc4__matmul_kernel_name(char *out, size_t n, uint32_t M, uint32_t N,
                             uint32_t K, int nq, int act)
{
    char sfx[8];
    act_suffix(sfx, sizeof(sfx), act);
    snprintf(out, n, "gemm10_%u_%u_%u_q%d%s%s", M, N, K, nq, sfx,
             vc4__matmul_compact(N) ? "_c" : "");
}

vc4_status vc4__gemm_run_q(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t b_bus, uint32_t c_bus, uint32_t panel_bus,
                           int gemm_q, int act, uint32_t c_rb)
{
    if (M % 8 || N % 64 || K % 16)
        return VC4_ERR_INVAL;
    if (act != VC4_ACT_NONE && act != VC4_ACT_RELU &&
        act != VC4_ACT_CLIP6) {
        /* The 8x64 tile holds 32 live C vectors over both files and leaves
         * one B register free after the k loop.  relu needs none (a small
         * immediate) and clip(0,6) needs two, both in registers the next
         * column block reloads; silu's twelve do not fit. */
        vc4__set_err(ctx, "matmul: activation %d needs more B registers than "
                     "the asm10 8x64 tile has free (relu and clip6 only)",
                     act);
        return VC4_ERR_UNSUPPORTED;
    }
    const uint32_t tr = M / 8;                 /* 8-row row blocks */
    int nq = gemm_q > 0 ? gemm_q : (tr < VC4_MAX_QPU ? (int)tr : VC4_MAX_QPU);
    if (nq < 1 || nq > VC4_MAX_QPU || (uint32_t)nq > tr)
        return VC4_ERR_INVAL;

    const int compact = vc4__matmul_compact(N);
    if (!c_rb)
        c_rb = 8u * N * 4u;
    char name[80], nm[16], nn[16], nk[16], nkpu[8], nact[8], ncmp[4];
    char *argv[] = { (char *)"gemm", (char *)"--M", nm, (char *)"--N", nn,
                     (char *)"--K", nk, (char *)"--nqpu", nkpu,
                     (char *)"--act", nact, (char *)"--compact", ncmp, NULL };
    snprintf(nm, sizeof(nm), "%u", M);
    snprintf(nn, sizeof(nn), "%u", N);
    snprintf(nk, sizeof(nk), "%u", K);
    snprintf(nkpu, sizeof(nkpu), "%d", nq);
    snprintf(nact, sizeof(nact), "%d", act);
    snprintf(ncmp, sizeof(ncmp), "%d", compact ? 1 : 0);
    vc4__matmul_kernel_name(name, sizeof(name), M, N, K, nq, act);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, nq, &k);
    if (st != VC4_OK)
        return st;

    const uint32_t a_rb = 32u * K;             /* one 8-row panel block     */
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

/* Scatter the compact tile layout into a row-major [M][N] result.
 *
 * The kernel writes one dense 8x64 tile per (row block, column block):
 * tile(rb, cb) at ((rb * NCB) + cb) * 2048, rows of 256 bytes inside it.
 * Both streams are sequential, so the copy is a row of 256-byte copies. */
static void unpack_compact(uint8_t *dst, const uint8_t *src, uint32_t M,
                           uint32_t N, uint32_t Np)
{
    const uint32_t cb_n = Np / 64u;
    const uint32_t rb_n = (M + 7u) / 8u;
    for (uint32_t rb = 0; rb < rb_n; ++rb) {
        const uint32_t rows = M - rb * 8u > 8u ? 8u : M - rb * 8u;
        for (uint32_t cb = 0; cb < cb_n; ++cb) {
            const uint32_t col0 = cb * 64u;
            if (col0 >= N)
                break;
            const uint32_t cols = N - col0 > 64u ? 64u : N - col0;
            const uint8_t *tile = src + ((size_t)rb * cb_n + cb) * 2048u;
            for (uint32_t r = 0; r < rows; ++r)
                memcpy(dst + (size_t)(rb * 8u + r) * N * 4u +
                       (size_t)col0 * 4u, tile + r * 256u, (size_t)cols * 4u);
        }
    }
}

vc4_status vc4__matmul_tuned(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                             vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                             vc4_layout layout, const vc4_tune_params *p,
                             int act)
{
    /* refuse before any work so an unsupported activation cannot leave a
     * panel behind in scratch */
    if (act != VC4_ACT_NONE && act != VC4_ACT_RELU &&
        act != VC4_ACT_CLIP6) {
        vc4__set_err(ctx, "matmul: activation %d needs more B registers than "
                     "the asm10 8x64 tile has free (relu and clip6 only)",
                     act);
        return VC4_ERR_UNSUPPORTED;
    }
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

    const uint32_t Mp = vc4__align8(M), Np = vc4__align64(N),
                   Kp = vc4__align16(K);
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

    /* A: the packer reads 8-row units of K floats with a constant pitch, so it
     * needs M 8-aligned and K 16-aligned. */
    uint32_t a_src_bus = a_bus;
    if (Mp != M || Kp != K) {
        float *ap = vc4__scratch_alloc(ctx, (size_t)Mp * Kp * 4, NULL);
        if (!ap)
            return VC4_ERR_NOMEM;
        zero((uint8_t *)ap, (size_t)Mp * Kp * 4);
        for (uint32_t r = 0; r < M; ++r)
            memcpy((uint8_t *)ap + (size_t)r * Kp * 4,
                   ctx->arena + A->off + (size_t)r * K * 4, (size_t)K * 4);
        a_src_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)ap - ctx->arena);
    }

    /* B: read row-major with row stride N*4; needs K 16-aligned and N
     * 64-aligned (one 64-column tile). */
    uint32_t b_src_bus = b_bus;
    if (Kp != K || Np != N) {
        float *bp = vc4__scratch_alloc(ctx, (size_t)Kp * Np * 4, NULL);
        if (!bp)
            return VC4_ERR_NOMEM;
        zero((uint8_t *)bp, (size_t)Kp * Np * 4);
        for (uint32_t r = 0; r < K; ++r)
            memcpy((uint8_t *)bp + (size_t)r * Np * 4,
                   ctx->arena + B->off + (size_t)r * N * 4, (size_t)N * 4);
        b_src_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)bp - ctx->arena);
    }

    /* C: the GEMM writes full 8x64 tiles.  Above 16384 columns the row pitch
     * does not fit the store's 16-bit stride, so the kernel writes a compact
     * tile layout instead (see vc4__matmul_compact). */
    const int compact = vc4__matmul_compact(Np);
    const uint32_t c_rb = compact ? Np * 32u : 8u * Np * 4u;
    uint32_t c_dst_bus = c_bus;
    float *c_tmp = NULL;
    if (Mp != M || Np != N || compact) {
        c_tmp = vc4__scratch_alloc(ctx, (size_t)Mp * Np * 4, NULL);
        if (!c_tmp)
            return VC4_ERR_NOMEM;
        memset((uint8_t *)c_tmp, 0, (size_t)Mp * Np * 4);
        c_dst_bus = ctx->gpu_bus + (uint32_t)((uint8_t *)c_tmp - ctx->arena);
    }

    vc4_status st = vc4__pack_run_q(ctx, a_src_bus, Kp, ctx->gpu_bus + pb.off,
                                    Mp / 8, p ? (int)p->pack_q : 0);
    if (st != VC4_OK) {
        vc4__scratch_reset(ctx);
        return st;
    }
    st = vc4__gemm_run_q(ctx, Mp, Np, Kp, b_src_bus, c_dst_bus,
                         ctx->gpu_bus + pb.off, p ? (int)p->gemm_q : 0, act,
                         c_rb);
    if (st == VC4_OK && c_tmp) {
        vc4__l2_flush(ctx);
        vc4__invalidate_off(ctx, C->off, (size_t)M * N * 4);
        if (compact)
            unpack_compact(ctx->arena + C->off, (const uint8_t *)c_tmp, M, N,
                           Np);
        else
            for (uint32_t r = 0; r < M; ++r)
                memcpy(ctx->arena + C->off + (size_t)r * N * 4,
                       (uint8_t *)c_tmp + (size_t)r * Np * 4, (size_t)N * 4);
        vc4__mark_dirty(ctx, C->off, (size_t)M * N * 4);
    }
    vc4__scratch_reset(ctx);
    return st;
}

/*
 * The public matmul: the tuned launch for this shape when a record exists
 * (VC4INFER_AUTOTUNE=1 or a vc4_autotune call), else the default.  A record
 * that no longer describes a legal launch - an older build's QPU count, say -
 * is not fatal: the default runs and is not used to overwrite the record.
 */
vc4_status vc4_matmul_act(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                          vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                          vc4_layout layout, vc4_act act)
{
    vc4_tune_shape sh;
    vc4_tune_params tp;
    vc4_status st;

    if (!ctx || !A || !B || !C || !M || !N || !K)
        return VC4_ERR_INVAL;
    vc4__tune_shape_mm(&sh, M, N, K);
    if (vc4__tune_lookup(ctx, VC4_OP_MATMUL, &sh, &tp)) {
        st = vc4__matmul_tuned(ctx, A, B, C, M, N, K, layout, &tp, act);
        if (st != VC4_ERR_INVAL)
            return st;
        vc4__set_err(ctx, "matmul: the tuned launch for %ux%ux%u was refused, "
                     "running the default", M, N, K);
    }
    return vc4__matmul_tuned(ctx, A, B, C, M, N, K, layout, NULL, act);
}

vc4_status vc4_matmul(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                      vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                      vc4_layout layout)
{
    return vc4_matmul_act(ctx, A, B, C, M, N, K, layout, VC4_ACT_NONE);
}

/* ==================================================================
 * asm3 GEMM/pack, kept for the conv path (proven end to end in
 * engine-fix).  The matmul uses the asm10 kernel above.
 * ================================================================== */
static int pack3_qpus(uint32_t tr)
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
vc4_status vc4__pack3_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen,
                         uint32_t src_rb_stride, uint32_t dst_bus,
                         uint32_t dst_rb_stride, uint32_t rowblocks)
{
    if (rowlen % 16 || rowblocks < 1)
        return VC4_ERR_INVAL;
    if ((uint64_t)rowlen * 4u > 8191u) {
        vc4__set_err(ctx, "pack3: source row %u floats (%llu B) exceeds the "
                     "13-bit VDR pitch field (8191 B); pack this K on the CPU "
                     "or in K-aligned chunks", rowlen,
                     (unsigned long long)rowlen * 4u);
        return VC4_ERR_UNSUPPORTED;
    }
    const uint32_t tr = rowblocks;
    const int pnq = pack3_qpus(tr);
    const uint32_t per = tr / (uint32_t)pnq;
    const uint32_t K = rowlen;
    const uint32_t kbytes = K * 4u;

    char nkb[16];
    char pname[48], *pargv[] = { (char *)"pack3", (char *)"--kblocks", nkb, NULL };
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

vc4_status vc4__gemm3_run_c(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                            int bmode, uint32_t b_bus, uint32_t c_bus,
                            uint32_t panel_bus, uint32_t cstride, int act)
{
    return vc4__gemm3_run_cq(ctx, M, N, K, bmode, b_bus, c_bus, panel_bus,
                             cstride, 0, act);
}

/* The asm3 GEMM's blob name for the conv path (bmode=1, cstride=N).  A
 * grouped conv passes a wider cstride and gets its own name; that path is not
 * autotuned, so it is not represented here. */
void vc4__conv_kernel_name(char *out, size_t n, uint32_t M, uint32_t N,
                           uint32_t K, int nq, int act)
{
    char sfx[8];
    act_suffix(sfx, sizeof(sfx), act);
    snprintf(out, n, "gemm_%u_%u_%u_b1_q%d%s", M, N, K, nq, sfx);
}

vc4_status vc4__gemm3_run_cq(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                             int bmode, uint32_t b_bus, uint32_t c_bus,
                             uint32_t panel_bus, uint32_t cstride, int gemm_q,
                             int act)
{
    if (M % 16 || N % 16 || K % 16)
        return VC4_ERR_INVAL;
    if (cstride == 0)
        cstride = N;
    if (cstride < N || cstride % 16)
        return VC4_ERR_INVAL;
    const uint32_t tr = M / 16;
    int nq = gemm_q > 0 ? gemm_q
                        : (tr < VC4_MAX_QPU ? (int)tr : VC4_MAX_QPU);
    if (nq < 1 || nq > VC4_MAX_QPU || (uint32_t)nq > tr)
        return VC4_ERR_INVAL;

    char name[96], nm[16], nn[16], nk[16], nb[8], nkpu[8], ncs[16], nact[8];
    char sfx[8];
    char *argv[] = { (char *)"gemm3", (char *)"--M", nm, (char *)"--N", nn,
                     (char *)"--K", nk, (char *)"--nqpu", nkpu,
                     (char *)"--bmode", nb, (char *)"--cstride", ncs,
                     (char *)"--act", nact, NULL };
    snprintf(nm, sizeof(nm), "%u", M);
    snprintf(nn, sizeof(nn), "%u", N);
    snprintf(nk, sizeof(nk), "%u", K);
    snprintf(nkpu, sizeof(nkpu), "%d", nq);
    snprintf(nb, sizeof(nb), "%d", bmode ? 1 : 0);
    snprintf(ncs, sizeof(ncs), "%u", cstride);
    snprintf(nact, sizeof(nact), "%d", act);
    act_suffix(sfx, sizeof(sfx), act);
    if (cstride == N)
        snprintf(name, sizeof(name), "gemm_%u_%u_%u_b%d_q%d%s", M, N, K,
                 bmode ? 1 : 0, nq, sfx);
    else
        snprintf(name, sizeof(name), "gemm_%u_%u_%u_b%d_q%d_cs%u%s", M, N, K,
                 bmode ? 1 : 0, nq, cstride, sfx);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, nq, &k);
    if (st != VC4_OK)
        return st;

    const uint32_t a_rb = 64u * K;
    const uint32_t c_rb = 16u * cstride * 4u;
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

vc4_status vc4__gemm3_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                          int bmode, uint32_t b_bus, uint32_t c_bus,
                          uint32_t panel_bus)
{
    return vc4__gemm3_run_c(ctx, M, N, K, bmode, b_bus, c_bus, panel_bus, N,
                            VC4_ACT_NONE);
}

/* zero3-fill helper for padded scratch copies */
static void zero3(uint8_t *p, size_t bytes) { memset(p, 0, bytes); }

