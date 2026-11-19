/*
 * vc4_train.c - the training primitives that the inference engine lacked:
 * transpose, B-transposed GEMM (C = A B^T), the fused softmax+CE backward,
 * and the smaller kernels' wrappers.
 *
 * Everything that *computes* runs on a QPU.  The ARM only
 *   - zero-fills and lays out padded panels (byte copies, as in vc4_gemm.c),
 *   - writes uniform/scalar words (hyperparameters),
 *   - downloads buffers for the CPU oracle in the tests (a checker, never a
 *     compute path).
 *
 * This file adds no new GEMM math: vc4_matmul_bt is the asm3 packer (which
 * already transposes A into a k-major panel) applied to B, followed by the
 * verified GEMM with bmode=1.  vc4_transpose is the same packer with a
 * strided VDW destination: a 16x16 block transpose with no ALU work.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================================================================
 * transpose: rows x cols -> cols x rows (both multiples of 16)
 *
 * The pack kernel loads 16 source rows x 16 words (VDR, pitch = the source
 * row stride) and writes the transposed 16-word rows with a vertical VDW.
 * For the k-major panel the rows are written contiguously; here the VDW
 * destination stride is the output row pitch, so each 1 KB block lands as a
 * 16x16 transposed tile.  Per-kblock and per-rowblock strides put the tiles
 * where a dense transpose expects them.
 * ================================================================== */
vc4_status vc4__tpose_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rows,
                          uint32_t cols, uint32_t dst_bus)
{
    if (rows % 16 || cols % 16 || rows < 16 || cols < 16)
        return VC4_ERR_INVAL;
    /* VDR pitch field is 13 bits: the whole source row must fit. */
    if (cols * 4u > 8191u)
        return VC4_ERR_UNSUPPORTED;
    const uint32_t dstride = rows * 4u - 64u;      /* extra VDW row stride */
    if (dstride > 0xFFFFu)
        return VC4_ERR_UNSUPPORTED;

    const uint32_t tr = rows / 16u;
    int pnq = (tr % 4 == 0) ? 4 : (tr % 2 == 0) ? 2 : 1;
    const uint32_t per = tr / (uint32_t)pnq;

    char nkb[16], nds[16], name[56];
    char *argv[] = { (char *)"pack3", (char *)"--kblocks", nkb,
                     (char *)"--dstride", nds, NULL };
    snprintf(nkb, sizeof(nkb), "%u", cols / 16u);
    snprintf(nds, sizeof(nds), "%u", dstride);
    snprintf(name, sizeof(name), "tpose_%u_%u", rows, cols);

    vc4_kernel *k = NULL;
    vc4_status st = vc4__kernel(ctx, name, argv, pnq, &k);
    if (st != VC4_OK)
        return st;

    const uint32_t src_rb = 64u * cols;            /* 16 rows x cols floats */
    for (int i = 0; i < pnq; ++i) {
        uint32_t *u = vc4__uni(ctx, k, i);
        u[0] = cols;                               /* source row length      */
        u[1] = (uint32_t)i * per * src_rb;         /* src rb byte offset     */
        u[2] = src_bus;
        u[3] = src_rb;                             /* src rb byte stride     */
        u[4] = dst_bus;
        u[5] = per;                                /* row blocks this QPU    */
        u[6] = (uint32_t)i;                        /* VPM row base = 16*q    */
        u[7] = 16u * rows * 4u;                    /* dst per k-block        */
        u[8] = 64u;                                /* dst per row block      */
        u[9] = (uint32_t)i * per * 64u;            /* dst row block offset   */
    }
    uint64_t ns = 0;
    st = vc4__launch(ctx, k, pnq, &ns);
    ctx->st.kernel_ns += ns;
    if (st == VC4_OK)
        ctx->st.flops += (uint64_t)rows * cols;    /* data movement, counted */
    return st;
}

vc4_status vc4_transpose(vc4_ctx *ctx, const vc4_buf *in, vc4_buf *out,
                         uint32_t rows, uint32_t cols)
{
    if (!ctx || !in || !out || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (rows % 16 || cols % 16)
        return VC4_ERR_UNSUPPORTED;                /* 16x16 tiles only       */
    if (in->bytes < (size_t)rows * cols * 4 ||
        out->bytes < (size_t)rows * cols * 4)
        return VC4_ERR_INVAL;
    return vc4__tpose_run(ctx, vc4_bus_addr(ctx, in), rows, cols,
                          vc4_bus_addr(ctx, out));
}

/* ==================================================================
 * C = A * B^T   (A is M x K, B is N x K)
 *
 * GEMM with bmode=1 reads B from a k-major panel whose element
 * PANEL[(nblk*K + k)*16 + r] = B[nblk*16+r][k]; the packer builds exactly
 * that from B's rows.  A is packed the same way (the GEMM always reads A
 * from a k-major panel).  This is the 16x16-tile asm3 GEMM, K%16 needed.
 * ================================================================== */
vc4_status vc4_matmul_bt(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                         vc4_buf *C, uint32_t M, uint32_t N, uint32_t K)
{
    if (!ctx || !A || !B || !C || !M || !N || !K)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (M % 16 || N % 16 || K % 16)
        return VC4_ERR_UNSUPPORTED;
    if (A->bytes < (size_t)M * K * 4 || B->bytes < (size_t)N * K * 4 ||
        C->bytes < (size_t)M * N * 4)
        return VC4_ERR_INVAL;

    vc4__scratch_reset(ctx);
    vc4_buf pa, pb;
    uint8_t *a_panel = vc4__scratch_alloc(ctx, (size_t)M * K * 4, &pa);
    uint8_t *b_panel = vc4__scratch_alloc(ctx, (size_t)N * K * 4, &pb);
    if (!a_panel || !b_panel) {
        vc4__set_err(ctx, "matmul_bt: no room for %ux%u/%ux%u panels",
                     M, K, N, K);
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }

    vc4_status st = vc4__pack3_run(ctx, vc4_bus_addr(ctx, A), K, 64u * K,
                                   pa.off + ctx->gpu_bus, 64u * K, M / 16u);
    if (st == VC4_OK)
        st = vc4__pack3_run(ctx, vc4_bus_addr(ctx, B), K, 64u * K,
                            pb.off + ctx->gpu_bus, 64u * K, N / 16u);
    if (st == VC4_OK)
        st = vc4__gemm3_run(ctx, M, N, K, 1, pb.off + ctx->gpu_bus,
                            vc4_bus_addr(ctx, C), pa.off + ctx->gpu_bus);
    vc4__scratch_reset(ctx);
    return st;
}

/* ==================================================================
 * fused softmax + cross-entropy backward
 *
 *   dlogits[r][c] = (softmax(logits[r])[c] - onehot(labels[r])[c]) / B
 *   partial[q][0] = sum over this QPU's rows of -log p_label
 *   partial[q][1] = count of rows whose argmax equals the label
 *
 * Rows are processed 8 at a time (one 512 B store block); the 16 columns are
 * the model's padded output width.  q kernel writes a 64 B slot, so the host
 * reads 12 slots and a GPU GEMM (ones^T * partial) reduces the totals.
 *
 * Buffer requirements: logits/dlogits >= rows*64 + 64 bytes (the label-offset
 * gather reads a 64 B line), labels >= rows*4 + 64, partial >= nqpu*64.
 * ================================================================== */
vc4_status vc4_softmax_ce(vc4_ctx *ctx, const vc4_buf *logits,
                          const vc4_buf *labels, vc4_buf *dlogits,
                          vc4_buf *partial, uint32_t rows, float inv_batch)
{
    if (!ctx || !logits || !labels || !dlogits || !partial || !rows)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (rows % 8)
        return VC4_ERR_UNSUPPORTED;                /* 8-row groups           */
    const uint32_t groups = rows / 8u;
    if (logits->bytes < (size_t)rows * 64 + 64 ||
        dlogits->bytes < (size_t)rows * 64 + 64 ||
        labels->bytes < (size_t)rows * 4 + 64 ||
        partial->bytes < (size_t)groups * 64)
        return VC4_ERR_INVAL;

    /* VC4_SOFTMAX_PROBE is a bring-up aid: it makes the kernel store an
       intermediate (max/sum/exp/p/zy/cor) instead of dlogits.  Not used in
       training. */
    const char *probe = getenv("VC4_SOFTMAX_PROBE");
    if (probe && strcmp(probe, "full") == 0)
        probe = NULL;
    char name[48], nqbuf[8];
    char *argv[] = { (char *)"softmax", (char *)"--probe",
                     (char *)(probe ? probe : "full"), NULL };

    uint32_t ib;
    memcpy(&ib, &inv_batch, 4);
    vc4_status st = VC4_OK;
    for (uint32_t base = 0; base < groups && st == VC4_OK; base += VC4_MAX_QPU) {
        uint32_t left = groups - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        snprintf(name, sizeof(name), "softmax8_%s_q%d", probe ? probe : "full", nq);
        snprintf(nqbuf, sizeof(nqbuf), "%d", nq);
        (void)nqbuf;
        vc4_kernel *k = NULL;
        st = vc4__kernel(ctx, name, argv, nq, &k);
        if (st != VC4_OK)
            break;
        for (int i = 0; i < nq; ++i) {
            uint32_t g = base + (uint32_t)i;
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = vc4_bus_addr(ctx, logits) + g * 512u;
            u[1] = vc4_bus_addr(ctx, dlogits) + g * 512u;
            u[2] = vc4_bus_addr(ctx, labels) + g * 32u;
            u[3] = vc4_bus_addr(ctx, partial) + g * 64u;
            u[4] = (uint32_t)i;          /* VPM row base = 4*i */
            u[5] = ib;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 8u * 16u * 40u;
    }
    return st;
}

/* Forward softmax: the same fused kernel with its 'p' probe, which stores
 * p = softmax(row) instead of dlogits.  The label path still executes (the
 * kernel is one straight-line block), so the dummy label buffer holds zeros
 * and the probe discards the one-hot term.  The host pads each row to 16
 * classes with -1e30 and compacts the probabilities back to `cols`. */
vc4_status vc4_softmax_fwd(vc4_ctx *ctx, const vc4_buf *logits,
                           uint32_t rows, uint32_t cols, vc4_buf *out)
{
    if (!ctx || !logits || !out || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if (cols > 16 || rows % 8)
        return VC4_ERR_UNSUPPORTED;
    if (logits->bytes < (size_t)rows * cols * 4 ||
        out->bytes < (size_t)rows * cols * 4)
        return VC4_ERR_INVAL;

    const uint32_t groups = rows / 8u;
    vc4__scratch_reset(ctx);

    vc4_buf lb, ob, pb, partb;
    float *lp = vc4__scratch_alloc(ctx, (size_t)rows * 64 + 64, &lb);
    float *op = vc4__scratch_alloc(ctx, (size_t)rows * 64 + 64, &ob);
    int32_t *labels = vc4__scratch_alloc(ctx, (size_t)rows * 4 + 64, &pb);
    float *partial = vc4__scratch_alloc(ctx, (size_t)groups * 64, &partb);
    if (!lp || !op || !labels || !partial) {
        vc4__scratch_reset(ctx);
        return VC4_ERR_NOMEM;
    }
    /* logits padding must contribute exp = 0; the labels are never used by
       the probe path but the load must be a valid in-arena address */
    for (uint32_t i = 0; i < rows * 16u + 16u; ++i)
        lp[i] = -1e30f;
    memset(labels, 0, (size_t)rows * 4 + 64);
    memset(op, 0, (size_t)rows * 64 + 64);
    /* the logits are arena data the QPU wrote: drop our cached copies first */
    vc4__invalidate_off(ctx, logits->off, (size_t)rows * cols * 4);
    for (uint32_t r = 0; r < rows; ++r)
        memcpy(lp + (size_t)r * 16u,
               ctx->arena + logits->off + (size_t)r * cols * 4,
               (size_t)cols * 4);
    vc4__mark_dirty(ctx, lb.off, (size_t)rows * 64 + 64);
    vc4__mark_dirty(ctx, pb.off, (size_t)rows * 4 + 64);

    char name[48];
    char *argv[] = { (char *)"softmax", (char *)"--probe", (char *)"p", NULL };
    uint32_t one_bits;
    {
        float one = 1.0f;
        memcpy(&one_bits, &one, 4);
    }
    vc4_status st = VC4_OK;
    for (uint32_t base = 0; base < groups && st == VC4_OK; base += VC4_MAX_QPU) {
        uint32_t left = groups - base;
        int nq = left < (uint32_t)VC4_MAX_QPU ? (int)left : VC4_MAX_QPU;
        snprintf(name, sizeof(name), "softmax8_p_q%d", nq);
        vc4_kernel *k = NULL;
        st = vc4__kernel(ctx, name, argv, nq, &k);
        if (st != VC4_OK)
            break;
        for (int i = 0; i < nq; ++i) {
            uint32_t g = base + (uint32_t)i;
            uint32_t *u = vc4__uni(ctx, k, i);
            u[0] = ctx->gpu_bus + lb.off + g * 512u;
            u[1] = ctx->gpu_bus + ob.off + g * 512u;
            u[2] = ctx->gpu_bus + pb.off + g * 32u;
            u[3] = ctx->gpu_bus + partb.off + g * 64u;
            u[4] = (uint32_t)i;
            u[5] = one_bits;
        }
        uint64_t ns = 0;
        st = vc4__launch(ctx, k, nq, &ns);
        ctx->st.kernel_ns += ns;
        if (st == VC4_OK)
            ctx->st.flops += (uint64_t)nq * 8u * 16u * 40u;
    }
    if (st == VC4_OK) {
        vc4__l2_flush(ctx);
        vc4__invalidate_off(ctx, out->off, (size_t)rows * cols * 4);
        for (uint32_t r = 0; r < rows; ++r)
            memcpy(ctx->arena + out->off + (size_t)r * cols * 4,
                   op + (size_t)r * 16u, (size_t)cols * 4);
        vc4__mark_dirty(ctx, out->off, (size_t)rows * cols * 4);
    }
    vc4__scratch_reset(ctx);
    return st;
}
