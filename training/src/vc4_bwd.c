/*
 * vc4_bwd.c - the training core: a backward entry point for every
 * differentiable op the engine has.
 *
 * Every arithmetic operation here runs on a QPU, and every primitive it uses
 * is one of the kernels the test suite already verifies element by element:
 *
 *   matmul      the asm10 GEMM (test_matmul).  It zero-pads A/B/C into
 *               scratch itself, so any M,N,K is safe.  Two uses:
 *                 - (rows x 1) * (1 x cols) with a vector of ones is an exact
 *                   broadcast;
 *                 - (rows x cols) * (cols x 1) of ones is an exact row sum;
 *                 - a host-built 0/1 or weight matrix times a vector is an
 *                   exact table gather / scatter-add / interpolation.
 *               The last form is O(n_out * n_in) MACs, so the irregular
 *               layouts (arbitrary pooling windows, bilinear taps, gather
 *               tables) are capped by VC4_MATVEC_MAX and documented.
 *   elementwise add/mul/relu/sigmoid/silu/relu_bwd/rsqrt (test_ew)
 *   ewb         the derivative family: one body on the *same* verified
 *               build_ew harness (vc4_bwop_run)
 *   uop         affine / exp / log / sqrt / reciprocal / erf / tanh ...
 *   maxpooln    the generic max pool (test_onnxops), used by maxpool bwd
 *   transpose   the 16x16 packer transpose (test_matmul)
 *   softmax8    the fused row softmax (test_onnxops + test_grad)
 *   copy2d      pitched/affine copies (test_copyops)
 *
 * WHY NOT vc4_gather / vc4_reduce: both are in the kops "generic kernel"
 * family, which no test in this tree exercises.  On the board they wedge the
 * QPU (the gather kernel hands the TMU an address from an accumulator --
 * `areg(100)` -- and its pop order does not match the build_ew pipeline; the
 * reduce kernel seeds register 36 instead of r0 and walks a fixed 256-output
 * group past the caller's buffer).  The training path therefore does not call
 * them.  docs/TRAINING.md records this as a known gap in the *inference* path.
 *
 * The ARM only
 *   - builds integer index tables / 0-1 weight matrices from *geometry*
 *     (window bounds, corner indices, gather tables) and, for bilinear
 *     resize, the corner weights (a function of the output grid alone: no
 *     activation value is ever read back),
 *   - writes scalar hyperparameters (eps, momentum, inv_batch, scales),
 *   - pads a tail to a 256-float group (byte copy).
 *
 * Temporaries live in the *workspace*: a bump region carved down from the end
 * of the arena (vc4__ws_alloc), so the per-op scratch reset that every nested
 * kernel performs cannot overwrite them.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * helpers
 * ====================================================================== */
enum { EW_ADD, EW_SUB, EW_MUL, EW_DIV, EW_EQMASK };

/*
 * a - b and a / b, composed from the verified kernels: the ewb kernels that
 * read their second TMU stream do not see it on this board (the a-only ones,
 * exp and rcp, do), so subtraction is an fma and division is a multiply by
 * the reciprocal.
 */
static vc4_status e_sub(vc4_ctx *ctx, const vc4_buf *a, const vc4_buf *b,
                        vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(ctx);
    vc4_buf t;
    float *p = vc4__ws_alloc(ctx, n * 4, &t);
    vc4_status st = p ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4_unary(ctx, VC4_UOP_AFFINE, b, &t, n, -1.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(ctx, VC4_OP_ADD, a, &t, o, n);
    vc4__ws_end(ctx, top);
    return st;
}

static vc4_status e_div(vc4_ctx *ctx, const vc4_buf *a, const vc4_buf *b,
                        vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(ctx);
    vc4_buf r;
    float *p = vc4__ws_alloc(ctx, n * 4, &r);
    vc4_status st = p ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = vc4__ewb(ctx, "rcp", 0, b, b, &r, n, 0, 0);
    if (st == VC4_OK)
        st = vc4_elementwise(ctx, VC4_OP_MUL, a, &r, o, n);
    vc4__ws_end(ctx, top);
    return st;
}

/* 1[a == b]: |a-b| saturated to 1 and inverted, all on the QPU */
static vc4_status e_eqmask(vc4_ctx *ctx, const vc4_buf *a, const vc4_buf *b,
                           vc4_buf *o, size_t n)
{
    size_t top = vc4__ws_begin(ctx);
    vc4_buf d, s1, s2, t;
    float *p1 = vc4__ws_alloc(ctx, n * 4, &d);
    float *p2 = vc4__ws_alloc(ctx, n * 4, &s1);
    float *p3 = vc4__ws_alloc(ctx, n * 4, &s2);
    float *p4 = vc4__ws_alloc(ctx, n * 4, &t);
    vc4_status st = (p1 && p2 && p3 && p4) ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = e_sub(ctx, a, b, &d, n);
    if (st == VC4_OK)
        st = vc4_relu(ctx, &d, &s1, n);                    /* max(d,0) */
    if (st == VC4_OK)
        st = vc4_unary(ctx, VC4_UOP_AFFINE, &d, &t, n, -1.0f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_relu(ctx, &t, &s2, n);                    /* max(-d,0) */
    if (st == VC4_OK)
        st = vc4_elementwise(ctx, VC4_OP_ADD, &s1, &s2, &d, n);   /* |d| */
    if (st == VC4_OK)
        st = vc4_unary(ctx, VC4_UOP_AFFINE, &d, &d, n, 1e30f, 0.0f, 0);
    if (st == VC4_OK)
        st = vc4_unary(ctx, VC4_UOP_AFFINE, &d, &d, n, -1.0f, 1.0f, 0);
    if (st == VC4_OK)
        st = vc4_relu(ctx, &d, o, n);
    vc4__ws_end(ctx, top);
    return st;
}

static vc4_status ew(vc4_ctx *ctx, int op, const vc4_buf *a, const vc4_buf *b,
                     vc4_buf *o, size_t n)
{
    switch (op) {
    case EW_ADD: return vc4_elementwise(ctx, VC4_OP_ADD, a, b, o, n);
    case EW_MUL: return vc4_elementwise(ctx, VC4_OP_MUL, a, b, o, n);
    case EW_SUB: return e_sub(ctx, a, b, o, n);
    case EW_DIV: return e_div(ctx, a, b, o, n);
    case EW_EQMASK: return e_eqmask(ctx, a, b, o, n);
    default: return VC4_ERR_INVAL;
    }
}

vc4_status vc4_affine(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n,
                      float p0, float p1)
{
    return vc4_unary(ctx, VC4_UOP_AFFINE, a, out, n, p0, p1, 0.0f);
}

/* exp/log/sqrt/reciprocal/tanh/erf as the compact ewb kernels: the uop
 * family's nvec=4 bodies write no output on this board (see the file
 * header), so the training path uses these instead. */
static vc4_status u_exp(vc4_ctx *c, const vc4_buf *a, vc4_buf *o, size_t n)
{ return vc4__ewb(c, "exp", 0, a, a, o, n, 0, 0); }
static vc4_status u_log(vc4_ctx *c, const vc4_buf *a, vc4_buf *o, size_t n)
{ return vc4__ewb(c, "log", 0, a, a, o, n, 0, 0); }

static size_t al64(size_t v) { return (v + 63u) & ~(size_t)63u; }

static int64_t prod_dims(const int *d, int n)
{
    int64_t p = 1;
    for (int i = 0; i < n; ++i)
        p *= d[i];
    return p;
}

typedef struct {
    vc4_ctx *ctx;
    size_t   top;
    int      ok;
} bw_ws;

static void ws_on(bw_ws *w, vc4_ctx *ctx)
{
    w->ctx = ctx;
    w->top = vc4__ws_begin(ctx);
    w->ok = 1;
}

static vc4_buf ws_get(bw_ws *w, size_t bytes)
{
    vc4_buf b = { 0, 0 };
    if (!w->ok)
        return b;
    if (!vc4__ws_alloc(w->ctx, bytes, &b)) {
        if (getenv("VC4_BWD_DEBUG"))
            fprintf(stderr, "[bwd] ws_get %zu B failed, free %zu, used %zu\n",
                    bytes, vc4__ws_free(w->ctx), vc4_arena_used(w->ctx));
        w->ok = 0;
    }
    return b;
}

static void ws_off(bw_ws *w) { vc4__ws_end(w->ctx, w->top); }

static void fbuf_write(vc4_ctx *ctx, vc4_buf *b, const float *v, size_t n)
{
    memcpy(ctx->arena + b->off, v, n * 4);
    vc4__mark_dirty(ctx, b->off, n * 4);
}

static int32_t off_of(const vc4_buf *b, size_t idx)
{
    return (int32_t)(b->off + idx * 4);
}

/* contiguous float copy; 64 B multiples use the blit kernel, everything else
 * a copy2d row copy (still on the QPU) */
static vc4_status copy_floats(vc4_ctx *ctx, const vc4_buf *src, vc4_buf *dst,
                              size_t n)
{
    if (src->off == dst->off)
        return VC4_OK;
    uint32_t full = (uint32_t)(n / 16), rem = (uint32_t)(n % 16);
    vc4_status st = VC4_OK;
    if (full) {
        st = vc4_copy2d_n(ctx, src, dst, full, 1, 64, 64, 0, 64, 4, 16, 1, 16);
    }
    if (st == VC4_OK && rem) {
        vc4_buf s2 = *src, d2 = *dst;
        s2.off += full * 64u;
        d2.off += full * 64u;
        st = vc4_copy2d_n(ctx, &s2, &d2, 1, 1, 0, 0, 0, rem * 4u, 4, 16, 1,
                          rem);
    }
    return st;
}

/* out[r][0..cols) = src[r]  (exact: ones-vector outer product) */
static vc4_status bcast_rows(vc4_ctx *ctx, const vc4_buf *src, uint32_t rows,
                             uint32_t cols, vc4_buf *out)
{
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf ones = ws_get(&w, al64((size_t)cols * 4));
    if (!w.ok) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    float *o = malloc((size_t)cols * 4);
    if (!o) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    for (uint32_t i = 0; i < cols; ++i) o[i] = 1.0f;
    fbuf_write(ctx, &ones, o, cols);
    free(o);
    vc4_status st = vc4_matmul(ctx, src, &ones, out, rows, cols, 1,
                               VC4_LAYOUT_ROW_MAJOR);
    ws_off(&w);
    return st;
}

/* out[r] = sum_c a[r][c]  (or the mean) */
static vc4_status row_sum(vc4_ctx *ctx, const vc4_buf *a, uint32_t rows,
                          uint32_t cols, int mean, vc4_buf *out)
{
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf ones = ws_get(&w, al64((size_t)cols * 4));
    if (!w.ok) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    float *o = malloc((size_t)cols * 4);
    if (!o) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    float s = mean ? 1.0f / (float)cols : 1.0f;
    for (uint32_t i = 0; i < cols; ++i) o[i] = s;
    fbuf_write(ctx, &ones, o, cols);
    free(o);
    vc4_status st = vc4_matmul(ctx, a, &ones, out, rows, 1, cols,
                               VC4_LAYOUT_ROW_MAJOR);
    ws_off(&w);
    return st;
}

/* out[j] = sum_i a[i][j]  (ones[1][red] * a) */
static vc4_status col_sum(vc4_ctx *ctx, const vc4_buf *a, uint32_t red,
                          uint32_t inner, vc4_buf *out)
{
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf ones = ws_get(&w, al64((size_t)red * 4));
    if (!w.ok) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    float *o = malloc((size_t)red * 4);
    if (!o) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    for (uint32_t i = 0; i < red; ++i) o[i] = 1.0f;
    fbuf_write(ctx, &ones, o, red);
    free(o);
    vc4_status st = vc4_matmul(ctx, &ones, a, out, 1, inner, red,
                               VC4_LAYOUT_ROW_MAJOR);
    ws_off(&w);
    return st;
}

/* An irregular layout as one matvec: out = W * x, W is [n_out][n_in]. */
#define VC4_MATVEC_MAX (32u << 20)

static vc4_status matvec_w(bw_ws *w, const float *W, uint32_t n_out,
                           uint32_t n_in, const vc4_buf *x, vc4_buf *y)
{
    if ((uint64_t)n_out * n_in * 4 > VC4_MATVEC_MAX) {
        vc4__set_err(w->ctx, "matvec: a %ux%u layout table is %.1f MiB, over "
                     "the %u MiB cap; this op's irregular form is only "
                     "supported for small tensors (see docs/TRAINING.md)",
                     n_out, n_in, (double)n_out * n_in * 4 / (1 << 20),
                     VC4_MATVEC_MAX >> 20);
        return VC4_ERR_UNSUPPORTED;
    }
    vc4_buf wb = ws_get(w, al64((size_t)n_out * n_in * 4) + 64);
    if (!w->ok)
        return VC4_ERR_NOMEM;
    fbuf_write(w->ctx, &wb, W, (size_t)n_out * n_in);
    return vc4_matmul(w->ctx, &wb, x, y, n_out, 1, n_in,
                      VC4_LAYOUT_ROW_MAJOR);
}

/* ones[1][n] as a matvec row */
static vc4_status ones_row(bw_ws *w, uint32_t n, vc4_buf *b)
{
    *b = ws_get(w, al64((size_t)n * 4));
    if (!w->ok)
        return VC4_ERR_NOMEM;
    float *o = malloc((size_t)n * 4);
    if (!o)
        return VC4_ERR_NOMEM;
    for (uint32_t i = 0; i < n; ++i) o[i] = 1.0f;
    fbuf_write(w->ctx, b, o, n);
    free(o);
    return VC4_OK;
}

/* out[r*block + k] = src[r] for every block of `block` floats: the affine
 * copy2d with a zero lane stride.  No table, one launch, no memory. */
static vc4_status bcast_blocks(vc4_ctx *ctx, const vc4_buf *src, uint32_t nblk,
                               uint32_t block, vc4_buf *out, uint32_t srow)
{
    if (block == 0 || nblk == 0)
        return VC4_ERR_INVAL;
    if (block <= 16)
        return vc4_copy2d_n(ctx, src, out, nblk, 1, srow, block * 4, 0,
                            block * 4u, 0, 16, 1, block);
    if (block % 16)
        return VC4_ERR_UNSUPPORTED;
    return vc4_copy2d_n(ctx, src, out, nblk, block / 16, srow, block * 4, 0,
                        64, 0, 16, 1, 16);
}

/* ======================================================================
 * generic broadcast and single-axis reduction
 * ====================================================================== */
static vc4_status bcast_gen(vc4_ctx *ctx, const vc4_buf *src, const int *sdims,
                            const int *odims, int ndim, vc4_buf *out,
                            size_t n_out)
{
    /* only the last axis differs: an outer product with ones */
    int last_only = 1;
    for (int i = 0; i < ndim - 1; ++i)
        if (sdims[i] != odims[i])
            last_only = 0;
    if (last_only && sdims[ndim - 1] == 1) {
        int64_t rows = 1;
        for (int i = 0; i < ndim - 1; ++i) rows *= odims[i];
        return bcast_rows(ctx, src, (uint32_t)rows, (uint32_t)odims[ndim - 1],
                          out);
    }
    /* A per-channel vector expanded over BOTH a leading (batch) and a
     * trailing (spatial) axis: two affine passes instead of a table.
     *   step 1  [1][C][1] -> [1][C][HW]   (repeat each element over the tail)
     *   step 2  [1][C][HW] -> [N][C][HW]  (repeat the whole block N times)
     */
    if (sdims[ndim - 1] == 1 && odims[ndim - 1] > 1) {
        /* every differing axis must be the leading one or the trailing one */
        int ok = 1;
        for (int i = 1; i < ndim - 1; ++i)
            if (sdims[i] != odims[i])
                ok = 0;
        if (sdims[0] != 1 && odims[0] > 1)
            ok = 0;
        const int64_t nblk1 = prod_dims(sdims, ndim - 1);
        const uint32_t run1 = (uint32_t)odims[ndim - 1];
        if (ok && (run1 % 16 == 0 || run1 <= 16)) {
            bw_ws w;
            ws_on(&w, ctx);
            vc4_buf t1 = ws_get(&w, al64((size_t)nblk1 * run1 * 4));
            vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
            if (st == VC4_OK)
                st = bcast_blocks(ctx, src, (uint32_t)nblk1, run1, &t1, 4);
            if (st == VC4_OK) {
                /* the block to repeat is nblk1*run1 floats, odims[0] times */
                const uint32_t blk = (uint32_t)(nblk1 * run1);
                if (blk <= 16)
                    st = vc4_copy2d_n(ctx, &t1, out, (uint32_t)odims[0], 1, 0,
                                      blk * 4, 0, blk * 4u, 0, 16, 1, blk);
                else if (blk % 16 == 0)
                    st = vc4_copy2d_n(ctx, &t1, out, (uint32_t)odims[0],
                                      blk / 16, 0, blk * 4, 0, 64, 0, 16, 1,
                                      16);
                else
                    st = VC4_ERR_UNSUPPORTED;
            }
            ws_off(&w);
            if (st == VC4_OK) {
                ctx->st.flops += (uint64_t)n_out;
                return st;
            }
            /* fall through to the table on an unsupported shape */
        }
    }
    /* general: build the 0/1 map and do one matvec */
    bw_ws w;
    ws_on(&w, ctx);
    float *W = calloc(n_out * (size_t)prod_dims(sdims, ndim), 4);
    if (!W) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    int64_t sstr[8], ostr[8];
    int64_t acc = 1;
    for (int i = ndim - 1; i >= 0; --i) { sstr[i] = acc; acc *= sdims[i]; }
    acc = 1;
    for (int i = ndim - 1; i >= 0; --i) { ostr[i] = acc; acc *= odims[i]; }
    int64_t idx[8] = { 0 };
    for (size_t o = 0; o < n_out; ++o) {
        int64_t s = 0;
        for (int i = 0; i < ndim; ++i)
            if (sdims[i] != 1)
                s += idx[i] * sstr[i];
        W[o * (size_t)prod_dims(sdims, ndim) + (size_t)s] = 1.0f;
        for (int i = ndim - 1; i >= 0; --i) {
            if (++idx[i] < odims[i]) break;
            idx[i] = 0;
        }
    }
    vc4_status st = matvec_w(&w, W, (uint32_t)n_out,
                             (uint32_t)prod_dims(sdims, ndim), src, out);
    free(W);
    (void)ostr;
    ws_off(&w);
    return st;
}

/* reduce one axis: [outer][red][inner] -> [outer][inner] */
static vc4_status reduce_one(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                             const int *dims, int ndim, int axis, int mean)
{
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= dims[i];
    for (int i = axis + 1; i < ndim; ++i) inner *= dims[i];
    const int64_t red = dims[axis];
    if (outer <= 0 || red <= 0 || inner <= 0)
        return VC4_ERR_INVAL;
    if (axis == ndim - 1)
        return row_sum(ctx, a, (uint32_t)outer, (uint32_t)red, mean, out);
    if (axis == 0)
        return col_sum(ctx, a, (uint32_t)red, (uint32_t)inner, out);
    if (outer > 256) {
        vc4__set_err(ctx, "reduce: %lld outer slices for a middle axis is "
                     "over the 256-launch cap", (long long)outer);
        return VC4_ERR_UNSUPPORTED;
    }
    /* middle axis: one ones-row matvec per outer slice */
    vc4_status st = VC4_OK;
    for (int64_t o = 0; o < outer && st == VC4_OK; ++o) {
        vc4_buf ai = { a->off + (uint32_t)((uint64_t)o * red * inner * 4),
                       (size_t)(red * inner * 4) };
        vc4_buf oi = { out->off + (uint32_t)((uint64_t)o * inner * 4),
                       (size_t)(inner * 4) };
        st = col_sum(ctx, &ai, (uint32_t)red, (uint32_t)inner, &oi);
    }
    return st;
}

/* Sum `g` over every axis where sdims[a] == 1 < odims[a]. */
static vc4_status reduce_bcast_axes(vc4_ctx *ctx, const vc4_buf *g,
                                    const int *odims, const int *sdims,
                                    int ndim, vc4_buf *out)
{
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf cur = *g;
    size_t cur_n = (size_t)prod_dims(odims, ndim);
    int shape[8];
    memcpy(shape, odims, sizeof(int) * (size_t)ndim);
    vc4_status st = VC4_OK;
    for (int a = ndim - 1; a >= 0 && st == VC4_OK; --a) {
        if (sdims[a] != 1 || odims[a] <= 1)
            continue;
        vc4_buf nxt = ws_get(&w, al64(cur_n / (size_t)odims[a] * 4));
        if (!w.ok) {
            st = VC4_ERR_NOMEM;
            break;
        }
        st = reduce_one(ctx, &cur, &nxt, shape, ndim, a, 0);
        if (st != VC4_OK)
            break;
        shape[a] = 1;
        cur = nxt;
        cur_n /= (size_t)odims[a];
    }
    if (st == VC4_OK)
        st = copy_floats(ctx, &cur, out, cur_n);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * softmax (any class count, last axis)
 * ====================================================================== */
vc4_status vc4_softmax(vc4_ctx *ctx, const vc4_buf *x, uint32_t rows,
                       uint32_t cols, vc4_buf *y)
{
    if (!ctx || !x || !y || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if ((uint64_t)x->bytes < (uint64_t)rows * cols * 4 ||
        (uint64_t)y->bytes < (uint64_t)rows * cols * 4)
        return VC4_ERR_INVAL;
    const size_t n = (size_t)rows * cols;
    /* the fused, numerically stable kernel covers the classification case */
    if (cols <= 16 && rows % 8 == 0)
        return vc4_softmax_fwd(ctx, x, rows, cols, y);
    if (cols > 262144u) {
        vc4__set_err(ctx, "vc4_softmax: %u classes is over the reduce cap",
                     cols);
        return VC4_ERR_UNSUPPORTED;
    }
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf s = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf sb = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[2] = { (int)rows, 1 }, od[2] = { (int)rows, (int)cols };
    /* A row maximum is not a linear map, so the *numerically stable* path is
     * the fused row kernel above (cols <= 16).  This generic path exponentiates
     * directly: it is exact for |x| below ~80 and the caller is told so. */
    if (st == VC4_OK)
        st = u_exp(ctx, x, &t, n);
    if (st == VC4_OK)
        st = row_sum(ctx, &t, rows, cols, 0, &s);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &s, sd, od, 2, &sb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_DIV, &t, &sb, y, n);
    ws_off(&w);
    return st;
}

vc4_status vc4_softmax_bwd(vc4_ctx *ctx, const vc4_buf *y, const vc4_buf *g,
                           vc4_buf *dx, uint32_t rows, uint32_t cols)
{
    if (!ctx || !y || !g || !dx || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)rows * cols;
    if ((uint64_t)y->bytes < n * 4 || (uint64_t)g->bytes < n * 4 ||
        (uint64_t)dx->bytes < n * 4)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf s = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf sb = ws_get(&w, al64(n * 4));
    vc4_buf d = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[2] = { (int)rows, 1 }, od[2] = { (int)rows, (int)cols };
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, y, g, &t, n);
    if (st == VC4_OK)
        st = row_sum(ctx, &t, rows, cols, 0, &s);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &s, sd, od, 2, &sb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, g, &sb, &d, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, y, &d, dx, n);
    ws_off(&w);
    return st;
}

/*
 * Softmax cross entropy backward, any class count:
 *   p       = softmax(logits)
 *   dlogits = (p - onehot(label)) * inv_batch
 *   loss    = -inv_batch * sum_rows log p[label]
 * The picked probability is a dot product with the one-hot row, so no gather
 * is needed: pick[r] = sum_c p[r][c]*onehot[r][c].
 */
vc4_status vc4_softmax_ce_bwd(vc4_ctx *ctx, const vc4_buf *logits,
                              const vc4_buf *labels, vc4_buf *dlogits,
                              vc4_buf *loss, vc4_buf *y_scratch, uint32_t rows,
                              uint32_t cols, float inv_batch)
{
    if (!ctx || !logits || !labels || !dlogits || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)rows * cols;
    if ((uint64_t)logits->bytes < n * 4 || (uint64_t)dlogits->bytes < n * 4 ||
        (uint64_t)labels->bytes < (uint64_t)rows * 4)
        return VC4_ERR_INVAL;
    if (loss && loss->bytes < 4)
        return VC4_ERR_INVAL;

    bw_ws w;
    ws_on(&w, ctx);
    vc4_status st = VC4_OK;

    if (cols == 16 && rows % 8 == 0) {
        /* The fused row kernel is the verified path: it gives dlogits and the
         * per-QPU loss totals.  It wants 64 B of slack after a 16-wide row, so
         * it runs on staged copies. */
        const size_t groups = rows / 8;
        vc4_buf lg = ws_get(&w, al64((size_t)rows * 64 + 64));
        vc4_buf dl = ws_get(&w, al64((size_t)rows * 64 + 64));
        vc4_buf part = ws_get(&w, al64(groups * 64));
        vc4_buf ls = ws_get(&w, al64(64));
        if (!w.ok)
            st = VC4_ERR_NOMEM;
        if (st == VC4_OK) {
            vc4_invalidate_buf(ctx, logits);
            memcpy(ctx->arena + lg.off, ctx->arena + logits->off,
                   (size_t)rows * 64);
            vc4__mark_dirty(ctx, lg.off, (size_t)rows * 64 + 64);
        }
        if (st == VC4_OK)
            st = vc4_softmax_ce(ctx, &lg, labels, &dl, &part, rows, inv_batch);
        if (st == VC4_OK) {
            vc4__l2_flush(ctx);
            vc4__invalidate_off(ctx, dlogits->off, (size_t)rows * 64);
            memcpy(ctx->arena + dlogits->off, ctx->arena + dl.off,
                   (size_t)rows * 64);
            vc4__mark_dirty(ctx, dlogits->off, (size_t)rows * 64);
        }
        if (st == VC4_OK && loss) {
            /* The fused kernel writes the *unscaled* -log p sum per group in
             * lane 0 of its slot, so the total is a column sum and the batch
             * normalisation is one more scale: loss is the mean CE. */
            st = col_sum(ctx, &part, (uint32_t)groups, 16, &ls);
            if (st == VC4_OK)
                st = vc4_affine(ctx, &ls, &ls, 1, inv_batch, 0.0f);
            if (st == VC4_OK) {
                vc4__l2_flush(ctx);
                vc4__invalidate_off(ctx, loss->off, 4);
                memcpy(ctx->arena + loss->off, ctx->arena + ls.off, 4);
                vc4__mark_dirty(ctx, loss->off, 4);
            }
        }
        ws_off(&w);
        return st;
    }

    /* Generic class count: dlogits = (p - onehot)*inv_batch.  The loss needs
     * log(p[label]) and there is no usable log kernel (the body is 39248 B,
     * the slot is 16384 B), so the caller is told to compute it from p. */
    if (loss) {
        vc4__set_err(ctx, "vc4_softmax_ce_bwd: the loss scalar is only "
                     "computed for the 16-class fused path (cols == 16, "
                     "rows %% 8 == 0); for %u classes pass loss = NULL and "
                     "take -log(p[label]) from vc4_softmax()", cols);
        ws_off(&w);
        return VC4_ERR_UNSUPPORTED;
    }
    vc4_buf p;
    if (y_scratch && y_scratch->bytes >= n * 4 && y_scratch->off)
        p = *y_scratch;
    else
        p = ws_get(&w, al64(n * 4));
    vc4_buf oh = ws_get(&w, al64(n * 4));
    vc4_buf d = ws_get(&w, al64(n * 4));
    if (!w.ok) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    float *ohv = calloc(n, 4);
    int32_t *lab = malloc((size_t)rows * 4);
    if (!ohv || !lab) {
        st = VC4_ERR_NOMEM;
    } else {
        vc4_invalidate_buf(ctx, labels);
        memcpy(lab, ctx->arena + labels->off, (size_t)rows * 4);
        for (uint32_t r = 0; r < rows; ++r) {
            int32_t c = lab[r];
            if (c < 0 || (uint32_t)c >= cols) {
                vc4__set_err(ctx, "vc4_softmax_ce_bwd: label %d at row %u is "
                             "outside [0,%u)", c, r, cols);
                st = VC4_ERR_INVAL;
                break;
            }
            ohv[(size_t)r * cols + (size_t)c] = 1.0f;
        }
    }
    if (st == VC4_OK)
        st = vc4_softmax(ctx, logits, rows, cols, &p);
    if (st == VC4_OK) {
        fbuf_write(ctx, &oh, ohv, n);
        st = ew(ctx, EW_SUB, &p, &oh, &d, n);
    }
    if (st == VC4_OK)
        st = vc4_affine(ctx, &d, dlogits, n, inv_batch, 0.0f);
    free(ohv);
    free(lab);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * batchnorm
 * ====================================================================== */
/* per-channel sum of a [N][C][HW] tensor -> [C] */
static vc4_status chan_sum(vc4_ctx *ctx, const vc4_buf *x, vc4_buf *out,
                           uint32_t N, uint32_t C, uint32_t HW)
{
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf s1 = ws_get(&w, al64((size_t)N * C * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK)
        st = row_sum(ctx, x, N * C, HW, 0, &s1);
    if (st == VC4_OK) {
        /* out[c] = sum_n s1[n*C + c]: an [C][N*C] selection matrix */
        float *W = calloc((size_t)C * N * C, 4);
        if (!W)
            st = VC4_ERR_NOMEM;
        else {
            for (uint32_t n = 0; n < N; ++n)
                for (uint32_t c = 0; c < C; ++c)
                    W[(size_t)c * N * C + (size_t)n * C + c] = 1.0f;
            st = matvec_w(&w, W, C, N * C, &s1, out);
            free(W);
        }
    }
    ws_off(&w);
    return st;
}

vc4_status vc4_batchnorm_fwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *beta,
                             vc4_buf *y, vc4_buf *save_mean,
                             vc4_buf *save_invstd, vc4_buf *running_mean,
                             vc4_buf *running_var, uint32_t N, uint32_t C,
                             uint32_t HW, float momentum, float eps,
                             int update_running)
{
    if (!ctx || !x || !gamma || !beta || !y || !save_mean || !save_invstd ||
        !N || !C || !HW)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)N * C * HW;
    const float M = (float)((double)N * HW);
    if ((uint64_t)x->bytes < n * 4 || (uint64_t)y->bytes < n * 4 ||
        (uint64_t)gamma->bytes < C * 4 || (uint64_t)beta->bytes < C * 4 ||
        (uint64_t)save_mean->bytes < C * 4 ||
        (uint64_t)save_invstd->bytes < C * 4)
        return VC4_ERR_INVAL;

    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf sum = ws_get(&w, al64((size_t)C * 4));
    vc4_buf mean = ws_get(&w, al64((size_t)C * 4));
    vc4_buf x2 = ws_get(&w, al64(n * 4));
    vc4_buf var = ws_get(&w, al64((size_t)C * 4));
    vc4_buf istd = ws_get(&w, al64((size_t)C * 4));
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf a = ws_get(&w, al64(n * 4));
    vc4_buf mb = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[3] = { 1, (int)C, 1 }, od[3] = { (int)N, (int)C, (int)HW };

    if (st == VC4_OK)
        st = chan_sum(ctx, x, &sum, N, C, HW);
    if (st == VC4_OK)
        st = vc4_affine(ctx, &sum, &mean, C, 1.0f / M, 0.0f);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, x, x, &x2, n);
    if (st == VC4_OK)
        st = chan_sum(ctx, &x2, &var, N, C, HW);
    if (st == VC4_OK)
        st = vc4_affine(ctx, &var, &var, C, 1.0f / M, 0.0f);
    if (st == VC4_OK) {
        vc4_buf mm = ws_get(&w, al64((size_t)C * 4));
        if (!w.ok)
            st = VC4_ERR_NOMEM;
        else {
            st = ew(ctx, EW_MUL, &mean, &mean, &mm, C);
            if (st == VC4_OK)
                st = ew(ctx, EW_SUB, &var, &mm, &var, C);
        }
    }
    if (st == VC4_OK)
        st = vc4_affine(ctx, &var, &var, C, 1.0f, eps);
    if (st == VC4_OK)
        st = vc4_rsqrt(ctx, &var, &istd, C);
    if (st == VC4_OK)
        st = copy_floats(ctx, &mean, save_mean, C);
    if (st == VC4_OK)
        st = copy_floats(ctx, &istd, save_invstd, C);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &mean, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, x, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &istd, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &t, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, gamma, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &t, &mb, &a, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, beta, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_ADD, &a, &mb, y, n);
    if (st == VC4_OK && update_running && running_mean && running_var) {
        vc4_buf vb = ws_get(&w, al64((size_t)C * 4));
        vc4_buf t2 = ws_get(&w, al64((size_t)C * 4));
        if (!w.ok)
            st = VC4_ERR_NOMEM;
        else {
            float ub = M > 1.5f ? M / (M - 1.0f) : 1.0f;
            st = vc4_affine(ctx, &var, &vb, C, ub, 0.0f);
            if (st == VC4_OK)
                st = vc4_affine(ctx, running_mean, &t2, C, 1.0f - momentum,
                                0.0f);
            if (st == VC4_OK)
                st = vc4_affine(ctx, &mean, &t, C, momentum, 0.0f);
            if (st == VC4_OK)
                st = ew(ctx, EW_ADD, &t2, &t, running_mean, C);
            if (st == VC4_OK)
                st = vc4_affine(ctx, running_var, &t2, C, 1.0f - momentum,
                                0.0f);
            if (st == VC4_OK)
                st = vc4_affine(ctx, &vb, &t, C, momentum, 0.0f);
            if (st == VC4_OK)
                st = ew(ctx, EW_ADD, &t2, &t, running_var, C);
        }
    }
    ws_off(&w);
    return st;
}

vc4_status vc4_batchnorm_infer(vc4_ctx *ctx, const vc4_buf *x,
                               const vc4_buf *gamma, const vc4_buf *beta,
                               const vc4_buf *running_mean,
                               const vc4_buf *running_var, vc4_buf *y,
                               uint32_t N, uint32_t C, uint32_t HW, float eps)
{
    if (!ctx || !x || !gamma || !beta || !running_mean || !running_var || !y ||
        !N || !C || !HW)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)N * C * HW;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf istd = ws_get(&w, al64((size_t)C * 4));
    vc4_buf ve = ws_get(&w, al64((size_t)C * 4));
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf mb = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[3] = { 1, (int)C, 1 }, od[3] = { (int)N, (int)C, (int)HW };
    if (st == VC4_OK)
        st = vc4_affine(ctx, running_var, &ve, C, 1.0f, eps);
    if (st == VC4_OK)
        st = vc4_rsqrt(ctx, &ve, &istd, C);
    if (st == VC4_OK)
        st = bcast_gen(ctx, running_mean, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, x, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &istd, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &t, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, gamma, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &t, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, beta, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_ADD, &t, &mb, y, n);
    ws_off(&w);
    return st;
}

vc4_status vc4_batchnorm_bwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *dy,
                             const vc4_buf *save_mean,
                             const vc4_buf *save_invstd, vc4_buf *dx,
                             vc4_buf *dgamma, vc4_buf *dbeta, uint32_t N,
                             uint32_t C, uint32_t HW)
{
    if (!ctx || !x || !gamma || !dy || !save_mean || !save_invstd || !dx ||
        !dgamma || !dbeta || !N || !C || !HW)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)N * C * HW;
    const float M = (float)((double)N * HW);
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf xhat = ws_get(&w, al64(n * 4));
    vc4_buf tmp = ws_get(&w, al64(n * 4));
    vc4_buf mb = ws_get(&w, al64(n * 4));
    vc4_buf gm = ws_get(&w, al64((size_t)C * 4));
    vc4_buf c2 = ws_get(&w, al64((size_t)C * 4));
    vc4_buf c3 = ws_get(&w, al64((size_t)C * 4));
    vc4_buf c2b = ws_get(&w, al64(n * 4));
    vc4_buf c3b = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[3] = { 1, (int)C, 1 }, od[3] = { (int)N, (int)C, (int)HW };

    if (st == VC4_OK)
        st = bcast_gen(ctx, save_mean, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, x, &mb, &xhat, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, save_invstd, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &xhat, &mb, &xhat, n);
    if (st == VC4_OK)
        st = chan_sum(ctx, dy, dbeta, N, C, HW);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, dy, &xhat, &tmp, n);
    if (st == VC4_OK)
        st = chan_sum(ctx, &tmp, dgamma, N, C, HW);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, gamma, save_invstd, &gm, C);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &gm, dbeta, &c2, C);
    if (st == VC4_OK)
        st = vc4_affine(ctx, &c2, &c2, C, 1.0f / M, 0.0f);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &gm, dgamma, &c3, C);
    if (st == VC4_OK)
        st = vc4_affine(ctx, &c3, &c3, C, 1.0f / M, 0.0f);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &gm, sd, od, 3, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, dy, &mb, &tmp, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &c2, sd, od, 3, &c2b, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, &tmp, &c2b, &tmp, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &c3, sd, od, 3, &c3b, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &c3b, &xhat, &xhat, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, &tmp, &xhat, dx, n);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * layernorm (last axis)
 * ====================================================================== */
vc4_status vc4_layernorm_fwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *beta,
                             vc4_buf *y, vc4_buf *save_mean,
                             vc4_buf *save_invstd, uint32_t rows,
                             uint32_t cols, float eps)
{
    if (!ctx || !x || !gamma || !beta || !y || !save_mean || !save_invstd ||
        !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)rows * cols;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf mean = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf d = ws_get(&w, al64(n * 4));
    vc4_buf mb = ws_get(&w, al64(n * 4));
    vc4_buf var = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf istd = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf xhat = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[2] = { (int)rows, 1 }, od[2] = { (int)rows, (int)cols };
    int sdcol[2] = { 1, (int)cols };   /* gamma/beta are per-column */

    if (st == VC4_OK)
        st = row_sum(ctx, x, rows, cols, 1, &mean);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &mean, sd, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, x, &mb, &d, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &d, &d, &t, n);
    if (st == VC4_OK)
        st = row_sum(ctx, &t, rows, cols, 1, &var);
    if (st == VC4_OK)
        st = vc4_affine(ctx, &var, &var, rows, 1.0f, eps);
    if (st == VC4_OK)
        st = vc4_rsqrt(ctx, &var, &istd, rows);
    if (st == VC4_OK)
        st = copy_floats(ctx, &mean, save_mean, rows);
    if (st == VC4_OK)
        st = copy_floats(ctx, &istd, save_invstd, rows);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &istd, sd, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &d, &mb, &xhat, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, gamma, sdcol, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &xhat, &mb, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, beta, sdcol, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_ADD, &t, &mb, y, n);
    ws_off(&w);
    return st;
}

vc4_status vc4_layernorm_bwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *dy,
                             const vc4_buf *save_mean,
                             const vc4_buf *save_invstd, vc4_buf *dx,
                             vc4_buf *dgamma, vc4_buf *dbeta, uint32_t rows,
                             uint32_t cols)
{
    if (!ctx || !x || !gamma || !dy || !save_mean || !save_invstd || !dx ||
        !dgamma || !dbeta || !rows || !cols)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)rows * cols;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf xhat = ws_get(&w, al64(n * 4));
    vc4_buf gx = ws_get(&w, al64(n * 4));
    vc4_buf mb = ws_get(&w, al64(n * 4));
    vc4_buf a = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf b = ws_get(&w, al64((size_t)rows * 4));
    vc4_buf t = ws_get(&w, al64(n * 4));
    vc4_buf ab = ws_get(&w, al64(n * 4));
    vc4_buf bb = ws_get(&w, al64(n * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[2] = { (int)rows, 1 }, od[2] = { (int)rows, (int)cols };
    int sdcol[2] = { 1, (int)cols };

    if (st == VC4_OK)
        st = bcast_gen(ctx, save_mean, sd, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, x, &mb, &xhat, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, save_invstd, sd, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &xhat, &mb, &xhat, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, gamma, sdcol, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, dy, &mb, &gx, n);
    if (st == VC4_OK)
        st = row_sum(ctx, &gx, rows, cols, 1, &a);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &gx, &xhat, &t, n);
    if (st == VC4_OK)
        st = row_sum(ctx, &t, rows, cols, 1, &b);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &a, sd, od, 2, &ab, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &b, sd, od, 2, &bb, n);
    /* dgamma = sum_rows dy*xhat: compute it BEFORE xhat is reused for the
     * xhat*b term of dx (it is the same buffer). */
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, dy, &xhat, &t, n);
    if (st == VC4_OK) {
        int dims[2] = { (int)rows, (int)cols };
        st = reduce_one(ctx, &t, dgamma, dims, 2, 0, 0);
        if (st == VC4_OK)
            st = reduce_one(ctx, dy, dbeta, dims, 2, 0, 0);
    }
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, &gx, &ab, &t, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &xhat, &bb, &xhat, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_SUB, &t, &xhat, &t, n);
    if (st == VC4_OK)
        st = bcast_gen(ctx, save_invstd, sd, od, 2, &mb, n);
    if (st == VC4_OK)
        st = ew(ctx, EW_MUL, &t, &mb, dx, n);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * pooling geometry: the (dx element -> covering windows) map
 *
 * For a max/avg pool with window N, stride S and pad P, position y is
 * covered by windows wy = y/S - ky with (y%S) + ky*S < N.  So the number of
 * candidate windows per position is K = ceil(N/S) in each dimension, and the
 * scatter is K*K matvecs with 0/1 (or 1/count) weight matrices.
 * ====================================================================== */
static int pool_outdim(int H, int N, int S, int pad)
{
    return (H + 2 * pad - N) / S + 1;
}

vc4_status vc4_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx,
                           int C, int H, int W, int N, int S, int pad,
                           int count_include_pad)
{
    if (!ctx || !dy || !dx || C <= 0 || H <= 0 || W <= 0 || N <= 0 || S <= 0 ||
        pad < 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const int Ho = pool_outdim(H, N, S, pad), Wo = pool_outdim(W, N, S, pad);
    if (Ho <= 0 || Wo <= 0) {
        vc4__set_err(ctx, "vc4_avgpool_bwd: window %d/%d does not fit %dx%d",
                     N, S, H, W);
        return VC4_ERR_UNSUPPORTED;
    }
    const size_t n_dy = (size_t)C * Ho * Wo, n_dx = (size_t)C * H * W;
    if ((uint64_t)dy->bytes < n_dy * 4 || (uint64_t)dx->bytes < n_dx * 4)
        return VC4_ERR_INVAL;
    const int K = (N + S - 1) / S;
    bw_ws w;
    ws_on(&w, ctx);
    float *M = calloc(n_dx * n_dy, 4);
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    if (!M)
        st = VC4_ERR_NOMEM;
    if (st == VC4_OK) {
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    size_t i = ((size_t)c * H + y) * W + x;
                    int wy = (y + pad) / S, wx = (x + pad) / S;
                    for (int ky = 0; ky < K; ++ky)
                        for (int kx = 0; kx < K; ++kx) {
                            int oy = wy - ky, ox = wx - kx;
                            if (oy < 0 || oy >= Ho || ox < 0 || ox >= Wo)
                                continue;
                            /* the window starts at oy*S - pad, so the offset
                             * of this position inside it is y+pad-oy*S */
                            int kyoff = y + pad - oy * S;
                            int kxoff = x + pad - ox * S;
                            if (kyoff < 0 || kyoff >= N || kxoff < 0 ||
                                kxoff >= N)
                                continue;
                            /* the average pool divides by the window area */
                            int cnt = N * N;
                            if (!count_include_pad) {
                                int y0 = oy * S - pad, x0 = ox * S - pad;
                                int y1 = y0 + N, x1 = x0 + N;
                                if (y0 < 0) y0 = 0;
                                if (x0 < 0) x0 = 0;
                                if (y1 > H) y1 = H;
                                if (x1 > W) x1 = W;
                                cnt = (y1 > y0 ? y1 - y0 : 0) *
                                      (x1 > x0 ? x1 - x0 : 0);
                                if (cnt <= 0) cnt = 1;
                            }
                            M[i * n_dy + ((size_t)c * Ho + oy) * Wo + ox] +=
                                1.0f / (float)cnt;
                        }
                }
        st = matvec_w(&w, M, (uint32_t)n_dx, (uint32_t)n_dy, dy, dx);
    }
    free(M);
    ws_off(&w);
    return st;
}

vc4_status vc4_global_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                  vc4_buf *dx, int C, int H, int W)
{
    if (!ctx || !dy || !dx || C <= 0 || H <= 0 || W <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n_dx = (size_t)C * H * W;
    if ((uint64_t)dy->bytes < (uint64_t)C * 4 || (uint64_t)dx->bytes < n_dx * 4)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf g = ws_get(&w, al64((size_t)C * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    int sd[3] = { 1, (int)C, 1 }, od[3] = { 1, (int)C, (int)(H * W) };
    if (st == VC4_OK)
        st = vc4_affine(ctx, dy, &g, C, 1.0f / (float)((double)H * W), 0.0f);
    if (st == VC4_OK)
        st = bcast_gen(ctx, &g, sd, od, 3, dx, n_dx);
    ws_off(&w);
    return st;
}

vc4_status vc4_adaptive_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                    vc4_buf *dx, int C, int H, int W,
                                    int outH, int outW)
{
    if (!ctx || !dy || !dx || C <= 0 || H <= 0 || W <= 0 || outH <= 0 ||
        outW <= 0 || outH > H || outW > W)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n_dy = (size_t)C * outH * outW, n_dx = (size_t)C * H * W;
    if ((uint64_t)dy->bytes < n_dy * 4 || (uint64_t)dx->bytes < n_dx * 4)
        return VC4_ERR_INVAL;
    int *ys = malloc((size_t)outH * 2 * sizeof(int));
    int *xs = malloc((size_t)outW * 2 * sizeof(int));
    bw_ws w;
    ws_on(&w, ctx);
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    float *M = calloc(n_dx * n_dy, 4);
    if (!ys || !xs || !M)
        st = VC4_ERR_NOMEM;
    for (int i = 0; i < outH && st == VC4_OK; ++i) {
        ys[2 * i] = (int)((int64_t)i * H / outH);
        ys[2 * i + 1] = (int)(((int64_t)(i + 1) * H + outH - 1) / outH);
        if (ys[2 * i + 1] <= ys[2 * i])
            ys[2 * i + 1] = ys[2 * i] + 1;
    }
    for (int i = 0; i < outW && st == VC4_OK; ++i) {
        xs[2 * i] = (int)((int64_t)i * W / outW);
        xs[2 * i + 1] = (int)(((int64_t)(i + 1) * W + outW - 1) / outW);
        if (xs[2 * i + 1] <= xs[2 * i])
            xs[2 * i + 1] = xs[2 * i] + 1;
    }
    if (st == VC4_OK) {
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < outH; ++oy)
                for (int ox = 0; ox < outW; ++ox) {
                    int area = (ys[2 * oy + 1] - ys[2 * oy]) *
                               (xs[2 * ox + 1] - xs[2 * ox]);
                    for (int y = ys[2 * oy]; y < ys[2 * oy + 1]; ++y)
                        for (int x = xs[2 * ox]; x < xs[2 * ox + 1]; ++x) {
                            size_t i = ((size_t)c * H + y) * W + x;
                            size_t o = ((size_t)c * outH + oy) * outW + ox;
                            M[i * n_dy + o] += 1.0f / (float)area;
                        }
                }
        st = matvec_w(&w, M, (uint32_t)n_dx, (uint32_t)n_dy, dy, dx);
    }
    free(ys);
    free(xs);
    free(M);
    ws_off(&w);
    return st;
}

vc4_status vc4_maxpooln_bwd(vc4_ctx *ctx, const vc4_buf *x,
                            const vc4_buf *dy, vc4_buf *dx, int C, int H,
                            int W, int N, int S, int pad)
{
    if (!ctx || !x || !dy || !dx || C <= 0 || H <= 0 || W <= 0 || N <= 0 ||
        S <= 0 || pad < 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const int Ho = pool_outdim(H, N, S, pad), Wo = pool_outdim(W, N, S, pad);
    const int dpitch = (Wo + 15) & ~15;           /* vc4_maxpooln layout */
    const int drows = (Ho + 3) & ~3;
    const size_t n_dy = (size_t)C * drows * dpitch;   /* padded dy layout */
    const size_t n_dx = (size_t)C * H * W;
    if (Ho <= 0 || Wo <= 0) {
        vc4__set_err(ctx, "vc4_maxpooln_bwd: window %d/%d does not fit %dx%d",
                     N, S, H, W);
        return VC4_ERR_UNSUPPORTED;
    }
    if ((uint64_t)dy->bytes < n_dy * 4 || (uint64_t)dx->bytes < n_dx * 4)
        return VC4_ERR_INVAL;
    const int K = (N + S - 1) / S;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf m = ws_get(&w, al64(n_dy * 4));
    vc4_buf t1 = ws_get(&w, al64(n_dx * 4));
    vc4_buf t2 = ws_get(&w, al64(n_dx * 4));
    vc4_buf acc = ws_get(&w, al64(n_dx * 4));
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    if (st == VC4_OK) {
        int oh = 0, ow = 0;
        st = vc4_maxpooln(ctx, x, C, H, W, N, S, pad, &m, &oh, &ow);
        if (st == VC4_OK && (oh != Ho || ow != Wo)) {
            vc4__set_err(ctx, "vc4_maxpooln_bwd: forward produced %dx%d, "
                         "expected %dx%d", oh, ow, Ho, Wo);
            st = VC4_ERR_GPU;
        }
    }
    if (st == VC4_OK) {
        /* per slot: m_d[i] = m[w_d(i)], dy_d[i] = dy[w_d(i)], then
         * dx[i] += dy_d[i] * (x[i] == m_d[i]); the max buffer and dy share
         * the padded [C][align4(Ho)][align16(Wo)] layout. */
        float *Md = calloc(n_dx * n_dy, 4);
        if (!Md)
            st = VC4_ERR_NOMEM;
        else {
            int first = 1;
            for (int ky = 0; ky < K && st == VC4_OK; ++ky)
                for (int kx = 0; kx < K && st == VC4_OK; ++kx) {
                    memset(Md, 0, n_dx * n_dy * 4);
                    for (int c = 0; c < C; ++c)
                        for (int y = 0; y < H; ++y)
                            for (int x = 0; x < W; ++x) {
                                size_t i = ((size_t)c * H + y) * W + x;
                                int wy = (y + pad) / S - ky;
                                int wx = (x + pad) / S - kx;
                                if (wy < 0 || wy >= Ho || wx < 0 || wx >= Wo)
                                    continue;
                                int kyoff = y + pad - wy * S;
                                int kxoff = x + pad - wx * S;
                                if (kyoff < 0 || kyoff >= N || kxoff < 0 ||
                                    kxoff >= N)
                                    continue;
                                size_t op = ((size_t)c * drows + wy) * dpitch
                                            + wx;
                                Md[i * n_dy + op] = 1.0f;
                            }
                    st = matvec_w(&w, Md, (uint32_t)n_dx, (uint32_t)n_dy, &m,
                                  &t2);
                    if (st == VC4_OK)
                        st = matvec_w(&w, Md, (uint32_t)n_dx, (uint32_t)n_dy,
                                      dy, &t1);
                    if (st == VC4_OK)
                        st = ew(ctx, EW_EQMASK, x, &t2, &t2, n_dx);
                    if (st == VC4_OK)
                        st = ew(ctx, EW_MUL, &t1, &t2, &t1, n_dx);
                    if (st == VC4_OK) {
                        if (first) {
                            st = copy_floats(ctx, &t1, &acc, n_dx);
                            first = 0;
                        } else
                            st = ew(ctx, EW_ADD, &acc, &t1, &acc, n_dx);
                    }
                }
            if (st == VC4_OK)
                st = copy_floats(ctx, &acc, dx, n_dx);
            free(Md);
        }
    }
    ws_off(&w);
    return st;
}

/* ======================================================================
 * broadcasting binary op backward
 * ====================================================================== */
vc4_status vc4_binop_bwd(vc4_ctx *ctx, vc4_binop op, const vc4_buf *x,
                         const vc4_buf *y, const vc4_buf *g, const int *odims,
                         const int *xdims, const int *ydims, int ndim,
                         vc4_buf *dx, vc4_buf *dy)
{
    if (!ctx || !x || !y || !g || !odims || !xdims || !ydims || ndim < 1 ||
        ndim > 8)
        return VC4_ERR_INVAL;
    if ((!dx && !dy) || (op > VC4_BIN_DIV))
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)prod_dims(odims, ndim);
    bw_ws w;
    ws_on(&w, ctx);
    vc4_status st = VC4_OK;
    int same_x = 1, same_y = 1;
    for (int i = 0; i < ndim; ++i) {
        if (xdims[i] != odims[i]) same_x = 0;
        if (ydims[i] != odims[i]) same_y = 0;
    }
    vc4_buf gx = ws_get(&w, al64(n * 4));
    vc4_buf gy = ws_get(&w, al64(n * 4));
    vc4_buf yb = ws_get(&w, al64(n * 4));
    vc4_buf xb = ws_get(&w, al64(n * 4));
    vc4_buf t = ws_get(&w, al64(n * 4));
    if (!w.ok) {
        ws_off(&w);
        return VC4_ERR_NOMEM;
    }
    st = copy_floats(ctx, g, &gx, n);
    if (st == VC4_OK)
        st = copy_floats(ctx, &gx, &gy, n);
    if (st == VC4_OK && same_y)
        st = copy_floats(ctx, y, &yb, n);
    if (st == VC4_OK && !same_y)
        st = bcast_gen(ctx, y, ydims, odims, ndim, &yb, n);
    if (st == VC4_OK && same_x)
        st = copy_floats(ctx, x, &xb, n);
    if (st == VC4_OK && !same_x)
        st = bcast_gen(ctx, x, xdims, odims, ndim, &xb, n);

    if (st == VC4_OK && op == VC4_BIN_MUL) {
        st = ew(ctx, EW_MUL, g, &yb, &gx, n);
        if (st == VC4_OK)
            st = ew(ctx, EW_MUL, g, &xb, &gy, n);
    } else if (st == VC4_OK && op == VC4_BIN_DIV) {
        st = ew(ctx, EW_DIV, g, &yb, &gx, n);
        if (st == VC4_OK)
            st = ew(ctx, EW_DIV, &xb, &yb, &t, n);
        if (st == VC4_OK)
            st = ew(ctx, EW_MUL, &gx, &t, &gy, n);
        if (st == VC4_OK)
            st = vc4_affine(ctx, &gy, &gy, n, -1.0f, 0.0f);
    } else if (st == VC4_OK && op == VC4_BIN_SUB) {
        st = vc4_affine(ctx, &gy, &gy, n, -1.0f, 0.0f);
    }
    if (st == VC4_OK && dx) {
        if (same_x)
            st = copy_floats(ctx, &gx, dx, n);
        else
            st = reduce_bcast_axes(ctx, &gx, odims, xdims, ndim, dx);
    }
    if (st == VC4_OK && dy) {
        if (same_y)
            st = copy_floats(ctx, &gy, dy, n);
        else
            st = reduce_bcast_axes(ctx, &gy, odims, ydims, ndim, dy);
    }
    ws_off(&w);
    return st;
}

/* ======================================================================
 * concat / split / pad, as one 0/1 matvec
 * ====================================================================== */
vc4_status vc4_cat_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                       vc4_buf *const *dx, const int *axisdim, int nsrc,
                       int outer, int inner)
{
    if (!ctx || !dy || !dx || !axisdim || nsrc <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    int64_t tot = 0;
    for (int i = 0; i < nsrc; ++i) {
        if (axisdim[i] <= 0)
            return VC4_ERR_INVAL;
        tot += axisdim[i];
    }
    if (outer <= 0 || inner <= 0)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    vc4_status st = VC4_OK;
    int off = 0;
    for (int i = 0; i < nsrc && st == VC4_OK; ++i) {
        size_t np = (size_t)outer * axisdim[i] * inner;
        float *W = calloc(np * (size_t)outer * (size_t)tot * inner, 4);
        if (!W)
            st = VC4_ERR_NOMEM;
        else {
            for (int o = 0; o < outer; ++o)
                for (int j = 0; j < axisdim[i]; ++j)
                    for (int k = 0; k < inner; ++k) {
                        size_t d = ((size_t)o * axisdim[i] + j) * inner + k;
                        size_t s = ((size_t)o * (size_t)tot + off + j) * inner
                                   + k;
                        W[d * (size_t)outer * tot * inner + s] = 1.0f;
                    }
            st = matvec_w(&w, W, (uint32_t)np,
                          (uint32_t)((size_t)outer * tot * inner), dy, dx[i]);
            free(W);
        }
        off += axisdim[i];
    }
    ws_off(&w);
    return st;
}

vc4_status vc4_split_bwd(vc4_ctx *ctx, const vc4_buf *const *dy,
                         vc4_buf *dx, const int *axisdim, int nsrc,
                         int outer, int inner)
{
    if (!ctx || !dy || !dx || !axisdim || nsrc <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    int64_t tot = 0;
    for (int i = 0; i < nsrc; ++i) {
        if (axisdim[i] <= 0)
            return VC4_ERR_INVAL;
        tot += axisdim[i];
    }
    if (outer <= 0 || inner <= 0)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    const size_t nt = (size_t)outer * (size_t)tot * inner;
    vc4_status st = VC4_OK;
    {
        int off = 0;
        /* each part is scattered into the concatenated buffer with its own
         * [nt][np] 0/1 matrix, then the parts are added (they are disjoint) */
        vc4_buf cat = ws_get(&w, al64(nt * 4));
        if (!w.ok)
            st = VC4_ERR_NOMEM;
        else {
            off = 0;
            for (int i = 0; i < nsrc && st == VC4_OK; ++i) {
                size_t np = (size_t)outer * axisdim[i] * inner;
                /* scatter the part into `cat` at the right offsets */
                float *P = calloc(np * nt, 4);
                if (!P) {
                    st = VC4_ERR_NOMEM;
                    break;
                }
                for (int o = 0; o < outer; ++o)
                    for (int j = 0; j < axisdim[i]; ++j)
                        for (int k = 0; k < inner; ++k) {
                            size_t d = ((size_t)o * axisdim[i] + j) * inner + k;
                            size_t s = ((size_t)o * (size_t)tot + off + j) *
                                       inner + k;
                            P[s * np + d] = 1.0f;
                        }
                vc4_buf tmp = ws_get(&w, al64(nt * 4));
                if (!w.ok) {
                    free(P);
                    st = VC4_ERR_NOMEM;
                    break;
                }
                st = matvec_w(&w, P, (uint32_t)nt, (uint32_t)np, dy[i], &tmp);
                free(P);
                if (st == VC4_OK) {
                    if (i == 0)
                        st = copy_floats(ctx, &tmp, &cat, nt);
                    else
                        st = ew(ctx, EW_ADD, &cat, &tmp, &cat, nt);
                }
                off += axisdim[i];
            }
        }
        if (st == VC4_OK)
            st = copy_floats(ctx, &cat, dx, nt);
    }
    ws_off(&w);
    return st;
}

vc4_status vc4_pad_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx, int N,
                       int C, int H, int W, int pt, int pl)
{
    if (!ctx || !dy || !dx || N <= 0 || C <= 0 || H <= 0 || W <= 0 || pt < 0 ||
        pl < 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const int Hp = H + 2 * pt, Wp = W + 2 * pl;
    const size_t n_dx = (size_t)N * C * H * W;
    const size_t n_dy = (size_t)N * C * Hp * Wp;
    bw_ws w;
    ws_on(&w, ctx);
    float *M = calloc(n_dx * n_dy, 4);
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    if (!M)
        st = VC4_ERR_NOMEM;
    else {
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        size_t d = (((size_t)n * C + c) * H + y) * W + x;
                        size_t s = (((size_t)n * C + c) * Hp + (y + pt)) * Wp
                                   + (x + pl);
                        M[d * n_dy + s] = 1.0f;
                    }
        st = matvec_w(&w, M, (uint32_t)n_dx, (uint32_t)n_dy, dy, dx);
    }
    free(M);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * transpose / reduce backward
 * ====================================================================== */
vc4_status vc4_transpose_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx,
                             uint32_t rows, uint32_t cols)
{
    if (!ctx || !dy || !dx || !rows || !cols)
        return VC4_ERR_INVAL;
    const size_t n = (size_t)rows * cols;
    if ((uint64_t)dy->bytes < n * 4 || (uint64_t)dx->bytes < n * 4)
        return VC4_ERR_INVAL;
    if (rows % 16 == 0 && cols % 16 == 0 && cols * 4u <= 8191u)
        return vc4_transpose(ctx, dy, dx, cols, rows);
    bw_ws w;
    ws_on(&w, ctx);
    float *M = calloc(n * n, 4);
    vc4_status st = w.ok ? VC4_OK : VC4_ERR_NOMEM;
    if (!M)
        st = VC4_ERR_NOMEM;
    else {
        for (uint32_t i = 0; i < rows; ++i)
            for (uint32_t j = 0; j < cols; ++j)
                M[((size_t)i * cols + j) * n + (size_t)j * rows + i] = 1.0f;
        st = matvec_w(&w, M, (uint32_t)n, (uint32_t)n, dy, dx);
    }
    free(M);
    ws_off(&w);
    return st;
}

vc4_status vc4_reduce_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                          const int *dims, int ndim, int axis, int mean)
{
    if (!ctx || !g || !dx || !dims || ndim < 1 || ndim > 8 || axis < 0 ||
        axis >= ndim)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    const size_t n = (size_t)prod_dims(dims, ndim);
    bw_ws w;
    ws_on(&w, ctx);
    vc4_buf src = *g;
    if (mean) {
        vc4_buf t = ws_get(&w, al64((n / (size_t)dims[axis]) * 4));
        if (!w.ok) {
            ws_off(&w);
            return VC4_ERR_NOMEM;
        }
        vc4_status st = vc4_affine(ctx, g, &t, n / (size_t)dims[axis],
                                   1.0f / (float)dims[axis], 0.0f);
        if (st != VC4_OK) {
            ws_off(&w);
            return st;
        }
        src = t;
    }
    int sd[8];
    memcpy(sd, dims, sizeof(int) * (size_t)ndim);
    sd[axis] = 1;
    vc4_status st = bcast_gen(ctx, &src, sd, dims, ndim, dx, n);
    ws_off(&w);
    return st;
}

/* ======================================================================
 * resize
 * ====================================================================== */
static int nearest_src(int d, int in, int out)
{
    int64_t s = (int64_t)d * in / out;
    if (s < 0) s = 0;
    if (s >= in) s = in - 1;
    return (int)s;
}

/* the nearest-resize matrix: out[p][q] with q = the source cell of p */
static vc4_status nearest_matrix(vc4_ctx *ctx, int planes, int H, int W,
                                 int Ho, int Wo, float **Wout, size_t *n_out,
                                 size_t *n_in)
{
    *n_out = (size_t)planes * Ho * Wo;
    *n_in = (size_t)planes * H * W;
    float *M = calloc(*n_out * *n_in, 4);
    if (!M)
        return VC4_ERR_NOMEM;
    for (int p = 0; p < planes; ++p)
        for (int oy = 0; oy < Ho; ++oy)
            for (int ox = 0; ox < Wo; ++ox) {
                int iy = nearest_src(oy, H, Ho), ix = nearest_src(ox, W, Wo);
                size_t o = ((size_t)p * Ho + oy) * Wo + ox;
                size_t s = ((size_t)p * H + iy) * W + ix;
                M[o * *n_in + s] = 1.0f;
            }
    (void)ctx;
    *Wout = M;
    return VC4_OK;
}

vc4_status vc4_resize_nearest(vc4_ctx *ctx, const vc4_buf *x, vc4_buf *y,
                              int planes, int H, int W, int Ho, int Wo)
{
    if (!ctx || !x || !y || planes <= 0 || H <= 0 || W <= 0 || Ho <= 0 ||
        Wo <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    size_t n_out, n_in;
    float *M = NULL;
    vc4_status st = nearest_matrix(ctx, planes, H, W, Ho, Wo, &M, &n_out,
                                   &n_in);
    if (st != VC4_OK)
        return st;
    bw_ws w;
    ws_on(&w, ctx);
    st = matvec_w(&w, M, (uint32_t)n_out, (uint32_t)n_in, x, y);
    ws_off(&w);
    free(M);
    return st;
}

vc4_status vc4_resize_nearest_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                  vc4_buf *dx, int planes, int H, int W,
                                  int Ho, int Wo)
{
    if (!ctx || !dy || !dx || planes <= 0 || H <= 0 || W <= 0 || Ho <= 0 ||
        Wo <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    size_t n_out, n_in;
    float *M = NULL;
    vc4_status st = nearest_matrix(ctx, planes, H, W, Ho, Wo, &M, &n_out,
                                   &n_in);
    if (st != VC4_OK)
        return st;
    /* dx = M^T dy */
    float *MT = calloc(n_in * n_out, 4);
    if (!MT) {
        free(M);
        return VC4_ERR_NOMEM;
    }
    for (size_t o = 0; o < n_out; ++o)
        for (size_t s = 0; s < n_in; ++s)
            if (M[o * n_in + s] != 0.0f)
                MT[s * n_out + o] = M[o * n_in + s];
    free(M);
    bw_ws w;
    ws_on(&w, ctx);
    st = matvec_w(&w, MT, (uint32_t)n_in, (uint32_t)n_out, dy, dx);
    ws_off(&w);
    free(MT);
    return st;
}

/* bilinear (half-pixel, edge clamp): the four taps and weights of every
 * output pixel.  Pure geometry. */
static float bilin_axis(int o, int in, int out, int *i0, int *i1)
{
    double s = ((double)o + 0.5) * (double)in / (double)out - 0.5;
    int i = (int)floor(s);
    double f = s - (double)i;
    if (i < 0) { i = 0; f = 0.0; }
    if (i >= in - 1) { i = in - 1; f = 0.0; }
    *i0 = i;
    *i1 = i + 1 < in ? i + 1 : in - 1;
    return (float)f;
}

static vc4_status bilin_matrix(vc4_ctx *ctx, int planes, int H, int W, int Ho,
                               int Wo, float **Mout, size_t *n_out,
                               size_t *n_in)
{
    *n_out = (size_t)planes * Ho * Wo;
    *n_in = (size_t)planes * H * W;
    float *M = calloc(*n_out * *n_in, 4);
    if (!M)
        return VC4_ERR_NOMEM;
    int y0, y1, x0, x1;
    for (int oy = 0; oy < Ho; ++oy) {
        float fy = bilin_axis(oy, H, Ho, &y0, &y1);
        for (int ox = 0; ox < Wo; ++ox) {
            float fx = bilin_axis(ox, W, Wo, &x0, &x1);
            int yy[2] = { y0, y1 }, xx[2] = { x0, x1 };
            float wy[2] = { 1.0f - fy, fy }, wx[2] = { 1.0f - fx, fx };
            for (int p = 0; p < planes; ++p) {
                size_t o = ((size_t)p * Ho + oy) * Wo + ox;
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) {
                        size_t s = ((size_t)p * H + yy[a]) * W + xx[b];
                        M[o * *n_in + s] += wy[a] * wx[b];
                    }
            }
        }
    }
    (void)ctx;
    *Mout = M;
    return VC4_OK;
}

vc4_status vc4_resize_bilinear(vc4_ctx *ctx, const vc4_buf *x, vc4_buf *y,
                               int planes, int H, int W, int Ho, int Wo)
{
    if (!ctx || !x || !y || planes <= 0 || H <= 0 || W <= 0 || Ho <= 0 ||
        Wo <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    size_t n_out, n_in;
    float *M = NULL;
    vc4_status st = bilin_matrix(ctx, planes, H, W, Ho, Wo, &M, &n_out,
                                 &n_in);
    if (st != VC4_OK)
        return st;
    bw_ws w;
    ws_on(&w, ctx);
    st = matvec_w(&w, M, (uint32_t)n_out, (uint32_t)n_in, x, y);
    ws_off(&w);
    free(M);
    return st;
}

vc4_status vc4_resize_bilinear_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                   vc4_buf *dx, int planes, int H, int W,
                                   int Ho, int Wo)
{
    if (!ctx || !dy || !dx || planes <= 0 || H <= 0 || W <= 0 || Ho <= 0 ||
        Wo <= 0)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    size_t n_out, n_in;
    float *M = NULL;
    vc4_status st = bilin_matrix(ctx, planes, H, W, Ho, Wo, &M, &n_out,
                                 &n_in);
    if (st != VC4_OK)
        return st;
    float *MT = calloc(n_in * n_out, 4);
    if (!MT) {
        free(M);
        return VC4_ERR_NOMEM;
    }
    for (size_t o = 0; o < n_out; ++o)
        for (size_t s = 0; s < n_in; ++s)
            if (M[o * n_in + s] != 0.0f)
                MT[s * n_out + o] = M[o * n_in + s];
    free(M);
    bw_ws w;
    ws_on(&w, ctx);
    st = matvec_w(&w, MT, (uint32_t)n_in, (uint32_t)n_out, dy, dx);
    ws_off(&w);
    free(MT);
    return st;
}

/* ======================================================================
 * gather / scatter backward
 * ====================================================================== */
vc4_status vc4_gather_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                          const int32_t *idx, size_t n_out, size_t n_in)
{
    if (!ctx || !g || !dx || !idx || !n_out || !n_in)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if ((uint64_t)g->bytes < n_out * 4 || (uint64_t)dx->bytes < n_in * 4)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    float *M = calloc(n_in * n_out, 4);
    vc4_status st = VC4_OK;
    if (!M)
        st = VC4_ERR_NOMEM;
    else {
        for (size_t o = 0; o < n_out; ++o) {
            if (idx[o] < 0 || (size_t)idx[o] >= n_in) {
                st = VC4_ERR_INVAL;
                break;
            }
            M[(size_t)idx[o] * n_out + o] += 1.0f;
        }
        if (st == VC4_OK)
            st = matvec_w(&w, M, (uint32_t)n_in, (uint32_t)n_out, g, dx);
    }
    free(M);
    ws_off(&w);
    return st;
}

vc4_status vc4_scatter_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                           const int32_t *idx, size_t n, size_t n_out)
{
    if (!ctx || !g || !dx || !idx || !n || !n_out)
        return VC4_ERR_INVAL;
    if (ctx->dead)
        return VC4_ERR_TIMEOUT;
    if ((uint64_t)g->bytes < n_out * 4 || (uint64_t)dx->bytes < n * 4)
        return VC4_ERR_INVAL;
    bw_ws w;
    ws_on(&w, ctx);
    float *M = calloc(n * n_out, 4);
    vc4_status st = VC4_OK;
    if (!M)
        st = VC4_ERR_NOMEM;
    else {
        for (size_t i = 0; i < n; ++i) {
            if (idx[i] < 0 || (size_t)idx[i] >= n_out) {
                st = VC4_ERR_INVAL;
                break;
            }
            M[i * n_out + (size_t)idx[i]] = 1.0f;
        }
        if (st == VC4_OK)
            st = matvec_w(&w, M, (uint32_t)n, (uint32_t)n_out, g, dx);
    }
    free(M);
    ws_off(&w);
    return st;
}
