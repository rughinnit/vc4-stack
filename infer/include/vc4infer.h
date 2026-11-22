/*
 * libvc4infer - a VideoCore IV inference engine layer.
 *
 * Builds on the hand-written, element-verified V3D QPU GEMM from
 * /home/raspi/work/asm3 (16x16 tile, k-major packed A panel, both TMUs
 * 2-deep, VPM->DMA stores) and adds the pieces needed to run real
 * inference: one dma-buf arena, a kernel blob cache with the two-launch
 * rule, conv2d via im2col, elementwise ops.
 *
 * STRICT DISPATCH: every operation goes through an op registry.  If an op,
 * a shape, a layout or a dtype is not implemented on the GPU this library
 * returns VC4_ERR_UNSUPPORTED.  There is no CPU compute fallback anywhere in
 * the library; the CPU only ever moves bytes (layout/padding), never MACs.
 *
 * Threading: a vc4_ctx is not thread safe.  Multiple processes are safe
 * (each gets its own arena), but only one GPU user should run at a time;
 * the test binaries take /tmp/vc4-gpu.lock with flock before touching the GPU.
 */
#ifndef VC4INFER_H
#define VC4INFER_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VC4INFER_VERSION_MAJOR 0
#define VC4INFER_VERSION_MINOR 1
#define VC4INFER_VERSION_STRING "1.2.3"

/* Default arena capacity.  One dma-buf, persistent user buffers + scratch. */
#define VC4_ARENA_DEFAULT (64u << 20)

typedef enum {
    VC4_OK            =  0,
    VC4_ERR_INVAL     = -1,  /* bad argument                                   */
    VC4_ERR_NOMEM     = -2,  /* arena exhausted / dma-buf allocation failed    */
    VC4_ERR_IO        = -3,  /* /dev/mem, /dev/dma_heap, /dev/vcio             */
    VC4_ERR_TIMEOUT   = -4,  /* QPU program did not complete; ctx is now dead  */
    VC4_ERR_UNSUPPORTED = -5,/* strict dispatch: not implemented on the GPU   */
    VC4_ERR_GPU       = -6,  /* V3D error status / queue error after a launch  */
    VC4_ERR_KERNEL    = -7   /* kernel blob missing and generation failed      */
} vc4_status;

typedef enum {
    VC4_DTYPE_F32 = 0,
    VC4_DTYPE_COUNT
} vc4_dtype;

typedef enum {
    VC4_LAYOUT_ROW_MAJOR = 0,   /* A[M][K], B[K][N], C[M][N], contiguous     */
    VC4_LAYOUT_COUNT
} vc4_layout;

typedef enum {
    VC4_OP_MATMUL = 0,
    VC4_OP_CONV2D,          /* 3x3 and 1x1, stride 1, zero padding        */
    VC4_OP_ADD,
    VC4_OP_MUL,
    VC4_OP_RELU,
    VC4_OP_SIGMOID,
    VC4_OP_SILU,
    VC4_OP_RELU_BWD,        /* out = b if a > 0 else 0 (masked copy)         */
    VC4_OP_RSQRT,           /* out = 1/sqrt(a), a > 0 (Adam)                 */
    VC4_OP_CONV2D_BWD,      /* conv dX + dW (im2col_r, gemm3, col2im)        */
    VC4_OP_COL2IM,          /* scatter-add the im2col gradient                */
    VC4_OP_IM2COL,          /* row-major im2col matrix                        */
    VC4_OP_MAXPOOL,         /* 2x2 stride-2 max                              */
    VC4_OP_MAXPOOL_BWD,     /* 2x2 stride-2 argmax-mask backward             */
    /* the training op set: forward and backward (see docs/TRAINING.md) */
    VC4_OP_SUB,             /* a - b (broadcast)                             */
    VC4_OP_DIV,             /* a / b (broadcast)                             */
    VC4_OP_SOFTMAX,         /* last-axis softmax, any class count            */
    VC4_OP_SOFTMAX_BWD,     /* y*(g - sum(g*y))                              */
    VC4_OP_SOFTMAX_CE_BWD,  /* softmax + cross-entropy backward, any classes */
    VC4_OP_BATCHNORM,       /* train-mode batch stats + running update       */
    VC4_OP_BATCHNORM_BWD,   /* dx, dgamma, dbeta                             */
    VC4_OP_LAYERNORM,       /* last-axis normalisation                       */
    VC4_OP_LAYERNORM_BWD,   /* dx, dgamma, dbeta                             */
    VC4_OP_AVGPOOL,         /* NxN / stride-S average pool                   */
    VC4_OP_AVGPOOL_BWD,     /* uniform scatter                               */
    VC4_OP_ADAPTIVE_AVGPOOL,     /* adaptive average pool to (outH,outW)     */
    VC4_OP_ADAPTIVE_AVGPOOL_BWD, /* window-area scatter                      */
    VC4_OP_MAXPOOLN_BWD,    /* NxN/stride-S argmax routing                   */
    VC4_OP_BWOP,            /* the elementwise-derivative family             */
    VC4_OP_BINOP_BWD,       /* add/sub/mul/div with broadcasting             */
    VC4_OP_CAT,             /* concat along one axis                         */
    VC4_OP_SPLIT,           /* split along one axis                          */
    VC4_OP_RESIZE_NEAREST,  /* nearest resize (asymmetric mapping)           */
    VC4_OP_RESIZE_BILINEAR, /* bilinear resize                               */
    VC4_OP_TRANSPOSE_BWD,   /* = transpose                                   */
    VC4_OP_PAD,             /* zero pad                                       */
    VC4_OP_PAD_BWD,         /* = crop                                         */
    VC4_OP_SUM,             /* reduce over an axis                           */
    VC4_OP_MEAN,            /* mean over an axis                             */
    VC4_OP_SUM_BWD,         /* broadcast the gradient                        */
    VC4_OP_GATHER,          /* table gather                                   */
    VC4_OP_GATHER_BWD,      /* additive scatter                               */
    VC4_OP_SCATTER_BWD,     /* = gather                                       */
    VC4_OP_GLOBAL_AVGPOOL,  /* mean over H,W                                  */
    VC4_OP_GLOBAL_AVGPOOL_BWD, /* uniform scatter over H,W                   */
    /* exact FP32 comparisons and predicates (0.0f / 1.0f, bit-exact) */
    VC4_OP_CMP_EQ,          /* a == b                                        */
    VC4_OP_CMP_GT,          /* a >  b                                        */
    VC4_OP_CMP_GE,          /* a >= b                                        */
    VC4_OP_CMP_LT,          /* a <  b                                        */
    VC4_OP_CMP_LE,          /* a <= b                                        */
    VC4_OP_ISNAN,           /* a is NaN                                      */
    VC4_OP_ISINF,           /* a is +inf or -inf                             */
    /* the elementwise selection pair: hardware FMAX/FMIN, the same bodies the
     * elementwise-derivative family already carries (kops.EWB 'max'/'min') */
    VC4_OP_MAX,             /* out = a >= b ? a : b                          */
    VC4_OP_MIN,             /* out = a <= b ? a : b                          */
    VC4_OP_COUNT
} vc4_op;

/* A buffer that lives inside the single dma-buf arena. */
typedef struct {
    uint32_t off;       /* byte offset from the arena base                   */
    size_t   bytes;     /* size requested at allocation                      */
} vc4_buf;

typedef struct vc4_ctx vc4_ctx;

/* ------------------------------------------------------------------ lifecycle */
/* cache_dir: where generated kernel blobs are cached (NULL = <cwd>/kcache or
 * $VC4INFER_CACHE).  arena_bytes: 0 = VC4_ARENA_DEFAULT (falls back smaller
 * if the CMA heap is tight; see vc4_arena_bytes()). */
vc4_status vc4_init(vc4_ctx **out, const char *cache_dir, size_t arena_bytes);
void       vc4_shutdown(vc4_ctx *ctx);
const char *vc4_strerror(vc4_status st);

/* -------------------------------------------------------------------- arena */
size_t vc4_arena_bytes(const vc4_ctx *ctx);
size_t vc4_arena_used(const vc4_ctx *ctx);
vc4_status vc4_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out);      /* persistent */
vc4_status vc4_upload(vc4_ctx *ctx, vc4_buf *b, const void *src, size_t bytes);
vc4_status vc4_download(vc4_ctx *ctx, const vc4_buf *b, void *dst, size_t bytes);
void      *vc4_host_ptr(vc4_ctx *ctx, const vc4_buf *b);   /* cacheable mapping */
/*
 * Drop the client CPU's cached copies of a buffer.  The client's arena mapping
 * is cacheable (the daemon's /dev/mem view is not), so a zero-copy reader must
 * invalidate before touching GPU-written bytes, and a zero-copy writer must
 * invalidate first or the clean that precedes the next launch writes the stale
 * rest of a cache line back over the QPU's data.  vc4_upload and vc4_download
 * already do this; use this only for vc4_host_ptr().
 */
void       vc4_invalidate_buf(vc4_ctx *ctx, const vc4_buf *b);
uint32_t   vc4_bus_addr(const vc4_ctx *ctx, const vc4_buf *b);

/* --------------------------------------------------------------------- ops */
/*
 * C = A * B, all f32 row-major, A is MxK, B is KxN, C is MxN, ld = K/N/N.
 * Implemented by the verified QPU GEMM: A is first transposed into the
 * k-major packed panel by a second QPU launch (the asm3 packer, generalised),
 * then the GEMM reads one 64 B line per k.  M,N,K are rounded up to a
 * multiple of 16 internally; the padding rows/cols are zero.
 */
vc4_status vc4_matmul(vc4_ctx *ctx,
                      const vc4_buf *A, const vc4_buf *B, vc4_buf *C,
                      uint32_t M, uint32_t N, uint32_t K, vc4_layout layout);

/*
 * out = conv2d(in, wt) + bias  (bias may be NULL), NCHW, f32.
 *   in  [Cin][H][W]      wt [Cout][Cin][KH][KW]      out [Cout][HO][WO]
 * HO = (H + 2*pad - KH)/stride + 1, WO likewise.
 * Implemented as im2col (QPU gather kernel) -> vc4_matmul.  Supported:
 * KH=KW in {1,3}, stride == 1, any pad >= 0, Cin % 16 == 0, Cout % 16 == 0.
 * (The ONNX runner composes stride 2 and depthwise from this op; see
 * docs/ONNX.md.)
 */
vc4_status vc4_conv2d(vc4_ctx *ctx,
                      const vc4_buf *in, int Cin, int H, int W,
                      const vc4_buf *wt, int Cout, int KH, int KW,
                      const vc4_buf *bias,
                      int stride, int pad,
                      vc4_buf *out, int *HO, int *WO);

/* Elementwise over n floats; b may be NULL for the unary ops. */
vc4_status vc4_elementwise(vc4_ctx *ctx, vc4_op op,
                           const vc4_buf *a, const vc4_buf *b, vc4_buf *out,
                           size_t n);
vc4_status vc4_relu(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n);
vc4_status vc4_sigmoid(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n);
vc4_status vc4_silu(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n);
vc4_status vc4_rsqrt(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n);
/* out = dy if a > 0 else 0 (relu backward mask; a = the forward input) */
vc4_status vc4_relu_backward(vc4_ctx *ctx, const vc4_buf *a,
                             const vc4_buf *dy, vc4_buf *out, size_t n);

/* ------------------------------------------------------------ NHWC plane */
/*
 * The YOLO path.  A tensor is [rows][pitch][C] fp32 contiguous, pixel p at
 * p*C*4 bytes; `pitch` is the storage width and `w` the logical width.  The
 * VDR pitch of a 16-pixel im2col row is stride*Cin*4 bytes (not the plane), so
 * any feature map with <= 2047 channels works, and the GEMM's C[M][N] (M =
 * pixels, N = channels) is exactly the NHWC output.  nimg batches are
 * contiguous images; every op is one launch (conv = im2col + GEMM) for the
 * whole batch.
 */
size_t vc4_conv_nhwc_wpanel_bytes(int Cin, int Cout, int KH, int KW, int bias);
/* wt [Cout][Cin][KH][KW] (ci-major) + optional bias -> packed B panel */
vc4_status vc4_conv_nhwc_pack_w(vc4_ctx *ctx, const vc4_buf *wt, int Cin,
                                int Cout, int KH, int KW, const vc4_buf *bias,
                                vc4_buf *wpanel);
/*
 * out = conv(in) + bias, NHWC.  in [nimg][H][W][Cin] (row pitch in_pitch),
 * out [nimg][ho][Wg][Cout] with Wg = align16(wo); the columns [wo, Wg) are
 * junk (never read back).  stride in {1,2}, KH,KW in {1,3,5}, Cin/Cout % 16 ==
 * 0, stride*Cin*4 <= 8191 (VDR pitch).  The stride-2 decimation happens in
 * the im2col gather (the VDR reads pixels `stride` apart), not by phase
 * images, so a stride-2 conv runs the same MAC count as stride 1.
 *
 * vc4_conv_nhwc_dil is the same op with an explicit dilation (1..8); the
 * output size is (H + 2p - dil*(K-1) - 1)/stride + 1.
 */
vc4_status vc4_conv_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int Cin,
                         int H, int W, int in_pitch, const vc4_buf *wpanel,
                         int Cout, int KH, int KW, int stride, int pad,
                         vc4_buf *out, int *ho, int *wo, int *opitch);
vc4_status vc4_conv_nhwc_dil(vc4_ctx *ctx, const vc4_buf *in, int nimg, int Cin,
                             int H, int W, int in_pitch,
                             const vc4_buf *wpanel, int Cout, int KH, int KW,
                             int stride, int pad, int dilation, vc4_buf *out,
                             int *ho, int *wo, int *opitch);
/*
 * The same op with independent top/left/bottom/right padding.  The output
 * size is (H + pt + pb - dil*(KH-1) - 1)/stride + 1 (and likewise for W).
 * It exists so a decomposed 6x6 stride-2 kernel can offset the input by one
 * row/column per phase without materialising a shifted activation copy.
 */
vc4_status vc4_conv_nhwc_pads(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                              int Cin, int H, int W, int in_pitch,
                              const vc4_buf *wpanel, int Cout, int KH, int KW,
                              int stride, int pt, int pl, int pb, int pr,
                              int dilation, vc4_buf *out, int *ho, int *wo,
                              int *opitch);
/*
 * Grouped conv: `groups` independent convs over channel blocks of Cin/groups
 * and Cout/groups.  wpanel[g] is the packed panel of group g (call
 * vc4_conv_nhwc_pack_w with the group's [Cout/groups][Cin/groups][KH][KW]
 * slice).  Cin/groups and Cout/groups must each be multiples of 16; the input
 * is padded once and the activation panel is reused, and each group's GEMM
 * writes straight into its channel range of the single NHWC output (the GEMM
 * C row stride is set to Cout), so no host interleave is needed.  MACs are
 * the real grouped MACs, not a block-diagonal expansion.
 */
vc4_status vc4_conv_nhwc_group(vc4_ctx *ctx, const vc4_buf *in, int nimg,
                               int Cin, int H, int W, int in_pitch,
                               const vc4_buf *const *wpanel, int Cout,
                               int groups, int KH, int KW, int stride, int pad,
                               int dilation, vc4_buf *out, int *ho, int *wo,
                               int *opitch);
/* depthwise 3x3 pad 1: wt [C][1][3][3] -> WP[tap][C] */
size_t vc4_depthwise_wpanel_bytes(int C);
vc4_status vc4_depthwise_pack_w(vc4_ctx *ctx, const vc4_buf *wt, int C,
                                vc4_buf *wp);
vc4_status vc4_depthwise_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                              int H, int W, int in_pitch, const vc4_buf *wp,
                              vc4_buf *out, int *ho, int *wo, int *opitch);
/* max pool kxk stride 1 with zero padding (SPPF: k=5, pad=2) */
vc4_status vc4_maxpool_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                            int H, int W, int in_pitch, int k, int pad,
                            vc4_buf *out, int *ho, int *wo, int *opitch);
/* nearest-neighbour 2x: [nimg][H][W][C] -> [nimg][2H][2W][C] */
vc4_status vc4_upsample2x_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int C,
                               int H, int W, int in_pitch, vc4_buf *out,
                               int *ho, int *wo, int *opitch);
/* channel concat: nsrc sources of Csrc[i] channels, M pixels per image */
vc4_status vc4_concat_nhwc(vc4_ctx *ctx, const vc4_buf *const *src,
                           const int *Csrc, int nsrc, int nimg,
                           size_t pixels_per_img, vc4_buf *out, int Cdst);

/* ------------------------------------------------------- training primitives */
/* Y = X^T (dense transpose, 16x16 tiles; both dims multiple of 16) */
vc4_status vc4_transpose(vc4_ctx *ctx, const vc4_buf *in, vc4_buf *out,
                         uint32_t rows, uint32_t cols);
/* C = A * B^T, A MxK, B NxK, all row-major; M,N,K multiples of 16 */
vc4_status vc4_matmul_bt(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                         vc4_buf *C, uint32_t M, uint32_t N, uint32_t K);
/*
 * dlogits[r][c] = (softmax(logits[r])[c] - onehot(labels[r])[c]) * inv_batch
 * plus per-QPU (loss, correct) totals in partial[] (lane 0 / lane 1 of a
 * 64 B slot per QPU).  logits/dlogits are rows x 16 floats.  The logits
 * padding columns must hold a very negative bias (-1e30) so they contribute
 * exp = 0.  See vc4_train.c for the exact buffer slack requirements.
 */
vc4_status vc4_softmax_ce(vc4_ctx *ctx, const vc4_buf *logits,
                          const vc4_buf *labels, vc4_buf *dlogits,
                          vc4_buf *partial, uint32_t rows, float inv_batch);

/*
 * Forward softmax over rows of at most 16 classes (the fused kernel's lane
 * width).  logits/out are rows x cols f32; internally each row is copied to
 * the kernel's 16-column padded form with -1e30 in the padding, and the
 * probability rows are copied back.  rows % 8 == 0.  The exponential and the
 * reciprocal run on the QPU (the same degree-5 2^f polynomial and 3x Newton
 * reciprocal as vc4_softmax_ce); the CPU only pads and compacts bytes.
 */
vc4_status vc4_softmax_fwd(vc4_ctx *ctx, const vc4_buf *logits,
                           uint32_t rows, uint32_t cols, vc4_buf *out);

/* ---------------------------------------------- conv / pool backward */
/*
 * A = im2col(X), row-major [M][K], M = ho*align16(W+2*pad), K = Cin*KH*KW,
 * k = tap*Cin + ci (the same tap-major order vc4_conv2d builds internally).
 * Same shape constraints as vc4_conv2d.  out must hold M*K*4 bytes.
 */
vc4_status vc4_im2col(vc4_ctx *ctx, const vc4_buf *in, int Cin, int H, int W,
                      int KH, int KW, int pad, vc4_buf *out, int *rows);

/*
 * dXpad = col2im(dA): accumulates the overlapping 3x3/1x1 patches.
 *   dA   [HO*Wp][K] row-major, K = Cin*KH*KW  (the im2col ordering)
 *   dXpad [Cin][HO+KH-1][Wp] zero-initialised, Wp = align16(WO+KW-1)
 * The QPU accumulates the taps in fixed order (deterministic); see
 * docs/NUMERICS.md for the fp32 truncation caveat.  The buffer needs 64 B of
 * slack after the last element.
 */
vc4_status vc4_col2im(vc4_ctx *ctx, const vc4_buf *dA, int Cin, int K,
                      int KH, int KW, int HO, int WO, vc4_buf *dXpad,
                      int *Hp, int *Wp);

/*
 * Conv2d backward (stride 1, KH == KW in {1,3}, Cin % 16 == 0, Cout % 16 == 0).
 *   dy   [Cout][ho][wo]     gradient of the conv output (NCHW)
 *   dW   [Cout][Cin][KH][KW]  (ci-major, same layout as wt)
 *   dX   [Cin][H][W]
 * Everything that computes runs on the QPU: im2col(row-major), gemm3
 * (dA = dY*W^T and dW = dY^T*A), and col2im.  Bias gradients are not
 * implemented (no extra kernel); conv bias itself is forward-only.
 */
vc4_status vc4_conv2d_backward(vc4_ctx *ctx,
                               const vc4_buf *in, int Cin, int H, int W,
                               const vc4_buf *wt, int Cout, int KH, int KW,
                               const vc4_buf *dy, int pad,
                               vc4_buf *dW, vc4_buf *dX);

/*
 * 2x2 stride-2 max pool over [C][H][W] (H, W even, W % 16 == 0).
 * The output row pitch is out_pitch (>= W/2, a multiple of 16; lanes past
 * W/2 are zero).  in needs 64 B of slack (the 2x2 gathers over-read a row).
 */
vc4_status vc4_maxpool2x2(vc4_ctx *ctx, const vc4_buf *in, int C, int H, int W,
                          int out_pitch, vc4_buf *out, int *Ho, int *Wo);
/* dx[c][i][x] = dy[c][i/2][x/2] when x is a 2x2 group maximum (ties share) */
vc4_status vc4_maxpool2x2_backward(vc4_ctx *ctx, const vc4_buf *in,
                                   const vc4_buf *dy, int C, int H, int W,
                                   int dy_pitch, vc4_buf *dx);

/*
 * Generic NxN / stride-S max pool over [C][H][W].
 *   out has align4(Ho) rows of pitch align16(Wo); only [..][Ho][Wo] is
 *   meaningful.  The host pads the border with -inf (bytes only), so any
 *   pad >= 0, N >= 1 and stride >= 1 are supported; all arithmetic is on the
 *   QPU.  ho/wo are the ONNX output dimensions:
 *   Ho = (H + 2*pad - N)/stride + 1, Wo likewise.
 */
vc4_status vc4_maxpooln(vc4_ctx *ctx, const vc4_buf *in, int C, int H, int W,
                        int N, int stride, int pad,
                        vc4_buf *out, int *Ho, int *Wo);


/* ------------------------------------------- generic kernels (vc4_gen.c) */
/*
 * Unary math, built as range reduction + minimax polynomial + exact integer
 * scaling on the QPU (the FPU truncates, so no hardware transcendental is
 * used).  Measured accuracy per op is in docs/ONNX.md section 10 and is
 * checked element-by-element by tests/test_genops.c.
 */
typedef enum {
    VC4_UOP_ABS = 0, VC4_UOP_NEG, VC4_UOP_IDENTITY, VC4_UOP_CLIP,
    VC4_UOP_LEAKYRELU, VC4_UOP_FLOOR, VC4_UOP_CEIL, VC4_UOP_ROUND,
    VC4_UOP_SIGN, VC4_UOP_EXP, VC4_UOP_LOG, VC4_UOP_SQRT,
    VC4_UOP_RECIPROCAL, VC4_UOP_TANH, VC4_UOP_ERF, VC4_UOP_ELU,
    VC4_UOP_SELU, VC4_UOP_SOFTPLUS, VC4_UOP_SOFTSIGN, VC4_UOP_GELU,
    VC4_UOP_GELU_ERF, VC4_UOP_HARDSIGMOID, VC4_UOP_HARDSWISH,
    VC4_UOP_SIN, VC4_UOP_COS,
    VC4_UOP_AFFINE,          /* out = a*p0 + p1 (gradient broadcast scale)   */
    VC4_UOP_MISH,            /* x*tanh(softplus(x))                          */
    /* inverse / hyperbolic family (2026-11-20): log- and exp-based bodies,
     * docs/OPS-UNARY.md has the method, domain and measured accuracy */
    VC4_UOP_ACOS, VC4_UOP_ACOSH, VC4_UOP_ASIN, VC4_UOP_ASINH,
    VC4_UOP_ATAN, VC4_UOP_ATANH, VC4_UOP_TAN, VC4_UOP_SINH, VC4_UOP_COSH,
    VC4_UOP_RNG,             /* uniform [0,1) from a counter (kops.b_rng)   */
    VC4_UOP_COUNT
} vc4_uop;

/* out = f(a, b, c): where (a != 0 ? b : c), clip3 (min(max(a,b),c)), prelu */
typedef enum { VC4_SEL_WHERE = 0, VC4_SEL_CLIP3, VC4_SEL_PRELU } vc4_selop;
typedef enum {
    VC4_RED_SUM = 0, VC4_RED_MEAN, VC4_RED_MAX, VC4_RED_MIN,
    VC4_RED_ARGMAX, VC4_RED_ARGMIN
} vc4_redop;

/* out[i] = uop(a[i]) for i < n; p0..p2 are the op attributes. */
vc4_status vc4_unary(vc4_ctx *ctx, vc4_uop op, const vc4_buf *a,
                     vc4_buf *out, size_t n, float p0, float p1, float p2);
/*
 * Diagnostic instrument: one 16-lane vector in, the raw 16-lane result out.
 * `seq` names a minimal ALU dependency chain (see kgen.build_probe_seq and
 * docs/NUMERICS.md): the caller compares the words bit-for-bit with a model
 * of the truncating FPU.  `a` must hold 16 floats, `out` 256 bytes (the
 * verified 4-row store).  Not part of the inference path.
 */
vc4_status vc4_probe(vc4_ctx *ctx, const char *seq, const vc4_buf *a,
                     vc4_buf *out, size_t n);
vc4_status vc4_select3(vc4_ctx *ctx, vc4_selop op, const vc4_buf *a,
                       const vc4_buf *b, const vc4_buf *c, vc4_buf *out,
                       size_t n);
/*
 * out[o] = arena[tab[o]]: `tab` is n int32 *arena byte offsets* (negative
 * entries write `fill`).  This is the layout workhorse: pad, tile, expand,
 * transpose, slice, gather (incl. GatherElements), depth/space, nearest and
 * cubic resize taps, and the compaction step of NonZero/TopK.  The table is
 * 4 B per output element and is built from integer index arithmetic.
 */
vc4_status vc4_gather(vc4_ctx *ctx, const vc4_buf *tab, vc4_buf *out,
                      size_t n, float fill);
/* Contiguous 64 B-granular copy (no table); bytes % 64 == 0. */
vc4_status vc4_blit(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t bytes);
/*
 * The affine layout copy: the QPU's layout workhorse (pad, channel concat,
 * split, slice, NHWC<->NCHW, transpose, nearest resize, broadcast expand).
 *
 * A *vector* is 16 consecutive destination words (64 B, the verified 4-row
 * VPM->DMA store) and its 16 source words are read one per lane from
 * ``src + lane*slane``, so ``slane == 4`` is a contiguous copy and any other
 * value is a strided gather.  A *row* is ``vpr`` vectors (a multiple of 4);
 * QPU q handles rows q, q+nqpu, ...  ``srow``/``drow`` are the byte steps
 * between consecutive rows and ``svec``/``dvec`` between consecutive vectors.
 * ``dvec`` is the stored row stride of the 16 destination words and must be a
 * 4-byte multiple in 64..65535.
 *
 * Nothing here is element arithmetic: the host passes byte strides, the QPUs
 * move the data.  Element-verified against a CPU oracle (tests/test_copyops.c)
 * for contiguous, pitched, channel-strided, laned (transpose) and 4-byte
 * lane-gather shapes, including W not a multiple of 4 and odd extents.
 */
vc4_status vc4_copy2d(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                      uint32_t rows, uint32_t vpr, uint32_t srow,
                      uint32_t drow, uint32_t svec, uint32_t dvec,
                      uint32_t slane);
/* As vc4_copy2d, with `units` vectors per DMA store (1, 2 or 4) and `depth`
 * words per vector (1..16).  units=1, depth=k moves a row of k < 16 floats
 * without writing past it. */
vc4_status vc4_copy2d_n(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                        uint32_t rows, uint32_t vpr, uint32_t srow,
                        uint32_t drow, uint32_t svec, uint32_t dvec,
                        uint32_t slane, uint32_t nlane, uint32_t units,
                        uint32_t depth);
/* The general form: `nblk` blocks per row, `sbstep`/`dbstep` bytes apart in
 * the two streams (a channel concat/split of more than 16 channels per
 * pixel: the step inside a row is 64 B while the step to the next row is a
 * different constant, so the two cannot be one affine stride), with the
 * store shape (units vectors x depth words) and a lane count. */
vc4_status vc4_copy2d_x(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                        uint32_t rows, uint32_t vpr, uint32_t srow,
                        uint32_t drow, uint32_t svec, uint32_t dvec,
                        uint32_t slane, uint32_t nlane, uint32_t nblk,
                        uint32_t sbstep, uint32_t dbstep, uint32_t units,
                        uint32_t depth);
/* out[rows][vpr*16] = value (the -inf MaxPool border, zero channel padding). */
vc4_status vc4_fill2d(vc4_ctx *ctx, vc4_buf *out, uint32_t rows, uint32_t vpr,
                      uint32_t drow, uint32_t dvec, float value);
/*
 * out[o] = reduce over a[o*red .. o*red+red) for o < outer.
 * ARGMAX/ARGMIN write raw int32 indices into `out` (interpret with a
 * .view(int32)); the value ops write fp32.
 */
vc4_status vc4_reduce(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                      uint32_t outer, uint32_t red, vc4_redop op);
/* Inclusive prefix sum over the last axis; cols % 256 == 0. */
vc4_status vc4_cumsum(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out,
                      uint32_t rows, uint32_t cols);

/* ======================================================================
 * training: forward + backward
 *
 * Every entry point here runs *all* its arithmetic on the QPUs.  The CPU only
 *   - builds integer index tables / geometry (byte offsets, window bounds),
 *   - writes scalar hyperparameters (eps, momentum, inv_batch),
 *   - pads tails to a 256-float group (byte copy).
 * The compositions call the already element-verified kernels (gather, reduce,
 * elementwise, unary, copy2d); no new CPU math exists.  Where a weight table
 * is needed (bilinear resize, adaptive pooling) it is derived from *geometry
 * only* -- no activation value ever reaches the ARM.
 *
 * Layouts are fp32 row-major.  "NCHW" tensors are [N][C][H][W] contiguous:
 * a single (C,H,W) image with N == 1 is the same thing.
 * ====================================================================== */

/* ------------------------------------------------- elementwise derivatives */
typedef enum {
    VC4_BW_SIGMOID = 0,   /* g * s * (1-s)                                  */
    VC4_BW_SILU,          /* g * (s + x*s*(1-s))                            */
    VC4_BW_TANH,          /* g * (1 - tanh(x)^2)                            */
    VC4_BW_EXP,           /* g * exp(x)                                     */
    VC4_BW_LOG,           /* g / x                                          */
    VC4_BW_SQRT,          /* g / (2 sqrt(x))                                */
    VC4_BW_ABS,           /* g * sign(x)                                    */
    VC4_BW_CLIP,          /* g inside [p0,p1], 0 outside                    */
    VC4_BW_LEAKYRELU,     /* g * (x>0 ? 1 : p0)                             */
    VC4_BW_HARDSWISH,     /* g * d/dx [x*clip(ax+b,0,1)]                    */
    VC4_BW_HARDSIGMOID,   /* g * (inside the linear band ? a : 0)           */
    VC4_BW_GELU,          /* g * d/dx gelu (tanh approximation)             */
    VC4_BW_GELU_ERF,      /* g * d/dx gelu (exact erf form)                 */
    VC4_BW_ERF,           /* g * 2/sqrt(pi) * exp(-x^2)                     */
    VC4_BW_COUNT
} vc4_bwop;
/* p0/p1 are the op attributes: clip lo/hi, leakyrelu alpha, hard* alpha/beta. */
vc4_status vc4_bwop_run(vc4_ctx *ctx, vc4_bwop op, const vc4_buf *x,
                        const vc4_buf *g, vc4_buf *out, size_t n, float p0,
                        float p1);
/* out = a*p0 + p1 (the scale an upstream reduction needs). */
vc4_status vc4_affine(vc4_ctx *ctx, const vc4_buf *a, vc4_buf *out, size_t n,
                      float p0, float p1);

/* ------------------------------------------------------------- softmax/CE */
/* Last-axis softmax over any class count: x, y are [rows][cols]. */
vc4_status vc4_softmax(vc4_ctx *ctx, const vc4_buf *x, uint32_t rows,
                       uint32_t cols, vc4_buf *y);
/* dx = y * (g - sum_c g*y), the row sum broadcast back over the columns. */
vc4_status vc4_softmax_bwd(vc4_ctx *ctx, const vc4_buf *y, const vc4_buf *g,
                           vc4_buf *dx, uint32_t rows, uint32_t cols);
/*
 * Fused softmax + cross-entropy backward for any class count.
 *   logits [rows][cols]  ->  dlogits = (softmax(logits) - onehot(label))/B
 *   loss (1 float, arena) = sum_row -log p[label] / B
 * y_scratch is optional (>= rows*cols*4 B) and reused for the probabilities;
 * pass NULL to let the library carve it from the arena's per-op scratch.
 * labels are int32, one per row.
 */
vc4_status vc4_softmax_ce_bwd(vc4_ctx *ctx, const vc4_buf *logits,
                              const vc4_buf *labels, vc4_buf *dlogits,
                              vc4_buf *loss, vc4_buf *y_scratch,
                              uint32_t rows, uint32_t cols, float inv_batch);

/* -------------------------------------------------------------- batchnorm */
/*
 * Train-mode batch normalisation over the N*H*W elements of each channel.
 *   x        [N][C][HW]
 *   y        [N][C][HW]           save_mean/save_invstd  [C]
 *   running_mean/running_var [C]  updated when update_running != 0:
 *      running = (1-momentum)*running + momentum*batch_stat   (on the QPU)
 * The batch variance is the biased one (as PyTorch's training pass uses);
 * running_var is updated with the *unbiased* estimate, like PyTorch.
 */
vc4_status vc4_batchnorm_fwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *beta,
                             vc4_buf *y, vc4_buf *save_mean,
                             vc4_buf *save_invstd, vc4_buf *running_mean,
                             vc4_buf *running_var, uint32_t N, uint32_t C,
                             uint32_t HW, float momentum, float eps,
                             int update_running);
/* Inference: y = (x - running_mean)/sqrt(running_var+eps)*gamma + beta. */
vc4_status vc4_batchnorm_infer(vc4_ctx *ctx, const vc4_buf *x,
                               const vc4_buf *gamma, const vc4_buf *beta,
                               const vc4_buf *running_mean,
                               const vc4_buf *running_var, vc4_buf *y,
                               uint32_t N, uint32_t C, uint32_t HW, float eps);
/* dgamma/dbeta need no saved stats; dx does. */
vc4_status vc4_batchnorm_bwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *dy,
                             const vc4_buf *save_mean,
                             const vc4_buf *save_invstd, vc4_buf *dx,
                             vc4_buf *dgamma, vc4_buf *dbeta, uint32_t N,
                             uint32_t C, uint32_t HW);

/* -------------------------------------------------------------- layernorm */
vc4_status vc4_layernorm_fwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *beta,
                             vc4_buf *y, vc4_buf *save_mean,
                             vc4_buf *save_invstd, uint32_t rows,
                             uint32_t cols, float eps);
vc4_status vc4_layernorm_bwd(vc4_ctx *ctx, const vc4_buf *x,
                             const vc4_buf *gamma, const vc4_buf *dy,
                             const vc4_buf *save_mean,
                             const vc4_buf *save_invstd, vc4_buf *dx,
                             vc4_buf *dgamma, vc4_buf *dbeta, uint32_t rows,
                             uint32_t cols);

/* ----------------------------------------------------------------- pooling */
/*
 * Average pool over [C][H][W], N x N window, stride S, zero padding `pad`.
 * dx is [C][H][W]; dy is [C][Ho][Wo] *packed* (Ho = (H+2pad-N)/S+1).
 * count_include_pad != 0 divides by N*N, else by the number of in-bounds
 * samples in the window (ONNX default).
 */
vc4_status vc4_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx,
                           int C, int H, int W, int N, int S, int pad,
                           int count_include_pad);
/* Global average pool: d/dx of (1/(H*W)) * sum over the plane. */
vc4_status vc4_global_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                  vc4_buf *dx, int C, int H, int W);
/* Adaptive average pool to (outH,outW) (ONNX window bounds). */
vc4_status vc4_adaptive_avgpool_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                    vc4_buf *dx, int C, int H, int W,
                                    int outH, int outW);
/*
 * Max-pool backward with argmax routing, any N/stride/pad (ties share the
 * gradient, as vc4_maxpool2x2_backward does).  dy uses the *same padded
 * layout vc4_maxpooln produces*: align16(Wo) floats per row, align4(Ho)
 * rows per channel.  dx is [C][H][W].
 */
vc4_status vc4_maxpooln_bwd(vc4_ctx *ctx, const vc4_buf *x,
                            const vc4_buf *dy, vc4_buf *dx, int C, int H,
                            int W, int N, int S, int pad);

/* ------------------------------------------------ broadcasting binary ops */
typedef enum { VC4_BIN_ADD = 0, VC4_BIN_SUB, VC4_BIN_MUL, VC4_BIN_DIV }
    vc4_binop;
/*
 * Backward of a broadcasting binary op.  Shapes are given as dimension
 * vectors of length ndim (right aligned, like numpy): a dimension of 1 in an
 * operand is broadcast to the output's extent.  dx/dy may be NULL when that
 * input does not need a gradient.
 */
vc4_status vc4_binop_bwd(vc4_ctx *ctx, vc4_binop op, const vc4_buf *x,
                         const vc4_buf *y, const vc4_buf *g, const int *odims,
                         const int *xdims, const int *ydims, int ndim,
                         vc4_buf *dx, vc4_buf *dy);

/* ------------------------------------------------------- concat / split */
/*
 * Forward concat: out is [outer][sum(axisdim)][inner], part i is
 * [outer][axisdim[i]][inner].  vc4_cat_bwd writes each part's slice of dy;
 * vc4_split_bwd is its transpose (concatenate the gradients).
 */
vc4_status vc4_cat_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                       vc4_buf *const *dx, const int *axisdim, int nsrc,
                       int outer, int inner);
vc4_status vc4_split_bwd(vc4_ctx *ctx, const vc4_buf *const *dy,
                         vc4_buf *dx, const int *axisdim, int nsrc,
                         int outer, int inner);

/* ------------------------------------------------------------- resize/pad */
/*
 * Nearest resize with the asymmetric mapping in = floor(out * in/out_size)
 * (the ONNX "asymmetric" mode, and what vc4_upsample2x_nhwc does).
 * Tensors are [planes][H][W].
 */
vc4_status vc4_resize_nearest(vc4_ctx *ctx, const vc4_buf *x, vc4_buf *y,
                              int planes, int H, int W, int Ho, int Wo);
/* dx[q] += dy[p] for every output pixel p that maps to q (scatter-add). */
vc4_status vc4_resize_nearest_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                  vc4_buf *dx, int planes, int H, int W,
                                  int Ho, int Wo);
/*
 * Which source coordinate an output index maps to, per axis.  These are the
 * ONNX Resize coordinate_transformation_mode names, and they are what
 * torch's two interpolate settings mean:
 *
 *   HALF_PIXEL     s = (o + 0.5) * in/out - 0.5, clamped at 0
 *                  (ONNX "half_pixel"; torch align_corners=False)
 *   ALIGN_CORNERS  s = o * (in-1)/(out-1), and s = 0 when out == 1
 *                  (ONNX "align_corners"; torch align_corners=True)
 *   ASYMMETRIC     s = o * in/out                   (ONNX "asymmetric")
 *
 * The taps are i0 = floor(s) and i1 = min(i0+1, in-1), the weight of i0 is
 * s - i0, and an s at or past in-1 keeps the edge sample with weight 0 on
 * the far tap (the clamped pair still sums to x[in-1]).
 */
typedef enum {
    VC4_RESIZE_HALF_PIXEL = 0,
    VC4_RESIZE_ALIGN_CORNERS,
    VC4_RESIZE_ASYMMETRIC
} vc4_resize_map;
/*
 * Bilinear resize over [planes][H][W] with the mapping selected by `map`.
 * Lowered to four vc4_gather passes plus the QPU elementwise multiply/add:
 * the four int32 index tables and the four weight vectors are built from
 * *geometry* on the ARM (no activation value is read), and every float
 * operation on the data runs on the QPUs -- 4 gathers, 4 muls and 3 adds
 * over the output, whatever the scale.
 */
vc4_status vc4_resize_bilinear_mode(vc4_ctx *ctx, const vc4_buf *x,
                                    vc4_buf *y, int planes, int H, int W,
                                    int Ho, int Wo, vc4_resize_map map);
/* vc4_resize_bilinear_mode with VC4_RESIZE_HALF_PIXEL. */
vc4_status vc4_resize_bilinear(vc4_ctx *ctx, const vc4_buf *x, vc4_buf *y,
                               int planes, int H, int W, int Ho, int Wo);
/* The adjoint (scatter-add of the four corner weights). */
vc4_status vc4_resize_bilinear_bwd(vc4_ctx *ctx, const vc4_buf *dy,
                                   vc4_buf *dx, int planes, int H, int W,
                                   int Ho, int Wo);
/* Zero pad [N][C][H][W] -> [N][C][H+pt+pb][W+pl+pr]; bwd is the crop. */
vc4_status vc4_pad_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx, int N,
                       int C, int H, int W, int pt, int pl);

/* ---------------------------------------------------- transpose / reduce */
/* dy is [cols][rows] -> dx is [rows][cols] (the same QPU 16x16 transpose). */
vc4_status vc4_transpose_bwd(vc4_ctx *ctx, const vc4_buf *dy, vc4_buf *dx,
                             uint32_t rows, uint32_t cols);
/*
 * bwd of a reduction over one axis of a dense [dims] tensor: broadcast g
 * (which has `axis` removed) back over it.  mean divides by the extent.
 */
vc4_status vc4_reduce_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                          const int *dims, int ndim, int axis, int mean);

/* ------------------------------------------------------ gather / scatter */
/*
 * Forward: out[o] = in[idx[o]] (a table of element indices into the *first*
 * dimension, i.e. in is [n_in] contiguous).  The backward is an additive
 * scatter; a target hit more than VC4_GATHER_BWD_MAX times is reported as
 * unsupported (the regular broadcast cases have dedicated backwards above).
 */
#define VC4_GATHER_BWD_MAX 64
vc4_status vc4_gather_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                          const int32_t *idx, size_t n_out, size_t n_in);
/* Forward: out[idx[i]] = x[i].  Backward: dx[i] = g[idx[i]]. */
vc4_status vc4_scatter_bwd(vc4_ctx *ctx, const vc4_buf *g, vc4_buf *dx,
                           const int32_t *idx, size_t n, size_t n_out);

/* -------------------------------------------------------------- autotuning */
/*
 * Per-shape kernel autotuning.
 *
 * The generated GEMMs bake their QPU count into the blob's row-block loop and
 * the host chooses the packer's QPU count per launch.  vc4_autotune measures
 * the legal combinations for one op and shape, records the winner under the
 * kernel cache directory, and (once enabled) every later launch of that shape
 * uses it.  It is off unless VC4INFER_AUTOTUNE=1 or vc4_autotune() was called
 * on the context, so existing timings stay reproducible.
 */
typedef struct {
    /* matmul: m,n,k of A[M][K] * B[K][N].  conv2d: the layer geometry below;
     * m, n and k are derived from it (n = Cout, k includes the bias block). */
    uint32_t m, n, k;
    uint32_t cin, cout, h, w, kh, kw;
    uint32_t pad, stride;
    uint32_t flags;             /* VC4_TUNE_F_*                            */
} vc4_tune_shape;

#define VC4_TUNE_F_BIAS 1u      /* conv2d: bias folded in as an extra k block */

typedef struct {
    uint32_t pack_q;            /* matmul: QPUs for the A-panel packer      */
    uint32_t gemm_q;            /* QPUs for the GEMM                        */
    uint32_t im2col_q;          /* conv2d: QPUs for the im2col gather       */
} vc4_tune_params;

typedef struct {
    uint32_t max_candidates;    /* 0 = 14; hard cap on measured sets        */
    uint32_t budget_ms;         /* 0 = 30000; wall clock for the whole call */
    uint32_t reps;              /* 0 = 5; timed runs per candidate          */
    uint32_t force;             /* 1 = re-measure even if a record exists   */
} vc4_tune_opts;

typedef struct {
    vc4_tune_params params;     /* the chosen launch                        */
    double   best_ms;           /* its median time                          */
    double   default_ms;        /* the executor's default, same protocol    */
    double   max_abs;           /* |chosen - default| max (0 = bit equal)   */
    uint32_t candidates;        /* parameter sets measured (incl. default)  */
    uint32_t over_tol;          /* elements outside atol=rtol=5e-4          */
    int      from_cache;        /* 1 = read from a record, not measured     */
    char     kernel[80];        /* blob the chosen parameters select        */
} vc4_tune_result;

/*
 * Measure (or read) the best launch parameters for one op and shape.
 *
 * op is VC4_OP_MATMUL or VC4_OP_CONV2D.  shape is normalised internally the
 * same way the executor does, so the record is keyed by the shape the kernel
 * actually runs.  opts may be NULL (defaults, overridable with
 * VC4INFER_AUTOTUNE_{MAX,BUDGET_MS,REPS}); out may be NULL.
 *
 * A valid record is returned immediately (from_cache = 1) unless opts->force.
 * A new measurement is only written if the chosen parameters produce the same
 * output as the default within the recorded tolerance; otherwise the call
 * fails and nothing is cached.
 *
 * On success autotuning is enabled for the context, so subsequent ops of that
 * shape consult the record.
 */
vc4_status vc4_autotune(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *shape,
                        const vc4_tune_opts *opts, vc4_tune_result *out);
/* 1 when this context consults tuned records (set by a successful
 * vc4_autotune, or by VC4INFER_AUTOTUNE=1 in the environment). */
int vc4_autotune_enabled(vc4_ctx *ctx);

/* ---------------------------------------------------------------- registry */
typedef struct {
    const char *name;       /* "matmul", "conv2d/3x3", "sigmoid", ...        */
    const char *backend;    /* which GPU path implements it                  */
    int         supported;  /* 1 = dispatchable, 0 = VC4_ERR_UNSUPPORTED     */
    const char *note;       /* constraints, or why it is unsupported         */
    int         differentiable; /* 1 = a vc4_*_bwd entry point exists       */
} vc4_op_info;

const vc4_op_info *vc4_supported_ops(size_t *count);   /* static table        */
int  vc4_op_supported(vc4_op op);
/* 1 when the op has a backward entry point in this library.  An op can be
 * supported (forward runs on the QPU) and still not differentiable here. */
int  vc4_op_differentiable(vc4_op op);
void vc4_print_supported_ops(FILE *f);
const char *vc4_op_name(vc4_op op);

/* --------------------------------------------------------------- statistics */
typedef struct {
    uint64_t launches;        /* QPU program launches                       */
    uint64_t pack_ns;         /* time in the A-panel / im2col packer         */
    uint64_t kernel_ns;       /* time in the GEMM / elementwise kernel       */
    uint64_t gpu_cycles;      /* V3D EXECUTION_CYCLES (per QPU, summed)      */
    uint64_t flops;           /* FLOPs actually issued on the GPU            */
} vc4_stats;
void vc4_get_stats(const vc4_ctx *ctx, vc4_stats *out);
void vc4_reset_stats(vc4_ctx *ctx);

/* Last error detail string (may include V3D ERRSTAT). */
const char *vc4_last_error(const vc4_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif /* VC4INFER_H */
