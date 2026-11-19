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
#define VC4INFER_VERSION_STRING "0.1.0"

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
void      *vc4_host_ptr(vc4_ctx *ctx, const vc4_buf *b);   /* uncached mapping */
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
 * junk (never read back).  stride in {1,2}, KH=KW in {1,3}, Cin/Cout % 16 == 0,
 * stride*Cin*4 <= 8191 (VDR pitch).
 */
vc4_status vc4_conv_nhwc(vc4_ctx *ctx, const vc4_buf *in, int nimg, int Cin,
                         int H, int W, int in_pitch, const vc4_buf *wpanel,
                         int Cout, int KH, int KW, int stride, int pad,
                         vc4_buf *out, int *ho, int *wo, int *opitch);
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

/* ---------------------------------------------------------------- registry */
typedef struct {
    const char *name;       /* "matmul", "conv2d/3x3", "sigmoid", ...        */
    const char *backend;    /* which GPU path implements it                  */
    int         supported;  /* 1 = dispatchable, 0 = VC4_ERR_UNSUPPORTED     */
    const char *note;       /* constraints, or why it is unsupported         */
} vc4_op_info;

const vc4_op_info *vc4_supported_ops(size_t *count);   /* static table        */
int  vc4_op_supported(vc4_op op);
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
