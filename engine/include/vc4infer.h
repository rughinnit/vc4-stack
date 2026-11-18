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
    VC4_OP_CONV2D,          /* 3x3 and 1x1, stride 1/2, zero padding      */
    VC4_OP_ADD,
    VC4_OP_MUL,
    VC4_OP_RELU,
    VC4_OP_SIGMOID,
    VC4_OP_SILU,
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
 * KH=KW in {1,3}, stride in {1,2}, any pad >= 0, Cin % 16 == 0 for 3x3.
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
