/* internal definitions shared by the libvc4infer sources */
#ifndef VC4_INT_H
#define VC4_INT_H

#include "vc4infer.h"
#include <sys/types.h>

#define VC4_MAX_KERNELS   32
#define VC4_MAX_QPU       12

/* arena layout (one dma-buf):
 *   0x000000  arena signature (unique per process, used for the phys scan)
 *   0x001000  kernel code slots   32 x 16 KiB
 *   0x090000  per-kernel uniform areas, 32 x 4 KiB (12 QPUs x 64 B each)
 *   0x100000  data heap (persistent buffers + per-op scratch)
 */
#define VC4_CODE_BASE     0x001000u
#define VC4_CODE_SLOT     0x4000u
#define VC4_UNI_BASE      0x090000u
#define VC4_UNI_SLOT      0x1000u
#define VC4_DATA_BASE     0x100000u

#define VC4_UNI_WORDS     16

typedef struct {
    char     name[80];
    uint32_t code_off;
    uint32_t uni_off;
    uint32_t code_words;
    uint32_t nqpu;
    int      used;
} vc4_kernel;

struct vc4_ctx {
    int      mem_fd;
    int      dmabuf_fd;
    volatile uint32_t *v3d;
    uint8_t *arena;
    uint32_t gpu_phys;
    uint32_t gpu_bus;
    size_t   arena_bytes;
    size_t   arena_top;      /* persistent bump cursor                 */
    size_t   scratch;        /* scratch bump cursor, reset per op      */
    int      dead;
    uint32_t errstat_base;   /* bits that read as set even on an idle GPU */
    char     err[256];

    vc4_kernel k[VC4_MAX_KERNELS];
    int      nkern;
    char     cache_dir[256];
    int      allow_jit;

    vc4_stats st;
};

/* ---- vc4_dev.c ---------------------------------------------------------- */
vc4_status vc4__launch(vc4_ctx *ctx, const vc4_kernel *k, int nqpu,
                       uint64_t *wall_ns);
void       vc4__set_err(vc4_ctx *ctx, const char *fmt, ...);
uint32_t  *vc4__uni(vc4_ctx *ctx, const vc4_kernel *k, int qpu);
int        vc4__health(vc4_ctx *ctx);
void       vc4__l2_flush(vc4_ctx *ctx);
void       vc4__write(vc4_ctx *ctx, uint32_t off, const void *src, size_t bytes);
void      *vc4__scratch_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out);
void       vc4__scratch_reset(vc4_ctx *ctx);

/* ---- vc4_kern.c --------------------------------------------------------- */
/* gen_argv[0] is the kernel kind, the rest are the kgen.py flags. */
vc4_status vc4__kernel(vc4_ctx *ctx, const char *name, char *const gen_argv[],
                       int nqpu, vc4_kernel **out);
vc4_status vc4__kernel_load_file(vc4_ctx *ctx, const char *name,
                                 const uint8_t *words, size_t bytes,
                                 int nqpu, vc4_kernel **out);

/* ---- vc4_gemm.c --------------------------------------------------------- */
/* one work unit = 8 rows x rowlen floats; grid-strided over 4 QPUs */
vc4_status vc4__pack_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen_floats,
                         uint32_t dst_bus, uint32_t rowblocks);
/* 8x64 tiles: M 8-aligned, N 64-aligned, K 16-aligned, B row-major */
vc4_status vc4__gemm_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t b_bus, uint32_t c_bus, uint32_t panel_bus);

/* ---- vc4_ew.c ----------------------------------------------------------- */
vc4_status vc4__ew_run(vc4_ctx *ctx, vc4_op op, uint64_t n,
                       uint32_t a_bus, uint32_t b_bus, uint32_t o_bus);

/* ---- helpers ------------------------------------------------------------ */
static inline uint32_t vc4__align8(uint32_t v)  { return (v + 7u) & ~7u; }
static inline uint32_t vc4__align16(uint32_t v) { return (v + 15u) & ~15u; }
static inline uint32_t vc4__align64(uint32_t v) { return (v + 63u) & ~63u; }

#endif

/* vc4_train.c: dense transpose (VDR + strided VDW) */
vc4_status vc4__tpose_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rows,
                          uint32_t cols, uint32_t dst_bus);

/* the asm3 packer/GEMM (conv path) */
vc4_status vc4__pack3_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen_floats,
                          uint32_t src_rb_stride, uint32_t dst_bus,
                          uint32_t dst_rb_stride, uint32_t rowblocks);
vc4_status vc4__gemm3_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                          int bmode, uint32_t b_bus, uint32_t c_bus,
                          uint32_t panel_bus);
