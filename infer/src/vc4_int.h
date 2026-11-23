/* internal definitions shared by the libvc4infer sources */
#ifndef VC4_INT_H
#define VC4_INT_H

#include "vc4infer.h"
#include <sys/types.h>

#define VC4_MAX_KERNELS   128
/* Names of every blob this context has loaded, in load order.  The kernel
 * table above is an LRU of 128 live slots (the code area holds no more); an
 * engine artifact needs the full set, which is larger for a detector. */
#define VC4_MAX_SEEN      512
#define VC4_MAX_QPU       12
#define VC4_TUNE_MEMO     32

/* arena layout (one dma-buf):
 *   0x000000  arena signature (unique per process, used for the phys scan)
 *   0x001000  kernel code slots   32 x 16 KiB
 *   0x090000  per-kernel uniform areas, 32 x 4 KiB (12 QPUs x 64 B each)
 *   0x100000  data heap (persistent buffers + per-op scratch)
 */
#define VC4_CODE_BASE     0x001000u
#define VC4_CODE_SLOT     0x4000u
#define VC4_UNI_BASE      0x210000u
#define VC4_UNI_SLOT      0x1000u
#define VC4_DATA_BASE     0x300000u

#define VC4_UNI_WORDS     16

typedef struct {
    char     name[80];
    uint32_t code_off;
    uint32_t uni_off;
    uint32_t code_words;
    uint32_t nqpu;
    int      used;
    /* G3: the daemon evicts program slots LRU-wise, so a launch can come back
     * E_NOENT for a kernel the client still knows.  The remote backend keeps
     * the code words to re-upload it without a JIT round trip; `used_at` is
     * this cache's own LRU stamp, and `slot` is the index in ctx->k[] (the
     * remote program table is parallel to it). */
    uint8_t *code;
    uint32_t code_len;
    uint64_t used_at;
} vc4_kernel;

/* One entry of the autotune lookup memo (vc4_autotune.c): a graph runs the
 * same shape repeatedly, and a record that is parsed once per launch would be
 * pure overhead. */
typedef struct {
    uint64_t        key;
    vc4_tune_params p;
    int             valid;
} vc4_tune_memo;

struct vc4_ctx {
    int      mem_fd;
    int      dmabuf_fd;
    int      lock_fd;        /* board-wide GPU lock, direct mode only */
    volatile uint32_t *v3d;
    uint8_t *arena;
    uint32_t gpu_phys;
    uint32_t gpu_bus;
    size_t   arena_bytes;
    size_t   arena_top;      /* persistent bump cursor                 */
    size_t   scratch;        /* scratch bump cursor, reset per op      */
    /* Remote depth-2 pipeline: the two scratch windows are carved from the
     * free arena when the pipeline is armed, and every op's scratch lives in
     * the window of the request it belongs to.  pipe_win == 0 means "not
     * armed", and then scratch_base/scratch_top are 0 and scratch starts at
     * arena_top exactly as it always did. */
    size_t   scratch_base;   /* base of the current window (0 = arena_top) */
    size_t   scratch_top;    /* end of the current window (0 = arena end)  */
    size_t   pipe_base;      /* arena_top when the windows were sized      */
    size_t   pipe_win;       /* bytes in each window                       */
    size_t   pipe_top;       /* end of window 1: no persistent alloc above */
    size_t   ws_top;         /* workspace cursor, carved *down* from the
                              * end of the arena by a bwd composition    */
    size_t   dirty_top;      /* remote: high-water of CPU-written bytes */
    int      dead;
    void    *remote;         /* struct vc4_remote * when VC4INFER_DAEMON=1 */
    uint32_t errstat_base;   /* bits that read as set even on an idle GPU */
    char     err[256];

    vc4_kernel k[VC4_MAX_KERNELS];
    int      nkern;
    char     seen[VC4_MAX_SEEN][80];   /* every blob loaded, in order */
    int      nseen;
    uint64_t kern_clock;     /* LRU stamp for the kernel cache (G3) */
    char     cache_dir[256];
    int      allow_jit;

    /* autotuning: 1 after an explicit vc4_autotune() (the env switch is read
     * on demand), plus the memo of the last lookups */
    int            autotune;
    vc4_tune_memo  tune_memo[VC4_TUNE_MEMO];
    char           tune_digest[65];   /* generator digest, computed once */

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
/* Drop the client cache lines for a range before the CPU writes it: the arena
 * mapping is cacheable, so a surviving clean line would be written back by
 * the pre-launch clean and would overlay what the QPU put there. */
void       vc4__invalidate_off(vc4_ctx *ctx, uint32_t off, size_t bytes);
void       vc4__invalidate(void *p, size_t bytes);
void      *vc4__scratch_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out);
void       vc4__scratch_reset(vc4_ctx *ctx);

/* ---- vc4_dev_remote.c (VC4INFER_DAEMON=1) -------------------------------- */
vc4_status vc4__remote_init(vc4_ctx **out, const char *cache_dir,
                            size_t arena_bytes);
void       vc4__remote_shutdown(vc4_ctx *ctx);
vc4_status vc4__remote_load(vc4_ctx *ctx, const char *name,
                            const uint8_t *words, size_t bytes,
                            int nqpu, vc4_kernel **out);
vc4_status vc4__remote_launch(vc4_ctx *ctx, const vc4_kernel *k, int nqpu,
                              uint64_t *wall_ns);
void       vc4__remote_batch_begin(vc4_ctx *ctx);
vc4_status vc4__remote_batch_end(vc4_ctx *ctx);
/* Depth-2 request pipelining over protocol 4's SUBMIT.  begin() reports
 * VC4_ERR_UNSUPPORTED when the daemon does not advertise it, so a caller can
 * fall back to the batch path without guessing. */
vc4_status vc4__remote_pipeline_begin(vc4_ctx *ctx);
vc4_status vc4__remote_pipeline_flush(vc4_ctx *ctx);
vc4_status vc4__remote_pipeline_end(vc4_ctx *ctx);
uint64_t   vc4__remote_pipeline_epoch(vc4_ctx *ctx);
uint64_t   vc4__remote_pipeline_reaped(vc4_ctx *ctx);
vc4_status vc4__remote_upload(vc4_ctx *ctx, vc4_buf *b, const void *src,
                              size_t bytes);
vc4_status vc4__remote_download(vc4_ctx *ctx, const vc4_buf *b, void *dst,
                                size_t bytes);
void       vc4__remote_l2_flush(vc4_ctx *ctx);
/* optional launch-path profiler (vc4_dev_remote.c) */
const void *vc4_prof_ptr(void);
void       vc4_prof_reset(void);
void       vc4_prof_report(void);
int        vc4_prof_enabled(void);
uint64_t   vc4_prof_now(void);
void       vc4_prof_add(int which, uint64_t ns);
void       vc4_prof_bytes(int which, uint64_t n);
void       vc4__remote_mark(vc4_ctx *ctx, uint32_t off, size_t bytes);
void       vc4__remote_scratch_reset(vc4_ctx *ctx);
void       vc4__mark_dirty(vc4_ctx *ctx, uint32_t off, size_t bytes);
/* resolves $VC4INFER_CACHE, else $XDG_CACHE_HOME|$HOME/.cache, and mkdir -p's it */
void       vc4__default_cache_dir(char *out, size_t n);

/* ---- vc4_kern.c --------------------------------------------------------- */
/* gen_argv[0] is the kernel kind, the rest are the kgen.py flags. */
vc4_status vc4__kernel(vc4_ctx *ctx, const char *name, char *const gen_argv[],
                       int nqpu, vc4_kernel **out);
vc4_status vc4__kernel_load_file(vc4_ctx *ctx, const char *name,
                                 const uint8_t *words, size_t bytes,
                                 int nqpu, vc4_kernel **out);
/* The cache entry a newly uploaded kernel goes in: a free one, or - when all
 * VC4_MAX_KERNELS are live - the least recently used one, whose code/uniform
 * area is then reused.  Shared with the remote backend so both paths have the
 * same behaviour when a session touches more kernels than the cache holds. */
vc4_kernel *vc4__kern_slot(vc4_ctx *ctx);

/* ---- vc4_gemm.c --------------------------------------------------------- */
/* one work unit = 8 rows x rowlen floats; grid-strided over 4 QPUs */
vc4_status vc4__pack_run(vc4_ctx *ctx, uint32_t src_bus, uint32_t rowlen_floats,
                         uint32_t dst_bus, uint32_t rowblocks);
/* same, with the packer's QPU count explicit (0 = the default: min(units,4)).
 * The packer has no in-kernel guard, so q must divide rowblocks. */
vc4_status vc4__pack_run_q(vc4_ctx *ctx, uint32_t src_bus,
                           uint32_t rowlen_floats, uint32_t dst_bus,
                           uint32_t rowblocks, int q);
/* 8x64 tiles: M 8-aligned, N 64-aligned, K 16-aligned, B row-major */
vc4_status vc4__gemm_run(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t b_bus, uint32_t c_bus, uint32_t panel_bus);
/* same, with the GEMM's QPU count explicit (0 = min(M/8, 12)) and the C
 * row-block stride in bytes (0 = the row-major 8*N*4; the compact layout of
 * a wide N uses N*32). */
vc4_status vc4__gemm_run_q(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t b_bus, uint32_t c_bus, uint32_t panel_bus,
                           int gemm_q, int act, uint32_t c_rb);
/* 1 when a row-major result of this padded width cannot be stored by the
 * asm10 8x64 store and the compact tile layout is used instead. */
int vc4__matmul_compact(uint32_t N);
/* The blob names the generator sees, shared with the autotuner so a record
 * names the kernel it selects instead of recomputing the scheme. */
void vc4__matmul_kernel_name(char *out, size_t n, uint32_t M, uint32_t N,
                             uint32_t K, int nq, int act);
void vc4__conv_kernel_name(char *out, size_t n, uint32_t M, uint32_t N,
                           uint32_t K, int nq, int act);

/* The matmul with an explicit launch choice (NULL = the executor's default).
 * vc4_matmul is this plus the tuned-record lookup. */
vc4_status vc4__matmul_tuned(vc4_ctx *ctx, const vc4_buf *A, const vc4_buf *B,
                             vc4_buf *C, uint32_t M, uint32_t N, uint32_t K,
                             vc4_layout layout, const vc4_tune_params *p,
                             int act);
/* Same for conv2d; vc4_conv2d is this plus the lookup. */
vc4_status vc4__conv2d_tuned(vc4_ctx *ctx,
                             const vc4_buf *in, int Cin, int H, int W,
                             const vc4_buf *wt, int Cout, int KH, int KW,
                             const vc4_buf *bias, int stride, int pad,
                             vc4_buf *out, int *HO, int *WO,
                             const vc4_tune_params *p);

/* ---- vc4_ew.c ----------------------------------------------------------- */
vc4_status vc4__ew_run(vc4_ctx *ctx, vc4_op op, uint64_t n,
                       uint32_t a_bus, uint32_t b_bus, uint32_t o_bus);

/* ---- vc4_guard.c: the unverified-kernel gate ---------------------------- */
/*
 * Kernels that no test exercises and that are confirmed to break the board
 * (gather / reduce / blit / the nvec=4 store path) refuse to launch unless
 * VC4_ALLOW_UNVERIFIED_KERNELS=1.  A refusal is VC4_ERR_UNSUPPORTED with the
 * defect named in the error string.
 */
int        vc4__unverified_allowed(void);
vc4_status vc4__refuse_unverified(vc4_ctx *ctx, const char *what,
                                  const char *why);

/* ---- vc4_ew.c: generic binary/backward elementwise ---------------------- */
/* A kernel from kops' EWB family: out = f(a, b) with p0/p1 as attributes.
 * `heavy` selects the 4-vector group the long bodies are generated with. */
vc4_status vc4__ewb(vc4_ctx *ctx, const char *opname, int heavy,
                    const vc4_buf *a, const vc4_buf *b, vc4_buf *out,
                    size_t n, float p0, float p1);

/* ---- workspace: temporaries that outlive nested per-op scratch resets ---- */
/*
 * Every op resets the scratch cursor (which bumps *up* from arena_top) before
 * and after it runs, so a backward composition cannot keep its intermediates
 * there.  The workspace carves *down* from the end of the arena instead:
 * nothing else allocates from that direction, and ws_end restores the cursor.
 * Both nest.
 */
size_t vc4__ws_begin(vc4_ctx *ctx);
void   vc4__ws_end(vc4_ctx *ctx, size_t top);
/* Carve `bytes` from the workspace; NULL (VC4_ERR_NOMEM) when it is full. */
void  *vc4__ws_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out);
/* How much workspace is left. */
size_t vc4__ws_free(const vc4_ctx *ctx);


/* A kernel from kops' EWB family: out = f(a, b) with p0/p1 as attributes.
 * `heavy` selects the 4-vector group the long bodies are generated with. */
vc4_status vc4__ewb(vc4_ctx *ctx, const char *opname, int heavy,
                    const vc4_buf *a, const vc4_buf *b, vc4_buf *out,
                    size_t n, float p0, float p1);

/* ---- workspace: temporaries that outlive nested per-op scratch resets ---- */
/*
 * Every op resets the scratch cursor before and after it runs, so a backward
 * composition cannot keep its intermediates in scratch naively.  ws_begin
 * raises the scratch *base* above whatever is live and returns the previous
 * base; every nested reset then rewinds to the raised base instead of 0.
 * ws_end restores.  Both are cheap and nest.
 */
size_t vc4__ws_begin(vc4_ctx *ctx);
void   vc4__ws_end(vc4_ctx *ctx, size_t base);

/* ---- vc4_autotune.c ----------------------------------------------------- */
/* A measured parameter set and its median time: one cache record. */
typedef struct {
    vc4_tune_params p;
    double          ms;
} vc4_tune_trial;

/* The canonical shape descriptors, used by the executor and by the tuner so
 * both key the same layer the same way. */
void vc4__tune_shape_mm(vc4_tune_shape *s, uint32_t M, uint32_t N, uint32_t K);
void vc4__tune_shape_conv(vc4_tune_shape *s, uint32_t Cin, uint32_t Cout,
                          uint32_t H, uint32_t W, uint32_t KH, uint32_t KW,
                          uint32_t pad, uint32_t stride, int bias);
/* 1 when the parameter set is a launch the kernels can run for this shape:
 * QPU counts within the 12-QPU ceiling and the row-block count, the auxiliary
 * kernel's work divisibly partitioned, and the unused op field zero. */
int  vc4__tune_params_legal(vc4_op op, const vc4_tune_shape *s,
                            const vc4_tune_params *p);
int  vc4__tune_params_same(const vc4_tune_params *a, const vc4_tune_params *b);
/* The coarse candidate list in measurement order (default first). */
int  vc4__tune_space(vc4_op op, const vc4_tune_shape *s, vc4_tune_params *out,
                     int max);
/* The fine pass around a coarse winner, deduplicated against `seen`. */
int  vc4__tune_refine(vc4_op op, const vc4_tune_shape *s,
                      const vc4_tune_params *best, const vc4_tune_params *seen,
                      int nseen, vc4_tune_params *out, int max);
/* <cache>/autotune/<16 hex>.tune plus the 64-bit key the memo uses. */
void vc4__tune_key(const char *cache_dir, vc4_op op, const vc4_tune_shape *s,
                   const char *digest, uint64_t *key, char *path, size_t pn);
/* 0 = a valid record for exactly this op, shape and generator; -1 = miss
 * (absent, stale, truncated, foreign or an illegal parameter set). */
int  vc4__tune_cache_read(const char *path, vc4_op op, const vc4_tune_shape *s,
                          const char *digest, vc4_tune_params *best,
                          char *kernel, size_t kn, double *best_ms,
                          double *default_ms);
vc4_status vc4__tune_cache_write(const char *path, vc4_op op,
                                 const vc4_tune_shape *s, const char *digest,
                                 const vc4_tune_trial *trials, int n,
                                 const vc4_tune_params *best,
                                 const char *kernel, double best_ms,
                                 double default_ms);
/* Consult the record (memo first).  1 = hit, 0 = miss or autotuning off. */
int  vc4__tune_lookup(vc4_ctx *ctx, vc4_op op, const vc4_tune_shape *s,
                      vc4_tune_params *out);

/* ---- vc4_kern.c: digests ------------------------------------------------ */
/* sha256 of the generator (kgen.py, kops.py beside it, the ABI tag), the same
 * digest the blob cache stamps into every .bin, as 64 lowercase hex chars. */
void vc4__generator_digest(char out[65]);
void vc4__sha256_hex(const void *data, size_t len, char out[65]);

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
/* same, with C's row pitch (floats) independent of N; a grouped conv passes
 * the full output channel count so every group writes into one image */
vc4_status vc4__gemm3_run_c(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                            int bmode, uint32_t b_bus, uint32_t c_bus,
                            uint32_t panel_bus, uint32_t cstride, int act);
/* same, with the QPU count explicit (0 = min(M/16, 12)) */
vc4_status vc4__gemm3_run_cq(vc4_ctx *ctx, uint32_t M, uint32_t N, uint32_t K,
                             int bmode, uint32_t b_bus, uint32_t c_bus,
                             uint32_t panel_bus, uint32_t cstride, int gemm_q,
                             int act);
