/*
 * vc4_dev.c - device layer: dma-buf arena, /dev/mem mapping, V3D launch.
 *
 * This is a port of the proven qrun_gemm.c harness (asm3), with three
 * deliberate differences:
 *
 *  1. the arena signature is unique per process ("VC4INFR1" + pid + nonce),
 *     so two concurrent users can never locate and scribble over each
 *     other's arena (qrun_gemm's fixed magic can);
 *  2. kernel blobs are cached and each gets its own code and uniform area,
 *     which is the asm3 "two-launch rule" (a fresh code block at the same
 *     address can be shadowed by the stale, clean-but-not-dirty L2 line of
 *     the previous program);
 *  3. every launch invalidates the slice caches and flushes the L2.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define V3D_PHYS_BASE 0x3FC00000UL
#define V3D_LENGTH    0x4000UL

#define V3D_IDENT0         (0x0000 / 4)
#define V3D_L2CACTL        (0x0020 / 4)
#define V3D_SLCACTL        (0x0024 / 4)
#define V3D_SRQPC          (0x0430 / 4)
#define V3D_SRQUA          (0x0434 / 4)
#define V3D_SRQCS          (0x043C / 4)
#define V3D_VPMBASE        (0x0504 / 4)
#define V3D_COUNTER_CLEAR  (0x0670 / 4)
#define V3D_COUNTER_ENABLE (0x0674 / 4)
#define V3D_COUNTER_VALUE  (0x0680 / 4)
#define V3D_COUNTER_MAP    (0x0684 / 4)
#define V3D_ERRORS         (0x0F20 / 4)

#define CTR_EXEC_CYCLES 16
#define CTR_INSTR_HITS  20

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)
#define IOCTL_MBOX_PROPERTY  _IOWR(100, 0, char *)

static const char ARENA_MAGIC[8] = { 'V','C','4','I','N','F','R','1' };

void vc4__set_err(vc4_ctx *ctx, const char *fmt, ...)
{
    va_list ap;
    if (!ctx)
        return;
    va_start(ap, fmt);
    vsnprintf(ctx->err, sizeof(ctx->err), fmt, ap);
    va_end(ap);
}

const char *vc4_last_error(const vc4_ctx *ctx)
{
    return ctx ? ctx->err : "";
}

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void barrier(void)
{
    __sync_synchronize();
    __asm__ volatile("dsb sy" ::: "memory");
}

static void cache_clean(void *addr, size_t bytes)
{
    uintptr_t line = (uintptr_t)addr & ~63ul;
    for (; line < (uintptr_t)addr + bytes; line += 64)
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}

static void *map_phys(uint32_t phys, size_t len, int prot, int *fd_out)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0)
        return NULL;
    void *p = mmap(NULL, len, prot, MAP_SHARED, fd, (off_t)phys);
    if (p == MAP_FAILED) {
        close(fd);
        return NULL;
    }
    if (fd_out)
        *fd_out = fd;
    else
        close(fd);
    return p;
}

/* --------------------------------------------------------------- V3D regs */
static int srq_reset(vc4_ctx *ctx)
{
    const uint32_t RESET = (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16);
    for (int attempt = 0; attempt < 20; ++attempt) {
        ctx->v3d[V3D_SRQCS] = RESET;
        barrier();
        uint32_t s = ctx->v3d[V3D_SRQCS];
        if ((s & 0x3F) == 0 && ((s >> 16) & 0xFF) == 0 && (s & (1u << 7)) == 0)
            return 1;
        usleep(100);
    }
    return 0;
}

static void srq_state(vc4_ctx *ctx, char *buf, size_t n)
{
    uint32_t s = ctx->v3d[V3D_SRQCS];
    snprintf(buf, n, "SRQCS=0x%08x queue=%u err=%u done=%u",
             s, s & 0x3F, (s >> 7) & 1, (s >> 16) & 0xFF);
}

int vc4__health(vc4_ctx *ctx)
{
    if (!ctx)
        return 0;
    if (ctx->remote)
        return !ctx->dead;         /* the daemon owns V3D health */
    if (!ctx->v3d)
        return 0;
    return ctx->v3d[V3D_IDENT0] != 0xdeadbeefu;
}

void vc4__l2_flush(vc4_ctx *ctx)
{
    if (!ctx)
        return;
    if (ctx->remote) {
        /* Invalidate the CPU cache lines for everything the GPU may have
         * written; they were cleaned before the launch, so this only drops
         * stale copies. */
        vc4__remote_l2_flush(ctx);
        return;
    }
    if (!ctx->v3d)
        return;
    ctx->v3d[V3D_L2CACTL] = 1u << 2;   /* write back + invalidate */
    barrier();
}

void vc4__mark_dirty(vc4_ctx *ctx, uint32_t off, size_t bytes)
{
    if (ctx && ctx->remote)
        vc4__remote_mark(ctx, off, bytes);
}

/* CPU write into the arena.  Two requirements, both about the cacheable
 * client mapping:
 *   - stale lines of *ours* must not survive into the range (a later clean
 *     would write them back over what the QPU has since written), so the
 *     range is invalidated before the memcpy;
 *   - the range is then marked dirty so the launch path cleans it, which is
 *     what makes the QPU see the data at all. */
void vc4__write(vc4_ctx *ctx, uint32_t off, const void *src, size_t bytes)
{
    if (!ctx || (!src && bytes))
        return;
    if (!ctx->v3d) {
        vc4__invalidate(ctx->arena + off, bytes);
        memcpy(ctx->arena + off, src, bytes);
        vc4__mark_dirty(ctx, off, bytes);
        __sync_synchronize();
        return;
    }
    vc4__l2_flush(ctx);
    memcpy(ctx->arena + off, src, bytes);
    __sync_synchronize();
}

vc4_status vc4__wait_idle(vc4_ctx *ctx)
{
    if (!ctx)
        return VC4_ERR_INVAL;
    if (ctx->remote)
        return VC4_OK;             /* one launch in flight, WAIT is synchronous */
    if (!srq_reset(ctx))
        return VC4_ERR_GPU;
    return VC4_OK;
}

uint32_t *vc4__uni(vc4_ctx *ctx, const vc4_kernel *k, int qpu)
{
    return (uint32_t *)(ctx->arena + k->uni_off + 64 * (size_t)qpu);
}

/* --------------------------------------------------------------- launching */
vc4_status vc4__launch(vc4_ctx *ctx, const vc4_kernel *k, int nqpu,
                       uint64_t *wall_ns)
{
    char st[96];
    if (!ctx || !k)
        return VC4_ERR_INVAL;
    if (ctx->remote)
        return vc4__remote_launch(ctx, k, nqpu, wall_ns);
    if (ctx->dead) {
        vc4__set_err(ctx, "context is dead after a previous timeout; reboot the board");
        return VC4_ERR_TIMEOUT;
    }
    if (nqpu < 1 || nqpu > VC4_MAX_QPU || nqpu > (int)k->nqpu) {
        vc4__set_err(ctx, "launch: nqpu=%d out of range (blob supports %u)", nqpu, k->nqpu);
        return VC4_ERR_INVAL;
    }
    if (!vc4__health(ctx)) {
        ctx->dead = 1;
        vc4__set_err(ctx, "V3D IDENT0=0xdeadbeef (suspended)");
        return VC4_ERR_GPU;
    }

    const uint32_t code_bus = ctx->gpu_bus + k->code_off;
    barrier();

    ctx->v3d[V3D_COUNTER_ENABLE] |= (1u << 31) | (1u << 0);
    ctx->v3d[V3D_COUNTER_MAP + 0 * 2] = CTR_EXEC_CYCLES & 0x1F;
    ctx->v3d[V3D_COUNTER_VALUE + 0 * 2] = 0;
    ctx->v3d[V3D_COUNTER_ENABLE] |= (1u << 31) | (1u << 1);
    ctx->v3d[V3D_COUNTER_MAP + 1 * 2] = CTR_INSTR_HITS & 0x1F;
    ctx->v3d[V3D_COUNTER_VALUE + 1 * 2] = 0;
    ctx->v3d[V3D_COUNTER_CLEAR] = 3;
    barrier();
    ctx->v3d[V3D_VPMBASE] = 16;
    barrier();
    /* the CPU wrote the arena through an uncached mapping; flush whatever the
       GPU left dirty in L2 so the GPU reads what we wrote */
    vc4__l2_flush(ctx);
    /* always invalidate the slice caches: the instruction cache must not serve
       a previous program's code */
    ctx->v3d[V3D_SLCACTL] = 0xFFFFFFFFu;
    barrier();
    ctx->v3d[V3D_ERRORS] = 0xFFFFFFFFu;   /* write-1-to-clear */
    barrier();
    if (!srq_reset(ctx)) {
        srq_state(ctx, st, sizeof(st));
        vc4__set_err(ctx, "launch: queue reset failed (%s)", st);
        return VC4_ERR_GPU;
    }
    for (int i = 0; i < nqpu; ++i) {
        ctx->v3d[V3D_SRQUA] = ctx->gpu_bus + k->uni_off + 64u * (uint32_t)i;
        ctx->v3d[V3D_SRQPC] = code_bus;
    }
    barrier();
    uint32_t s = ctx->v3d[V3D_SRQCS];
    if (s & (1u << 7)) {
        srq_state(ctx, st, sizeof(st));
        vc4__set_err(ctx, "launch: queue error (%s)", st);
        return VC4_ERR_GPU;
    }

    uint64_t limit_ns = 20000000000ull;
    const char *e = getenv("VC4INFER_TMO_MS");
    if (e)
        limit_ns = strtoull(e, NULL, 0) * 1000000ull;

    uint64_t t0 = now_ns();
    for (;;) {
        s = ctx->v3d[V3D_SRQCS];
        if (((s >> 16) & 0xFF) == (uint32_t)nqpu && (s & 0x3F) == 0 && !(s & (1u << 7)))
            break;
        if (now_ns() - t0 > limit_ns) {
            /* A stuck QPU cannot be reset from the ARM.  Clear the queue word
               so we do not wedge the next user, and refuse further launches. */
            ctx->v3d[V3D_SRQCS] = (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16);
            ctx->dead = 1;
            srq_state(ctx, st, sizeof(st));
            vc4__set_err(ctx, "launch: TIMEOUT after %llu ms (%s); board may need a reboot",
                         (unsigned long long)(limit_ns / 1000000ull), st);
            return VC4_ERR_TIMEOUT;
        }
    }
    if (wall_ns)
        *wall_ns = now_ns() - t0;
    ctx->st.gpu_cycles += ctx->v3d[V3D_COUNTER_VALUE];
    uint32_t errs = ctx->v3d[V3D_ERRORS] & ~ctx->errstat_base;
    ctx->st.launches++;
    if (errs) {
        vc4__set_err(ctx, "launch: V3D_ERRORS=0x%08x", errs);
        if (getenv("VC4INFER_DEBUG"))
            fprintf(stderr, "[vc4] launch %s: nqpu=%d errs=0x%08x\n",
                    k->name, nqpu, errs);
        if (!getenv("VC4INFER_IGNORE_ERRSTAT"))
            return VC4_ERR_GPU;
    }
    return VC4_OK;
}

/* ------------------------------------------------------------------ arena */
static void mbox_enable_qpu(void)
{
    int fd = open("/dev/vcio", O_RDWR);
    if (fd < 0)
        return;
    unsigned msg[7] = { 0 };
    msg[0] = sizeof(msg);
    msg[2] = 0x00030012;   /* ENABLE_QPU */
    msg[3] = 1 * 4;
    msg[5] = 1;
    (void)ioctl(fd, IOCTL_MBOX_PROPERTY, msg);
    close(fd);
}

/* Allocate the dma-buf and locate its physical base with a unique signature. */
static vc4_status gpu_alloc(vc4_ctx *ctx, size_t want)
{
    size_t sizes[5];
    int ns = 0;
    sizes[ns++] = want;
    sizes[ns++] = want / 2;
    sizes[ns++] = want / 4;
    sizes[ns++] = 16u << 20;
    sizes[ns++] = 8u << 20;

    for (int si = 0; si < ns; ++si) {
        size_t len = sizes[si];
        if (len < (8u << 20))
            continue;
        int heap = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
        if (heap < 0) {
            vc4__set_err(ctx, "open /dev/dma_heap/linux,cma: %s", strerror(errno));
            return VC4_ERR_IO;
        }
        struct dma_heap_allocation_data a;
        memset(&a, 0, sizeof(a));
        a.len = len;
        a.fd_flags = O_RDWR | O_CLOEXEC;
        int rc = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &a);
        close(heap);
        if (rc != 0)
            continue;   /* try a smaller arena */

        ctx->dmabuf_fd = (int)a.fd;
        void *cached = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->dmabuf_fd, 0);
        if (cached == MAP_FAILED) {
            close(ctx->dmabuf_fd);
            ctx->dmabuf_fd = -1;
            continue;
        }
        /* unique signature: 8 byte tag + pid + nonce */
        uint32_t sig[4];
        memcpy(&sig[0], ARENA_MAGIC, 8);
        sig[2] = (uint32_t)getpid();
        sig[3] = (uint32_t)(now_ns() ^ (uintptr_t)cached);
        memset(cached, 0, 4096);
        memcpy(cached, sig, sizeof(sig));
        cache_clean(cached, 64);
        __sync_synchronize();

        int mem = open("/dev/mem", O_RDONLY | O_SYNC);
        if (mem < 0) {
            munmap(cached, len);
            close(ctx->dmabuf_fd);
            ctx->dmabuf_fd = -1;
            vc4__set_err(ctx, "open /dev/mem: %s", strerror(errno));
            return VC4_ERR_IO;
        }
        uint32_t *window = malloc(1u << 20);
        uint32_t hit = 0;
        const uint32_t limit = 0x3B400000u;
        for (uint32_t base = 0; base < limit && !hit && window; base += (1u << 20)) {
            ssize_t got = pread(mem, window, 1u << 20, (off_t)base);
            if (got <= 0)
                continue;
            for (ssize_t off = 0; off + 16 <= got; off += 4096) {
                if (memcmp((char *)window + off, sig, 16) == 0) {
                    hit = base + (uint32_t)off;
                    break;
                }
            }
        }
        free(window);
        close(mem);
        if (!hit) {
            munmap(cached, len);
            close(ctx->dmabuf_fd);
            ctx->dmabuf_fd = -1;
            vc4__set_err(ctx, "arena: dma-buf not found in physical RAM (%zu MiB)", len >> 20);
            return VC4_ERR_IO;
        }
        ctx->gpu_phys = hit;
        ctx->gpu_bus = hit | 0xC0000000u;
        munmap(cached, len);   /* never touched again: use the uncached map */

        ctx->arena = map_phys(ctx->gpu_phys, len, PROT_READ | PROT_WRITE, NULL);
        if (!ctx->arena) {
            close(ctx->dmabuf_fd);
            ctx->dmabuf_fd = -1;
            vc4__set_err(ctx, "arena: mmap /dev/mem failed: %s", strerror(errno));
            return VC4_ERR_IO;
        }
        ctx->arena_bytes = len;
        ctx->arena_top = VC4_DATA_BASE;
        ctx->scratch = ctx->arena_top;
        return VC4_OK;
    }
    vc4__set_err(ctx, "arena: no dma-buf of %zu MiB could be allocated (CMA free?)", want >> 20);
    return VC4_ERR_NOMEM;
}

static size_t env_arena(void)
{
    const char *e = getenv("VC4INFER_ARENA_MB");
    if (!e)
        return VC4_ARENA_DEFAULT;
    unsigned long mb = strtoul(e, NULL, 0);
    if (mb < 8)
        mb = 8;
    return (size_t)mb << 20;
}

/* VC4INFER_DAEMON=1 selects the vc4d backend; unset or "0" keeps the direct
 * root path bit-for-bit. */
static int daemon_mode(void)
{
    const char *e = getenv("VC4INFER_DAEMON");
    return e && *e && strcmp(e, "0") != 0;
}

vc4_status vc4_init(vc4_ctx **out, const char *cache_dir, size_t arena_bytes)
{
    if (!out)
        return VC4_ERR_INVAL;
    *out = NULL;
    if (daemon_mode())
        return vc4__remote_init(out, cache_dir, arena_bytes);
    vc4_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return VC4_ERR_NOMEM;
    ctx->mem_fd = -1;
    ctx->dmabuf_fd = -1;

    if (arena_bytes == 0)
        arena_bytes = env_arena();

    vc4_status st = gpu_alloc(ctx, arena_bytes);
    if (st != VC4_OK) {
        free(ctx);
        return st;
    }

    ctx->v3d = map_phys(V3D_PHYS_BASE, V3D_LENGTH, PROT_READ | PROT_WRITE, &ctx->mem_fd);
    if (!ctx->v3d) {
        vc4__set_err(ctx, "map V3D registers: %s (running as root?)", strerror(errno));
        munmap(ctx->arena, ctx->arena_bytes);
        close(ctx->dmabuf_fd);
        free(ctx);
        return VC4_ERR_IO;
    }
    mbox_enable_qpu();

    if (cache_dir)
        snprintf(ctx->cache_dir, sizeof(ctx->cache_dir), "%s", cache_dir);
    else
        vc4__default_cache_dir(ctx->cache_dir, sizeof(ctx->cache_dir));
    ctx->allow_jit = getenv("VC4INFER_NO_JIT") ? 0 : 1;

    /* some ERRSTAT bits read as set on an idle GPU right after boot; they are
       status, not errors, so only newly raised bits count as a failure */
    ctx->errstat_base = ctx->v3d[V3D_ERRORS];

    if (!vc4__health(ctx)) {
        vc4__set_err(ctx, "V3D is suspended (IDENT0=0xdeadbeef)");
        vc4_shutdown(ctx);
        return VC4_ERR_GPU;
    }
    *out = ctx;
    return VC4_OK;
}

void vc4_shutdown(vc4_ctx *ctx)
{
    if (!ctx)
        return;
    if (ctx->remote) {
        vc4__remote_shutdown(ctx);
        return;
    }
    if (ctx->arena && ctx->arena_bytes)
        munmap(ctx->arena, ctx->arena_bytes);
    if (ctx->v3d)
        munmap((void *)ctx->v3d, V3D_LENGTH);
    if (ctx->mem_fd >= 0)
        close(ctx->mem_fd);
    if (ctx->dmabuf_fd >= 0)
        close(ctx->dmabuf_fd);
    free(ctx);
}

size_t vc4_arena_bytes(const vc4_ctx *ctx) { return ctx ? ctx->arena_bytes : 0; }
size_t vc4_arena_used(const vc4_ctx *ctx)  { return ctx ? ctx->arena_top : 0; }

vc4_status vc4_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out)
{
    if (!ctx || !out || bytes == 0)
        return VC4_ERR_INVAL;
    size_t off = (ctx->arena_top + 63u) & ~(size_t)63u;
    if (off + bytes > ctx->arena_bytes) {
        vc4__set_err(ctx, "arena exhausted: need %zu B, have %zu B",
                     bytes, ctx->arena_bytes - off);
        return VC4_ERR_NOMEM;
    }
    out->off = (uint32_t)off;
    out->bytes = bytes;
    ctx->arena_top = off + bytes;
    if (ctx->scratch < ctx->arena_top)
        ctx->scratch = ctx->arena_top;
    if (ctx->dirty_top < ctx->arena_top)
        ctx->dirty_top = ctx->arena_top;
    return VC4_OK;
}

void *vc4__scratch_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out)
{
    if (!ctx || !bytes)
        return NULL;
    size_t off = (ctx->scratch + 63u) & ~(size_t)63u;
    if (off + bytes > ctx->arena_bytes)
        return NULL;
    if (out) {
        out->off = (uint32_t)off;
        out->bytes = bytes;
    }
    ctx->scratch = off + bytes;
    if (ctx->dirty_top < ctx->scratch)
        ctx->dirty_top = ctx->scratch;
    return ctx->arena + off;
}

void vc4__scratch_reset(vc4_ctx *ctx)
{
    if (ctx) {
        ctx->scratch = ctx->arena_top;
        if (ctx->remote)
            vc4__remote_scratch_reset(ctx);
    }
}

/* ------------------------------------------------------------- workspace */
size_t vc4__ws_begin(vc4_ctx *ctx)
{
    if (!ctx)
        return 0;
    if (!ctx->ws_top)
        ctx->ws_top = ctx->arena_bytes;
    return ctx->ws_top;
}

void vc4__ws_end(vc4_ctx *ctx, size_t top)
{
    if (ctx)
        ctx->ws_top = top;
}

size_t vc4__ws_free(const vc4_ctx *ctx)
{
    size_t floor = ctx->arena_top > ctx->scratch ? ctx->arena_top
                                                 : ctx->scratch;
    size_t top = ctx->ws_top ? ctx->ws_top : ctx->arena_bytes;
    return top > floor ? top - floor : 0;
}

void *vc4__ws_alloc(vc4_ctx *ctx, size_t bytes, vc4_buf *out)
{
    if (!ctx || !bytes)
        return NULL;
    if (!ctx->ws_top)
        ctx->ws_top = ctx->arena_bytes;
    size_t floor = ctx->arena_top > ctx->scratch ? ctx->arena_top
                                                 : ctx->scratch;
    size_t need = (bytes + 63u) & ~(size_t)63u;
    if (ctx->ws_top < floor + need) {
        vc4__set_err(ctx, "workspace exhausted: need %zu B, have %zu B "
                     "(arena %zu B, %zu B in use)", need,
                     ctx->ws_top > floor ? ctx->ws_top - floor : 0,
                     ctx->arena_bytes, ctx->arena_top);
        return NULL;
    }
    size_t off = (ctx->ws_top - need) & ~(size_t)63u;
    ctx->ws_top = off;
    if (out) {
        out->off = (uint32_t)off;
        out->bytes = bytes;
    }
    return ctx->arena + off;
}

vc4_status vc4_upload(vc4_ctx *ctx, vc4_buf *b, const void *src, size_t bytes)
{
    if (!ctx || !b || (!src && bytes))
        return VC4_ERR_INVAL;
    if (bytes > b->bytes)
        return VC4_ERR_INVAL;
    if (ctx->remote)
        return vc4__remote_upload(ctx, b, src, bytes);
    if (ctx->v3d)
        vc4__write(ctx, b->off, src, bytes);
    else
        memcpy(ctx->arena + b->off, src, bytes);
    return VC4_OK;
}

vc4_status vc4_download(vc4_ctx *ctx, const vc4_buf *b, void *dst, size_t bytes)
{
    if (!ctx || !b || (!dst && bytes))
        return VC4_ERR_INVAL;
    if (bytes > b->bytes)
        return VC4_ERR_INVAL;
    if (ctx->remote)
        return vc4__remote_download(ctx, b, dst, bytes);
    if (ctx->v3d)
        vc4__l2_flush(ctx);       /* write the GPU result back to DRAM */
    barrier();
    memcpy(dst, ctx->arena + b->off, bytes);
    return VC4_OK;
}

void *vc4_host_ptr(vc4_ctx *ctx, const vc4_buf *b)
{
    if (!ctx || !b)
        return NULL;
    return ctx->arena + b->off;
}

uint32_t vc4_bus_addr(const vc4_ctx *ctx, const vc4_buf *b)
{
    if (!ctx || !b)
        return 0;
    return ctx->gpu_bus + b->off;
}

void vc4_get_stats(const vc4_ctx *ctx, vc4_stats *out)
{
    if (ctx && out)
        *out = ctx->st;
}

void vc4_reset_stats(vc4_ctx *ctx)
{
    if (ctx)
        memset(&ctx->st, 0, sizeof(ctx->st));
}

const char *vc4_strerror(vc4_status st)
{
    switch (st) {
    case VC4_OK:              return "ok";
    case VC4_ERR_INVAL:       return "invalid argument";
    case VC4_ERR_NOMEM:       return "out of arena memory";
    case VC4_ERR_IO:          return "device I/O error";
    case VC4_ERR_TIMEOUT:     return "GPU timeout (context is dead)";
    case VC4_ERR_UNSUPPORTED: return "unsupported by the GPU backend (strict dispatch)";
    case VC4_ERR_GPU:         return "GPU error status";
    case VC4_ERR_KERNEL:      return "kernel blob missing or generation failed";
    default:                  return "unknown";
    }
}
