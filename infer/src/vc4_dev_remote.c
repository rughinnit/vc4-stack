/*
 * vc4_dev_remote.c - the unprivileged backend: libvc4infer over vc4d.
 *
 * Selected by VC4INFER_DAEMON=1 (any value but "0").  The direct /dev/mem
 * backend in vc4_dev.c is untouched and is still what runs when the flag is
 * unset.
 *
 * Why this is small: the op code (vc4_gemm.c, vc4_ew.c, vc4_conv.c, ...)
 * never touches a V3D register.  It writes uniforms through vc4__uni() - a
 * pointer into the mapped arena - and calls vc4__launch().  So the remote
 * backend only has to provide the same four primitives:
 *
 *   vc4__remote_init()    connect, ALLOC the arena, mmap the dma-buf fd
 *   vc4__remote_load()    UPLOAD the first time a kernel is used
 *   vc4__remote_launch()  LAUNCH + WAIT in one round trip
 *   upload/download       memcpy into/out of the write-combining mapping
 *
 * The client's arena is one ALLOC region; ctx->arena is the mmap of it and
 * ctx->gpu_bus is the region's GPU bus address (the daemon returns it in the
 * ALLOC reply).  Kernel code, uniform blocks and data all keep the engine's
 * existing offsets inside that region, so nothing else changes.
 *
 * Uniforms: the daemon's UNI_IN_ARENA mode reads the 16 words per QPU
 * straight out of this region and validates each word as "inside this arena"
 * or "a literal outside the bus/arena/MMIO windows".  That is the per-arena
 * containment the vc4d README recommends for exactly this port, and it is why
 * vc4_gemm.c / vc4_ew.c / vc4_conv.c need no changes.
 *
 * Coherency: the dma-buf mapping is write-combining, so CPU writes reach DRAM
 * without cache maintenance; the daemon's LAUNCH does the GPU-side L2 flush
 * before the kick, and it flushes L2 write-back before the LAUNCH_WAIT reply,
 * so a CPU read of the mapping after a launch sees the result.
 *
 * No /dev/mem, no capabilities, no ioctl: only connect/sendmsg/recvmsg/mmap on
 * a dma-buf fd the daemon passed over the socket.
 */
#define _GNU_SOURCE
#include "vc4_int.h"
#include "vc4client.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* ---- optional launch-path profiler (VC4INFER_PROF=1) ------------------
 * CPU time in each phase of the remote backend, accumulated and exposed two
 * ways: as a JSON line at shutdown, and through vc4_prof_ptr() for
 * tools/cost/breakdown.py.  Off unless the variable is set; every timestamp
 * is a predictable branch when it is not. */
struct vc4_prof {
    int      on;
    uint64_t pre, mark, clean, mask, rt, tail;
    uint64_t up_clean, up_copy, down_clean, down_copy, load, flush;
    uint64_t pad_zero, pad_copy, civac_off, panel_zero;
    uint64_t pad_zero_bytes, pad_copy_bytes, civac_off_bytes;
    uint64_t n_launch, n_upload, n_download, n_load, n_flush;
    uint64_t n_batch, n_batch_launch;
    uint64_t clean_bytes, up_bytes, down_bytes;
};
static struct vc4_prof g_prof;

static void prof_init(void)
{
    const char *e = getenv("VC4INFER_PROF");
    g_prof.on = e && *e && *e != '0';
}
static inline int prof_on(void)
{
    return __builtin_expect(g_prof.on, 0);
}
static inline uint64_t cpu_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline uint64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
/* PROF_T: thread CPU (a phase that may block); PROF_W: monotonic (userspace
 * only).  The latter is vDSO and costs ~20 ns against ~1.5 us for the former. */
#define PROF_T(x) uint64_t x = prof_on() ? cpu_ns() : 0
#define PROF_W(x) uint64_t x = prof_on() ? mono_ns() : 0
#define PROF_ADD(f, a, b) do { if (prof_on()) g_prof.f += (b) - (a); } while (0)

const void *vc4_prof_ptr(void)
{
    return &g_prof;
}

int vc4_prof_enabled(void)
{
    return g_prof.on;
}

uint64_t vc4_prof_now(void)
{
    return mono_ns();
}

/* 0 pad zero, 1 pad copy, 2 invalidate_off, 3 panel zero */
void vc4_prof_add(int which, uint64_t ns)
{
    if (!g_prof.on)
        return;
    switch (which) {
    case 0: g_prof.pad_zero += ns; break;
    case 1: g_prof.pad_copy += ns; break;
    case 2: g_prof.civac_off += ns; break;
    case 3: g_prof.panel_zero += ns; break;
    }
}

/* which: 0 pad zero, 1 pad copy, 2 invalidate_off */
void vc4_prof_bytes(int which, uint64_t n)
{
    if (!g_prof.on)
        return;
    if (which == 0) g_prof.pad_zero_bytes += n;
    else if (which == 1) g_prof.pad_copy_bytes += n;
    else if (which == 2) g_prof.civac_off_bytes += n;
}

void vc4_prof_reset(void)
{
    int on = g_prof.on;
    memset(&g_prof, 0, sizeof(g_prof));
    g_prof.on = on;
}

void vc4_prof_report(void)
{
    if (!g_prof.n_launch && !g_prof.n_upload && !g_prof.n_download)
        return;
    fprintf(stderr,
            "VC4PROF {\"pre_ns\":%llu,\"mark_ns\":%llu,\"clean_ns\":%llu,"
            "\"mask_ns\":%llu,\"rt_ns\":%llu,\"tail_ns\":%llu,"
            "\"up_clean_ns\":%llu,\"up_copy_ns\":%llu,"
            "\"down_clean_ns\":%llu,\"down_copy_ns\":%llu,"
            "\"load_ns\":%llu,\"flush_ns\":%llu,"
            "\"pad_zero_ns\":%llu,\"pad_copy_ns\":%llu,"
            "\"civac_off_ns\":%llu,\"panel_zero_ns\":%llu,"
            "\"pad_zero_bytes\":%llu,\"pad_copy_bytes\":%llu,"
            "\"civac_off_bytes\":%llu,"
            "\"n_launch\":%llu,\"n_upload\":%llu,\"n_download\":%llu,"
            "\"n_load\":%llu,\"n_flush\":%llu,"
            "\"n_batch\":%llu,\"n_batch_launch\":%llu,"
            "\"clean_bytes\":%llu,\"up_bytes\":%llu,\"down_bytes\":%llu}\n",
            (unsigned long long)g_prof.pre, (unsigned long long)g_prof.mark,
            (unsigned long long)g_prof.clean, (unsigned long long)g_prof.mask,
            (unsigned long long)g_prof.rt, (unsigned long long)g_prof.tail,
            (unsigned long long)g_prof.up_clean,
            (unsigned long long)g_prof.up_copy,
            (unsigned long long)g_prof.down_clean,
            (unsigned long long)g_prof.down_copy,
            (unsigned long long)g_prof.load, (unsigned long long)g_prof.flush,
            (unsigned long long)g_prof.pad_zero,
            (unsigned long long)g_prof.pad_copy,
            (unsigned long long)g_prof.civac_off,
            (unsigned long long)g_prof.panel_zero,
            (unsigned long long)g_prof.pad_zero_bytes,
            (unsigned long long)g_prof.pad_copy_bytes,
            (unsigned long long)g_prof.civac_off_bytes,
            (unsigned long long)g_prof.n_launch,
            (unsigned long long)g_prof.n_upload,
            (unsigned long long)g_prof.n_download,
            (unsigned long long)g_prof.n_load,
            (unsigned long long)g_prof.n_flush,
            (unsigned long long)g_prof.n_batch,
            (unsigned long long)g_prof.n_batch_launch,
            (unsigned long long)g_prof.clean_bytes,
            (unsigned long long)g_prof.up_bytes,
            (unsigned long long)g_prof.down_bytes);
}

#define REMOTE_SOCKET      "/run/vc4d.sock"
#define REMOTE_TIMEOUT_MS  10000u     /* vc4d's default --max-timeout-ms */
#define REMOTE_MIN_ARENA   (8u << 20)
#define REMOTE_DIRTY_MAX   24
/* Shadow uniform slots for batched launches.
 *
 * The arena has 128 kernel slots: code at 0x1000 + slot*16 KiB, uniforms at
 * 0x210000 + slot*4 KiB, so the uniform areas end at 0x290000 and the data
 * heap starts at 0x300000.  The 448 KiB between them is unused in both
 * backends; the 48 KiB this needs sits at its start.  (The first attempt put
 * this at 0x230000, which is kernel slot 32's uniform block - it corrupted
 * uniforms of larger graphs and hung the QPU.)
 *
 * One 12-QPU block per queued launch means a launch's uniforms can never be
 * overwritten by the next launch while it waits in the queue. */
#define REMOTE_BATCH_UNI_BASE   0x290000u
#define REMOTE_BATCH_UNI_STRIDE 768u
_Static_assert(REMOTE_BATCH_UNI_BASE +
               VC4D_BATCH_MAX * REMOTE_BATCH_UNI_STRIDE <= VC4_DATA_BASE,
               "batch uniform shadows must stay below the data heap");

struct vc4_remote {
    vc4c_conn *conn;
    vc4c_buf   region;                /* the one ALLOC: the whole arena */
    vc4c_prog  prog[VC4_MAX_KERNELS]; /* parallel to ctx->k[] */
    int        nprog;
    int        have_named;            /* daemon keys slots by program name */
    int        have_uni_mask;         /* daemon takes the address mask     */
    int        have_batch;            /* daemon takes LAUNCH_BATCH         */
    int        batch_on;              /* between vc4_batch_begin/end       */
    int        npend;                 /* launches queued, not submitted    */
    int        batch_err;             /* latched failure from a barrier    */
    struct {
        int      idx;                 /* slot in ctx->k[] / r->prog[]      */
        unsigned nqpu;
        uint32_t uni_off;             /* shadow uniform block              */
        uint32_t mask;
        uint32_t timeout_ms;
    } pend[VC4D_BATCH_MAX];
    /* byte intervals the CPU has written since the last clean */
    struct { uint32_t off, len; } dirty[REMOTE_DIRTY_MAX];
    int        ndirty;
    size_t     scratch_clean;         /* scratch high-water already cleaned */
};

/* Clean+invalidate by VA.  Contrary to the vc4d demo's assumption, the CMA
 * dma-buf mapping this kernel hands out is cacheable (the daemon's uncached
 * /dev/mem read saw zeros until we cleaned), so the client must do the cache
 * maintenance the demo does with vc4c_sync().  The ranges are the engine's
 * high-water marks, not the whole 62 MiB region.
 *
 * DC IVAC (invalidate without writeback) is undefined at EL0 on arm64, so this
 * is CIVAC.  That is safe because every line is cleaned before a launch: a
 * later CIVAC finds only clean (or invalid) lines and therefore cannot write
 * a stale copy back over the GPU's result. */
static void cache_civac(void *p, size_t n)
{
#if defined(__aarch64__)
    uintptr_t line = (uintptr_t)p & ~63ul;
    uintptr_t end = (uintptr_t)p + n;
    for (; line < end; line += 64)
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)p;
    (void)n;
    __sync_synchronize();
#endif
}

/* Drop the client's cached copies of [p, p+n) *before* the CPU writes them.
 *
 * The dma-buf mapping is cacheable, so a stale line left behind by an earlier
 * read would be written back by the clean that precedes the next launch, and
 * would overlay whatever the QPU has since put there (a partial-line write
 * modifies only its words; the rest of the line comes from the cache).  ARM
 * has no cross-process invalidate and the daemon cannot reach another
 * process's D-cache, so this is the one place it can be done: the client that
 * owns the mapping.  DC IVAC is undefined at EL0, so this is the same CIVAC
 * the launch path uses, and it is safe here because it runs *before* the
 * write: it may only write back the client's own earlier data, which the
 * memcpy is about to replace. */
void vc4__invalidate(void *p, size_t n)
{
    cache_civac(p, n);
}

vc4_status vc4__remote_batch_end(vc4_ctx *ctx);
static void remote_drain(vc4_ctx *ctx);

/* Same, by arena offset, for the C code that fills a persistent buffer with
 * memset + partial memcpys before a launch (weight panels, padded images,
 * scratch, split-K C rows). */
void vc4__invalidate_off(vc4_ctx *ctx, uint32_t off, size_t n)
{
    if (ctx && ctx->arena) {
        /* The CPU is about to rewrite this range.  A queued launch may still
         * read it, so the queue goes first: this is the barrier that lets
         * batching stay correct without per-launch dependency information. */
        remote_drain(ctx);
        uint64_t t = g_prof.on ? mono_ns() : 0;
        cache_civac(ctx->arena + off, n);
        if (g_prof.on) {
            g_prof.civac_off += mono_ns() - t;
            g_prof.civac_off_bytes += n;
        }
    }
}

/* Remember a CPU-written [off, off+bytes) interval; merged on insertion so a
 * fill + a copy over the same buffer costs one clean.  No barrier: used for
 * ranges that belong to the launch being queued (its shadow uniform block and
 * the scratch it is about to read), not for data another queued launch may
 * already be reading. */
static void remote_mark_nb(vc4_ctx *ctx, uint32_t off, size_t bytes)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!r || !bytes)
        return;
    uint64_t s = (uint64_t)off & ~63ull;
    uint64_t e = ((uint64_t)off + bytes + 63ull) & ~63ull;
    if (e > ctx->arena_bytes)
        e = ctx->arena_bytes;
    if (s >= e)
        return;
    for (int i = 0; i < r->ndirty; ++i) {
        uint64_t ds = r->dirty[i].off;
        uint64_t de = ds + r->dirty[i].len;
        if (s <= de && ds <= e) {              /* overlap or touch: merge */
            if (s > ds)
                s = ds;
            if (e < de)
                e = de;
            r->dirty[i] = r->dirty[--r->ndirty];
            i = -1;
        }
    }
    if (r->ndirty == REMOTE_DIRTY_MAX) {       /* collapse to the hull */
        uint64_t ds = s, de = e;
        for (int i = 0; i < r->ndirty; ++i) {
            if (r->dirty[i].off < ds)
                ds = r->dirty[i].off;
            uint64_t x = (uint64_t)r->dirty[i].off + r->dirty[i].len;
            if (x > de)
                de = x;
        }
        r->ndirty = 0;
        s = ds;
        e = de;
    }
    r->dirty[r->ndirty].off = (uint32_t)s;
    r->dirty[r->ndirty].len = (uint32_t)(e - s);
    r->ndirty++;
}

/* A CPU write that other queued launches may be reading: the batch must
 * complete first.  (The write itself has already happened in some callers,
 * which is why the launch-time calls use remote_mark_nb; this form exists for
 * the upload path and for marking a buffer whose contents changed.) */
static void remote_mark(vc4_ctx *ctx, uint32_t off, size_t bytes)
{
    remote_drain(ctx);
    remote_mark_nb(ctx, off, bytes);
}

/* Write every CPU-dirtied line back to DRAM before the GPU (or the daemon's
 * uncached view) reads it. */
static void remote_clean(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!r)
        return;
    for (int i = 0; i < r->ndirty; ++i) {
        if (prof_on())
            g_prof.clean_bytes += (r->dirty[i].len + 63u) / 64u;
        cache_civac(ctx->arena + r->dirty[i].off, r->dirty[i].len);
    }
    r->ndirty = 0;
}

void vc4__remote_mark(vc4_ctx *ctx, uint32_t off, size_t bytes)
{
    remote_mark(ctx, off, bytes);
}

void vc4__remote_scratch_reset(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (r) {
        /* The scratch cursor is about to hand out memory a queued launch may
         * still be reading, so the queue has to drain first. */
        remote_drain(ctx);
        r->scratch_clean = ctx->arena_top;   /* rewritten from the base */
    }
}

static const char *remote_socket(void)
{
    const char *e = getenv("VC4D_SOCKET");
    return e && *e ? e : REMOTE_SOCKET;
}

static unsigned remote_timeout_ms(void)
{
    unsigned long v = 0;
    const char *e = getenv("VC4INFER_TMO_MS");
    if (e)
        v = strtoul(e, NULL, 0);
    if (!v || v > REMOTE_TIMEOUT_MS)
        v = REMOTE_TIMEOUT_MS;
    return (unsigned)v;
}

/* Map a -VC4D_E_* to the engine's status space, keeping a useful message. */
static vc4_status remote_status(vc4_ctx *ctx, int rc, const char *what)
{
    int e = rc < 0 ? -rc : rc;
    vc4__set_err(ctx, "%s: %s", what, vc4c_strerror(rc));
    switch (e) {
    case VC4D_E_TIMEOUT:
    case VC4D_E_DEAD:
        ctx->dead = 1;
        return VC4_ERR_TIMEOUT;
    case VC4D_E_GPU:
        return VC4_ERR_GPU;
    case VC4D_E_NOMEM:
        return VC4_ERR_NOMEM;
    case VC4D_E_BUSY:
        return VC4_ERR_GPU;
    case VC4D_E_TOOLARGE:
    case VC4D_E_INVAL:
        return VC4_ERR_INVAL;
    default:
        return VC4_ERR_IO;
    }
}

vc4_status vc4__remote_init(vc4_ctx **out, const char *cache_dir,
                            size_t arena_bytes)
{
    if (!out)
        return VC4_ERR_INVAL;
    *out = NULL;
    vc4_ctx *ctx = calloc(1, sizeof(*ctx));
    struct vc4_remote *r = calloc(1, sizeof(*r));
    if (!ctx || !r) {
        free(ctx);
        free(r);
        return VC4_ERR_NOMEM;
    }
    ctx->mem_fd = -1;
    ctx->dmabuf_fd = -1;
    ctx->remote = r;

    prof_init();
    int rc = vc4c_connect(&r->conn, remote_socket());
    if (rc != 0) {
        vc4__set_err(ctx, "connect %s: %s (is vc4d running?)",
                     remote_socket(), vc4c_strerror(rc));
        free(r);
        free(ctx);
        return VC4_ERR_IO;
    }

    struct vc4d_hello_rsp info;
    rc = vc4c_info(r->conn, &info);
    if (rc != 0) {
        vc4__set_err(ctx, "vc4d HELLO: %s", vc4c_strerror(rc));
        vc4c_close(r->conn);
        free(r);
        free(ctx);
        return VC4_ERR_IO;
    }
    r->have_named = (info.flags & VC4D_F_NAMED_PROGS) != 0;
    r->have_uni_mask = (info.flags & VC4D_F_UNI_MASK) != 0;
    r->have_batch = (info.flags & VC4D_F_LAUNCH_BATCH) != 0;
    /* A/B switch for correctness comparisons and for a daemon whose batch
     * path is suspect: the single-launch path is unchanged.  Like the other
     * VC4 switches, an empty value or "0" leaves the feature on. */
    {
        const char *nb = getenv("VC4INFER_NO_BATCH");
        if (nb && *nb && *nb != '0')
            r->have_batch = 0;
    }
    if (getenv("VC4INFER_DEBUG"))
        fprintf(stderr, "[vc4] HELLO flags=0x%x batch=%d uni_mask=%d\n",
                info.flags, r->have_batch, r->have_uni_mask);

    /* The daemon hands out regions from its data heap only. */
    size_t max = info.arena_bytes > info.data_base
                     ? (size_t)(info.arena_bytes - info.data_base) : 0;
    if (arena_bytes == 0) {
        /* same override as the direct path (vc4_dev.c env_arena), so a bigger
           model can be run against a daemon that serves a bigger arena */
        const char *e = getenv("VC4INFER_ARENA_MB");
        long mb = e ? strtol(e, NULL, 10) : 0;
        arena_bytes = mb > 0 ? (size_t)mb << 20 : VC4_ARENA_DEFAULT;
    }
    size_t want = arena_bytes < max ? arena_bytes : max;
    want &= ~(size_t)4095;
    if (want < REMOTE_MIN_ARENA) {
        vc4__set_err(ctx, "vc4d arena too small: %zu MiB total, %zu MiB usable",
                     (size_t)info.arena_bytes >> 20, max >> 20);
        vc4c_close(r->conn);
        free(r);
        free(ctx);
        return VC4_ERR_NOMEM;
    }

    /* Fall back to smaller regions like the direct path does (CMA pressure,
     * another client already holding part of the arena). */
    size_t got = 0;
    for (size_t len = want; len >= REMOTE_MIN_ARENA; len /= 2) {
        rc = vc4c_alloc(r->conn, len, 0, &r->region);
        if (rc == 0) {
            got = len;
            break;
        }
        if (rc != -VC4D_E_NOMEM && rc != -VC4D_E_TOOLARGE)
            break;
    }
    if (!r->region.map || rc != 0) {
        vc4__set_err(ctx, "vc4d ALLOC: %s (wanted %zu MiB, daemon arena %u MiB, "
                     "data base %u B)", vc4c_strerror(rc), want >> 20,
                     info.arena_bytes >> 20, info.data_base);
        vc4c_close(r->conn);
        free(r);
        free(ctx);
        return VC4_ERR_NOMEM;
    }

    /* Never accept a grant that cannot hold data.
     *
     * The engine keeps its code and uniform slots in the first VC4_DATA_BASE
     * bytes of its region and allocates upwards from there, so a region of
     * exactly VC4_DATA_BASE is an arena with zero usable bytes - the client
     * would report "ok" and then fail every vc4_alloc() (measured: B12, a
     * struct-layout mismatch between two objects of the library, which made
     * the client read the daemon's `off` as its `len`).  The reply is checked
     * against what HELLO advertised, so a disagreement between client and
     * daemon about the alloc reply fails here, with the numbers, instead of
     * much later inside a graph. */
    if (r->region.len <= VC4_DATA_BASE ||
        r->region.off < info.data_base ||
        (uint64_t)r->region.off + r->region.len > info.arena_bytes ||
        r->region.len < got) {
        vc4__set_err(ctx,
                     "vc4d granted a %zu B region at off 0x%x but the daemon "
                     "advertised %u B with a %u B data base: the region has "
                     "%s data space (requested %zu MiB, granted %zu MiB). "
                     "Client and daemon disagree about the allocation reply - "
                     "rebuild both from the same tree (`make -C vc4d && make "
                     "-C infer clean && make -C infer`) and check that "
                     "struct vc4_ctx / struct vc4d_alloc_rsp are the same in "
                     "every object of libvc4infer.a",
                     (size_t)r->region.len, r->region.off, info.arena_bytes,
                     info.data_base,
                     r->region.len > VC4_DATA_BASE ? "some"
                                                   : "no",
                     want >> 20, (size_t)got >> 20);
        vc4c_free(r->conn, &r->region);
        vc4c_close(r->conn);
        free(r);
        free(ctx);
        return VC4_ERR_NOMEM;
    }

    ctx->arena = r->region.map;
    ctx->arena_bytes = r->region.len;
    ctx->arena_top = VC4_DATA_BASE;
    ctx->scratch = ctx->arena_top;
    ctx->dirty_top = VC4_DATA_BASE;    /* covers the uniform slots below it */
    ctx->gpu_bus = r->region.bus;      /* bus address of region.map[0] */
    /* ctx->v3d stays NULL: every register path is replaced by the daemon. */
    if (getenv("VC4INFER_DEBUG"))
        fprintf(stderr, "[vc4] vc4d arena: off=0x%x len=0x%x bus=0x%08x map=%p\n",
                r->region.off, r->region.len, r->region.bus, r->region.map);

    if (cache_dir)
        snprintf(ctx->cache_dir, sizeof(ctx->cache_dir), "%s", cache_dir);
    else
        vc4__default_cache_dir(ctx->cache_dir, sizeof(ctx->cache_dir));
    ctx->allow_jit = getenv("VC4INFER_NO_JIT") ? 0 : 1;

    *out = ctx;
    return VC4_OK;
}

void vc4__remote_shutdown(vc4_ctx *ctx)
{
    if (!ctx)
        return;
    remote_drain(ctx);
    struct vc4_remote *r = ctx->remote;
    for (int i = 0; i < ctx->nkern; ++i) {
        free(ctx->k[i].code);
        ctx->k[i].code = NULL;
    }
    if (r) {
        if (r->region.map)
            munmap(r->region.map, r->region.len);
        if (r->conn)
            vc4c_close(r->conn);
        free(r);
        ctx->remote = NULL;
    }
    vc4_prof_report();
    free(ctx);
}

vc4_status vc4__remote_load(vc4_ctx *ctx, const char *name,
                            const uint8_t *words, size_t bytes,
                            int nqpu, vc4_kernel **out)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    /* A load can evict the program slot a queued launch names. */
    remote_drain(ctx);
    if (!ctx || !r || !words || !out)
        return VC4_ERR_INVAL;
    if (bytes < 8 || (bytes & 7u) || bytes > VC4_CODE_SLOT) {
        vc4__set_err(ctx, "remote upload %s: %zu B is not a QPU program",
                     name, bytes);
        return VC4_ERR_KERNEL;
    }
    if (nqpu < 1 || nqpu > VC4_MAX_QPU) {
        vc4__set_err(ctx, "remote upload %s: nqpu=%d", name, nqpu);
        return VC4_ERR_INVAL;
    }

    /* A free cache entry, or the least recently used one (G3).  Its uniform
     * block and its daemon program slot are both indexed by the entry, so the
     * new kernel simply takes them over. */
    PROF_T(t_load);
    vc4_kernel *k = vc4__kern_slot(ctx);
    if (!k) {
        vc4__set_err(ctx, "remote kernel cache: no slot (all %d in use)",
                     VC4_MAX_KERNELS);
        return VC4_ERR_NOMEM;
    }
    int slot = (int)(k - ctx->k);
    k->code_off = VC4_CODE_BASE + (uint32_t)slot * VC4_CODE_SLOT;
    k->uni_off = VC4_UNI_BASE + (uint32_t)slot * VC4_UNI_SLOT;
    if (k->uni_off + VC4_UNI_SLOT > VC4_DATA_BASE ||
        (size_t)k->uni_off + 64u * VC4_MAX_QPU > ctx->arena_bytes) {
        vc4__set_err(ctx, "kernel cache: uniform area does not fit the arena");
        return VC4_ERR_NOMEM;
    }
    /* The daemon owns the code slot; the client only owns this uniform block,
     * which it materialises in place and names in the LAUNCH request. */
    memset(ctx->arena + k->uni_off, 0, VC4_UNI_SLOT);
    if (ctx->dirty_top < (size_t)k->uni_off + VC4_UNI_SLOT)
        ctx->dirty_top = (size_t)k->uni_off + VC4_UNI_SLOT;
    __sync_synchronize();

    /* Keep the code so a launch the daemon evicted can be re-uploaded without
     * going back to the blob cache (and without a JIT). */
    uint8_t *copy = malloc(bytes);
    if (!copy) {
        k->used = 0;
        return VC4_ERR_NOMEM;
    }
    memcpy(copy, words, bytes);

    int rc = vc4c_upload_named(r->conn, r->have_named ? name : NULL,
                               words, bytes, (unsigned)nqpu,
                               VC4_UNI_WORDS, &r->prog[slot]);
    if (rc != 0) {
        free(copy);
        k->used = 0;
        vc4_status st = remote_status(ctx, rc, "UPLOAD");
        return st == VC4_ERR_INVAL ? VC4_ERR_KERNEL : st;
    }

    k->code = copy;
    k->code_len = (uint32_t)bytes;
    k->code_words = (uint32_t)(bytes / 8);
    k->nqpu = (uint32_t)nqpu;
    k->used = 1;
    k->used_at = ++ctx->kern_clock;
    snprintf(k->name, sizeof(k->name), "%s", name);
    r->nprog = ctx->nkern;
    *out = k;
    if (prof_on()) {
        g_prof.load += mono_ns() - t_load;
        g_prof.n_load++;
    }
    return VC4_OK;
}

/* The daemon evicted the program slot (G3): upload the same code again, under
 * the same name, and let the LRU give it some other slot.  Returns 0 or a
 * negative VC4D_E_*. */
static int remote_reupload(struct vc4_remote *r, const vc4_kernel *k, int slot)
{
    if (!k->code || !k->code_len)
        return -VC4D_E_NOENT;
    return vc4c_upload_named(r->conn, r->have_named ? k->name : NULL,
                             k->code, k->code_len, k->nqpu,
                             VC4_UNI_WORDS, &r->prog[slot]);
}

/* Which uniform slots hold a GPU bus address?  A word is an address exactly
 * when it points into this client's arena; the engine's pointers are always
 * (arena offset + gpu_bus), and a value that happens to land in the same
 * window is still inside the arena, so the mask can only ever be *more*
 * permissive than the daemon's heuristic, never reject a valid launch.  The
 * mask is per slot across QPUs: an address is marked only when every QPU's
 * word at that slot is one. */
static uint32_t remote_uni_mask(vc4_ctx *ctx, const vc4_kernel *k, int nqpu)
{
    uint32_t mask = 0;
    for (uint32_t u = 0; u < VC4_UNI_WORDS; ++u) {
        int all_addr = 1;
        for (int i = 0; i < nqpu && all_addr; ++i) {
            uint32_t w;
            memcpy(&w, ctx->arena + k->uni_off + 64u * (uint32_t)i + 4u * u, 4);
            if (!(w >= ctx->gpu_bus &&
                  (uint64_t)w < (uint64_t)ctx->gpu_bus + ctx->arena_bytes))
                all_addr = 0;
        }
        if (all_addr)
            mask |= 1u << u;
    }
    return mask;
}


/* Queue one launch: snapshot its uniforms into a private shadow slot and
 * remember the program.  Nothing is sent here.  The snapshot is what makes
 * deferral safe with kernels whose uniform block is rewritten for the next
 * launch - the common case, since one uniform area belongs to one blob. */
static vc4_status remote_queue_launch(vc4_ctx *ctx, const vc4_kernel *k,
                                      int nqpu, uint64_t *wall_ns)
{
    struct vc4_remote *r = ctx->remote;
    ptrdiff_t idx = k - ctx->k;
    if (r->npend == (int)VC4D_BATCH_MAX) {
        vc4_status st = vc4__remote_batch_end(ctx);
        if (st != VC4_OK)
            return st;
    }
    int slot = r->npend;
    uint32_t off = REMOTE_BATCH_UNI_BASE +
                   (uint32_t)slot * REMOTE_BATCH_UNI_STRIDE;
    uint32_t mask = r->have_uni_mask ? remote_uni_mask(ctx, k, nqpu) : 0;
    if ((uint64_t)off + 64ull * (unsigned)nqpu > ctx->arena_bytes) {
        vc4__set_err(ctx, "launch: batch uniform area is outside the arena");
        return VC4_ERR_INVAL;
    }
    memcpy(ctx->arena + off, ctx->arena + k->uni_off, 64u * (size_t)nqpu);
    remote_mark_nb(ctx, off, 64u * (size_t)nqpu);
    /* The scratch a launch is about to read was written before the call, so
     * preserve the high-water mark the synchronous path would have cleaned. */
    r->scratch_clean = r->scratch_clean < ctx->arena_top
                           ? ctx->arena_top : r->scratch_clean;
    if (r->scratch_clean > ctx->scratch)
        r->scratch_clean = ctx->arena_top;
    if (ctx->scratch > r->scratch_clean) {
        remote_mark_nb(ctx, (uint32_t)r->scratch_clean,
                       ctx->scratch - r->scratch_clean);
        r->scratch_clean = ctx->scratch;
    }
    r->pend[slot].idx = (int)idx;
    r->pend[slot].nqpu = (unsigned)nqpu;
    r->pend[slot].uni_off = off;
    r->pend[slot].mask = mask;
    r->pend[slot].timeout_ms = remote_timeout_ms();
    r->npend++;
    if (wall_ns)
        *wall_ns = 0;          /* completion time is known at the flush */
    return VC4_OK;
}

/* Submit the queued launches as one request and account for the completions.
 *
 * Retries mirror the single-launch path: E_NOENT means the daemon evicted a
 * program slot (re-upload it and resume at that entry), E_BUSY means another
 * process holds the GPU lock (wait and resume at that entry).  Entries before
 * the failing one have run and are never resent, which is why the resume index
 * comes from the reply's per-entry statuses and not from a guess. */
/* Submit what is queued.  The public vc4__remote_batch_end() reports the
 * result to its caller; the hazard barriers call remote_drain(), which latches
 * a failure for the next checked call instead of discarding it.  That matters
 * because a mid-node barrier (a dirty mark, a download) flushes launches whose
 * caller is not the one that will see the return value. */
static vc4_status remote_submit(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx->remote;
    if (!r->npend)
        return VC4_OK;

    int n = r->npend;
    vc4_status ret = VC4_OK;
    int from = 0;
    unsigned busy_tries = 0, noent_tries = 0;

    while (from < n) {
        PROF_W(bc0);
        remote_clean(ctx);
        PROF_W(bc1);
        PROF_ADD(clean, bc0, bc1);
        vc4c_batch_ent ent[VC4D_BATCH_MAX];
        memset(ent, 0, sizeof(ent));
        for (int i = from; i < n; ++i) {
            const vc4c_prog *p = &r->prog[r->pend[i].idx];
            ent[i - from].kind = VC4D_BENT_LAUNCH;
            ent[i - from].prog = p->handle;
            ent[i - from].timeout_ms = r->pend[i].timeout_ms;
            ent[i - from].nqpu = r->pend[i].nqpu;
            ent[i - from].nuni = p->nuni;
            ent[i - from].flags = r->have_uni_mask ? VC4D_LAUNCH_UNI_MASK : 0;
            ent[i - from].uni_off = r->pend[i].uni_off;
            ent[i - from].uni_addr = r->pend[i].mask;
        }
        struct vc4d_batch_item item[VC4D_BATCH_MAX];
        memset(item, 0, sizeof(item));
        unsigned done = 0;
        PROF_T(br0);
        int rc = vc4c_launch_batch(r->conn, &r->region, ent, (unsigned)(n - from),
                                   item, &done);
        PROF_T(br1);
        PROF_ADD(rt, br0, br1);
        if (prof_on()) {
            g_prof.n_batch++;
            g_prof.n_batch_launch += (unsigned)(n - from);
        }
        for (unsigned i = 0; i < done && i < (unsigned)(n - from); ++i) {
            if (item[i].status != VC4D_OK)
                continue;
            int j = from + (int)i;
            ctx->st.launches++;
            ctx->st.kernel_ns += item[i].wall_ns;
            ctx->st.gpu_cycles += item[i].cycles;
            if (item[i].errs) {
                const vc4_kernel *k = &ctx->k[r->pend[j].idx];
                vc4__set_err(ctx, "launch %s: V3D_ERR_STAT=0x%08x",
                             k->name, item[i].errs);
                if (!getenv("VC4INFER_IGNORE_ERRSTAT")) {
                    ret = VC4_ERR_GPU;
                    goto out;
                }
            }
        }
        if (rc == 0) {
            if (done != (unsigned)(n - from)) {
                vc4__set_err(ctx, "batch: %u of %d entries completed",
                             done, n - from);
                ret = VC4_ERR_IO;
            }
            break;
        }
        /* done includes the entry that failed (its status is the failure),
         * so execution stopped there and the entries after it never ran. */
        int fail = (int)done - 1;
        if (fail < 0) {
            /* Nothing ran: the daemon refused the request itself (another
             * launch is in flight).  Same bounded wait as the single path. */
            if (rc == -VC4D_E_BUSY && busy_tries < 60) {
                busy_tries++;
                usleep(1000000);
                continue;
            }
            ret = remote_status(ctx, rc, "LAUNCH");
            break;
        }
        fail += from;
        if (rc == -VC4D_E_NOENT && noent_tries < 2) {
            noent_tries++;
            int idx = r->pend[fail].idx;
            if (getenv("VC4INFER_DEBUG"))
                fprintf(stderr, "[vc4] %s: program slot evicted, re-uploading\n",
                        ctx->k[idx].name);
            if (remote_reupload(r, &ctx->k[idx], idx) != 0) {
                ret = remote_status(ctx, rc, "LAUNCH");
                break;
            }
            from = fail;
            continue;
        }
        if (rc == -VC4D_E_BUSY && busy_tries < 60) {
            busy_tries++;
            from = fail;
            usleep(1000000);
            continue;
        }
        ret = remote_status(ctx, rc, "LAUNCH");
        break;
    }
out:
    r->npend = 0;
    return ret;
}

vc4_status vc4__remote_batch_end(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!r)
        return VC4_OK;
    vc4_status st = remote_submit(ctx);
    if (st == VC4_OK && r->batch_err) {
        st = (vc4_status)r->batch_err;
        r->batch_err = 0;
    }
    return st;
}

static void remote_drain(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!r || !r->npend)
        return;
    vc4_status st = remote_submit(ctx);
    if (st != VC4_OK && !r->batch_err)
        r->batch_err = (int)st;
}

/* Arm/disarm explicit batching (public vc4_batch_begin/vc4_batch_end).  A
 * program that never arms it gets the synchronous single-launch path, byte for
 * byte; the Python binding arms around one C API call so a launch error or a
 * timeout is reported by the call that issued it. */
void vc4__remote_batch_begin(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (r && r->have_batch)
        r->batch_on = 1;
}

vc4_status vc4__remote_launch(vc4_ctx *ctx, const vc4_kernel *k, int nqpu,
                              uint64_t *wall_ns)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!ctx || !k || !r)
        return VC4_ERR_INVAL;
    PROF_W(t0);
    if (ctx->dead) {
        vc4__set_err(ctx, "context is dead after a previous timeout; reboot the board");
        return VC4_ERR_TIMEOUT;
    }
    if (nqpu < 1 || nqpu > VC4_MAX_QPU || nqpu > (int)k->nqpu) {
        vc4__set_err(ctx, "launch: nqpu=%d out of range (blob supports %u)",
                     nqpu, k->nqpu);
        return VC4_ERR_INVAL;
    }
    ptrdiff_t idx = k - ctx->k;
    if (idx < 0 || idx >= ctx->nkern || !r->prog[idx].handle) {
        vc4__set_err(ctx, "launch: kernel %s was not uploaded", k->name);
        return VC4_ERR_INVAL;
    }
    ((vc4_kernel *)k)->used_at = ++ctx->kern_clock;
    PROF_W(t1);

    if (r->batch_on && r->have_batch) {
        vc4_status st = remote_queue_launch(ctx, k, nqpu, wall_ns);
        if (prof_on()) {
            g_prof.n_launch++;
            g_prof.pre += t1 - t0;
        }
        return st;
    }

    vc4c_result res;
    memset(&res, 0, sizeof(res));
    /* Put every CPU-written line in DRAM before the daemon's uncached read of
     * the uniforms and before the QPUs read their inputs.  The uniform block
     * and the per-op scratch are not routed through vc4_upload(), so mark
     * them here; persistent buffers were marked where they were written. */
    remote_mark(ctx, k->uni_off, 64u * (size_t)nqpu);
    r->scratch_clean = r->scratch_clean < ctx->arena_top
                           ? ctx->arena_top : r->scratch_clean;
    if (r->scratch_clean > ctx->scratch)
        r->scratch_clean = ctx->arena_top;      /* scratch was reset */
    if (ctx->scratch > r->scratch_clean) {
        remote_mark(ctx, (uint32_t)r->scratch_clean,
                    ctx->scratch - r->scratch_clean);
        r->scratch_clean = ctx->scratch;
    }
    PROF_W(t2);
    PROF_ADD(mark, t1, t2);
    remote_clean(ctx);
    PROF_W(t3);
    PROF_ADD(clean, t2, t3);
    if (getenv("VC4INFER_DEBUG")) {
        const uint32_t *u = (const uint32_t *)(ctx->arena + k->uni_off);
        fprintf(stderr, "[vc4] launch %s nqpu=%d uni=%08x %08x %08x %08x"
                " %08x %08x %08x %08x (bus=0x%08x)\n",
                k->name, nqpu, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                ctx->gpu_bus + k->uni_off);
    }
    int rc;
    for (unsigned attempt = 0;; ++attempt) {
        PROF_W(tm0);
        uint32_t mask = r->have_uni_mask ? remote_uni_mask(ctx, k, nqpu) : 0;
        PROF_W(tm1);
        PROF_ADD(mask, tm0, tm1);
        PROF_T(tr0);
        if (r->have_uni_mask)
            rc = vc4c_launch_wait_u(r->conn, &r->prog[idx], (unsigned)nqpu,
                                    &r->region, k->uni_off, remote_timeout_ms(),
                                    &res, mask);
        else
            rc = vc4c_launch_wait(r->conn, &r->prog[idx], (unsigned)nqpu,
                                  &r->region, k->uni_off, remote_timeout_ms(),
                                  &res);
        PROF_T(tr1);
        PROF_ADD(rt, tr0, tr1);
        /* The daemon evicted this program slot for a newer one (G3).  The code
         * is in the cache, so put it back and launch again.  Attempts are
         * bounded so a daemon that answers E_NOENT for a different reason
         * cannot spin here. */
        if (rc == -VC4D_E_NOENT && attempt < 2) {
            if (getenv("VC4INFER_DEBUG"))
                fprintf(stderr, "[vc4] %s: program slot evicted, re-uploading\n",
                        k->name);
            rc = remote_reupload(r, k, (int)idx);
            if (rc != 0)
                break;
            continue;
        }
        /* On a shared board another user may hold /tmp/vc4-gpu.lock for a
         * whole test run; the daemon's bounded wait then reports BUSY without
         * kicking anything, so wait and try again. */
        if (rc != -VC4D_E_BUSY || attempt >= 60)
            break;
        usleep(1000000);
    }
    if (rc != 0) {
        vc4_status st = remote_status(ctx, rc, "LAUNCH");
        if (st == VC4_ERR_GPU && res.errs) {
            /* the daemon reported a V3D error with the wait payload */
        }
        return st;
    }

    PROF_W(t4);
    if (prof_on()) {
        g_prof.pre += t1 - t0;
        g_prof.n_launch++;
    }
    if (wall_ns)
        *wall_ns = res.wall_ns;
    ctx->st.gpu_cycles += res.cycles;
    ctx->st.launches++;
    if (res.errs) {
        vc4__set_err(ctx, "launch %s: V3D_ERR_STAT=0x%08x", k->name, res.errs);
        if (getenv("VC4INFER_DEBUG"))
            fprintf(stderr, "[vc4] launch %s: nqpu=%d errs=0x%08x\n",
                    k->name, nqpu, res.errs);
        if (!getenv("VC4INFER_IGNORE_ERRSTAT"))
            return VC4_ERR_GPU;
    }
    if (prof_on())
        g_prof.tail += mono_ns() - t4;
    return VC4_OK;
}

vc4_status vc4__remote_upload(vc4_ctx *ctx, vc4_buf *b, const void *src,
                              size_t bytes)
{
    if (!ctx || !b || (!src && bytes))
        return VC4_ERR_INVAL;
    if (bytes > b->bytes)
        return VC4_ERR_INVAL;
    /* The CPU is about to write memory a queued launch may read. */
    remote_drain(ctx);
    /* no stale line of ours may survive into the range we are writing: the
     * clean before the next launch writes whole lines back */
    PROF_W(u0);
    cache_civac(ctx->arena + b->off, bytes);
    PROF_W(u1);
    memcpy(ctx->arena + b->off, src, bytes);
    PROF_W(u2);
    PROF_ADD(up_clean, u0, u1);
    PROF_ADD(up_copy, u1, u2);
    if (prof_on()) {
        g_prof.n_upload++;
        g_prof.up_bytes += bytes;
    }
    if (ctx->dirty_top < (size_t)b->off + bytes)
        ctx->dirty_top = (size_t)b->off + bytes;
    remote_mark(ctx, b->off, bytes);
    __sync_synchronize();
    return VC4_OK;
}

vc4_status vc4__remote_download(vc4_ctx *ctx, const vc4_buf *b, void *dst,
                                size_t bytes)
{
    if (!ctx || !b || (!dst && bytes))
        return VC4_ERR_INVAL;
    if (bytes > b->bytes)
        return VC4_ERR_INVAL;
    /* The CPU is about to read memory a queued launch may write. */
    remote_drain(ctx);
    /* Drop the CPU's stale (clean) copies of this range before reading the
     * GPU's result out of DRAM. */
    PROF_W(d0);
    cache_civac(ctx->arena + b->off, bytes);
    PROF_W(d1);
    memcpy(dst, ctx->arena + b->off, bytes);
    PROF_W(d2);
    PROF_ADD(down_clean, d0, d1);
    PROF_ADD(down_copy, d1, d2);
    if (prof_on()) {
        g_prof.n_download++;
        g_prof.down_bytes += bytes;
    }
    return VC4_OK;
}

void vc4__remote_l2_flush(vc4_ctx *ctx)
{
    if (!ctx || !ctx->remote)
        return;
    /* Callers use this to make a launch's result visible to the CPU. */
    remote_drain(ctx);
    /* Every path that calls this right after a launch reads a scratch buffer
     * (c_tmp, cmat, o_tmp, DXp, dWp).  Invalidate those clean lines so the
     * CPU sees what the QPUs wrote. */
    if (ctx->scratch > ctx->arena_top) {
        PROF_W(f0);
        cache_civac(ctx->arena + ctx->arena_top,
                    ctx->scratch - ctx->arena_top);
        PROF_W(f1);
        PROF_ADD(flush, f0, f1);
        if (prof_on())
            g_prof.n_flush++;
    }
}
