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

#define REMOTE_SOCKET      "/run/vc4d.sock"
#define REMOTE_TIMEOUT_MS  10000u     /* vc4d's default --max-timeout-ms */
#define REMOTE_MIN_ARENA   (8u << 20)
#define REMOTE_DIRTY_MAX   24

struct vc4_remote {
    vc4c_conn *conn;
    vc4c_buf   region;                /* the one ALLOC: the whole arena */
    vc4c_prog  prog[VC4_MAX_KERNELS]; /* parallel to ctx->k[] */
    int        nprog;
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

/* Remember a CPU-written [off, off+bytes) interval; merged on insertion so a
 * fill + a copy over the same buffer costs one clean. */
static void remote_mark(vc4_ctx *ctx, uint32_t off, size_t bytes)
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

/* Write every CPU-dirtied line back to DRAM before the GPU (or the daemon's
 * uncached view) reads it. */
static void remote_clean(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!r)
        return;
    for (int i = 0; i < r->ndirty; ++i)
        cache_civac(ctx->arena + r->dirty[i].off, r->dirty[i].len);
    r->ndirty = 0;
}

void vc4__remote_mark(vc4_ctx *ctx, uint32_t off, size_t bytes)
{
    remote_mark(ctx, off, bytes);
}

void vc4__remote_scratch_reset(vc4_ctx *ctx)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (r)
        r->scratch_clean = ctx->arena_top;   /* rewritten from the base */
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
    for (size_t len = want; len >= REMOTE_MIN_ARENA; len /= 2) {
        rc = vc4c_alloc(r->conn, len, 0, &r->region);
        if (rc == 0)
            break;
        if (rc != -VC4D_E_NOMEM && rc != -VC4D_E_TOOLARGE)
            break;
    }
    if (!r->region.map) {
        vc4__set_err(ctx, "vc4d ALLOC: %s (wanted %zu MiB)",
                     vc4c_strerror(rc), want >> 20);
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
        fprintf(stderr, "[vc4] vc4d arena: off=0x%x len=0x%zx bus=0x%08x map=%p\n",
                r->region.off, r->region.len, r->region.bus, r->region.map);

    const char *cd = cache_dir;
    if (!cd) {
        cd = getenv("VC4INFER_CACHE");
        if (!cd)
            cd = "kcache";
    }
    snprintf(ctx->cache_dir, sizeof(ctx->cache_dir), "%s", cd);
    ctx->allow_jit = getenv("VC4INFER_NO_JIT") ? 0 : 1;

    *out = ctx;
    return VC4_OK;
}

void vc4__remote_shutdown(vc4_ctx *ctx)
{
    if (!ctx)
        return;
    struct vc4_remote *r = ctx->remote;
    if (r) {
        if (r->region.map)
            munmap(r->region.map, r->region.len);
        if (r->conn)
            vc4c_close(r->conn);
        free(r);
        ctx->remote = NULL;
    }
    free(ctx);
}

vc4_status vc4__remote_load(vc4_ctx *ctx, const char *name,
                            const uint8_t *words, size_t bytes,
                            int nqpu, vc4_kernel **out)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
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
    if (ctx->nkern >= VC4_MAX_KERNELS) {
        vc4__set_err(ctx, "kernel cache full (%d entries)", VC4_MAX_KERNELS);
        return VC4_ERR_NOMEM;
    }

    vc4_kernel *k = &ctx->k[ctx->nkern];
    k->code_off = VC4_CODE_BASE + (uint32_t)ctx->nkern * VC4_CODE_SLOT;
    k->uni_off = VC4_UNI_BASE + (uint32_t)ctx->nkern * VC4_UNI_SLOT;
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

    int rc = vc4c_upload(r->conn, words, bytes, (unsigned)nqpu,
                         VC4_UNI_WORDS, &r->prog[ctx->nkern]);
    if (rc != 0) {
        vc4_status st = remote_status(ctx, rc, "UPLOAD");
        return st == VC4_ERR_INVAL ? VC4_ERR_KERNEL : st;
    }

    k->code_words = (uint32_t)(bytes / 8);
    k->nqpu = (uint32_t)nqpu;
    k->used = 1;
    snprintf(k->name, sizeof(k->name), "%s", name);
    ctx->nkern++;
    r->nprog = ctx->nkern;
    *out = k;
    return VC4_OK;
}

vc4_status vc4__remote_launch(vc4_ctx *ctx, const vc4_kernel *k, int nqpu,
                              uint64_t *wall_ns)
{
    struct vc4_remote *r = ctx ? ctx->remote : NULL;
    if (!ctx || !k || !r)
        return VC4_ERR_INVAL;
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
    remote_clean(ctx);
    if (getenv("VC4INFER_DEBUG")) {
        const uint32_t *u = (const uint32_t *)(ctx->arena + k->uni_off);
        fprintf(stderr, "[vc4] launch %s nqpu=%d uni=%08x %08x %08x %08x"
                " %08x %08x %08x %08x (bus=0x%08x)\n",
                k->name, nqpu, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                ctx->gpu_bus + k->uni_off);
    }
    int rc;
    for (unsigned attempt = 0;; ++attempt) {
        rc = vc4c_launch_wait(r->conn, &r->prog[idx], (unsigned)nqpu,
                              &r->region, k->uni_off, remote_timeout_ms(),
                              &res);
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
    return VC4_OK;
}

vc4_status vc4__remote_upload(vc4_ctx *ctx, vc4_buf *b, const void *src,
                              size_t bytes)
{
    if (!ctx || !b || (!src && bytes))
        return VC4_ERR_INVAL;
    if (bytes > b->bytes)
        return VC4_ERR_INVAL;
    memcpy(ctx->arena + b->off, src, bytes);
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
    /* Drop the CPU's stale (clean) copies of this range before reading the
     * GPU's result out of DRAM. */
    cache_civac(ctx->arena + b->off, bytes);
    memcpy(dst, ctx->arena + b->off, bytes);
    return VC4_OK;
}

void vc4__remote_l2_flush(vc4_ctx *ctx)
{
    if (!ctx || !ctx->remote)
        return;
    /* Every path that calls this right after a launch reads a scratch buffer
     * (c_tmp, cmat, o_tmp, DXp, dWp).  Invalidate those clean lines so the
     * CPU sees what the QPUs wrote. */
    if (ctx->scratch > ctx->arena_top)
        cache_civac(ctx->arena + ctx->arena_top,
                    ctx->scratch - ctx->arena_top);
}
