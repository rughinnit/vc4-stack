/*
 * libtf_vc4_plugin.so - a TensorFlow PluggableDevice for the VideoCore IV QPUs.
 *
 * The plugin is two C API surfaces in one shared object:
 *
 *   SE_InitPlugin   the StreamExecutor (SE, API 0.0.1) platform: one device,
 *                   a host-visible allocator carved out of the engine's
 *                   dma-buf arena, a synchronous stream, events, and the
 *                   H2D / D2H / D2D memcpy paths.
 *   TF_InitKernel   the op kernels, registered for this device type:
 *                   Add/AddV2, Mul, Sub, Div/RealDiv, Maximum, Minimum,
 *                   Relu, Relu6, Sigmoid, Tanh, Neg, Exp, Log and Sqrt, all
 *                   fp32.  Every kernel hands the arithmetic to a libvc4infer
 *                   QPU kernel and refuses (rather than computing on the
 *                   host) when the engine cannot take it.
 *
 * The device type is "VC" by default.  TensorFlow 2.21 validates a
 * StreamExecutor platform name and type against [A-Z][A-Z_]*, so a digit is
 * rejected before the plugin's callbacks run:
 *
 *   Check failed: ... Device name/type 'VC4' must match [A-Z][A-Z_]*.
 *
 * and TF_LoadPluggableDeviceLibrary CHECK-fails, which aborts the process.
 * Build with -DVC4_TF_DEVICE_TYPE='"VC4"' against a TensorFlow whose
 * tensorflow::device_utils::ValidateDeviceType allows digits.
 *
 * All engine calls are serialised by one mutex: a vc4_ctx is not thread safe
 * and this plugin is the only engine user in the process.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"

#include "vc4infer.h"

#ifndef VC4_TF_DEVICE_TYPE
#define VC4_TF_DEVICE_TYPE "VC"
#endif

#define VC4_TF_MAX_BLOCKS 512
#define VC4_TF_DEFAULT_ARENA_MB 64
#define VC4_TF_MIN_ARENA_MB 8
#define VC4_TF_MAX_ARENA_MB 96

/* The first 3 MiB of the arena hold the QPU code slots and the uniform
 * blocks: vc4d reserves [0, A_DATA_BASE) and libvc4infer's vc4_init() checks
 * that the context starts at VC4_DATA_BASE (= 0x300000 in both vc4d.h and
 * infer/src/vc4_int.h).  Only the rest is a data heap, so that is what the
 * allocator may hand out. */
#define VC4_TF_DATA_BASE (3u << 20u)

/* ------------------------------------------------------------------ state */

typedef struct {
    uint32_t off;      /* offset inside the engine arena, 0 in host mode */
    size_t   bytes;    /* block size, 64 B aligned */
    void    *host;     /* vc4_host_ptr() of off, or a malloc'd buffer */
    int      used;
} block_t;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static vc4_ctx *g_ctx;
static int      g_engine;                  /* 0 untried, 1 up, -1 failed */
static char     g_engine_err[320];
static int      g_host_arena;              /* VC4_TF_ARENA=host diagnostic */
static size_t   g_arena_bytes;             /* configured, then actual */
static size_t   g_data_limit;              /* what TensorFlow may allocate */
static block_t  g_blocks[VC4_TF_MAX_BLOCKS];
static int      g_nblocks;
static size_t   g_in_use;
static int64_t  g_allocs;
static int64_t  g_peak_in_use;
static int64_t  g_largest_alloc;

static void fail(TF_Status *status, const char *msg)
{
    if (status)
        TF_SetStatus(status, TF_INTERNAL, msg);
}

/* ---------------------------------------------------------------- config */

static size_t configured_arena_bytes(void)
{
    const char *e = getenv("VC4_TF_ARENA_MB");
    long mb = e && *e ? strtol(e, NULL, 10) : VC4_TF_DEFAULT_ARENA_MB;
    if (mb < VC4_TF_MIN_ARENA_MB) mb = VC4_TF_MIN_ARENA_MB;
    if (mb > VC4_TF_MAX_ARENA_MB) mb = VC4_TF_MAX_ARENA_MB;
    return (size_t)mb << 20;
}

/* Caller holds g_mu. */
static void config_locked(void)
{
    if (g_arena_bytes)
        return;
    g_host_arena = getenv("VC4_TF_ARENA") &&
                   strcmp(getenv("VC4_TF_ARENA"), "host") == 0;
    g_arena_bytes = configured_arena_bytes();
    size_t heap = g_arena_bytes > VC4_TF_DATA_BASE
                      ? g_arena_bytes - VC4_TF_DATA_BASE : 0;
    /* Three quarters of the data heap for tensors, one quarter for the
     * engine's per-op scratch (im2col panels, the rounded-up tail of an
     * elementwise op), which grows from the top of the pool. */
    g_data_limit = heap - heap / 4;
}

/* Caller holds g_mu.  Brings the engine up on first use. */
static int engine_up_locked(void)
{
    if (g_engine == 1)
        return 1;
    if (g_engine == -1)
        return 0;

    /* A TensorFlow process is normally unprivileged, so default the engine to
     * the vc4d backend.  vc4_init() selects its backend from this variable;
     * an explicit VC4INFER_DAEMON=0 still selects the direct /dev/mem path. */
    if (!getenv("VC4INFER_DAEMON"))
        setenv("VC4INFER_DAEMON", "1", 1);

    vc4_status st = vc4_init(&g_ctx, getenv("VC4INFER_CACHE"), g_arena_bytes);
    if (st != VC4_OK) {
        g_engine = -1;
        snprintf(g_engine_err, sizeof(g_engine_err),
                 "vc4_init(%zu MiB) failed: %s", g_arena_bytes >> 20,
                 vc4_strerror(st));
        if (g_ctx) {
            const char *detail = vc4_last_error(g_ctx);
            if (detail && *detail) {
                size_t n = strlen(g_engine_err);
                snprintf(g_engine_err + n, sizeof(g_engine_err) - n,
                         " [%s]", detail);
            }
            vc4_shutdown(g_ctx);
            g_ctx = NULL;
        }
        fprintf(stderr, "tf-vc4: %s\n", g_engine_err);
        return 0;
    }

    g_engine = 1;
    /* The daemon may serve a smaller region than requested (its own arena,
     * or the CMA).  Recompute the heap from the context's real start, which
     * vc4_init() has already checked is VC4_DATA_BASE. */
    size_t actual = vc4_arena_bytes(g_ctx);
    size_t start = vc4_arena_used(g_ctx);
    if (actual) {
        g_arena_bytes = actual;
        size_t heap = actual > start ? actual - start : 0;
        size_t limit = heap - heap / 4;
        if (limit < g_data_limit)
            g_data_limit = limit;
    }
    fprintf(stderr,
            "tf-vc4: engine up, arena %zu MiB, %zu MiB for tensors, "
            "device type %s\n", g_arena_bytes >> 20, g_data_limit >> 20,
            VC4_TF_DEVICE_TYPE);
    return 1;
}

/* ------------------------------------------------------------- allocator */

/* Caller holds g_mu. */
static int find_range_locked(const void *ptr, uint64_t bytes, vc4_buf *out)
{
    if (!ptr)
        return 0;
    for (int i = 0; i < g_nblocks; ++i) {
        const char *base = (const char *)g_blocks[i].host;
        const char *p = (const char *)ptr;
        if (p < base || p > base + g_blocks[i].bytes)
            continue;
        uint64_t delta = (uint64_t)(p - base);
        if (delta + bytes > g_blocks[i].bytes)
            continue;
        out->off = g_blocks[i].off + (uint32_t)delta;
        out->bytes = g_blocks[i].bytes - (size_t)delta;
        return 1;
    }
    return 0;
}

static void se_allocate(const SP_Device *device, uint64_t size,
                        int64_t memory_space, SP_DeviceMemoryBase *mem)
{
    (void)device;
    (void)memory_space;
    mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
    mem->opaque = NULL;
    mem->size = 0;
    mem->payload = 0;
    if (size == 0)
        return;

    size_t want = ((size_t)size + 63u) & ~(size_t)63u;

    pthread_mutex_lock(&g_mu);
    config_locked();

    if (g_in_use + want > g_data_limit) {
        pthread_mutex_unlock(&g_mu);
        return;
    }

    /* Reuse a block of ours before asking the engine for more: vc4_alloc()
     * is a bump allocator with no free, and the arena is shared with the
     * engine's per-op scratch. */
    for (int i = 0; i < g_nblocks; ++i) {
        if (g_blocks[i].used || g_blocks[i].bytes < want)
            continue;
        g_blocks[i].used = 1;
        g_in_use += g_blocks[i].bytes;
        mem->opaque = g_blocks[i].host;
        mem->size = g_blocks[i].bytes;
        mem->payload = g_blocks[i].off;
        g_allocs++;
        if ((int64_t)g_in_use > g_peak_in_use) g_peak_in_use = g_in_use;
        if ((int64_t)g_blocks[i].bytes > g_largest_alloc)
            g_largest_alloc = (int64_t)g_blocks[i].bytes;
        pthread_mutex_unlock(&g_mu);
        return;
    }

    if (g_nblocks == VC4_TF_MAX_BLOCKS) {
        pthread_mutex_unlock(&g_mu);
        fprintf(stderr, "tf-vc4: allocator block table full (%d)\\n",
                VC4_TF_MAX_BLOCKS);
        return;
    }

    block_t *b = &g_blocks[g_nblocks];
    if (g_host_arena) {
        void *p = NULL;
        if (posix_memalign(&p, 64, want) != 0) {
            pthread_mutex_unlock(&g_mu);
            return;
        }
        memset(p, 0, want);
        b->off = 0;
        b->bytes = want;
        b->host = p;
    } else {
        if (!engine_up_locked()) {
            pthread_mutex_unlock(&g_mu);
            return;
        }
        vc4_buf buf;
        if (vc4_alloc(g_ctx, want, &buf) != VC4_OK) {
            pthread_mutex_unlock(&g_mu);
            fprintf(stderr, "tf-vc4: vc4_alloc(%zu) failed: %s\\n", want,
                    vc4_last_error(g_ctx));
            return;
        }
        b->off = buf.off;
        b->bytes = want;
        b->host = vc4_host_ptr(g_ctx, &buf);
    }
    b->used = 1;
    g_nblocks++;
    g_in_use += b->bytes;
    g_allocs++;
    if ((int64_t)g_in_use > g_peak_in_use) g_peak_in_use = g_in_use;
    if ((int64_t)b->bytes > g_largest_alloc) g_largest_alloc = (int64_t)b->bytes;

    mem->opaque = b->host;
    mem->size = b->bytes;
    mem->payload = b->off;
    pthread_mutex_unlock(&g_mu);
}

static void se_deallocate(const SP_Device *device, SP_DeviceMemoryBase *mem)
{
    (void)device;
    if (!mem || !mem->opaque)
        return;

    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nblocks; ++i) {
        if (g_blocks[i].host != mem->opaque)
            continue;
        if (g_blocks[i].used) {
            g_blocks[i].used = 0;
            g_in_use = g_in_use > g_blocks[i].bytes
                           ? g_in_use - g_blocks[i].bytes
                           : 0;
        }
        break;
    }
    pthread_mutex_unlock(&g_mu);
    mem->opaque = NULL;
    mem->size = 0;
    mem->payload = 0;
}

static void *se_host_memory_allocate(const SP_Device *device, uint64_t size)
{
    (void)device;
    return size ? malloc((size_t)size) : NULL;
}

static void se_host_memory_deallocate(const SP_Device *device, void *mem)
{
    (void)device;
    free(mem);
}

/* Caller holds g_mu. */
static size_t largest_free_block_locked(void)
{
    size_t best = 0;
    for (int i = 0; i < g_nblocks; ++i)
        if (!g_blocks[i].used && g_blocks[i].bytes > best)
            best = g_blocks[i].bytes;
    if (g_data_limit > g_in_use) {
        size_t tail = g_data_limit - g_in_use;
        if (tail > best) best = tail;
    }
    return best;
}

static TF_Bool se_get_allocator_stats(const SP_Device *device,
                                      SP_AllocatorStats *stats)
{
    (void)device;
    if (!stats)
        return false;
    pthread_mutex_lock(&g_mu);
    config_locked();
    memset(stats, 0, sizeof(*stats));
    stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
    stats->num_allocs = g_allocs;
    stats->bytes_in_use = (int64_t)g_in_use;
    stats->peak_bytes_in_use = g_peak_in_use;
    stats->largest_alloc_size = g_largest_alloc;
    stats->has_bytes_limit = 1;
    stats->bytes_limit = (int64_t)g_data_limit;
    stats->bytes_reserved = (int64_t)g_in_use;
    stats->peak_bytes_reserved = g_peak_in_use;
    stats->has_bytes_reservable_limit = 1;
    stats->bytes_reservable_limit = (int64_t)g_data_limit;
    stats->largest_free_block_bytes = (int64_t)largest_free_block_locked();
    pthread_mutex_unlock(&g_mu);
    return true;
}

static TF_Bool se_device_memory_usage(const SP_Device *device, int64_t *free_b,
                                      int64_t *total_b)
{
    (void)device;
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (total_b) *total_b = (int64_t)g_data_limit;
    if (free_b) {
        *free_b = g_in_use < g_data_limit
                      ? (int64_t)(g_data_limit - g_in_use) : 0;
    }
    pthread_mutex_unlock(&g_mu);
    return true;
}

/* ------------------------------------------------------------------ copy */

static void xfer_htod(void *dst, const void *src, uint64_t size,
                      TF_Status *status)
{
    if (size == 0)
        return;
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (g_host_arena) {
        memcpy(dst, src, (size_t)size);
    } else if (!engine_up_locked()) {
        fail(status, g_engine_err);
    } else {
        vc4_buf b;
        if (!find_range_locked(dst, size, &b)) {
            fail(status, "accepted a copy into memory outside the VC arena");
        } else {
            vc4_status st = vc4_upload(g_ctx, &b, src, (size_t)size);
            if (st != VC4_OK) {
                char msg[256];
                snprintf(msg, sizeof(msg), "vc4_upload(%llu): %s",
                         (unsigned long long)size, vc4_strerror(st));
                fail(status, msg);
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
}

static void xfer_dtoh(void *dst, const void *src, uint64_t size,
                      TF_Status *status)
{
    if (size == 0)
        return;
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (g_host_arena) {
        memcpy(dst, src, (size_t)size);
    } else if (!engine_up_locked()) {
        fail(status, g_engine_err);
    } else {
        vc4_buf b;
        if (!find_range_locked(src, size, &b)) {
            fail(status, "accepted a copy out of memory outside the VC arena");
        } else {
            vc4_status st = vc4_download(g_ctx, &b, dst, (size_t)size);
            if (st != VC4_OK) {
                char msg[256];
                snprintf(msg, sizeof(msg), "vc4_download(%llu): %s",
                         (unsigned long long)size, vc4_strerror(st));
                fail(status, msg);
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
}

static void xfer_dtod(void *dst, const void *src, uint64_t size,
                      TF_Status *status)
{
    if (size == 0)
        return;
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (g_host_arena) {
        memmove(dst, src, (size_t)size);
    } else if (!engine_up_locked()) {
        fail(status, g_engine_err);
    } else {
        vc4_buf sb, db;
        if (!find_range_locked(src, size, &sb) ||
            !find_range_locked(dst, size, &db)) {
            fail(status, "device-to-device copy outside the VC arena");
        } else {
            /* The arena mapping is cacheable; a plain memcpy between two
             * offsets would read lines the GPU wrote after the last
             * download.  Round-trip through the host (the engine's
             * download does the cache maintenance). */
            void *tmp = malloc((size_t)size);
            if (!tmp) {
                fail(status, "out of host memory for a device copy");
            } else {
                vc4_status st = vc4_download(g_ctx, &sb, tmp, (size_t)size);
                if (st == VC4_OK)
                    st = vc4_upload(g_ctx, &db, tmp, (size_t)size);
                if (st != VC4_OK) {
                    char msg[256];
                    snprintf(msg, sizeof(msg), "device copy(%llu): %s",
                             (unsigned long long)size, vc4_strerror(st));
                    fail(status, msg);
                }
                free(tmp);
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
}

/* mem_zero / memset / memset32 are enqueued by TF, so they are implemented
 * on top of the upload path (which does the cache maintenance). */
static void fill_device(void *dst, const void *pattern, size_t pattern_len,
                        uint64_t size, TF_Status *status)
{
    if (size == 0)
        return;
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (g_host_arena) {
        for (uint64_t done = 0; done < size; done += pattern_len) {
            size_t n = (size - done) < pattern_len ? (size_t)(size - done)
                                                   : pattern_len;
            memcpy((char *)dst + done, pattern, n);
        }
    } else if (!engine_up_locked()) {
        fail(status, g_engine_err);
    } else {
        vc4_buf b;
        if (!find_range_locked(dst, size, &b)) {
            fail(status, "filled memory outside the VC arena");
        } else {
            enum { CHUNK = 4096 };
            char *buf = malloc(CHUNK);
            if (!buf) {
                fail(status, "out of host memory for a device fill");
            } else {
                const uint8_t *pat = pattern;
                for (size_t i = 0; i < CHUNK; ++i)
                    buf[i] = (char)pat[i % pattern_len];
                vc4_status st = VC4_OK;
                for (uint64_t done = 0; done < size && st == VC4_OK;) {
                    size_t n = (size - done) < CHUNK ? (size_t)(size - done)
                                                     : CHUNK;
                    vc4_buf cb = { b.off + (uint32_t)done,
                                   b.bytes - (size_t)done };
                    st = vc4_upload(g_ctx, &cb, buf, n);
                    done += n;
                }
                if (st != VC4_OK) {
                    char msg[256];
                    snprintf(msg, sizeof(msg), "device fill(%llu): %s",
                             (unsigned long long)size, vc4_strerror(st));
                    fail(status, msg);
                }
                free(buf);
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
}

static void se_memcpy_dtoh(const SP_Device *device, SP_Stream stream,
                           void *host_dst, const SP_DeviceMemoryBase *src,
                           uint64_t size, TF_Status *status)
{
    (void)device;
    (void)stream;
    xfer_dtoh(host_dst, src ? src->opaque : NULL, size, status);
}

static void se_memcpy_htod(const SP_Device *device, SP_Stream stream,
                           SP_DeviceMemoryBase *dst, const void *host_src,
                           uint64_t size, TF_Status *status)
{
    (void)device;
    (void)stream;
    xfer_htod(dst ? dst->opaque : NULL, host_src, size, status);
}

static void se_memcpy_dtod(const SP_Device *device, SP_Stream stream,
                           SP_DeviceMemoryBase *dst,
                           const SP_DeviceMemoryBase *src, uint64_t size,
                           TF_Status *status)
{
    (void)device;
    (void)stream;
    xfer_dtod(dst ? dst->opaque : NULL, src ? src->opaque : NULL, size,
              status);
}

static void se_sync_memcpy_dtoh(const SP_Device *device, void *host_dst,
                                const SP_DeviceMemoryBase *src, uint64_t size,
                                TF_Status *status)
{
    se_memcpy_dtoh(device, NULL, host_dst, src, size, status);
}

static void se_sync_memcpy_htod(const SP_Device *device,
                                SP_DeviceMemoryBase *dst, const void *host_src,
                                uint64_t size, TF_Status *status)
{
    se_memcpy_htod(device, NULL, dst, host_src, size, status);
}

static void se_sync_memcpy_dtod(const SP_Device *device,
                                SP_DeviceMemoryBase *dst,
                                const SP_DeviceMemoryBase *src, uint64_t size,
                                TF_Status *status)
{
    se_memcpy_dtod(device, NULL, dst, src, size, status);
}

static void se_mem_zero(const SP_Device *device, SP_Stream stream,
                        SP_DeviceMemoryBase *location, uint64_t size,
                        TF_Status *status)
{
    (void)device;
    (void)stream;
    const uint32_t zero = 0;
    fill_device(location ? location->opaque : NULL, &zero, sizeof(zero), size,
                status);
}

static void se_memset(const SP_Device *device, SP_Stream stream,
                      SP_DeviceMemoryBase *location, uint8_t pattern,
                      uint64_t size, TF_Status *status)
{
    (void)device;
    (void)stream;
    fill_device(location ? location->opaque : NULL, &pattern,
                sizeof(pattern), size, status);
}

static void se_memset32(const SP_Device *device, SP_Stream stream,
                        SP_DeviceMemoryBase *location, uint32_t pattern,
                        uint64_t size, TF_Status *status)
{
    (void)device;
    (void)stream;
    fill_device(location ? location->opaque : NULL, &pattern,
                sizeof(pattern), size, status);
}

/* --------------------------------------------------------- streams, etc. */

struct SP_Stream_st {
    uint64_t id;
};

struct SP_Event_st {
    int done;
};

struct SP_Timer_st {
    uint64_t start_ns;
    uint64_t stop_ns;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void se_create_stream(const SP_Device *device, SP_Stream *stream,
                             TF_Status *status)
{
    (void)device;
    static uint64_t next_id;
    struct SP_Stream_st *s = calloc(1, sizeof(*s));
    if (!s) {
        fail(status, "out of host memory for a VC stream");
        return;
    }
    s->id = ++next_id;
    *stream = s;
}

static void se_destroy_stream(const SP_Device *device, SP_Stream stream)
{
    (void)device;
    free(stream);
}

static void se_create_stream_dependency(const SP_Device *device,
                                        SP_Stream dependent, SP_Stream other,
                                        TF_Status *status)
{
    (void)device;
    (void)dependent;
    (void)other;
    (void)status;
}

static void se_get_stream_status(const SP_Device *device, SP_Stream stream,
                                 TF_Status *status)
{
    (void)device;
    (void)stream;
    (void)status;
}

static void se_create_event(const SP_Device *device, SP_Event *event,
                            TF_Status *status)
{
    (void)device;
    struct SP_Event_st *e = calloc(1, sizeof(*e));
    if (!e) {
        fail(status, "out of host memory for a VC event");
        return;
    }
    e->done = 1;   /* work completes inside the enqueue call */
    *event = e;
}

static void se_destroy_event(const SP_Device *device, SP_Event event)
{
    (void)device;
    free(event);
}

static SE_EventStatus se_get_event_status(const SP_Device *device,
                                          SP_Event event)
{
    (void)device;
    return event && event->done ? SE_EVENT_COMPLETE : SE_EVENT_PENDING;
}

static void se_record_event(const SP_Device *device, SP_Stream stream,
                            SP_Event event, TF_Status *status)
{
    (void)device;
    (void)stream;
    (void)status;
    if (event)
        event->done = 1;
}

static void se_wait_for_event(const SP_Device *const device, SP_Stream stream,
                              SP_Event event, TF_Status *const status)
{
    (void)device;
    (void)stream;
    (void)event;
    (void)status;
}

static void se_block_host_for_event(const SP_Device *device, SP_Event event,
                                    TF_Status *status)
{
    (void)device;
    (void)status;
    if (event)
        event->done = 1;
}

static void se_create_timer(const SP_Device *device, SP_Timer *timer,
                            TF_Status *status)
{
    (void)device;
    struct SP_Timer_st *t = calloc(1, sizeof(*t));
    if (!t) {
        fail(status, "out of host memory for a VC timer");
        return;
    }
    *timer = t;
}

static void se_destroy_timer(const SP_Device *device, SP_Timer timer)
{
    (void)device;
    free(timer);
}

static void se_start_timer(const SP_Device *device, SP_Stream stream,
                           SP_Timer timer, TF_Status *status)
{
    (void)device;
    (void)stream;
    (void)status;
    if (timer)
        timer->start_ns = now_ns();
}

static void se_stop_timer(const SP_Device *device, SP_Stream stream,
                          SP_Timer timer, TF_Status *status)
{
    (void)device;
    (void)stream;
    (void)status;
    if (timer)
        timer->stop_ns = now_ns();
}

static uint64_t timer_nanoseconds(SP_Timer timer)
{
    if (!timer || timer->stop_ns < timer->start_ns)
        return 0;
    return timer->stop_ns - timer->start_ns;
}

static void se_synchronize_all_activity(const SP_Device *device,
                                        TF_Status *status)
{
    (void)device;
    (void)status;
}

static TF_Bool se_host_callback(const SP_Device *device, SP_Stream stream,
                                SE_StatusCallbackFn fn, void *arg)
{
    (void)device;
    (void)stream;
    if (!fn)
        return false;
    TF_Status *status = TF_NewStatus();
    fn(arg, status);
    TF_DeleteStatus(status);
    return true;
}

/* ------------------------------------------------------ platform, device */

static void se_get_device_count(const SP_Platform *platform, int *count,
                                TF_Status *status)
{
    (void)platform;
    (void)status;
    *count = 1;
}

static void se_create_device(const SP_Platform *platform,
                             SE_CreateDeviceParams *params, TF_Status *status)
{
    (void)platform;
    (void)status;
    SP_Device *d = params->device;
    d->struct_size = SP_DEVICE_STRUCT_SIZE;
    d->ext = NULL;
    d->ordinal = params->ordinal;
    d->device_handle = NULL;
    d->hardware_name = "VideoCore IV QPU";
    d->device_vendor = "Broadcom";
    d->pci_bus_id = "";
}

static void se_destroy_device(const SP_Platform *platform, SP_Device *device)
{
    (void)platform;
    (void)device;
}

static void se_create_device_fns(const SP_Platform *platform,
                                 SE_CreateDeviceFnsParams *params,
                                 TF_Status *status)
{
    (void)platform;
    (void)status;
    SP_DeviceFns *fns = params->device_fns;
    fns->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
    fns->get_numa_node = NULL;
    fns->get_memory_bandwidth = NULL;
    fns->get_gflops = NULL;
}

static void se_destroy_device_fns(const SP_Platform *platform,
                                  SP_DeviceFns *fns)
{
    (void)platform;
    (void)fns;
}

static void se_create_stream_executor(const SP_Platform *platform,
                                      SE_CreateStreamExecutorParams *params,
                                      TF_Status *status)
{
    (void)platform;
    (void)status;
    SP_StreamExecutor *se = params->stream_executor;
    se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;

    se->allocate = se_allocate;
    se->deallocate = se_deallocate;
    se->host_memory_allocate = se_host_memory_allocate;
    se->host_memory_deallocate = se_host_memory_deallocate;
    se->unified_memory_allocate = NULL;
    se->unified_memory_deallocate = NULL;
    se->get_allocator_stats = se_get_allocator_stats;
    se->device_memory_usage = se_device_memory_usage;

    se->create_stream = se_create_stream;
    se->destroy_stream = se_destroy_stream;
    se->create_stream_dependency = se_create_stream_dependency;
    se->get_stream_status = se_get_stream_status;

    se->create_event = se_create_event;
    se->destroy_event = se_destroy_event;
    se->get_event_status = se_get_event_status;
    se->record_event = se_record_event;
    se->wait_for_event = se_wait_for_event;

    se->create_timer = se_create_timer;
    se->destroy_timer = se_destroy_timer;
    se->start_timer = se_start_timer;
    se->stop_timer = se_stop_timer;

    se->memcpy_dtoh = se_memcpy_dtoh;
    se->memcpy_htod = se_memcpy_htod;
    se->memcpy_dtod = se_memcpy_dtod;
    se->sync_memcpy_dtoh = se_sync_memcpy_dtoh;
    se->sync_memcpy_htod = se_sync_memcpy_htod;
    se->sync_memcpy_dtod = se_sync_memcpy_dtod;
    se->block_host_for_event = se_block_host_for_event;
    se->block_host_until_done = NULL;
    se->synchronize_all_activity = se_synchronize_all_activity;

    se->mem_zero = se_mem_zero;
    se->memset = se_memset;
    se->memset32 = se_memset32;
    se->host_callback = se_host_callback;
}

static void se_destroy_stream_executor(const SP_Platform *platform,
                                       SP_StreamExecutor *se)
{
    (void)platform;
    (void)se;
}

static void se_create_timer_fns(const SP_Platform *platform, SP_TimerFns *fns,
                                TF_Status *status)
{
    (void)platform;
    (void)status;
    fns->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
    fns->nanoseconds = timer_nanoseconds;
}

static void se_destroy_timer_fns(const SP_Platform *platform, SP_TimerFns *fns)
{
    (void)platform;
    (void)fns;
}

static void se_destroy_platform(SP_Platform *platform)
{
    (void)platform;
    pthread_mutex_lock(&g_mu);
    if (g_ctx) {
        vc4_shutdown(g_ctx);
        g_ctx = NULL;
        g_engine = 0;
    }
    pthread_mutex_unlock(&g_mu);
}

static void se_destroy_platform_fns(SP_PlatformFns *fns)
{
    (void)fns;
}

void SE_InitPlugin(SE_PlatformRegistrationParams *params, TF_Status *status)
{
    (void)status;
    SP_Platform *platform = params->platform;
    SP_PlatformFns *fns = params->platform_fns;

    platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
    platform->ext = NULL;
    platform->name = VC4_TF_DEVICE_TYPE;
    platform->type = VC4_TF_DEVICE_TYPE;
    platform->supports_unified_memory = 0;
    platform->use_bfc_allocator = 1;
    platform->force_memory_growth = 0;

    fns->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
    fns->ext = NULL;
    fns->get_device_count = se_get_device_count;
    fns->create_device = se_create_device;
    fns->destroy_device = se_destroy_device;
    fns->create_device_fns = se_create_device_fns;
    fns->destroy_device_fns = se_destroy_device_fns;
    fns->create_stream_executor = se_create_stream_executor;
    fns->destroy_stream_executor = se_destroy_stream_executor;
    fns->create_timer_fns = se_create_timer_fns;
    fns->destroy_timer_fns = se_destroy_timer_fns;

    params->destroy_platform = se_destroy_platform;
    params->destroy_platform_fns = se_destroy_platform_fns;
}

/* ---------------------------------------------------------------- kernels */
/*
 * One path for every kernel: fetch the inputs, refuse anything the engine
 * cannot represent, allocate the output, translate the tensor pointers into
 * arena offsets, and call the engine under the single mutex.
 *
 * No kernel computes on the host.  The arena mapping *is* the device memory,
 * so a tensor pointer either resolves to an arena range - and then the QPUs do
 * the arithmetic - or the kernel refuses: a pointer outside the arena, the
 * host-arena diagnostic, and an engine that did not come up are all named
 * refusals, never a quiet CPU result.
 *
 * The elementwise and unary ops share one table.  TensorFlow's C kernel API
 * gives create_func no cookie (TF_NewKernelBuilder takes only the function
 * pointers), so each op gets its own construction callback, expanded from the
 * table below; the descriptor that callback returns is compute's `user`.
 */

typedef enum {
    VC4_TF_EW = 0,      /* vc4_elementwise: same-shape binary or unary  */
    VC4_TF_UOP          /* vc4_unary: one input, polynomial body        */
} vc4_tf_sel;

typedef struct {
    const char *op;     /* TensorFlow op name, used in every refusal     */
    int         inputs; /* 1 or 2                                        */
    vc4_tf_sel  sel;
    vc4_op      ew;     /* sel == VC4_TF_EW: elementwise selector        */
    vc4_uop     uop;    /* sel == VC4_TF_UOP: unary selector             */
    float       p0, p1; /* unary attributes (CLIP's low and high bound)  */
} vc4_tf_ew;

/*
 * Add/AddV2, Mul, Sub, Div/RealDiv, Maximum and Minimum take two same-shape
 * inputs and one selector each.  Relu and Sigmoid use the dedicated
 * elementwise bodies (exact select / degree-5 2^f); the rest are the verified
 * unary bodies (docs/ACCURACY.md): Neg, Clip (Relu6) bit-exact, Tanh 1.2e-7
 * absolute, Log 4.8e-7, Sqrt 1.1e-7 and Exp 5.7e-6 relative.
 *
 * RealDiv is registered because that is what tf.math.divide lowers to; Div
 * stays for graphs built with the raw op.  The elementwise selector column is
 * 0 (unused) on the VC4_TF_UOP rows.
 */
#define VC4_TF_EW_KERNELS(X)                                                  \
    X(Add,     2, VC4_TF_EW,  VC4_OP_ADD,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(AddV2,   2, VC4_TF_EW,  VC4_OP_ADD,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Mul,     2, VC4_TF_EW,  VC4_OP_MUL,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Sub,     2, VC4_TF_EW,  VC4_OP_SUB,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Div,     2, VC4_TF_EW,  VC4_OP_DIV,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(RealDiv, 2, VC4_TF_EW,  VC4_OP_DIV,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Maximum, 2, VC4_TF_EW,  VC4_OP_MAX,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Minimum, 2, VC4_TF_EW,  VC4_OP_MIN,     VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Relu,    1, VC4_TF_EW,  VC4_OP_RELU,    VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Sigmoid, 1, VC4_TF_EW,  VC4_OP_SIGMOID, VC4_UOP_IDENTITY, 0.0f, 0.0f)   \
    X(Relu6,   1, VC4_TF_UOP, 0,              VC4_UOP_CLIP,     0.0f, 6.0f)   \
    X(Tanh,    1, VC4_TF_UOP, 0,              VC4_UOP_TANH,     0.0f, 0.0f)   \
    X(Neg,     1, VC4_TF_UOP, 0,              VC4_UOP_NEG,      0.0f, 0.0f)   \
    X(Exp,     1, VC4_TF_UOP, 0,              VC4_UOP_EXP,      0.0f, 0.0f)   \
    X(Log,     1, VC4_TF_UOP, 0,              VC4_UOP_LOG,      0.0f, 0.0f)   \
    X(Sqrt,    1, VC4_TF_UOP, 0,              VC4_UOP_SQRT,     0.0f, 0.0f)

typedef struct {
    const char *op;
    void *(*create)(TF_OpKernelConstruction *);
} vc4_tf_reg;

#define VC4_TF_EW_DEFINE(N, NIN, SEL, EW, UOP, P0, P1)                        \
    static void *vc4_##N##_create(TF_OpKernelConstruction *c)                 \
    {                                                                         \
        static const vc4_tf_ew d = { #N, NIN, SEL, EW, UOP, P0, P1 };         \
        (void)c;    /* no attributes: the table already holds the op */       \
        return (void *)&d;                                                    \
    }                                                                         \
    static const vc4_tf_reg vc4_reg_##N = { #N, vc4_##N##_create };

VC4_TF_EW_KERNELS(VC4_TF_EW_DEFINE)

#define VC4_TF_EW_ENTRY(N, ...) vc4_reg_##N,
static const vc4_tf_reg g_ew_regs[] = { VC4_TF_EW_KERNELS(VC4_TF_EW_ENTRY) };
#undef VC4_TF_EW_ENTRY
#undef VC4_TF_EW_DEFINE

static void refuse(TF_Status *status, TF_Code code, const char *fmt, ...)
{
    char msg[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    TF_SetStatus(status, code, msg);
}

/* "[2,3]" for the refusal text; a shape never needs more than 8 dims here. */
static void shape_text(TF_Tensor *t, char *buf, size_t len)
{
    size_t used = 0;
    int nd = TF_NumDims(t);
    if (len < 8) {
        if (len)
            buf[0] = 0;
        return;
    }
    buf[used++] = '[';
    for (int i = 0; i < nd; ++i) {
        int w = snprintf(buf + used, len - used - 1, i ? ",%lld" : "%lld",
                         (long long)TF_Dim(t, i));
        if (w < 0 || (size_t)w >= len - used - 1) {
            used = len - 2;
            break;
        }
        used += (size_t)w;
    }
    buf[used++] = ']';
    buf[used] = 0;
}

static int same_shape(TF_Tensor *a, TF_Tensor *b)
{
    if (TF_NumDims(a) != TF_NumDims(b))
        return 0;
    for (int i = 0; i < TF_NumDims(a); ++i)
        if (TF_Dim(a, i) != TF_Dim(b, i))
            return 0;
    return 1;
}

/* Lock the engine and require the arena backend.  Returns 1 with g_mu held;
 * on failure the status is set and the mutex is not held. */
static int engine_enter(const char *op, TF_Status *status)
{
    pthread_mutex_lock(&g_mu);
    config_locked();
    if (g_host_arena) {
        refuse(status, TF_FAILED_PRECONDITION,
               "tf-vc4 %s needs the engine arena; VC4_TF_ARENA=host is a "
               "copy-path diagnostic only", op);
        pthread_mutex_unlock(&g_mu);
        return 0;
    }
    if (!engine_up_locked()) {
        TF_SetStatus(status, TF_INTERNAL, g_engine_err);
        pthread_mutex_unlock(&g_mu);
        return 0;
    }
    return 1;
}

/* Caller holds g_mu.  A tensor that is not inside one of the allocator's
 * arena blocks was placed by something else (a host tensor, a host-memory
 * argument, a stale pointer) and may not be read by the QPUs. */
static int arena_range(const char *op, const void *p, size_t bytes,
                       vc4_buf *out, TF_Status *status)
{
    if (find_range_locked(p, (uint64_t)bytes, out))
        return 1;
    refuse(status, TF_INTERNAL,
           "tf-vc4 %s: a tensor pointer is outside the arena (was it placed "
           "on the host?)", op);
    return 0;
}

/* Fetch the inputs, refuse what the engine cannot represent, and allocate an
 * output with the first input's shape.  Returns 0 with the status set. */
static int ew_check(const vc4_tf_ew *d, TF_OpKernelContext *ctx,
                    TF_Tensor *in[2], TF_Tensor **out, int64_t *n,
                    TF_Status *status)
{
    for (int i = 0; i < d->inputs; ++i) {
        TF_GetInput(ctx, i, &in[i], status);
        if (TF_GetCode(status) != TF_OK)
            return 0;
    }

    for (int i = 0; i < d->inputs; ++i) {
        if (TF_TensorType(in[i]) != TF_FLOAT) {
            refuse(status, TF_INVALID_ARGUMENT,
                   "tf-vc4 %s: only float32 is registered", d->op);
            return 0;
        }
    }

    int nd = TF_NumDims(in[0]);
    if (nd > 8) {
        refuse(status, TF_INVALID_ARGUMENT,
               "tf-vc4 %s: rank %d is over the kernel's limit of 8", d->op,
               nd);
        return 0;
    }
    int64_t dims[8];
    for (int i = 0; i < nd; ++i)
        dims[i] = TF_Dim(in[0], i);

    if (d->inputs == 2 && !same_shape(in[0], in[1])) {
        char sa[64], sb[64];
        shape_text(in[0], sa, sizeof(sa));
        shape_text(in[1], sb, sizeof(sb));
        refuse(status, TF_INVALID_ARGUMENT,
               "tf-vc4 %s: inputs are not the same shape (%s and %s); "
               "broadcasting is not implemented", d->op, sa, sb);
        return 0;
    }

    *n = TF_TensorElementCount(in[0]);
    *out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, (size_t)*n * 4,
                             status);
    return *out && TF_GetCode(status) == TF_OK;
}

/* Translate the tensors and run the selector.  Caller does not hold g_mu. */
static int ew_run(const vc4_tf_ew *d, const void *pa, const void *pb, void *po,
                  size_t n, TF_Status *status)
{
    if (!engine_enter(d->op, status))
        return 0;

    vc4_buf ba, bb, bo;
    int ok = arena_range(d->op, pa, n * 4, &ba, status);
    if (ok && pb)
        ok = arena_range(d->op, pb, n * 4, &bb, status);
    if (ok)
        ok = arena_range(d->op, po, n * 4, &bo, status);

    vc4_status st = VC4_OK;
    if (ok) {
        if (d->sel == VC4_TF_EW) {
            st = vc4_elementwise(g_ctx, d->ew, &ba, pb ? &bb : NULL, &bo, n);
        } else {
            st = vc4_unary(g_ctx, d->uop, &ba, &bo, n, d->p0, d->p1, 0.0f);
        }
    }
    pthread_mutex_unlock(&g_mu);

    if (!ok)
        return 0;
    if (st != VC4_OK) {
        /* The ewb-backed selectors (sub, div, max, min) need an engine built
         * from this tree: an older libvc4infer refuses them as unimplemented
         * even though the QPU bodies are in its generator. */
        refuse(status, TF_INTERNAL,
               "tf-vc4 %s: %s failed on %zu elements: %s%s", d->op,
               d->sel == VC4_TF_EW ? vc4_op_name(d->ew) : "the unary kernel",
               n, vc4_strerror(st),
               st == VC4_ERR_UNSUPPORTED
                   ? " (this selector needs the libvc4infer from this tree)"
                   : "");
        return 0;
    }
    return 1;
}

static void ew_compute(void *user, TF_OpKernelContext *ctx)
{
    const vc4_tf_ew *d = user;
    TF_Status *status = TF_NewStatus();
    TF_Tensor *in[2] = { NULL, NULL };
    TF_Tensor *out = NULL;
    int64_t n = 0;

    if (!d) {
        TF_SetStatus(status, TF_INTERNAL,
                     "tf-vc4: the kernel descriptor is missing");
        goto finished;
    }
    if (!ew_check(d, ctx, in, &out, &n, status))
        goto finished;
    if (n == 0)             /* an empty tensor is a legal op with no work */
        goto finished;
    ew_run(d, TF_TensorData(in[0]),
           d->inputs == 2 ? TF_TensorData(in[1]) : NULL, TF_TensorData(out),
           (size_t)n, status);

finished:
    if (TF_GetCode(status) != TF_OK)
        TF_OpKernelContext_Failure(ctx, status);
    /* TF_AllocateOutput hands the caller a TF_Tensor that holds a reference to
     * the output buffer; leaving it unreleased keeps the arena block alive
     * after the graph has finished with the tensor. */
    if (out)
        TF_DeleteTensor(out);
    if (in[0])
        TF_DeleteTensor(in[0]);
    if (in[1])
        TF_DeleteTensor(in[1]);
    TF_DeleteStatus(status);
}

/* TypeConstraint("T", float) is the registry's dtype gate: a tensor of any
 * other type cannot select one of these kernels.  No kernel declares a host
 * memory argument (TF_KernelBuilder_HostMemory): every argument of these ops
 * is arithmetic data and belongs in the arena.  If TensorFlow ever pins one to
 * host memory the pointer would not resolve and the kernel would refuse. */
static void register_kernel(const char *op,
                            void *(*create)(TF_OpKernelConstruction *),
                            void (*compute)(void *, TF_OpKernelContext *),
                            TF_Status *status)
{
    TF_KernelBuilder *builder = TF_NewKernelBuilder(
        op, VC4_TF_DEVICE_TYPE, create, compute, NULL);
    TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "tf-vc4: type constraint for %s: %s\n", op,
                TF_Message(status));
        TF_DeleteKernelBuilder(builder);
        TF_SetStatus(status, TF_OK, "");
        return;
    }
    TF_RegisterKernelBuilder(op, builder, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "tf-vc4: registering %s on %s: %s\n", op,
                VC4_TF_DEVICE_TYPE, TF_Message(status));
        TF_SetStatus(status, TF_OK, "");
    }
}

void TF_InitKernel(void)
{
    TF_Status *status = TF_NewStatus();

    for (size_t i = 0; i < sizeof(g_ew_regs) / sizeof(g_ew_regs[0]); ++i)
        register_kernel(g_ew_regs[i].op, g_ew_regs[i].create, ew_compute,
                        status);

    TF_DeleteStatus(status);
}

/* ----------------------------------------------------------- diagnostics */
/*
 * Read by tests through ctypes.  None of these touches the engine, so a
 * process that has not placed anything on the device can be inspected
 * without allocating the arena.
 */
const char *vc4_tf_device_type(void) { return VC4_TF_DEVICE_TYPE; }

/* 0 = engine not needed yet, 1 = up, -1 = init failed. */
int vc4_tf_engine_state(void) { return g_engine; }

const char *vc4_tf_last_error(void)
{
    return g_engine_err[0] ? g_engine_err : "";
}

/* "arena MiB for tensors, MiB in use, host|engine" for the log line. */
const char *vc4_tf_arena_status(void)
{
    static char buf[640];
    pthread_mutex_lock(&g_mu);
    config_locked();
    snprintf(buf, sizeof(buf),
             "arena %zu MiB, %zu MiB of data heap, %zu MiB for tensors, "
             "%zu MiB in use, %s%s%s",
             g_arena_bytes >> 20,
             (g_host_arena ? g_arena_bytes
                           : (g_arena_bytes > VC4_TF_DATA_BASE
                                  ? g_arena_bytes - VC4_TF_DATA_BASE : 0)) >> 20,
             g_data_limit >> 20, g_in_use >> 20,
             g_host_arena ? "host-backed (diagnostic)" : "engine arena",
             g_engine == -1 ? ", engine down: " : "",
             g_engine == -1 ? g_engine_err : "");
    pthread_mutex_unlock(&g_mu);
    return buf;
}
