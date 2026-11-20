/*
 * vc4d_arena.c - the dma-buf/CMA arena.
 *
 * This is the part that needs CAP_SYS_RAWIO and is therefore the reason vc4d
 * exists.  It does three things:
 *
 *   1. allocate one contiguous CMA dma-buf,
 *   2. find its physical base so the QPU can be given bus addresses
 *      (dma-buf CPU mappings are VM_PFNMAP, so pagemap has no PFN; instead we
 *      stamp a per-run nonce into the first 2 MiB of the mapping and look for
 *      it with 4 KiB reads of /dev/mem at 1 MiB steps - one read per MiB of
 *      RAM, not a full RAM scan),
 *   3. map it uncached through /dev/mem and hand out page-aligned regions.
 *
 * After arena_open() the process only touches `map`; the /dev/mem and
 * /dev/dma_heap descriptors are closed.
 */
#define _GNU_SOURCE
#include "vc4d.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

#define SCAN_PAGES 512u                  /* 2 MiB of nonce-stamped pages */
#define SCAN_LIMIT 0x3F000000u           /* below the ARM peripheral window */
#define TAG_MAGIC  0x48595034u           /* "4PYH" */

struct tag {
    uint32_t magic;
    uint32_t nonce;
    uint32_t index;
    uint32_t inv;
};

static void page_align_cache(void *addr, size_t bytes)
{
#if defined(__aarch64__)
    uintptr_t line = (uintptr_t)addr & ~63ul;
    for (; line < (uintptr_t)addr + bytes; line += 64)
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)addr; (void)bytes;
#endif
}
static uint32_t run_nonce(void)
{
    uint32_t v = 0;
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        if (read(fd, &v, sizeof(v)) != (ssize_t)sizeof(v))
            v = 0;
        close(fd);
    }
    if (!v) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        v = (uint32_t)(ts.tv_nsec ^ (uintptr_t)&v ^ (uint32_t)getpid()) | 1u;
    }
    return v;
}

/* Stamp the nonce into the first 2 MiB and look for it through /dev/mem. */
static int find_phys(uint8_t *map, size_t len, uint32_t *out, char *err, size_t errn)
{
    if (len < (size_t)SCAN_PAGES * 4096u) {
        snprintf(err, errn, "arena too small for physical discovery");
        return -1;
    }
    const uint32_t nonce = run_nonce();
    for (uint32_t i = 0; i < SCAN_PAGES; ++i) {
        struct tag t = { TAG_MAGIC, nonce, i, ~i };
        memcpy(map + (size_t)i * 4096u, &t, sizeof(t));
    }
    page_align_cache(map, (size_t)SCAN_PAGES * 4096u);

    int mem = open("/dev/mem", O_RDONLY | O_SYNC | O_CLOEXEC);
    if (mem < 0) {
        snprintf(err, errn, "open /dev/mem: %s", strerror(errno));
        return -1;
    }
    uint8_t *buf = malloc(4096);
    if (!buf) {
        close(mem);
        snprintf(err, errn, "out of memory");
        return -1;
    }
    int rc = -1;
    for (uint32_t base = 0; base < SCAN_LIMIT; base += (1u << 20)) {
        ssize_t got = pread(mem, buf, 4096, (off_t)base);
        if (got != 4096)
            continue;
        struct tag t;
        memcpy(&t, buf, sizeof(t));
        if (t.magic != TAG_MAGIC || t.nonce != nonce || t.index >= SCAN_PAGES ||
            t.inv != ~t.index)
            continue;
        uint32_t p = base - t.index * 4096u;
        /* one confirmation read at the claimed base */
        struct tag v;
        if (pread(mem, &v, sizeof(v), (off_t)p) == (ssize_t)sizeof(v) &&
            v.magic == TAG_MAGIC && v.nonce == nonce && v.index == 0 && v.inv == ~0u) {
            *out = p;
            rc = 0;
        }
        break;
    }
    free(buf);
    close(mem);
    if (rc != 0)
        snprintf(err, errn, "dma-buf not found in physical RAM (%zu MiB)", len >> 20);
    return rc;
}

/* ---- page bitmap ------------------------------------------------------- */
static inline int page_used(const struct arena *a, uint32_t p)
{
    return (a->bits[p >> 5] >> (p & 31)) & 1u;
}

static inline void page_set(struct arena *a, uint32_t p)
{
    a->bits[p >> 5] |= 1u << (p & 31);
}

static inline int page_clear(struct arena *a, uint32_t p)
{
    uint32_t *w = &a->bits[p >> 5];
    uint32_t m = 1u << (p & 31);
    int was = (*w & m) != 0;
    *w &= ~m;
    return was;
}

/* Confirm the discovered physical base.  The tag scan that found it already
 * proved it, but this second check uses a different address, a different read
 * length and a different pattern, so an alignment/aliasing mistake in the scan
 * cannot pass.  A wrong base would point QPU DMA at arbitrary memory, so this
 * is fatal rather than a warning. */
static int verify_phys(int mem, uint8_t *unc, uint32_t phys, size_t len,
                       char *err, size_t errn)
{
    static const uint32_t offs[] = { 0x3000, 0x123000, 0x1f0000 };
    uint8_t pat[32];
    uint8_t page[4096];
    for (size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) {
        uint32_t off = offs[i];
        if ((size_t)off + sizeof(pat) > len)
            continue;
        for (size_t k = 0; k < sizeof(pat); ++k)
            pat[k] = (uint8_t)(0x5Au ^ (off >> 12) ^ (k * 7u) ^ 0x3C);
        memcpy(unc + off, pat, sizeof(pat));
        __sync_synchronize();
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#endif
        if (pread(mem, page, sizeof(page), (off_t)(phys + (off & ~4095u)))
            != (ssize_t)sizeof(page) ||
            memcmp(page + (off & 4095u), pat, sizeof(pat)) != 0) {
            snprintf(err, errn, "arena verify failed at phys 0x%08x+0x%x "
                     "(wrong physical base; refusing to run QPU DMA)",
                     phys, off);
            return -1;
        }
    }
    return 0;
}

void arena_cma_info(size_t *total, size_t *free_bytes)
{
    if (total)
        *total = 0;
    if (free_bytes)
        *free_bytes = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned long kb;
        if (sscanf(line, "CmaTotal: %lu kB", &kb) == 1) {
            if (total)
                *total = (size_t)kb << 10;
        } else if (sscanf(line, "CmaFree: %lu kB", &kb) == 1) {
            if (free_bytes)
                *free_bytes = (size_t)kb << 10;
        }
    }
    fclose(f);
}

/* One attempt at exactly `len`.  On failure nothing is left behind: the
 * dma-buf is closed (which returns the CMA reservation immediately) and every
 * mapping is unmapped before the caller retries or falls back. */
static int arena_try(struct arena *a, size_t len, char *err, size_t errn)
{
    int heap = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap < 0) {
        snprintf(err, errn, "open /dev/dma_heap/linux,cma: %s", strerror(errno));
        return -1;
    }
    struct dma_heap_allocation_data d;
    memset(&d, 0, sizeof(d));
    d.len = len;
    d.fd_flags = O_RDWR | O_CLOEXEC;
    int rc = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &d);
    close(heap);
    if (rc != 0) {
        size_t total = 0, cfree = 0;
        arena_cma_info(&total, &cfree);
        snprintf(err, errn, "%zu MiB: cma_heap_allocate failed (%s; CmaFree %zu MiB "
                 "of %zu MiB)", len >> 20, strerror(errno), cfree >> 20, total >> 20);
        return -1;
    }

    uint8_t *cached = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, (int)d.fd, 0);
    if (cached == MAP_FAILED) {
        snprintf(err, errn, "%zu MiB: mmap: %s", len >> 20, strerror(errno));
        close((int)d.fd);
        return -1;
    }
    uint32_t phys = 0;
    int fr = find_phys(cached, len, &phys, err, errn);
    munmap(cached, len);
    if (fr != 0) {
        close((int)d.fd);
        return -1;
    }

    int mem = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (mem < 0) {
        snprintf(err, errn, "open /dev/mem: %s", strerror(errno));
        close((int)d.fd);
        return -1;
    }
    void *unc = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)phys);
    if (unc == MAP_FAILED) {
        snprintf(err, errn, "mmap arena at 0x%08x: %s", phys, strerror(errno));
        close(mem);
        close((int)d.fd);
        return -1;
    }
    if (verify_phys(mem, unc, phys, len, err, errn) != 0) {
        munmap(unc, len);
        close(mem);
        close((int)d.fd);
        return -1;
    }
    close(mem);

    a->fd = (int)d.fd;
    a->map = unc;
    a->phys = phys;
    a->bus = phys | 0xC0000000u;
    a->bytes = len;
    memset(a->map, 0, A_DATA_BASE);       /* header, code and uni areas */
    a->npages = (uint32_t)((len - A_DATA_BASE) / ARENA_PAGE);
    a->bits = calloc((a->npages + 31u) / 32u, sizeof(uint32_t));
    if (!a->bits) {
        munmap(unc, len);
        close((int)d.fd);
        a->fd = -1;
        a->map = NULL;
        snprintf(err, errn, "out of memory for a %u-page arena bitmap",
                 a->npages);
        return -1;
    }
    a->hint = 0;
    a->used = 0;
    return 0;
}

/*
 * Open the arena at `want`, and only shrink if the full block cannot be had.
 *
 * The old code went straight from one failed allocation to half the size (and
 * then a quarter, an eighth) and kept the result for the daemon's lifetime, so
 * a transient block of CMA pressure - another client exiting, the DRM driver
 * holding scanout buffers for a moment - silently halved the arena until the
 * next reboot (G2).  Now:
 *
 *   1. the requested size is retried three times over ~600 ms, each attempt
 *      closing its dma-buf so the CMA reservation is dropped before the retry;
 *   2. only then does the ladder try half/quarter/eighth, and that is logged
 *      loudly with the sizes and the current CmaFree, because it changes what
 *      a client can run;
 *   3. A.degraded records it so the daemon can restore the full block once CMA
 *      has recovered and no client is using the arena (see main()).
 */
int arena_open(struct arena *a, size_t want, char *err, size_t errn)
{
    memset(a, 0, sizeof(*a));
    a->fd = -1;
    *err = 0;
    if (want < ARENA_MIN)
        want = ARENA_MIN;
    want &= ~4095ul;
    a->want = want;

    static const unsigned retry_ms[] = { 0, 200, 400 };
    char last[256] = "";
    for (size_t i = 0; i < sizeof(retry_ms) / sizeof(retry_ms[0]); ++i) {
        if (retry_ms[i])
            usleep(retry_ms[i] * 1000u);
        char e2[256] = "";
        if (arena_try(a, want, e2, sizeof(e2)) == 0) {
            a->degraded = 0;
            if (i)
                fprintf(stderr, "vc4d: arena: full %zu MiB obtained on attempt "
                        "%zu/%zu (CMA was released in between)\n", want >> 20,
                        i + 1, sizeof(retry_ms) / sizeof(retry_ms[0]));
            return 0;
        }
        /* Every failed attempt is worth a line: it is the difference between
         * "the board is full" and "a client was exiting at the wrong moment". */
        fprintf(stderr, "vc4d: arena attempt %zu/%zu for %zu MiB: %s\n",
                i + 1, sizeof(retry_ms) / sizeof(retry_ms[0]), want >> 20, e2);
        snprintf(last, sizeof(last), "%s", e2);
        snprintf(err, errn, "%s", e2);
    }

    for (size_t len = want / 2; len >= ARENA_MIN; len /= 2) {
        char e2[256] = "";
        if (arena_try(a, len, e2, sizeof(e2)) == 0) {
            size_t total = 0, cfree = 0;
            arena_cma_info(&total, &cfree);
            a->degraded = 1;
            fprintf(stderr,
                "vc4d: ARENA DEGRADED: obtained %zu MiB of the %zu MiB requested.\n"
                "vc4d:   %s\n"
                "vc4d:   CMA: %zu MiB free of %zu MiB total (/proc/meminfo); CMA is one\n"
                "vc4d:   contiguous pool shared by every GPU client, so a large block can\n"
                "vc4d:   fail while plenty of CMA is free in smaller pieces.\n"
                "vc4d:   Until the full block is available a model whose client arena\n"
                "vc4d:   needs more than %zu MiB fails with `pool exhausted`.\n"
                "vc4d:   What to do: stop the other GPU client (another vc4d, VC4CL, a\n"
                "vc4d:   camera/KMS buffer) and the daemon restores the full arena by\n"
                "vc4d:   itself as soon as it is idle - no reboot needed.  Or restart it\n"
                "vc4d:   now: sudo systemctl restart vc4d\n",
                len >> 20, want >> 20, last, cfree >> 20, total >> 20,
                (len - A_DATA_BASE) >> 20);
            return 0;
        }
        snprintf(last, sizeof(last), "%s", e2);
        snprintf(err, errn, "%s", e2);
    }
    if (!*err)
        snprintf(err, errn, "no dma-buf of %zu MiB (CMA free?)", want >> 20);
    return -1;
}

void arena_close(struct arena *a)
{
    if (a->map && a->bytes)
        munmap(a->map, a->bytes);
    if (a->fd >= 0)
        close(a->fd);
    free(a->bits);
    memset(a, 0, sizeof(*a));
    a->fd = -1;
}

int arena_alloc(struct arena *a, uint32_t bytes, uint32_t *off)
{
    if (!bytes || !a->bits || !a->npages)
        return -1;
    uint32_t need = (bytes + ARENA_PAGE - 1u) / ARENA_PAGE;
    if (need < 1 || need > a->npages)
        return -1;

    /* First fit from the cursor, then wrap once to page 0.  The scan is over
       a bitmap, so it can never "lose" free space to bookkeeping. */
    for (int pass = 0; pass < 2; ++pass) {
        uint32_t from = pass ? 0 : a->hint;
        uint32_t to = pass ? a->hint : a->npages;
        uint32_t run = 0, start = 0;
        for (uint32_t p = from; p < to; ++p) {
            if (page_used(a, p)) {
                run = 0;
                continue;
            }
            if (!run)
                start = p;
            if (++run == need) {
                for (uint32_t q = start; q < start + need; ++q)
                    page_set(a, q);
                a->used += (size_t)need * ARENA_PAGE;
                a->hint = start + need;
                if (a->hint >= a->npages)
                    a->hint = 0;
                *off = A_DATA_BASE + start * ARENA_PAGE;
                return 0;
            }
        }
        if (!a->hint)
            break;                    /* the second pass would repeat pass 1 */
    }
    return -1;
}

int arena_free(struct arena *a, uint32_t off, uint32_t len)
{
    if (!a->bits || !len || (off & (ARENA_PAGE - 1)) ||
        (len & (ARENA_PAGE - 1)) || off < A_DATA_BASE ||
        (size_t)off + len > a->bytes ||
        (size_t)off + len < A_DATA_BASE)
        return -1;
    uint32_t first = (off - A_DATA_BASE) / ARENA_PAGE;
    uint32_t np = len / ARENA_PAGE;
    uint32_t cleared = 0;
    for (uint32_t i = 0; i < np; ++i)
        if (page_clear(a, first + i))
            cleared++;
    /* Only pages that were actually allocated are subtracted, so the counter
       stays exact even if a caller frees the same region twice. */
    size_t bytes = (size_t)cleared * ARENA_PAGE;
    a->used = a->used >= bytes ? a->used - bytes : 0;
    if (first < a->hint)
        a->hint = first;
    return 0;
}

uint32_t arena_bus(const struct arena *a, uint32_t off)
{
    return a->bus + off;
}
