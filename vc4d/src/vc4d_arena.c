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

static void freeblk_insert(struct arena *a, uint32_t off, uint32_t len)
{
    int i = 0;
    while (i < a->nfree && a->free[i].off < off)
        ++i;
    if (a->nfree == MAX_FREEBLK)
        return;                       /* leak-safe: region stays unusable */
    memmove(&a->free[i + 1], &a->free[i], (size_t)(a->nfree - i) * sizeof(a->free[0]));
    a->free[i].off = off;
    a->free[i].len = len;
    a->nfree++;
    /* coalesce with the next, then with the previous */
    if (i + 1 < a->nfree && a->free[i].off + a->free[i].len == a->free[i + 1].off) {
        a->free[i].len += a->free[i + 1].len;
        memmove(&a->free[i + 1], &a->free[i + 2],
                (size_t)(a->nfree - i - 2) * sizeof(a->free[0]));
        a->nfree--;
    }
    if (i > 0 && a->free[i - 1].off + a->free[i - 1].len == a->free[i].off) {
        a->free[i - 1].len += a->free[i].len;
        memmove(&a->free[i], &a->free[i + 1],
                (size_t)(a->nfree - i - 1) * sizeof(a->free[0]));
        a->nfree--;
    }
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

int arena_open(struct arena *a, size_t want, char *err, size_t errn)
{
    memset(a, 0, sizeof(*a));
    a->fd = -1;
    *err = 0;
    if (want < ARENA_MIN)
        want = ARENA_MIN;
    want &= ~4095ul;

    static const size_t shrink[] = { 1, 1, 1, 2, 4, 8 };
    for (size_t si = 0; si < sizeof(shrink) / sizeof(shrink[0]); ++si) {
        size_t len = want / shrink[si];
        if (len < ARENA_MIN)
            break;
        if (si)
            usleep(200 * 1000);   /* let a just-exited user's CMA drain */
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
        if (rc != 0)
            continue;

        uint8_t *cached = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, (int)d.fd, 0);
        if (cached == MAP_FAILED) {
            close((int)d.fd);
            continue;
        }
        uint32_t phys = 0;
        int fr = find_phys(cached, len, &phys, err, errn);
        munmap(cached, len);
        if (fr != 0) {
            close((int)d.fd);
            continue;
        }

        int mem = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem < 0) {
            close((int)d.fd);
            snprintf(err, errn, "open /dev/mem: %s", strerror(errno));
            return -1;
        }
        void *unc = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)phys);
        if (unc == MAP_FAILED) {
            close(mem);
            close((int)d.fd);
            snprintf(err, errn, "mmap arena at 0x%08x: %s", phys, strerror(errno));
            return -1;
        }
        if (verify_phys(mem, unc, phys, len, err, errn) != 0) {
            munmap(unc, len);
            close(mem);
            close((int)d.fd);
            continue;                      /* try a smaller arena */
        }
        close(mem);

        a->fd = (int)d.fd;
        a->map = unc;
        a->phys = phys;
        a->bus = phys | 0xC0000000u;
        a->bytes = len;
        memset(a->map, 0, A_DATA_BASE);       /* header, code and uni areas */
        a->nfree = 1;
        a->free[0].off = A_DATA_BASE;
        a->free[0].len = (uint32_t)(len - A_DATA_BASE);
        return 0;
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
    memset(a, 0, sizeof(*a));
    a->fd = -1;
}

int arena_alloc(struct arena *a, uint32_t bytes, uint32_t *off)
{
    if (!bytes)
        return -1;
    uint32_t need = (bytes + 4095u) & ~4095u;
    if (need < bytes)
        return -1;
    for (int i = 0; i < a->nfree; ++i) {
        if (a->free[i].len < need)
            continue;
        uint32_t o = a->free[i].off;
        a->free[i].off += need;
        a->free[i].len -= need;
        if (!a->free[i].len) {
            memmove(&a->free[i], &a->free[i + 1],
                    (size_t)(a->nfree - i - 1) * sizeof(a->free[0]));
            a->nfree--;
        }
        a->used += need;
        *off = o;
        return 0;
    }
    return -1;
}

int arena_free(struct arena *a, uint32_t off, uint32_t len)
{
    if (!len || (off & 4095u) || (len & 4095u) || off < A_DATA_BASE ||
        (size_t)off + len > a->bytes)
        return -1;
    freeblk_insert(a, off, len);
    a->used -= len;
    return 0;
}

uint32_t arena_bus(const struct arena *a, uint32_t off)
{
    return a->bus + off;
}
