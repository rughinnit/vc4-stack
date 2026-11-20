/*
 * cmahold.c - hold CMA, the way another GPU client would.
 *
 * The G2 test needs the daemon's requested dma-buf block to be unavailable
 * while *some* CMA is still free, and it needs the pressure to be released
 * again without touching the board's other allocations.  This helper takes
 * N chunks of the CMA heap through /dev/dma_heap/linux,cma, prints how much it
 * got, and keeps them until it is killed (or --seconds elapses).  On exit the
 * fds close, which returns the memory to CMA immediately.
 *
 * It is deliberately not a GPU client: it never opens /dev/mem or the V3D, so
 * it can run next to the installed daemon without disturbing it.
 *
 * usage: cmahold --mb N [--chunk-mb M] [--seconds S]
 *        cmahold --free              (print CmaTotal/CmaFree and exit)
 *
 * Needs root (the dma-heap device is root-only on this kernel).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

static volatile sig_atomic_t stop_flag;

static void on_signal(int sig)
{
    (void)sig;
    stop_flag = 1;
}

static void cma_info(unsigned long *total_mb, unsigned long *free_mb)
{
    *total_mb = *free_mb = 0;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned long kb;
        if (sscanf(line, "CmaTotal: %lu kB", &kb) == 1)
            *total_mb = kb / 1024;
        else if (sscanf(line, "CmaFree: %lu kB", &kb) == 1)
            *free_mb = kb / 1024;
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    unsigned long want_mb = 0, chunk_mb = 4, seconds = 0;
    const char *pidfile = NULL;
    int just_free = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc)
            want_mb = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--chunk-mb") && i + 1 < argc)
            chunk_mb = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--pidfile") && i + 1 < argc)
            pidfile = argv[++i];
        else if (!strcmp(argv[i], "--free"))
            just_free = 1;
        else {
            fprintf(stderr, "usage: cmahold --mb N [--chunk-mb M] [--seconds S]\n"
                            "               [--pidfile FILE]\n");
            return 2;
        }
    }
    unsigned long tot = 0, fr = 0;
    cma_info(&tot, &fr);
    if (just_free) {
        printf("cma %lu MiB total, %lu MiB free\n", tot, fr);
        return 0;
    }
    if (geteuid() != 0) {
        fprintf(stderr, "cmahold: needs root (dma-heap is root-only)\n");
        return 2;
    }
    if (!want_mb || !chunk_mb)
        return 2;

    int heap = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap < 0) {
        fprintf(stderr, "cmahold: open /dev/dma_heap/linux,cma: %s\n",
                strerror(errno));
        return 1;
    }
    unsigned long got = 0;
    int nfds = 0, cap = (int)(want_mb / chunk_mb) + 2;
    int *fds = calloc((size_t)cap, sizeof(int));
    if (!fds) {
        close(heap);
        return 1;
    }
    while (got + chunk_mb <= want_mb) {
        if (nfds >= cap)
            break;
        struct dma_heap_allocation_data d;
        memset(&d, 0, sizeof(d));
        d.len = chunk_mb << 20;
        d.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &d) != 0) {
            /* stop at the first failure: the caller asked for an amount and
             * the interesting case is "this much is held", not the exact sum */
            break;
        }
        fds[nfds++] = (int)d.fd;
        got += chunk_mb;
    }
    close(heap);
    cma_info(&tot, &fr);
    printf("cmahold: holding %lu MiB in %d chunk(s); cma %lu/%lu MiB free\n",
           got, nfds, fr, tot);
    fflush(stdout);
    if (!got) {
        free(fds);
        return 1;
    }
    if (pidfile) {
        /* the caller kills this pid (through sudo) instead of guessing which
         * of sudo/child owns the fds */
        FILE *pf = fopen(pidfile, "w");
        if (pf) {
            fprintf(pf, "%d\n", (int)getpid());
            fclose(pf);
        }
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    if (seconds) {
        for (unsigned i = 0; i < seconds * 10 && !stop_flag; ++i)
            usleep(100 * 1000);
    } else {
        while (!stop_flag)
            pause();
    }
    for (int i = 0; i < nfds; ++i)
        close(fds[i]);
    free(fds);
    return 0;
}
