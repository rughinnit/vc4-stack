/*
 * vc4d_gpu.c - the privileged V3D register mapping and the launch sequence.
 *
 * The launch sequence is the engine's proven one (engine-fast2/src/vc4_dev.c,
 * itself asm3's qrun harness): counters, VPMBASE=16, L2 flush, SLC invalidate,
 * ERR_STAT W1C, SRQ reset, then SRQUA/SRQPC per QPU and the queue check.
 *
 * Register writes are restricted to an allowlist.  Reads are unrestricted
 * (a read cannot change hardware state) and READ_REGS can dump the block.
 */
#define _GNU_SOURCE
#include "vc4d.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define IOCTL_MBOX_PROPERTY _IOWR(100, 0, char *)

#define CTR_EXEC_CYCLES 16
#define CTR_INSTR_HITS  20

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void barrier(void)
{
    __sync_synchronize();
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#endif
}

static void mbox_enable_qpu(void)
{
    int fd = open("/dev/vcio", O_RDWR | O_CLOEXEC);
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

int gpu_open(struct gpu *g, char *err, size_t errn)
{
    memset(g, 0, sizeof(*g));
    int mem = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (mem < 0) {
        snprintf(err, errn, "open /dev/mem: %s", strerror(errno));
        return -1;
    }
    void *p = mmap(NULL, V3D_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, mem,
                   (off_t)V3D_PHYS_BASE);
    close(mem);
    if (p == MAP_FAILED) {
        snprintf(err, errn, "mmap V3D at 0x%08lx: %s",
                 (unsigned long)V3D_PHYS_BASE, strerror(errno));
        return -1;
    }
    g->regs = p;
    mbox_enable_qpu();
    g->errstat_base = g->regs[R_ERRSTAT / 4];
    if (!gpu_health(g)) {
        snprintf(err, errn, "V3D IDENT0=0xdeadbeef (suspended)");
        gpu_close(g);
        return -1;
    }
    return 0;
}

void gpu_close(struct gpu *g)
{
    if (g->regs)
        munmap((void *)g->regs, V3D_LENGTH);
    g->regs = NULL;
}

int gpu_health(struct gpu *g)
{
    return g && g->regs && g->regs[R_IDENT0 / 4] != 0xdeadbeefu;
}

static int srq_reset(struct gpu *g)
{
    const uint32_t RESET = (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16);
    for (int attempt = 0; attempt < 20; ++attempt) {
        g->regs[R_SRQCS / 4] = RESET;
        barrier();
        uint32_t s = g->regs[R_SRQCS / 4];
        if ((s & 0x3F) == 0 && ((s >> 16) & 0xFF) == 0 && (s & (1u << 7)) == 0)
            return 1;
        usleep(100);
    }
    return 0;
}

int gpu_kick(struct gpu *g, uint32_t code_bus, const uint32_t *uni_bus,
             int nqpu, uint32_t timeout_ms, char *err, size_t errn)
{
    if (g->dead) {
        snprintf(err, errn, "daemon is dead after a previous GPU timeout");
        return VC4D_E_DEAD;
    }
    if (g->inflight) {
        snprintf(err, errn, "a launch is already in flight");
        return VC4D_E_BUSY;
    }
    if (nqpu < 1 || nqpu > (int)VC4D_MAX_QPU) {
        snprintf(err, errn, "nqpu %d out of range", nqpu);
        return VC4D_E_INVAL;
    }
    if (!gpu_health(g)) {
        g->dead = 1;
        snprintf(err, errn, "V3D IDENT0=0xdeadbeef (suspended)");
        return VC4D_E_GPU;
    }
    barrier();
    g->regs[R_PCTR_EN / 4] |= (1u << 31) | (1u << 0);
    g->regs[R_PCTR_MAP / 4 + 0 * 2] = CTR_EXEC_CYCLES & 0x1F;
    g->regs[R_PCTR_VAL / 4 + 0 * 2] = 0;
    g->regs[R_PCTR_EN / 4] |= (1u << 31) | (1u << 1);
    g->regs[R_PCTR_MAP / 4 + 1 * 2] = CTR_INSTR_HITS & 0x1F;
    g->regs[R_PCTR_VAL / 4 + 1 * 2] = 0;
    g->regs[R_PCTR_CLR / 4] = 3;
    barrier();
    g->regs[R_VPMBASE / 4] = 16;
    barrier();
    g->regs[R_L2CACTL / 4] = 1u << 2;          /* write back + invalidate */
    barrier();
    g->regs[R_SLCACTL / 4] = 0xFFFFFFFFu;      /* instruction/data caches */
    barrier();
    g->regs[R_ERRSTAT / 4] = 0xFFFFFFFFu;      /* write-1-to-clear */
    barrier();
    if (!srq_reset(g)) {
        snprintf(err, errn, "SRQ reset failed");
        return VC4D_E_GPU;
    }
    for (int i = 0; i < nqpu; ++i) {
        g->regs[R_SRQUA / 4] = uni_bus[i];
        g->regs[R_SRQPC / 4] = code_bus;
    }
    barrier();
    uint32_t s = g->regs[R_SRQCS / 4];
    if (s & (1u << 7)) {
        snprintf(err, errn, "queue error after launch (SRQCS=0x%08x)", s);
        return VC4D_E_GPU;
    }
    g->inflight = 1;
    g->nqpu = nqpu;
    g->t0 = now_ns();
    g->deadline = g->t0 + (uint64_t)timeout_ms * 1000000ull;
    return 0;
}

/* 1 = done, 0 = still running, -1 = queue error observed */
int gpu_poll(struct gpu *g)
{
    uint32_t s = g->regs[R_SRQCS / 4];
    if (s & (1u << 7))
        return -1;
    if (((s >> 16) & 0xFF) == (uint32_t)g->nqpu && (s & 0x3F) == 0)
        return 1;
    return 0;
}

int gpu_finish(struct gpu *g, struct launch_result *r, char *err, size_t errn)
{
    uint64_t t = now_ns();
    uint32_t errs = g->regs[R_ERRSTAT / 4] & ~g->errstat_base;
    uint32_t s = g->regs[R_SRQCS / 4];
    r->wall_ns = t - g->t0;
    r->cycles = g->regs[R_PCTR_VAL / 4];
    r->errs = errs;
    g->gpu_ns += r->wall_ns;
    g->inflight = 0;
    if (errs) {
        snprintf(err, errn, "V3D_ERR_STAT=0x%08x (SRQCS=0x%08x)", errs, s);
        return VC4D_E_GPU;
    }
    return 0;
}

void gpu_abort(struct gpu *g)
{
    /* A stuck QPU cannot be reset from the ARM.  Clear the queue word so the
       board is not wedged for the next user, then refuse further launches. */
    if (g->regs)
        g->regs[R_SRQCS / 4] = (1u << 0) | (1u << 7) | (1u << 8) | (1u << 16);
    g->inflight = 0;
    g->dead = 1;
}

static const struct { uint32_t off; const char *name; } wreg_allow[] = {
    { R_SRQPC,   "SRQPC"   },
    { R_SRQUA,   "SRQUA"   },
    { R_SRQUL,   "SRQUL"   },
    { R_SRQCS,   "SRQCS"   },
    { R_VPMBASE, "VPMBASE" },
    { R_L2CACTL, "L2CACTL" },
    { R_SLCACTL, "SLCACTL" },
    { R_VPACNTL, "VPACNTL" },
    { R_INTCTL,  "INTCTL"  },
    { R_INTENA,  "INTENA"  },
    { R_INTDIS,  "INTDIS"  },
};

int gpu_wreg(struct gpu *g, uint32_t off, uint32_t val, char *err, size_t errn)
{
    for (size_t i = 0; i < sizeof(wreg_allow) / sizeof(wreg_allow[0]); ++i) {
        if (wreg_allow[i].off == off) {
            g->regs[off / 4] = val;
            return 0;
        }
    }
    snprintf(err, errn, "register 0x%04x is not write-allowlisted", off);
    return VC4D_E_DENIED;
}
