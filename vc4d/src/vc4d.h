/*
 * vc4d.h - internal daemon definitions.
 *
 * Only three pieces of state are privileged: the dma-buf fd, its physical
 * mapping (/dev/mem) and the V3D register mapping.  All of them are created in
 * vc4d_main() before the process drops to an unprivileged uid; after that the
 * daemon is pure memory access on mappings it already owns.
 */
#ifndef VC4D_H
#define VC4D_H

#include <stddef.h>
#include <stdint.h>
#include "vc4d_proto.h"

#define V3D_PHYS_BASE   0x3FC00000UL     /* legacy V3D window used by the engine */
#define V3D_LENGTH      0x4000UL
#define V3D_REG_WORDS   (V3D_LENGTH / 4)

#define ARENA_DEFAULT   (64u << 20)
#define ARENA_MIN       (8u << 20)

/* arena layout (offsets inside the dma-buf) */
#define A_HDR_BASE      0x000000u
#define A_CODE_BASE     0x001000u
#define A_CODE_SLOTS    64u
#define A_CODE_SLOT     0x4000u          /* 16 KiB per program */
#define A_UNI_BASE      0x110000u
#define A_UNI_SLOT      0x1000u          /* 12 QPUs x 64 B */
#define A_DATA_BASE     0x200000u        /* client-allocatable heap */

#define MAX_CLIENTS     8
#define MAX_REGIONS     64
/* One client may cache up to all the daemon's program slots (64): the engine
 * uploads one program per kernel shape, and a test that runs several conv
 * shapes legitimately needs ~25.  MAX_PROGS_SLOTS bounds the total. */
#define MAX_PROGS_CLI   64
#define MAX_FREEBLK     64

/* V3D registers used by name */
#define R_IDENT0    0x0000u
#define R_IDENT1    0x0004u
#define R_L2CACTL   0x0020u
#define R_SLCACTL   0x0024u
#define R_INTCTL    0x0030u
#define R_INTENA    0x0034u
#define R_INTDIS    0x0038u
#define R_SRQPC     0x0430u
#define R_SRQUA     0x0434u
#define R_SRQUL     0x0438u
#define R_SRQCS     0x043c
#define R_VPMBASE   0x0504u
#define R_VPACNTL   0x0508u
#define R_PCTR_CLR  0x0670u
#define R_PCTR_EN   0x0674u
#define R_PCTR_VAL  0x0680u
#define R_PCTR_MAP  0x0684u
#define R_ERRSTAT   0x0f20u

/* ---- arena ------------------------------------------------------------- */
struct arena {
    int      fd;              /* dma-buf, kept for SCM_RIGHTS dups */
    uint8_t *map;             /* uncached /dev/mem mapping        */
    uint32_t phys;
    uint32_t bus;             /* phys | 0xC0000000                */
    size_t   bytes;
    size_t   used;            /* bytes currently handed out       */
    int      nfree;
    struct { uint32_t off, len; } free[MAX_FREEBLK];
};

int  arena_open(struct arena *a, size_t want, char *err, size_t errn);
void arena_close(struct arena *a);
int  arena_alloc(struct arena *a, uint32_t bytes, uint32_t *off);
int  arena_free(struct arena *a, uint32_t off, uint32_t len);
uint32_t arena_bus(const struct arena *a, uint32_t off);

/* ---- gpu --------------------------------------------------------------- */
struct launch_result {
    uint64_t wall_ns;
    uint32_t cycles;
    uint32_t errs;
};

struct gpu {
    volatile uint32_t *regs;
    uint32_t errstat_base;
    int      dead;
    int      inflight;
    int      nqpu;
    uint64_t t0;
    uint64_t deadline;
    uint64_t gpu_ns;
};

int  gpu_open(struct gpu *g, char *err, size_t errn);
void gpu_close(struct gpu *g);
int  gpu_health(struct gpu *g);
int  gpu_kick(struct gpu *g, uint32_t code_bus, const uint32_t *uni_bus,
              int nqpu, uint32_t timeout_ms, char *err, size_t errn);
int  gpu_poll(struct gpu *g);
int  gpu_finish(struct gpu *g, struct launch_result *r, char *err, size_t errn);
void gpu_abort(struct gpu *g);
int  gpu_wreg(struct gpu *g, uint32_t off, uint32_t val, char *err, size_t errn);

/* ---- sha256 ------------------------------------------------------------ */
struct sha256 { uint32_t h[8]; uint64_t bytes; uint8_t buf[64]; size_t n; };
void sha256_init(struct sha256 *s);
void sha256_update(struct sha256 *s, const void *p, size_t n);
void sha256_final(struct sha256 *s, uint8_t out[32]);
int  sha256_allowlisted(const char *path, const uint8_t hash[32],
                        char *err, size_t errn);

#endif /* VC4D_H */
