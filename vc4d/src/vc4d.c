/*
 * vc4d.c - privileged VC4 QPU helper daemon.
 *
 * What the daemon is for: the engine must mmap /dev/mem twice (the V3D
 * register block and the physical base of its dma-buf), which needs
 * CAP_SYS_RAWIO.  vc4d does that once, at startup, and then serves a small
 * length-prefixed protocol over /run/vc4d.sock so unprivileged clients can
 * allocate GPU-visible buffers, upload QPU programs, launch them and read
 * results.
 *
 * Security model (see README.md for the long form):
 *   - the socket is 0660 root:vc4 and there is no network transport;
 *   - every pointer uniform is resolved and range-checked by the daemon;
 *   - literal uniforms may not be GPU bus / arena / MMIO addresses;
 *   - register writes are allowlisted, reads are unrestricted;
 *   - one launch in flight, bounded timeouts, dead-after-timeout;
 *   - after mapping, the process drops to an unprivileged uid with an empty
 *     capability set, so it keeps no path back to /dev/mem.
 *
 * What it cannot do: a deliberately hostile client can still upload a crafted
 * QPU program and make the GPU DMA anywhere in the 32-bit bus space, because
 * QPU programs are raw machine code and the daemon does not interpret them.
 * --strict (program sha256 allowlist + no raw SRQ writes) closes that hole for
 * deployments that only run known kernels.
 */
#define _GNU_SOURCE
#include "vc4d.h"

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <linux/capability.h>

#define DEFAULT_SOCKET   "/run/vc4d.sock"
#define DEFAULT_TIMEOUT  2000u
#define MAX_TIMEOUT      10000u
#define MAX_PROGS_SLOTS  64
#define DEFAULT_GPU_LOCK "/tmp/vc4-gpu.lock"
#define MAX_DENIALS_LOG  20

struct region { int used; uint32_t off, len; };
struct prog {
    int      used;
    int      owner;         /* client index */
    uint32_t gen;
    uint32_t code_off, uni_off;
    uint32_t nqpu_max, nuni, words;
    uint8_t  hash[32];
};

struct client {
    int      idx;
    int      fd;
    uid_t    uid;
    pid_t    pid;
    struct region reg[MAX_REGIONS];
    uint32_t prog[MAX_PROGS_CLI];
    int      nprog;
    int      inflight;
    uint32_t launch_id;
    uint64_t denials;
    int      deny_logged;
    time_t   deny_window;
};

struct opts {
    const char *socket_path;
    const char *allowlist;
    const char *drop_user;
    const char *group;
    size_t   arena_bytes;
    uint32_t max_timeout_ms;
    int      strict;
    int      raw_srq;
    int      drop;
    int      gpu_lock;
    int      verbose;
};

static struct opts opt;
static struct arena A;
static struct gpu G;
static struct client *clients[MAX_CLIENTS];
static struct prog progs[MAX_PROGS_SLOTS];
static int listen_fd = -1;
static int gpu_lock_fd = -1;
static uint32_t next_launch = 1;
static volatile sig_atomic_t stop_flag;

static uint64_t st_launches, st_failed, st_bytes_up, st_wreg, st_rejects;
static int st_clients, st_clients_max, st_progs;

static void logmsg(const char *fmt, ...)
{
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    fprintf(stderr, "vc4d: %s\n", b);
}

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Bounded wait for the board-wide GPU lock.  Non-blocking + sleep rather than
 * a blocking flock() so a client request can never stall forever. */
static int gpu_lock_take(uint32_t wait_ms)
{
    uint64_t deadline = now_ns() + (uint64_t)wait_ms * 1000000ull;
    for (;;) {
        if (flock(gpu_lock_fd, LOCK_EX | LOCK_NB) == 0)
            return 0;
        if (errno != EWOULDBLOCK && errno != EINTR)
            return -1;
        if (now_ns() > deadline)
            return -2;
        usleep(2000);
    }
}

static void gpu_lock_drop(void)
{
    if (opt.gpu_lock && gpu_lock_fd >= 0)
        flock(gpu_lock_fd, LOCK_UN);
}

static void on_signal(int sig) { (void)sig; stop_flag = 1; }

static void log_denial(struct client *c, const char *op, const char *why)
{
    if (!c)
        return;
    c->denials++;
    time_t now = time(NULL);
    if (now != c->deny_window) {
        c->deny_window = now;
        c->deny_logged = 0;
    }
    if (c->deny_logged < MAX_DENIALS_LOG)
        logmsg("DENY uid=%u pid=%d op=%s: %s", (unsigned)c->uid, (int)c->pid, op, why);
    else if (c->deny_logged == MAX_DENIALS_LOG)
        logmsg("DENY uid=%u pid=%d: further denials suppressed this second",
               (unsigned)c->uid, (int)c->pid);
    c->deny_logged++;
}

/* ---------------------------------------------------------------- replies */
static int reply(int fd, const struct vc4d_hdr *req, int32_t status,
                 const void *payload, size_t plen, int pass_fd,
                 uint32_t d0, uint32_t d1)
{
    struct vc4d_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC;
    h.len = (uint32_t)(sizeof(h) + plen);
    h.op = req->op;
    h.version = VC4D_VERSION;
    h.seq = req->seq;
    h.status = status;
    h.detail0 = d0;
    h.detail1 = d1;

    struct iovec iov[2] = {
        { .iov_base = &h, .iov_len = sizeof(h) },
        { .iov_base = (void *)payload, .iov_len = plen },
    };
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = plen ? 2 : 1;

    char cbuf[CMSG_SPACE(sizeof(int))];
    if (pass_fd >= 0) {
        memset(cbuf, 0, sizeof(cbuf));
        msg.msg_control = cbuf;
        msg.msg_controllen = sizeof(cbuf);
        struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cm), &pass_fd, sizeof(int));
    }
    for (;;) {
        ssize_t n = sendmsg(fd, &msg, MSG_NOSIGNAL);
        if (n >= 0)
            return 0;
        if (errno == EINTR)
            continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            struct pollfd p = { .fd = fd, .events = POLLOUT };
            if (poll(&p, 1, 2000) <= 0)
                return -1;
            continue;
        }
        return -1;
    }
}

/* ------------------------------------------------------------- region/prog */
static struct region *find_region(struct client *c, uint32_t handle)
{
    if (handle >= MAX_REGIONS || !c->reg[handle].used)
        return NULL;
    return &c->reg[handle];
}

static struct prog *find_prog(struct client *c, uint32_t handle)
{
    uint32_t idx = handle & 0xFFu;
    if (!idx || idx > MAX_PROGS_SLOTS)
        return NULL;
    struct prog *p = &progs[idx - 1];
    if (!p->used || p->owner != c->idx)
        return NULL;
    if (((p->gen << 8) | idx) != handle)
        return NULL;
    return p;
}

static void release_client_resources(struct client *c)
{
    for (int i = 0; i < MAX_REGIONS; ++i) {
        if (c->reg[i].used) {
            arena_free(&A, c->reg[i].off, c->reg[i].len);
            c->reg[i].used = 0;
        }
    }
    for (int i = 0; i < MAX_PROGS_SLOTS; ++i) {
        if (progs[i].used && progs[i].owner == c->idx) {
            progs[i].used = 0;
            progs[i].owner = -1;
            st_progs--;
        }
    }
    c->nprog = 0;
}

/* --------------------------------------------------------------- handlers */
static int handle_hello(struct client *c, const struct vc4d_hdr *h,
                        const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_hello_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_hello_req q;
    memcpy(&q, pl, sizeof(q));
    if (q.version != VC4D_VERSION) {
        log_denial(c, "HELLO", "version mismatch");
        return reply(c->fd, h, VC4D_E_VERSION, NULL, 0, -1, VC4D_VERSION, 0);
    }
    struct vc4d_hello_rsp r;
    memset(&r, 0, sizeof(r));
    r.version = VC4D_VERSION;
    r.arena_bytes = (uint32_t)A.bytes;
    r.data_base = A_DATA_BASE;
    r.page_size = 4096;
    r.max_msg = VC4D_MAX_MSG;
    r.max_qpu = VC4D_MAX_QPU;
    r.max_uni = VC4D_MAX_UNI;
    r.max_prog = VC4D_MAX_PROG;
    r.flags = (opt.raw_srq ? VC4D_F_RAW_SRQ : 0) |
              (opt.allowlist ? VC4D_F_ALLOWLIST : 0);
    r.ident0 = G.regs[R_IDENT0 / 4];
    r.ident1 = G.regs[R_IDENT1 / 4];
    return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
}

static int handle_alloc(struct client *c, const struct vc4d_hdr *h,
                        const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_alloc_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_alloc_req q;
    memcpy(&q, pl, sizeof(q));
    if (!q.bytes || (q.flags & ~VC4D_ALLOC_ZERO)) {
        log_denial(c, "ALLOC", "bad size or flags");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (q.bytes > A.bytes - A_DATA_BASE) {
        log_denial(c, "ALLOC", "size larger than the arena");
        return reply(c->fd, h, VC4D_E_TOOLARGE, NULL, 0, -1, 0, 0);
    }
    int slot = -1;
    for (int i = 0; i < MAX_REGIONS; ++i)
        if (!c->reg[i].used) { slot = i; break; }
    if (slot < 0)
        return reply(c->fd, h, VC4D_E_NOMEM, NULL, 0, -1, 0, 0);

    uint32_t off = 0;
    if (arena_alloc(&A, q.bytes, &off) != 0) {
        log_denial(c, "ALLOC", "arena exhausted");
        return reply(c->fd, h, VC4D_E_NOMEM, NULL, 0, -1, 0, 0);
    }
    uint32_t len = (q.bytes + 4095u) & ~4095u;
    if (q.flags & VC4D_ALLOC_ZERO)
        memset(A.map + off, 0, len);
    c->reg[slot].used = 1;
    c->reg[slot].off = off;
    c->reg[slot].len = len;

    struct vc4d_alloc_rsp r = { .handle = (uint32_t)slot, .off = off,
                                 .len = len, .bus = arena_bus(&A, off) };
    if (reply(c->fd, h, VC4D_OK, &r, sizeof(r), A.fd, 0, 0) != 0) {
        arena_free(&A, off, len);
        c->reg[slot].used = 0;
        return -1;
    }
    return 0;
}

static int handle_free(struct client *c, const struct vc4d_hdr *h,
                       const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_free_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_free_req q;
    memcpy(&q, pl, sizeof(q));
    struct region *r = find_region(c, q.handle);
    if (!r) {
        log_denial(c, "FREE", "unknown handle");
        return reply(c->fd, h, VC4D_E_NOENT, NULL, 0, -1, 0, 0);
    }
    arena_free(&A, r->off, r->len);
    r->used = 0;
    return reply(c->fd, h, VC4D_OK, NULL, 0, -1, 0, 0);
}

static int handle_upload(struct client *c, const struct vc4d_hdr *h,
                         const uint8_t *pl, size_t plen)
{
    if (plen < sizeof(struct vc4d_upload_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_upload_req q;
    memcpy(&q, pl, sizeof(q));
    const uint8_t *code = pl + sizeof(q);
    size_t clen = plen - sizeof(q);

    if (q.flags != 0 || q.bytes != clen) {
        log_denial(c, "UPLOAD", "payload length disagrees with bytes");
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    }
    if (q.bytes < 8 || (q.bytes & 7u) || q.bytes > VC4D_MAX_PROG ||
        q.max_qpu < 1 || q.max_qpu > VC4D_MAX_QPU ||
        q.nuni < 1 || q.nuni > VC4D_MAX_UNI) {
        log_denial(c, "UPLOAD", "bad size/alignment/max_qpu");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (c->nprog >= MAX_PROGS_CLI)
        return reply(c->fd, h, VC4D_E_NOMEM, NULL, 0, -1, 0, 0);

    uint8_t hash[32];
    struct sha256 sh;
    sha256_init(&sh);
    sha256_update(&sh, code, clen);
    sha256_final(&sh, hash);

    if (opt.allowlist) {
        char err[256];
        int ok = sha256_allowlisted(opt.allowlist, hash, err, sizeof(err));
        if (ok <= 0) {
            char hex[65], why[128];
            for (int i = 0; i < 32; ++i)
                snprintf(hex + 2 * i, 3, "%02x", hash[i]);
            snprintf(why, sizeof(why), "program sha256 %s not allowlisted", hex);
            log_denial(c, "UPLOAD", why);
            return reply(c->fd, h, ok < 0 ? VC4D_E_IO : VC4D_E_DENIED,
                         NULL, 0, -1, 0, 0);
        }
    }

    int slot = -1;
    for (int i = 0; i < MAX_PROGS_SLOTS; ++i)
        if (!progs[i].used) { slot = i; break; }
    if (slot < 0)
        return reply(c->fd, h, VC4D_E_NOMEM, NULL, 0, -1, 0, 0);

    struct prog *p = &progs[slot];
    uint32_t gen = (p->gen + 1u) & 0xFFFFFFu;
    if (!gen)
        gen = 1;
    memset(p, 0, sizeof(*p));
    p->used = 1;
    p->owner = c->idx;
    p->gen = gen;
    p->code_off = A_CODE_BASE + (uint32_t)slot * A_CODE_SLOT;
    p->uni_off = A_UNI_BASE + (uint32_t)slot * A_UNI_SLOT;
    p->nqpu_max = q.max_qpu;
    p->nuni = q.nuni;
    p->words = q.bytes / 8;
    memcpy(p->hash, hash, 32);
    st_progs++;

    memcpy(A.map + p->code_off, code, clen);
    memset(A.map + p->code_off + clen, 0, A_CODE_SLOT - clen);
    memset(A.map + p->uni_off, 0, A_UNI_SLOT);
    __sync_synchronize();
    st_bytes_up += clen;

    uint32_t hnd = (gen << 8) | (uint32_t)(slot + 1);
    c->prog[c->nprog++] = hnd;
    struct vc4d_upload_rsp r = { .prog = hnd, .code_off = p->code_off,
                                 .words = p->words };
    return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
}

static int literal_forbidden(uint32_t v)
{
    if (v >= 0xC0000000u)
        return 1;                                   /* GPU bus window */
    if (v >= (uint32_t)V3D_PHYS_BASE &&
        v < (uint32_t)V3D_PHYS_BASE + (uint32_t)V3D_LENGTH)
        return 1;                                   /* V3D MMIO */
    if (v >= A.phys && (uint64_t)v < (uint64_t)A.phys + A.bytes)
        return 1;                                   /* arena physical */
    return 0;
}

/* A word materialised in the client's own arena by UNI_IN_ARENA: it must be
 * either a bus address inside this arena, or a literal outside every window a
 * device address could name.  Containment is per-arena here (the engine's
 * pointers are arena offsets, not buffer handles), which is the price of not
 * touching the op code. */
static int arena_word_ok(uint32_t v)
{
    if (v >= A.bus && (uint64_t)v < (uint64_t)A.bus + A.bytes)
        return 1;
    return !literal_forbidden(v);
}

/* Poll an in-flight launch to completion, release the GPU lock and make the
 * GPU's results visible to unprivileged readers (L2 write-back + invalidate).
 * Fills lr and returns a VC4D_* status.  Shared by WAIT and LAUNCH_WAIT. */
static int await_launch(struct client *c, uint32_t id, struct launch_result *lr,
                        char *err, size_t errn)
{
    int rc;
    for (;;) {
        rc = gpu_poll(&G);
        if (rc != 0)
            break;
        if (now_ns() > G.deadline) {
            rc = -2;
            break;
        }
        usleep(10);
    }
    memset(lr, 0, sizeof(*lr));
    if (rc == -2) {
        gpu_abort(&G);
        c->inflight = 0;
        gpu_lock_drop();
        st_failed++;
        logmsg("TIMEOUT uid=%u pid=%d launch=%u; daemon marked dead",
               (unsigned)c->uid, (int)c->pid, id);
        return VC4D_E_TIMEOUT;
    }
    int status;
    if (rc < 0) {
        status = VC4D_E_GPU;
        snprintf(err, errn, "queue error (SRQCS=0x%08x)", G.regs[R_SRQCS / 4]);
        gpu_finish(&G, lr, err, errn);           /* clears inflight */
    } else {
        status = gpu_finish(&G, lr, err, errn);
    }
    c->inflight = 0;
    gpu_lock_drop();
    G.regs[R_L2CACTL / 4] = 1u << 2;             /* result to DRAM */
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
    if (status != VC4D_OK) {
        st_failed++;
        logmsg("launch %u error: %s", id, err);
    }
    return status;
}

static int handle_launch(struct client *c, const struct vc4d_hdr *h,
                         const uint8_t *pl, size_t plen)
{
    if (plen < sizeof(struct vc4d_launch_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_launch_req q;
    memcpy(&q, pl, sizeof(q));
    const struct vc4d_uni *uni = (const struct vc4d_uni *)(pl + sizeof(q));
    const int in_arena = (q.flags & VC4D_LAUNCH_UNI_ARENA) != 0;

    if (q.flags & ~VC4D_LAUNCH_FLAGS_ALL) {
        log_denial(c, "LAUNCH", "unknown launch flags");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (q.nqpu < 1 || q.nqpu > VC4D_MAX_QPU || q.nuni < 1 ||
        q.nuni > VC4D_MAX_UNI) {
        log_denial(c, "LAUNCH", "nqpu/nuni out of range");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (in_arena) {
        if (plen != sizeof(q)) {
            log_denial(c, "LAUNCH", "UNI_IN_ARENA carries extra payload");
            return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
        }
    } else if (plen != sizeof(q) + (size_t)q.nqpu * q.nuni * sizeof(*uni)) {
        log_denial(c, "LAUNCH", "uniform payload length mismatch");
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    }
    struct prog *p = find_prog(c, q.prog);
    if (!p) {
        log_denial(c, "LAUNCH", "unknown program handle");
        return reply(c->fd, h, VC4D_E_NOENT, NULL, 0, -1, 0, 0);
    }
    if (q.nqpu > p->nqpu_max) {
        log_denial(c, "LAUNCH", "nqpu exceeds the program's declared maximum");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (q.nuni != p->nuni) {
        log_denial(c, "LAUNCH", "nuni differs from the count declared at UPLOAD");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (G.inflight || c->inflight) {
        log_denial(c, "LAUNCH", "a launch is already in flight");
        return reply(c->fd, h, VC4D_E_BUSY, NULL, 0, -1, 0, 0);
    }

    uint32_t words[VC4D_MAX_QPU][VC4D_MAX_UNI];
    uint32_t uni_bus[VC4D_MAX_QPU];

    if (in_arena) {
        struct region *r = find_region(c, q.uni_handle);
        if (!r) {
            log_denial(c, "LAUNCH", "uniform block in an unknown region");
            return reply(c->fd, h, VC4D_E_NOENT, NULL, 0, -1, 0, 0);
        }
        if ((q.uni_off & 63u) ||
            (uint64_t)q.uni_off + 64ull * q.nqpu > r->len) {
            log_denial(c, "LAUNCH", "uniform block leaves its region");
            return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
        }
        for (uint32_t qpu = 0; qpu < q.nqpu; ++qpu) {
            const uint32_t *w = (const uint32_t *)(A.map + r->off +
                                                   q.uni_off + 64u * qpu);
            if (qpu == 0 && opt.verbose)
                logmsg("UNI_IN_ARENA prog=%u nqpu=%u off=0x%x:"
                       " %08x %08x %08x %08x %08x %08x %08x %08x",
                       q.prog, q.nqpu, q.uni_off, w[0], w[1], w[2], w[3],
                       w[4], w[5], w[6], w[7]);
            for (uint32_t u = 0; u < q.nuni; ++u) {
                if (!arena_word_ok(w[u])) {
                    log_denial(c, "LAUNCH",
                               "arena uniform word is outside the arena");
                    return reply(c->fd, h, VC4D_E_DENIED, NULL, 0, -1, 0, 0);
                }
                words[qpu][u] = w[u];
            }
            uni_bus[qpu] = arena_bus(&A, r->off + q.uni_off + 64u * qpu);
        }
    } else {
        for (uint32_t qpu = 0; qpu < q.nqpu; ++qpu) {
            for (uint32_t u = 0; u < q.nuni; ++u) {
                const struct vc4d_uni *e = &uni[(size_t)qpu * q.nuni + u];
                if (e->kind == VC4D_UNI_LIT) {
                    if (e->handle != 0 || e->len != 0) {
                        log_denial(c, "LAUNCH", "literal uniform with handle/len");
                        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
                    }
                    if (literal_forbidden(e->off)) {
                        log_denial(c, "LAUNCH",
                                   "literal uniform is a bus/arena/MMIO address");
                        return reply(c->fd, h, VC4D_E_DENIED, NULL, 0, -1, 0, 0);
                    }
                    words[qpu][u] = e->off;
                } else if (e->kind == VC4D_UNI_PTR) {
                    struct region *r = find_region(c, e->handle);
                    if (!r) {
                        log_denial(c, "LAUNCH", "uniform points at an unknown buffer");
                        return reply(c->fd, h, VC4D_E_NOENT, NULL, 0, -1, 0, 0);
                    }
                    if ((e->off & 3u) || (e->len & 3u) || !e->len ||
                        (uint64_t)e->off + e->len > r->len) {
                        log_denial(c, "LAUNCH", "uniform [off,len) leaves its buffer");
                        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
                    }
                    words[qpu][u] = arena_bus(&A, r->off + e->off);
                } else {
                    log_denial(c, "LAUNCH", "unknown uniform kind");
                    return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
                }
            }
        }

        /* materialise uniforms in the program's own 64 B x 12 area */
        for (uint32_t qpu = 0; qpu < q.nqpu; ++qpu) {
            uint8_t *dst = A.map + p->uni_off + (size_t)qpu * 64;
            memset(dst, 0, 64);             /* each launch is self-contained */
            memcpy(dst, words[qpu], (size_t)q.nuni * 4);
            uni_bus[qpu] = arena_bus(&A, p->uni_off + (uint32_t)qpu * 64);
        }
    }
    __sync_synchronize();

    uint32_t tmo = q.timeout_ms ? q.timeout_ms : DEFAULT_TIMEOUT;
    if (tmo > opt.max_timeout_ms)
        tmo = opt.max_timeout_ms;

    if (opt.gpu_lock && gpu_lock_fd >= 0) {
        int lr = gpu_lock_take(tmo < 30000u ? tmo : 30000u);
        if (lr != 0) {
            log_denial(c, "LAUNCH", lr == -2 ? "GPU lock wait timed out"
                                             : "GPU lock failed");
            return reply(c->fd, h, lr == -2 ? VC4D_E_BUSY : VC4D_E_IO,
                         NULL, 0, -1, 0, 0);
        }
    }

    char err[256];
    int rc = gpu_kick(&G, arena_bus(&A, p->code_off), uni_bus,
                      (int)q.nqpu, tmo, err, sizeof(err));
    if (rc != 0) {
        gpu_lock_drop();
        logmsg("launch failed: %s", err);
        st_failed++;
        return reply(c->fd, h, rc, NULL, 0, -1, 0, 0);
    }
    c->inflight = 1;
    c->launch_id = next_launch++;
    if (!next_launch)
        next_launch = 1;
    st_launches++;
    uint32_t id = c->launch_id;
    if (q.flags & VC4D_LAUNCH_WAIT) {
        struct launch_result lr;
        char werr[256] = "";
        int wst = await_launch(c, id, &lr, werr, sizeof(werr));
        if (wst == VC4D_E_TIMEOUT)
            return reply(c->fd, h, wst, NULL, 0, -1, id, 0);
        struct vc4d_wait_rsp wr = { .launch = id, .errs = lr.errs,
                                    .wall_ns = lr.wall_ns, .cycles = lr.cycles };
        return reply(c->fd, h, wst, &wr, sizeof(wr), -1, 0, 0);
    }
    struct vc4d_launch_rsp r = { .launch = id, .nqpu = q.nqpu };
    return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
}

static int handle_wait(struct client *c, const struct vc4d_hdr *h,
                       const uint8_t *pl, size_t plen)
{
    (void)pl;
    if (plen != 0)
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    if (!c->inflight) {
        log_denial(c, "WAIT", "this client has no launch in flight");
        return reply(c->fd, h, VC4D_E_NOENT, NULL, 0, -1, 0, 0);
    }
    uint32_t id = c->launch_id;
    struct launch_result lr;
    char err[256] = "";
    int status = await_launch(c, id, &lr, err, sizeof(err));
    if (status == VC4D_E_TIMEOUT)
        return reply(c->fd, h, status, NULL, 0, -1, id, 0);
    struct vc4d_wait_rsp r = { .launch = id, .errs = lr.errs,
                               .wall_ns = lr.wall_ns, .cycles = lr.cycles };
    return reply(c->fd, h, status, &r, sizeof(r), -1, 0, 0);
}

static int handle_read_regs(struct client *c, const struct vc4d_hdr *h,
                            const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_regs_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_regs_req q;
    memcpy(&q, pl, sizeof(q));
    if (q.count == 0) {
        struct vc4d_regs r;
        memset(&r, 0, sizeof(r));
        r.ident0 = G.regs[R_IDENT0 / 4];
        r.ident1 = G.regs[R_IDENT1 / 4];
        r.ident2 = G.regs[0x0008 / 4];
        r.ident3 = G.regs[0x000c / 4];
        r.l2cactl = G.regs[R_L2CACTL / 4];
        r.slcactl = G.regs[R_SLCACTL / 4];
        r.vpmbase = G.regs[R_VPMBASE / 4];
        r.vpacntl = G.regs[R_VPACNTL / 4];
        r.intctl = G.regs[R_INTCTL / 4];
        r.intena = G.regs[R_INTENA / 4];
        r.intdis = G.regs[R_INTDIS / 4];
        r.errors = G.regs[R_ERRSTAT / 4];
        r.srqcs = G.regs[R_SRQCS / 4];
        r.srqpc = G.regs[R_SRQPC / 4];
        r.srqua = G.regs[R_SRQUA / 4];
        r.srqul = G.regs[R_SRQUL / 4];
        r.cycles = G.regs[R_PCTR_VAL / 4];
        return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
    }
    if ((q.off & 3u) || q.count > 1024 || q.off > V3D_LENGTH ||
        (uint64_t)q.off + (uint64_t)q.count * 4 > V3D_LENGTH) {
        log_denial(c, "READ_REGS", "range outside the register block");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    uint8_t buf[sizeof(struct vc4d_regs_req) + 1024 * 4];
    memcpy(buf, &q, sizeof(q));
    for (uint32_t i = 0; i < q.count; ++i)
        memcpy(buf + sizeof(q) + (size_t)i * 4,
               (const void *)&G.regs[q.off / 4 + i], 4);
    return reply(c->fd, h, VC4D_OK, buf, sizeof(q) + (size_t)q.count * 4, -1, 0, 0);
}

static int handle_writereg(struct client *c, const struct vc4d_hdr *h,
                           const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_wreg_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_wreg_req q;
    memcpy(&q, pl, sizeof(q));
    if ((q.off & 3u) || q.off >= V3D_LENGTH)
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    if (!opt.raw_srq &&
        (q.off == R_SRQPC || q.off == R_SRQUA || q.off == R_SRQUL ||
         q.off == R_SRQCS || q.off == R_VPMBASE)) {
        log_denial(c, "WRITEREG", "raw SRQ/VPMBASE writes are disabled");
        return reply(c->fd, h, VC4D_E_DENIED, NULL, 0, -1, 0, 0);
    }
    char err[128];
    uint32_t old = G.regs[q.off / 4];
    int rc = gpu_wreg(&G, q.off, q.val, err, sizeof(err));
    if (rc != 0) {
        log_denial(c, "WRITEREG", err);
        return reply(c->fd, h, rc, NULL, 0, -1, 0, 0);
    }
    st_wreg++;
    return reply(c->fd, h, VC4D_OK, NULL, 0, -1, old, 0);
}

static int handle_stats(struct client *c, const struct vc4d_hdr *h,
                        const uint8_t *pl, size_t plen)
{
    (void)pl;
    if (plen != 0)
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_stats_rsp r;
    memset(&r, 0, sizeof(r));
    r.launches = st_launches;
    r.launches_failed = st_failed;
    r.bytes_uploaded = st_bytes_up;
    r.reg_writes = st_wreg;
    r.rejects = st_rejects;
    r.arena_bytes = A.bytes;
    r.arena_used = A.used;
    r.gpu_ns = G.gpu_ns;
    r.clients = (uint32_t)st_clients;
    r.max_clients = MAX_CLIENTS;
    r.programs = (uint32_t)st_progs;
    r.dead = (uint32_t)G.dead;
    return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
}

static int handle_sync(struct client *c, const struct vc4d_hdr *h,
                       const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_sync_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_sync_req q;
    memcpy(&q, pl, sizeof(q));
    if (!find_region(c, q.handle) || q.dir > VC4D_SYNC_G2R) {
        log_denial(c, "SYNC", "bad handle or direction");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    G.regs[R_L2CACTL / 4] = 1u << 2;    /* write back + invalidate */
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
    return reply(c->fd, h, VC4D_OK, NULL, 0, -1, 0, 0);
}

static int dispatch(struct client *c, const struct vc4d_hdr *h,
                    const uint8_t *pl, size_t plen)
{
    switch (h->op) {
    case VC4D_OP_HELLO:     return handle_hello(c, h, pl, plen);
    case VC4D_OP_ALLOC:     return handle_alloc(c, h, pl, plen);
    case VC4D_OP_FREE:      return handle_free(c, h, pl, plen);
    case VC4D_OP_UPLOAD:    return handle_upload(c, h, pl, plen);
    case VC4D_OP_LAUNCH:    return handle_launch(c, h, pl, plen);
    case VC4D_OP_WAIT:      return handle_wait(c, h, pl, plen);
    case VC4D_OP_READ_REGS: return handle_read_regs(c, h, pl, plen);
    case VC4D_OP_WRITEREG:  return handle_writereg(c, h, pl, plen);
    case VC4D_OP_STATS:     return handle_stats(c, h, pl, plen);
    case VC4D_OP_SYNC:      return handle_sync(c, h, pl, plen);
    case VC4D_OP_PING:      return reply(c->fd, h, VC4D_OK, NULL, 0, -1, 1, 0);
    case VC4D_OP_CLOSE:
        reply(c->fd, h, VC4D_OK, NULL, 0, -1, 0, 0);
        return 1;
    default:
        log_denial(c, "?", "unknown opcode");
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    }
}

/* --------------------------------------------------------------- clients */
static void client_destroy(int idx)
{
    struct client *c = clients[idx];
    if (!c)
        return;
    if (c->inflight) {
        logmsg("client pid=%d disconnected with launch %u in flight",
               (int)c->pid, c->launch_id);
        int rc = 0;
        while ((rc = gpu_poll(&G)) == 0 && now_ns() <= G.deadline)
            usleep(50);
        if (rc <= 0) {
            gpu_abort(&G);
            logmsg("launch %u aborted; daemon marked dead", c->launch_id);
        } else {
            struct launch_result lr;
            char err[128];
            gpu_finish(&G, &lr, err, sizeof(err));
        }
        gpu_lock_drop();
    }
    release_client_resources(c);
    close(c->fd);
    free(c);
    clients[idx] = NULL;
    st_clients--;
}

static void accept_client(void)
{
    struct sockaddr_un sa;
    socklen_t sl = sizeof(sa);
    int fd = accept4(listen_fd, (struct sockaddr *)&sa, &sl, SOCK_CLOEXEC);
    if (fd < 0)
        return;
    int slot = -1;
    for (int i = 0; i < MAX_CLIENTS; ++i)
        if (!clients[i]) { slot = i; break; }
    if (slot < 0) {
        logmsg("DENY connection: too many clients");
        close(fd);
        return;
    }
    struct client *c = calloc(1, sizeof(*c));
    if (!c) {
        close(fd);
        return;
    }
    c->idx = slot;
    c->fd = fd;
    struct ucred cr;
    socklen_t cl = sizeof(cr);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cr, &cl) == 0) {
        c->uid = cr.uid;
        c->pid = cr.pid;
    }
    int sz = 128 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    clients[slot] = c;
    st_clients++;
    if (st_clients > st_clients_max)
        st_clients_max = st_clients;
    if (opt.verbose)
        logmsg("connect uid=%u pid=%d", (unsigned)c->uid, (int)c->pid);
}

/* ------------------------------------------------------------ main loop */
static void drop_privileges(void)
{
    struct passwd *pw = getpwnam(opt.drop_user);
    uid_t uid = pw ? pw->pw_uid : 65534;
    gid_t dgid = pw ? pw->pw_gid : 65534;
    if (!pw)
        logmsg("user '%s' not found, dropping to 65534", opt.drop_user);
    if (setgroups(0, NULL) != 0) {
        logmsg("setgroups: %s", strerror(errno));
        exit(1);
    }
    if (setresgid(dgid, dgid, dgid) != 0 || setresuid(uid, uid, uid) != 0) {
        logmsg("setresuid/gid: %s", strerror(errno));
        exit(1);
    }
    if (prctl(PR_SET_DUMPABLE, 0) != 0 ||
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0)
        logmsg("prctl: %s", strerror(errno));

    struct __user_cap_header_struct hdr;
    struct __user_cap_data_struct data[2];
    memset(&hdr, 0, sizeof(hdr));
    memset(data, 0, sizeof(data));
    hdr.version = _LINUX_CAPABILITY_VERSION_3;
    if (syscall(SYS_capset, &hdr, data) != 0)
        logmsg("capset(clear): %s", strerror(errno));
    if (geteuid() != uid || getegid() != dgid) {
        logmsg("privilege drop did not take effect");
        exit(1);
    }
    logmsg("dropped to uid=%u gid=%u, capabilities cleared",
           (unsigned)uid, (unsigned)dgid);
}

int main(int argc, char **argv)
{
    memset(&opt, 0, sizeof(opt));
    opt.socket_path = DEFAULT_SOCKET;
    opt.drop_user = "nobody";
    opt.group = "vc4";
    opt.arena_bytes = ARENA_DEFAULT;
    opt.max_timeout_ms = MAX_TIMEOUT;
    opt.raw_srq = 1;
    opt.drop = 1;

    const char *e;
    if ((e = getenv("VC4D_SOCKET"))) opt.socket_path = e;
    if ((e = getenv("VC4D_ALLOWLIST"))) opt.allowlist = e;
    if ((e = getenv("VC4D_STRICT"))) opt.strict = atoi(e) != 0;
    if ((e = getenv("VC4D_ARENA_MB"))) opt.arena_bytes = (size_t)strtoul(e, NULL, 0) << 20;
    if ((e = getenv("VC4D_GPU_LOCK"))) opt.gpu_lock = atoi(e) != 0;
    if ((e = getenv("VC4D_NO_DROP"))) opt.drop = atoi(e) == 0;
    if ((e = getenv("VC4D_VERBOSE"))) opt.verbose = atoi(e) != 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--socket") && i + 1 < argc) opt.socket_path = argv[++i];
        else if (!strcmp(argv[i], "--allowlist") && i + 1 < argc) opt.allowlist = argv[++i];
        else if (!strcmp(argv[i], "--arena-mb") && i + 1 < argc) opt.arena_bytes = (size_t)strtoul(argv[++i], NULL, 0) << 20;
        else if (!strcmp(argv[i], "--group") && i + 1 < argc) opt.group = argv[++i];
        else if (!strcmp(argv[i], "--drop-user") && i + 1 < argc) opt.drop_user = argv[++i];
        else if (!strcmp(argv[i], "--max-timeout-ms") && i + 1 < argc) opt.max_timeout_ms = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--strict")) opt.strict = 1;
        else if (!strcmp(argv[i], "--no-raw-srq")) opt.raw_srq = 0;
        else if (!strcmp(argv[i], "--no-drop")) opt.drop = 0;
        else if (!strcmp(argv[i], "--gpu-lock")) opt.gpu_lock = 1;
        else if (!strcmp(argv[i], "--verbose")) opt.verbose = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: vc4d [--socket PATH] [--arena-mb N] [--group NAME]\n"
                   "            [--drop-user NAME] [--allowlist FILE] [--strict]\n"
                   "            [--max-timeout-ms N] [--no-raw-srq] [--no-drop]\n"
                   "            [--gpu-lock] [--verbose]\n");
            return 0;
        } else {
            fprintf(stderr, "vc4d: unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (opt.strict) {
        opt.raw_srq = 0;
        if (!opt.allowlist)
            opt.allowlist = "/etc/vc4d/programs.sha256";
        if (opt.max_timeout_ms > 2000)
            opt.max_timeout_ms = 2000;
    }
    if (!opt.max_timeout_ms)
        opt.max_timeout_ms = MAX_TIMEOUT;
    if (opt.allowlist && access(opt.allowlist, R_OK) != 0) {
        logmsg("allowlist %s: %s", opt.allowlist, strerror(errno));
        return 1;
    }

    struct sigaction sact;
    memset(&sact, 0, sizeof(sact));
    sact.sa_handler = on_signal;
    sigaction(SIGTERM, &sact, NULL);
    sigaction(SIGINT, &sact, NULL);
    signal(SIGPIPE, SIG_IGN);

    char err[256];
    if (arena_open(&A, opt.arena_bytes, err, sizeof(err)) != 0) {
        logmsg("arena: %s", err);
        return 1;
    }
    logmsg("arena %zu MiB at phys 0x%08x bus 0x%08x",
           A.bytes >> 20, A.phys, A.bus);

    if (gpu_open(&G, err, sizeof(err)) != 0) {
        logmsg("gpu: %s", err);
        arena_close(&A);
        return 1;
    }
    logmsg("V3D IDENT0=0x%08x IDENT1=0x%08x errstat=0x%08x",
           G.regs[R_IDENT0 / 4], G.regs[R_IDENT1 / 4], G.errstat_base);

    struct group *gr = getgrnam(opt.group);
    if (!gr) {
        logmsg("group '%s' does not exist (create it and add clients)", opt.group);
        gpu_close(&G);
        arena_close(&A);
        return 1;
    }
    gid_t gid = gr->gr_gid;

    listen_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (listen_fd < 0) {
        logmsg("socket: %s", strerror(errno));
        return 1;
    }
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", opt.socket_path);
    unlink(opt.socket_path);
    if (bind(listen_fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        logmsg("bind %s: %s", opt.socket_path, strerror(errno));
        return 1;
    }
    if (chmod(opt.socket_path, 0660) != 0 || chown(opt.socket_path, 0, gid) != 0) {
        logmsg("chmod/chown %s: %s", opt.socket_path, strerror(errno));
        return 1;
    }
    if (listen(listen_fd, 16) != 0) {
        logmsg("listen: %s", strerror(errno));
        return 1;
    }
    logmsg("listening on %s (mode 0660 group %s gid %u)",
           opt.socket_path, opt.group, (unsigned)gid);

    if (opt.gpu_lock) {
        gpu_lock_fd = open(DEFAULT_GPU_LOCK, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (gpu_lock_fd < 0)
            /* fs.protected_regular refuses O_CREAT on a /tmp file owned by
               another user; flock does not need write access */
            gpu_lock_fd = open(DEFAULT_GPU_LOCK, O_RDONLY | O_CLOEXEC);
        if (gpu_lock_fd < 0) {
            logmsg("open %s: %s (continuing without interop lock)",
                   DEFAULT_GPU_LOCK, strerror(errno));
            opt.gpu_lock = 0;
        }
    }

    if (opt.drop)
        drop_privileges();

    /* close anything inherited that we do not need (0,1,2 are the journal) */
    for (int fd = 3; fd < 256; ++fd) {
        if (fd != A.fd && fd != listen_fd && fd != gpu_lock_fd)
            close(fd);
    }

    struct pollfd pfds[MAX_CLIENTS + 1];
    struct client *who[MAX_CLIENTS + 1];
    while (!stop_flag) {
        int n = 0;
        pfds[n].fd = listen_fd;
        pfds[n].events = POLLIN;
        pfds[n].revents = 0;
        who[n] = NULL;
        n++;
        for (int i = 0; i < MAX_CLIENTS; ++i) {
            if (clients[i]) {
                pfds[n].fd = clients[i]->fd;
                pfds[n].events = POLLIN;
                pfds[n].revents = 0;
                who[n] = clients[i];
                n++;
            }
        }
        int pr = poll(pfds, n, 1000);
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            logmsg("poll: %s", strerror(errno));
            break;
        }
        for (int pi = 0; pi < n; ++pi) {
            if (!pfds[pi].revents)
                continue;
            if (!who[pi]) {
                if (pfds[pi].revents & POLLIN)
                    accept_client();
                continue;
            }
            struct client *c = who[pi];
            if (!(pfds[pi].revents & (POLLIN | POLLHUP | POLLERR)))
                continue;

            static uint8_t rbuf[VC4D_MAX_MSG];
            static uint8_t cbuf[CMSG_SPACE(4 * sizeof(int))];
            struct iovec iov = { .iov_base = rbuf, .iov_len = sizeof(rbuf) };
            struct msghdr m;
            memset(&m, 0, sizeof(m));
            m.msg_iov = &iov;
            m.msg_iovlen = 1;
            m.msg_control = cbuf;
            m.msg_controllen = sizeof(cbuf);
            ssize_t got = recvmsg(c->fd, &m, MSG_TRUNC | MSG_CMSG_CLOEXEC);
            if (got < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                client_destroy(c->idx);
                continue;
            }
            int extra = 0;
            for (struct cmsghdr *cm = CMSG_FIRSTHDR(&m); cm; cm = CMSG_NXTHDR(&m, cm)) {
                if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) {
                    int nfd = (int)((cm->cmsg_len - CMSG_LEN(0)) / sizeof(int));
                    for (int k = 0; k < nfd; ++k) {
                        int f;
                        memcpy(&f, CMSG_DATA(cm) + k * sizeof(int), sizeof(int));
                        close(f);
                        extra++;
                    }
                }
            }
            if (got == 0) {
                client_destroy(c->idx);
                continue;
            }
            if (extra) {
                st_rejects++;
                log_denial(c, "?", "client sent an unexpected file descriptor");
                struct vc4d_hdr hh;
                memset(&hh, 0, sizeof(hh));
                reply(c->fd, &hh, VC4D_E_DENIED, NULL, 0, -1, 0, 0);
                continue;
            }
            if ((size_t)got > sizeof(rbuf) || (size_t)got < sizeof(struct vc4d_hdr) ||
                (m.msg_flags & MSG_CTRUNC)) {
                st_rejects++;
                int too_big = (size_t)got > sizeof(rbuf);
                log_denial(c, "?", too_big ? "oversized datagram" : "truncated datagram");
                struct vc4d_hdr hh;
                memset(&hh, 0, sizeof(hh));
                reply(c->fd, &hh, too_big ? VC4D_E_TOOLARGE : VC4D_E_PROTO,
                      NULL, 0, -1, (uint32_t)got, 0);
                if (c->denials > 1000)
                    client_destroy(c->idx);
                continue;
            }
            struct vc4d_hdr h;
            memcpy(&h, rbuf, sizeof(h));
            if (h.magic != VC4D_MAGIC || h.len != (uint32_t)got ||
                h.version != VC4D_VERSION || h.status != 0) {
                st_rejects++;
                char why[128];
                snprintf(why, sizeof(why),
                         "bad header (magic=0x%08x len=%u got=%zd ver=%u status=%d)",
                         h.magic, h.len, got, h.version, h.status);
                log_denial(c, "?", why);
                struct vc4d_hdr hh = h;
                hh.op = 0;
                hh.seq = 0;
                int st = (h.magic == VC4D_MAGIC && h.version != VC4D_VERSION)
                         ? VC4D_E_VERSION : VC4D_E_PROTO;
                reply(c->fd, &hh, st, NULL, 0, -1, 0, 0);
                if (c->denials > 1000)
                    client_destroy(c->idx);
                continue;
            }
            int rc = dispatch(c, &h, rbuf + sizeof(h), got - sizeof(h));
            if (rc != 0)
                client_destroy(c->idx);
        }
    }

    for (int i = 0; i < MAX_CLIENTS; ++i)
        if (clients[i])
            client_destroy(i);
    unlink(opt.socket_path);
    gpu_close(&G);
    arena_close(&A);
    logmsg("exit");
    return 0;
}
