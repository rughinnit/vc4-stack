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
#define MAX_PROGS_SLOTS  128
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
    /* G3: the slot table is an LRU keyed by name (or by code hash for a
     * client that sends no name).  Re-uploading a name reuses its slot; when
     * the table is full the least recently used slot that is not executing is
     * freed.  `busy` marks the one program in flight. */
    char     name[VC4D_NAME_MAX];
    uint64_t lru;
    int      busy;
};

struct client {
    int      idx;
    int      fd;
    uid_t    uid;
    pid_t    pid;
    struct region reg[MAX_REGIONS];
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
static uint64_t st_reclaimed, st_recoveries, st_evicted;
static int st_clients, st_clients_max, st_progs;
static int restart_needed;       /* recovery failed: exit after replying */
static const char *restart_reason = "GPU context could not be reset";
static uint64_t lru_clock;       /* monotonic use counter for the program LRU */
static int busy_slot = -1;       /* the program slot that is executing */

static void client_destroy(int idx);
static void reap_dead_clients(int keep, const char *why);

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

/* The daemon's known-good program: `thrend` (sig 3) followed by the three
 * branch-delay slots every generated kernel ends with, i.e. exactly the tail
 * of kgen's output.  One QPU, no operands, no memory access: ~1 us, 12 cycles,
 * zero error bits (measured on this board).  It lives in the reserved last
 * code slot and is the only thing gpu_recover() trusts as proof that the
 * context works again. */
static const uint32_t probe_words[8] = {
    0x9e70009eu, 0x30000000u,   /* thrend  */
    0x9e70009eu, 0x10000000u,   /* nop (delay 1) */
    0x9e70009eu, 0x10000000u,   /* nop (delay 2) */
    0x9e70009eu, 0x10000000u,   /* nop (delay 3) */
};

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

/* A launch timed out.  The old behaviour was to set G.dead and keep it set:
 * the daemon stayed up but every later client failed with
 *   daemon is dead (GPU timeout earlier)
 * until someone ran `systemctl restart vc4d` (G1 in docs/BUGS.md).
 *
 * Now the context is reset and verified with the probe program, under the same
 * board-wide GPU lock a launch takes, and only then are launches accepted
 * again.  If the reset cannot be verified (the QPU is genuinely stuck, the V3D
 * is powered down, the queue will not drain) the daemon leaves instead of
 * becoming a zombie: systemd's Restart=always brings up a clean one in ~2 s.
 * Exiting after a reply keeps the client's error meaningful (E_TIMEOUT /
 * E_GPU) rather than turning it into a socket EOF. */
static void try_recover(const char *why)
{
    char err[256] = "";
    uint64_t t0 = now_ns();
    int locked = 1;
    if (opt.gpu_lock && gpu_lock_fd >= 0) {
        int lr = gpu_lock_take(2000);
        if (lr != 0) {
            locked = 0;
            snprintf(err, sizeof(err), "GPU lock %s",
                     lr == -2 ? "wait timed out" : "failed");
        }
    }
    int rc = -1;
    if (locked)
        rc = gpu_recover(&G, err, sizeof(err));
    gpu_lock_drop();
    if (rc == 0) {
        st_recoveries++;
        logmsg("%s: GPU context reset and verified with a probe launch "
               "(%.1f ms); accepting launches again", why,
               (double)(now_ns() - t0) / 1e6);
        return;
    }
    logmsg("%s: GPU context reset FAILED: %s", why, err);
    logmsg("leaving so systemd (Restart=always) starts a clean daemon");
    restart_needed = 1;
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

/* ----------------------------------------------------- program slot LRU (G3)
 *
 * The old table had 128 slots (one reserved for the probe) and no way to
 * reclaim one: a session that touched more than 127 distinct kernels - the
 * full YOLOv8 graph plus a few shapes - got E_NOMEM from UPLOAD for the rest
 * of its life and every later launch failed until the daemon was restarted.
 * That is a transient condition (more shapes than slots) turned permanent.
 *
 * A slot is now freed by LRU.  The key is the program *name* the client sends
 * with UPLOAD (the code sha256 for a client that sends none), so re-uploading
 * a kernel after an eviction costs the slot that kernel had, not a new one.
 * The program in flight and the daemon's probe are never evicted.
 */
static void prog_touch(struct prog *p)
{
    p->lru = ++lru_clock;
}

static void prog_release(int slot)
{
    struct prog *p = &progs[slot];
    if (!p->used)
        return;
    p->used = 0;
    p->owner = -1;
    p->name[0] = 0;
    if (st_progs > 0)
        st_progs--;
}

/* Return the slot to write this upload into, or -1.  found_out tells the
 * caller that the slot is the same program (same name) and must keep its
 * handle generation; evicted_out counts an LRU eviction. */
static int prog_slot_for(struct client *c, const char *name,
                         const uint8_t hash[32], uint32_t max_qpu,
                         uint32_t nuni, int *same, uint64_t *evicted_out)
{
    if (same)
        *same = 0;
    if (evicted_out)
        *evicted_out = 0;

    /* 1. the same program, already resident: reuse it in place */
    for (int i = 0; i < PROBE_SLOT; ++i) {
        struct prog *p = &progs[i];
        if (!p->used || p->owner != c->idx)
            continue;
        int match;
        if (name && *name)
            match = strcmp(p->name, name) == 0;
        else
            match = !p->name[0] && p->nqpu_max == max_qpu &&
                    p->nuni == nuni && memcmp(p->hash, hash, 32) == 0;
        if (!match)
            continue;
        if (p->busy)
            return -1;              /* the copy that is executing; refuse */
        if (same)
            *same = 1;
        return i;
    }

    /* 2. a free slot */
    for (int i = 0; i < PROBE_SLOT; ++i)
        if (!progs[i].used)
            return i;

    /* 3. evict the least recently used slot that is not executing: first from
     * this client (its own handles go stale, which it can recover from by
     * re-uploading), then from another client if this one owns nothing else. */
    for (int pass = 0; pass < 2; ++pass) {
        int best = -1;
        for (int i = 0; i < PROBE_SLOT; ++i) {
            struct prog *p = &progs[i];
            if (!p->used || p->busy)
                continue;
            if (pass == 0 && p->owner != c->idx)
                continue;
            if (best < 0 || p->lru < progs[best].lru)
                best = i;
        }
        if (best >= 0) {
            logmsg("evict prog slot %d uid=%u name=%s (LRU, %llu slots in use)",
                   best, (unsigned)c->uid,
                   progs[best].name[0] ? progs[best].name : "(unnamed)",
                   (unsigned long long)st_progs);
            prog_release(best);
            st_evicted++;
            if (evicted_out)
                *evicted_out = 1;
            return best;
        }
    }
    return -1;
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
            if (progs[i].busy)
                progs[i].busy = 0;
            progs[i].used = 0;
            progs[i].owner = -1;
            progs[i].name[0] = 0;
            if (st_progs > 0)
                st_progs--;
        }
    }
}

/* --------------------------------------------------------------- handlers */
static int handle_hello(struct client *c, const struct vc4d_hdr *h,
                        const uint8_t *pl, size_t plen)
{
    if (plen != sizeof(struct vc4d_hello_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_hello_req q;
    memcpy(&q, pl, sizeof(q));
    if (q.version < VC4D_VERSION_MIN || q.version > VC4D_VERSION) {
        log_denial(c, "HELLO", "version mismatch");
        return reply(c->fd, h, VC4D_E_VERSION, NULL, 0, -1, VC4D_VERSION, 0);
    }
    /* A client arriving is the natural moment to sweep: anything left over
       from a process that has already exited is released before this client
       asks for arena of its own. */
    reap_dead_clients(c->idx, "gone before a new client's HELLO");
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
              (opt.allowlist ? VC4D_F_ALLOWLIST : 0) |
              VC4D_F_NAMED_PROGS | VC4D_F_UNI_MASK | VC4D_F_ARENA_RESTORE |
              VC4D_F_LAUNCH_BATCH | VC4D_F_ASYNC_SUBMIT;
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
    /* One line per grant: when a client later reports a nonsense arena, the
       numbers it was given are the first thing to compare (B12). */
    if (opt.verbose)
        logmsg("ALLOC uid=%u pid=%d bytes=%u -> off=0x%x len=0x%x bus=0x%08x",
               (unsigned)c->uid, (int)c->pid, q.bytes, off, len, r.bus);
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

    if (q.flags & ~VC4D_UPLOAD_NAMED) {
        log_denial(c, "UPLOAD", "unknown upload flags");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    const int named = (q.flags & VC4D_UPLOAD_NAMED) != 0;
    size_t avail = plen - sizeof(q);
    if (q.bytes > avail)
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    size_t extra = avail - q.bytes;
    const uint8_t *namep = NULL;
    size_t namelen = 0;
    if (named) {
        if (extra < 1 || extra >= VC4D_NAME_MAX) {
            log_denial(c, "UPLOAD", "program name has a bad length");
            return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
        }
        namep = code + q.bytes;
        namelen = extra;
        for (size_t i = 0; i < namelen; ++i) {
            if (namep[i] < 0x21 || namep[i] > 0x7e) {
                log_denial(c, "UPLOAD", "program name has a non-printable byte");
                return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
            }
        }
    } else if (extra != 0) {
        log_denial(c, "UPLOAD", "payload length disagrees with bytes");
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    }
    char name[VC4D_NAME_MAX];
    name[0] = 0;
    if (named) {
        memcpy(name, namep, namelen);
        name[namelen] = 0;
    }
    const size_t clen = q.bytes;
    if (q.bytes < 8 || (q.bytes & 7u) || q.bytes > VC4D_MAX_PROG ||
        q.max_qpu < 1 || q.max_qpu > VC4D_MAX_QPU ||
        q.nuni < 1 || q.nuni > VC4D_MAX_UNI) {
        log_denial(c, "UPLOAD", "bad size/alignment/max_qpu");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }

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

    int same = 0;
    uint64_t evicted = 0;
    int slot = prog_slot_for(c, name, hash, q.max_qpu, q.nuni, &same, &evicted);
    if (slot < 0) {
        log_denial(c, "UPLOAD", "every program slot is executing or reserved");
        return reply(c->fd, h, VC4D_E_NOMEM, NULL, 0, -1, 0, 0);
    }

    struct prog *p = &progs[slot];
    uint32_t gen = same ? p->gen : ((p->gen + 1u) & 0xFFFFFFu);
    if (!gen)
        gen = 1;
    int was_used = same;
    p->used = 1;
    p->owner = c->idx;
    p->gen = gen;
    p->code_off = A_CODE_BASE + (uint32_t)slot * A_CODE_SLOT;
    p->uni_off = A_UNI_BASE + (uint32_t)slot * A_UNI_SLOT;
    p->nqpu_max = q.max_qpu;
    p->nuni = q.nuni;
    p->words = q.bytes / 8;
    memcpy(p->hash, hash, 32);
    snprintf(p->name, sizeof(p->name), "%s", name);
    p->busy = 0;
    prog_touch(p);
    if (!was_used)
        st_progs++;

    memcpy(A.map + p->code_off, code, clen);
    memset(A.map + p->code_off + clen, 0, A_CODE_SLOT - clen);
    memset(A.map + p->uni_off, 0, A_UNI_SLOT);
    __sync_synchronize();
    st_bytes_up += clen;

    uint32_t hnd = (gen << 8) | (uint32_t)(slot + 1);
    struct vc4d_upload_rsp r = { .prog = hnd, .code_off = p->code_off,
                                 .words = p->words };
    if (opt.verbose)
        logmsg("UPLOAD uid=%u pid=%d %s%s%s slot=%d prog=%u words=%u%s",
               (unsigned)c->uid, (int)c->pid,
               name[0] ? "name=" : "", name[0] ? name : "(unnamed)",
               same ? " (resident)" : "", slot, hnd, p->words,
               evicted ? " [LRU eviction]" : "");
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
 * touching the op code.
 *
 * This guess is only used for a client that sends no VC4D_LAUNCH_UNI_MASK: a
 * value it cannot classify (every float <= -2.0f is >= 0xC0000000) is refused.
 * With the mask the client says which words are addresses and the guess is
 * not made at all. */
static int arena_word_ok(uint32_t v)
{
    if (v >= A.bus && (uint64_t)v < (uint64_t)A.bus + A.bytes)
        return 1;
    return !literal_forbidden(v);
}

/* UNI_MASK: this slot was declared an address, so it must land in this
 * client's arena - that is the whole check, and it is exact. */
static int arena_addr_ok(uint32_t v)
{
    return v >= A.bus && (uint64_t)v < (uint64_t)A.bus + A.bytes;
}

/* Poll an in-flight launch to completion, release the GPU lock and make the
 * GPU's results visible to unprivileged readers (L2 write-back + invalidate).
 * Fills lr and returns a VC4D_* status.  Shared by WAIT and LAUNCH_WAIT. */
static void prog_unbusy(void)
{
    if (busy_slot >= 0 && busy_slot < MAX_PROGS_SLOTS)
        progs[busy_slot].busy = 0;
    busy_slot = -1;
}

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
        /* The deadline passed, but the launch may simply be slower than the
         * client asked for.  Give it a bounded grace to drain by itself: the
         * context reset below must never land under a kernel that is still
         * executing, or the completion counter it relies on is meaningless
         * (measured: the probe then never reports done).  1000 ms covers any
         * legitimate overrun of a timeout that is measured in milliseconds;
         * a genuinely stuck QPU does not finish in it, and is handled by the
         * abort + reset + probe path. */
        uint64_t grace = now_ns() + 1000ull * 1000000ull;
        int drained = 0;
        while (now_ns() < grace) {
            int pr = gpu_poll(&G);
            if (pr == 1) {
                drained = 1;
                break;
            }
            if (pr < 0)
                break;
            usleep(100);
        }
        if (drained) {
            struct launch_result tmp;
            char e2[128];
            gpu_finish(&G, &tmp, e2, sizeof(e2));   /* clears inflight */
        } else {
            gpu_abort(&G);                          /* clears the queue word */
        }
        c->inflight = 0;
        prog_unbusy();
        gpu_lock_drop();
        st_failed++;
        logmsg("TIMEOUT uid=%u pid=%d launch=%u (%s)", (unsigned)c->uid,
               (int)c->pid, id,
               drained ? "the kernel finished after its deadline"
                       : "the kernel did not finish; aborting the queue");
        try_recover("after a launch timeout");
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
    prog_unbusy();
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

/* Validate and execute one launch.  LAUNCH and LAUNCH_BATCH share this: it
 * never writes a reply (a batch needs the per-entry status instead), it
 * returns the VC4D_* status, and on a wait-mode launch it fills *lrp.  `pl`
 * is the launch payload: struct vc4d_launch_req followed, in the tagged
 * uniform form only, by nqpu*nuni struct vc4d_uni.  *id_out receives the
 * launch id whenever a kick happened. */
static int launch_one(struct client *c, const uint8_t *pl, size_t plen,
                      int wait, struct launch_result *lrp, uint32_t *id_out)
{
    if (plen < sizeof(struct vc4d_launch_req))
        return VC4D_E_PROTO;
    struct vc4d_launch_req q;
    memcpy(&q, pl, sizeof(q));
    const struct vc4d_uni *uni = (const struct vc4d_uni *)(pl + sizeof(q));
    const int in_arena = (q.flags & VC4D_LAUNCH_UNI_ARENA) != 0;
    const int have_mask = (q.flags & VC4D_LAUNCH_UNI_MASK) != 0;

    if (q.flags & ~VC4D_LAUNCH_FLAGS_ALL) {
        log_denial(c, "LAUNCH", "unknown launch flags");
        return VC4D_E_INVAL;
    }
    if (have_mask && !in_arena) {
        log_denial(c, "LAUNCH", "UNI_MASK without UNI_IN_ARENA");
        return VC4D_E_INVAL;
    }
    if (q.nqpu < 1 || q.nqpu > VC4D_MAX_QPU || q.nuni < 1 ||
        q.nuni > VC4D_MAX_UNI) {
        log_denial(c, "LAUNCH", "nqpu/nuni out of range");
        return VC4D_E_INVAL;
    }
    if (have_mask && (q.uni_addr >> q.nuni) != 0) {
        log_denial(c, "LAUNCH", "UNI_MASK sets a bit past the last uniform word");
        return VC4D_E_INVAL;
    }
    if (in_arena) {
        if (plen != sizeof(q)) {
            log_denial(c, "LAUNCH", "UNI_IN_ARENA carries extra payload");
            return VC4D_E_PROTO;
        }
    } else if (plen != sizeof(q) + (size_t)q.nqpu * q.nuni * sizeof(*uni)) {
        log_denial(c, "LAUNCH", "uniform payload length mismatch");
        return VC4D_E_PROTO;
    }
    struct prog *p = find_prog(c, q.prog);
    if (!p) {
        log_denial(c, "LAUNCH", "unknown program handle");
        return VC4D_E_NOENT;
    }
    if (q.nqpu > p->nqpu_max) {
        log_denial(c, "LAUNCH", "nqpu exceeds the program's declared maximum");
        return VC4D_E_INVAL;
    }
    if (q.nuni != p->nuni) {
        log_denial(c, "LAUNCH", "nuni differs from the count declared at UPLOAD");
        return VC4D_E_INVAL;
    }
    if (G.inflight || c->inflight) {
        log_denial(c, "LAUNCH", "a launch is already in flight");
        return VC4D_E_BUSY;
    }

    uint32_t words[VC4D_MAX_QPU][VC4D_MAX_UNI];
    uint32_t uni_bus[VC4D_MAX_QPU];

    if (in_arena) {
        struct region *r = find_region(c, q.uni_handle);
        if (!r) {
            log_denial(c, "LAUNCH", "uniform block in an unknown region");
            return VC4D_E_NOENT;
        }
        if ((q.uni_off & 63u) ||
            (uint64_t)q.uni_off + 64ull * q.nqpu > r->len) {
            log_denial(c, "LAUNCH", "uniform block leaves its region");
            return VC4D_E_INVAL;
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
                if (have_mask) {
                    /* The client declared where its addresses are.  Validate
                     * exactly those; a word with the bit clear is a value, so
                     * a float whose bits look like a bus address is legal. */
                    if (((q.uni_addr >> u) & 1u) && !arena_addr_ok(w[u])) {
                        log_denial(c, "LAUNCH",
                                   "declared address uniform is outside the arena");
                        return VC4D_E_DENIED;
                    }
                } else if (!arena_word_ok(w[u])) {
                    log_denial(c, "LAUNCH",
                               "arena uniform word is outside the arena");
                    return VC4D_E_DENIED;
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
                        return VC4D_E_INVAL;
                    }
                    if (literal_forbidden(e->off)) {
                        log_denial(c, "LAUNCH",
                                   "literal uniform is a bus/arena/MMIO address");
                        return VC4D_E_DENIED;
                    }
                    words[qpu][u] = e->off;
                } else if (e->kind == VC4D_UNI_PTR) {
                    struct region *r = find_region(c, e->handle);
                    if (!r) {
                        log_denial(c, "LAUNCH", "uniform points at an unknown buffer");
                        return VC4D_E_NOENT;
                    }
                    if ((e->off & 3u) || (e->len & 3u) || !e->len ||
                        (uint64_t)e->off + e->len > r->len) {
                        log_denial(c, "LAUNCH", "uniform [off,len) leaves its buffer");
                        return VC4D_E_INVAL;
                    }
                    words[qpu][u] = arena_bus(&A, r->off + e->off);
                } else {
                    log_denial(c, "LAUNCH", "unknown uniform kind");
                    return VC4D_E_INVAL;
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

    prog_touch(p);          /* LRU: a launch is the strongest use */

    if (opt.gpu_lock && gpu_lock_fd >= 0) {
        int lr = gpu_lock_take(tmo < 30000u ? tmo : 30000u);
        if (lr != 0) {
            log_denial(c, "LAUNCH", lr == -2 ? "GPU lock wait timed out"
                                             : "GPU lock failed");
            return lr == -2 ? VC4D_E_BUSY : VC4D_E_IO;
        }
    }

    char err[256];
    int rc = gpu_kick(&G, arena_bus(&A, p->code_off), uni_bus,
                      (int)q.nqpu, tmo, err, sizeof(err));
    if (rc != 0) {
        gpu_lock_drop();
        logmsg("launch failed: %s", err);
        st_failed++;
        return rc;
    }
    /* This program must not be evicted while it is executing (G3). */
    p->busy = 1;
    busy_slot = (int)(p - progs);
    c->inflight = 1;
    c->launch_id = next_launch++;
    if (!next_launch)
        next_launch = 1;
    st_launches++;
    uint32_t id = c->launch_id;
    if (id_out)
        *id_out = id;
    if (!wait)
        return VC4D_OK;
    char werr[256] = "";
    return await_launch(c, id, lrp, werr, sizeof(werr));
}

static int handle_launch(struct client *c, const struct vc4d_hdr *h,
                         const uint8_t *pl, size_t plen)
{
    if (plen < sizeof(struct vc4d_launch_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_launch_req q;
    memcpy(&q, pl, sizeof(q));
    const int wait = (q.flags & VC4D_LAUNCH_WAIT) != 0;
    struct launch_result lr;
    uint32_t id = 0;
    memset(&lr, 0, sizeof(lr));
    int st = launch_one(c, pl, plen, wait, &lr, &id);
    if (st != VC4D_OK) {
        if (wait && st == VC4D_E_TIMEOUT)
            return reply(c->fd, h, st, NULL, 0, -1, id, 0);
        return reply(c->fd, h, st, NULL, 0, -1, 0, 0);
    }
    if (wait) {
        struct vc4d_wait_rsp wr = { .launch = id, .errs = lr.errs,
                                    .wall_ns = lr.wall_ns, .cycles = lr.cycles };
        return reply(c->fd, h, VC4D_OK, &wr, sizeof(wr), -1, 0, 0);
    }
    struct vc4d_launch_rsp r = { .launch = id, .nqpu = q.nqpu };
    return reply(c->fd, h, VC4D_OK, &r, sizeof(r), -1, 0, 0);
}

/* One request, an ordered list of launches, one completion.
 *
 * The entries are executed exactly as separate LAUNCH messages would be: in
 * order, each kicked only after the previous one completed, with the same
 * validation, the same per-launch timeout and the same L2 write-back.  The
 * difference is the host cost: one sendmsg/recvmsg pair and one daemon
 * dispatch for the whole list instead of one each.
 *
 * An entry that fails validation or execution stops the batch: entries before
 * it have run and completed, entries after it are reported VC4D_E_NOTRUN and
 * never touched.  That is what makes the reply usable for a client that has
 * queued dependent kernels: it never has to guess which ones ran. */
static int handle_batch(struct client *c, const struct vc4d_hdr *h,
                        const uint8_t *pl, size_t plen)
{
    if (plen < sizeof(struct vc4d_batch_req))
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    struct vc4d_batch_req q;
    memcpy(&q, pl, sizeof(q));
    if (q.flags != 0) {
        log_denial(c, "BATCH", "reserved flags set");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (q.count < 1 || q.count > VC4D_BATCH_MAX) {
        log_denial(c, "BATCH", "entry count out of range");
        return reply(c->fd, h, VC4D_E_INVAL, NULL, 0, -1, 0, 0);
    }
    if (plen != sizeof(q) + (size_t)q.count * sizeof(struct vc4d_batch_ent)) {
        log_denial(c, "BATCH", "payload length does not match the count");
        return reply(c->fd, h, VC4D_E_PROTO, NULL, 0, -1, 0, 0);
    }
    if (G.inflight || c->inflight) {
        log_denial(c, "BATCH", "a launch is already in flight");
        return reply(c->fd, h, VC4D_E_BUSY, NULL, 0, -1, 0, 0);
    }

    const struct vc4d_batch_ent *ent =
        (const struct vc4d_batch_ent *)(pl + sizeof(q));

    struct vc4d_batch_item item[VC4D_BATCH_MAX];
    memset(item, 0, sizeof(item));
    for (uint32_t i = 0; i < q.count; ++i)
        item[i].status = VC4D_E_NOTRUN;

    int first = VC4D_OK;
    uint32_t fail_idx = 0xffffffffu;

    for (uint32_t i = 0; i < q.count; ++i) {
        const struct vc4d_batch_ent *e = &ent[i];
        if (e->kind == VC4D_BENT_FENCE) {
            if (e->dir > VC4D_SYNC_G2R || !find_region(c, e->handle)) {
                log_denial(c, "BATCH", "fence with a bad handle or direction");
                item[i].status = VC4D_E_INVAL;
                first = VC4D_E_INVAL;
                fail_idx = i;
                break;
            }
            G.regs[R_L2CACTL / 4] = 1u << 2;    /* write back + invalidate */
#if defined(__aarch64__)
            __asm__ volatile("dsb sy" ::: "memory");
#else
            __sync_synchronize();
#endif
            item[i].status = VC4D_OK;
            continue;
        }
        if (e->kind != VC4D_BENT_LAUNCH) {
            log_denial(c, "BATCH", "unknown entry kind");
            item[i].status = VC4D_E_INVAL;
            first = VC4D_E_INVAL;
            fail_idx = i;
            break;
        }
        /* A batch carries no per-word uniform array: the words are in the
         * client's arena, which is also what bounds the message size.  WAIT
         * is implied and must not be requested explicitly. */
        if (!(e->flags & VC4D_LAUNCH_UNI_ARENA) ||
            (e->flags & VC4D_LAUNCH_WAIT) || e->handle || e->dir) {
            log_denial(c, "BATCH",
                       "launch entry is not a UNI_IN_ARENA launch");
            item[i].status = VC4D_E_INVAL;
            first = VC4D_E_INVAL;
            fail_idx = i;
            break;
        }
        struct vc4d_launch_req lq = {
            .prog = e->prog,
            .timeout_ms = e->timeout_ms,
            .nqpu = e->nqpu,
            .nuni = e->nuni,
            .flags = e->flags,
            .uni_handle = e->uni_handle,
            .uni_off = e->uni_off,
            .uni_addr = e->uni_addr,
        };
        struct launch_result lr;
        uint32_t id = 0;
        memset(&lr, 0, sizeof(lr));
        int st = launch_one(c, (const uint8_t *)&lq, sizeof(lq), 1, &lr, &id);
        item[i].status = (uint32_t)st;
        item[i].launch = id;
        item[i].errs = lr.errs;
        item[i].wall_ns = lr.wall_ns;
        item[i].cycles = lr.cycles;
        if (st != VC4D_OK) {
            first = st;
            fail_idx = i;
            break;
        }
    }

    uint32_t done = 0;
    for (uint32_t i = 0; i < q.count; ++i)
        if (item[i].status != VC4D_E_NOTRUN)
            done = i + 1;

    size_t plen_out = (size_t)q.count * sizeof(struct vc4d_batch_item);
    return reply(c->fd, h, first, item, plen_out, -1, done, fail_idx);
}

/* SUBMIT (protocol 4) is LAUNCH_BATCH with a looser connection contract: the
 * client may have VC4D_SUBMIT_MAX of them outstanding and the daemon serves
 * them in arrival order.  The execution is identical - the same handler, the
 * same validation, the same completion - because the overlap happens on the
 * client's side of the socket, not the daemon's.  Anything else here (an id
 * table, an accepted-but-unrun queue) would add state that the arena and
 * program-slot lifetimes would then have to cover, and would buy nothing: a
 * single-threaded daemon cannot start a second request while the first is
 * executing anyway. */
static int handle_submit(struct client *c, const struct vc4d_hdr *h,
                         const uint8_t *pl, size_t plen)
{
    return handle_batch(c, h, pl, plen);
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
    r.evictions = st_evicted;
    r.arena_want = (uint32_t)A.want;
    r.arena_degraded = (uint32_t)A.degraded;
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
    case VC4D_OP_LAUNCH_BATCH: return handle_batch(c, h, pl, plen);
    case VC4D_OP_SUBMIT:    return handle_submit(c, h, pl, plen);
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
            logmsg("launch %u aborted", c->launch_id);
            gpu_lock_drop();
            try_recover("after a client left a launch in flight");
        } else {
            struct launch_result lr;
            char err[128];
            gpu_finish(&G, &lr, err, sizeof(err));
            gpu_lock_drop();
        }
        prog_unbusy();
    }
    release_client_resources(c);
    close(c->fd);
    free(c);
    clients[idx] = NULL;
    st_clients--;
}

/* ------------------------------------------------------- client liveness --
 *
 * A region belongs to the socket that allocated it, and until now it was
 * released only when the daemon saw that socket close (POLLHUP / a 0-byte
 * read).  A client whose socket outlives the process that allocated -- a
 * fork, an inherited or leaked descriptor -- therefore pinned its arena
 * forever: measured on this board, one 8 MiB region survived its allocating
 * process and the next client was denied with `op=ALLOC: arena exhausted`
 * while nothing was running (G9 in docs/BUGS.md).
 *
 * Liveness is anchored on the peer pid recorded at accept time (SO_PEERCRED),
 * not on the socket.  kill(pid, 0) answers "gone" with ESRCH; EPERM means the
 * process exists but belongs to another user, which is the normal case for a
 * daemon that dropped to `nobody`.  In addition the socket itself is peeked:
 * a peer that has shut its end down reads as EOF even if its pid is alive, so
 * reclaiming never depends on the poll loop having seen the HUP.  The check is
 * two syscalls, cheap enough to run on every poll iteration, on HELLO and
 * before refusing a connection for lack of a slot.
 *
 * Trade-off, documented in vc4d/README.md: a client that forks and keeps
 * *using* the inherited connection after the parent exits is reaped.  It must
 * reconnect instead; the engine never does this.
 */
static int peer_alive(const struct client *c)
{
    if (c->pid > 0) {
        if (kill(c->pid, 0) != 0 && errno != EPERM)
            return 0;                   /* ESRCH: the peer process is gone */
    }
    /* Independent of poll's POLLHUP: a peer that has shut down its end of the
       socket reads as EOF.  A half-closed connection whose pid is still alive
       (a client that leaked the fd, a fork, a process that closed the socket
       but not its pid) would otherwise pin its regions until the pid exits;
       the sweep must not depend on the poll loop having seen the HUP.  MSG_PEEK
       does not consume a pending datagram, and -1/EAGAIN means "no data". */
    char b;
    ssize_t n = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0)
        return 0;                       /* EOF: the peer has gone away */
    return 1;
}

static int client_nregions(const struct client *c)
{
    int n = 0;
    for (int i = 0; i < MAX_REGIONS; ++i)
        if (c->reg[i].used)
            n++;
    return n;
}

/* Reap every client whose peer process has exited.  `keep` is a client index
 * to skip (the one currently being served): its pid can only be gone between
 * accept and HELLO, and destroying it under dispatch would use freed memory. */
static void reap_dead_clients(int keep, const char *why)
{
    for (int i = 0; i < MAX_CLIENTS; ++i) {
        struct client *c = clients[i];
        if (!c || i == keep || peer_alive(c))
            continue;
        int nr = client_nregions(c);
        logmsg("reclaim pid=%d uid=%u (%s): releasing %d region(s)%s",
               (int)c->pid, (unsigned)c->uid, why, nr,
               c->inflight ? " and a launch in flight" : "");
        client_destroy(i);
        st_reclaimed++;
    }
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
        /* a slot held by a process that has already exited is not a full
           table: reclaim it instead of turning the new client away */
        reap_dead_clients(-1, "a departed peer held the last slot");
        for (int i = 0; i < MAX_CLIENTS; ++i)
            if (!clients[i]) { slot = i; break; }
    }
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

/* ------------------------------------------------------------ arena restore
 *
 * arena_open() may have had to settle for less than `want` (G2).  That must not
 * last: once CMA has recovered and nothing is using the arena, exit and let
 * systemd start a clean daemon, which will get the full block.  A restart is
 * the only way to redo the allocation, because the daemon has dropped
 * CAP_SYS_RAWIO and closed /dev/mem by the time it serves clients - it cannot
 * open a bigger dma-buf itself, and keeping /dev/mem open in the unprivileged
 * process would give up the property the daemon exists for.
 *
 * The exit is gated on CMA *being able* to satisfy the request.  The daemon's
 * own arena counts towards that: exiting releases it.  That is the common
 * upgrade case - the daemon holds 48 MiB, CMA has 72 MiB free, and 96 MiB only
 * becomes available once the daemon lets go of its 48.
 *
 * A fragmented board can still fail the allocation after the restart, so each
 * deliberate restart is spaced out (5 min, doubling to 6 h) by a stamp file in
 * /run, which survives the restart but not a reboot.  A start that gets the
 * full block removes the stamp, so a real recovery is immediate next time.
 */
/* Keyed by the requested size: two daemons with different --arena-mb (the
 * installed one and a test one, say) must not reset each other's backoff.
 * The directory is created (owned by the drop user) before vc4d gives up its
 * privileges, because the stamp is written after it has dropped them; /run is
 * the unit's ReadWritePaths entry and survives a restart but not a reboot,
 * which is exactly the lifetime this state should have. */
#define ARENA_STAMP_FMT "%s/arena-restore-%zu.state"
static const char *state_dir = "/tmp";
#define ARENA_BACKOFF_MIN_S 60
#define ARENA_BACKOFF_MAX_S 21600

static uint64_t arena_restore_next;
static int arena_wait_logged;      /* "waiting for CMA" said once */
static int arena_backoff_logged;   /* "a restore already failed" said once */

static void arena_stamp_path(size_t want, char *out, size_t n)
{
    snprintf(out, n, ARENA_STAMP_FMT, state_dir, want >> 20);
}

static unsigned arena_backoff_attempts(size_t want)
{
    char path[128];
    arena_stamp_path(want, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    unsigned n = 0;
    if (fscanf(f, "%u", &n) != 1)
        n = 0;
    fclose(f);
    return n;
}

static time_t arena_stamp_mtime(size_t want)
{
    char path[128];
    arena_stamp_path(want, path, sizeof(path));
    struct stat st;
    if (stat(path, &st) != 0)
        return 0;
    return st.st_mtime;
}

static void arena_stamp_write(size_t want, unsigned attempts)
{
    char path[128];
    arena_stamp_path(want, path, sizeof(path));
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    fprintf(f, "%u\n", attempts);
    fclose(f);
}

static void arena_stamp_clear(size_t want)
{
    char path[128];
    arena_stamp_path(want, path, sizeof(path));
    unlink(path);
}

static void arena_maybe_restore(void)
{
    if (!A.degraded || restart_needed)
        return;
    uint64_t now = now_ns();
    if (now < arena_restore_next)
        return;
    arena_restore_next = now + 2000ull * 1000000ull;
    if (A.used || st_clients || G.inflight)
        return;                     /* wait for an idle arena, quietly */
    if (A.want <= A.bytes)
        return;
    size_t total = 0, cfree = 0;
    arena_cma_info(&total, &cfree);
    /* our own block is released by the exit, so it counts as available */
    uint64_t avail = (uint64_t)cfree + (uint64_t)A.bytes;
    if (avail < (uint64_t)A.want + (8u << 20)) {
        if (!arena_wait_logged)
            logmsg("arena is %zu MiB of the %zu MiB requested; CMA has %zu MiB "
                   "free, %zu MiB would be free after a restart: waiting for "
                   "CMA before restoring it", A.bytes >> 20, A.want >> 20,
                   cfree >> 20, avail >> 20);
        arena_wait_logged = 1;
        return;
    }
    unsigned n = arena_backoff_attempts(A.want);
    unsigned backoff = ARENA_BACKOFF_MIN_S;
    for (unsigned i = 1; i < n && backoff < ARENA_BACKOFF_MAX_S; ++i)
        backoff *= 2;
    if (backoff > ARENA_BACKOFF_MAX_S)
        backoff = ARENA_BACKOFF_MAX_S;
    time_t last = arena_stamp_mtime(A.want);
    if (n && last && time(NULL) < last + (time_t)backoff) {
        if (!arena_backoff_logged)
            logmsg("arena restore: %u attempt(s) already failed; next try in "
                   "%ld s (a fragmented CMA can refuse the full block even when "
                   "enough is free in pieces).  Stop the other CMA user and "
                   "restart vc4d to retry now.", n,
                   (long)(last + (time_t)backoff - time(NULL)));
        arena_backoff_logged = 1;
        return;
    }
    logmsg("arena was %zu MiB of the %zu MiB requested; CMA now has %zu MiB "
           "free (%zu MiB after this daemon releases its block) and the arena "
           "is idle - exiting so a fresh daemon claims the full block "
           "(no reboot needed)", A.bytes >> 20, A.want >> 20, cfree >> 20,
           avail >> 20);
    arena_stamp_write(A.want, n + 1);
    restart_reason = "the arena was degraded and CMA has recovered";
    restart_needed = 1;
}

/* ------------------------------------------------------------ main loop */
/* A directory the unprivileged daemon can write its own state into (the
 * arena-restore backoff).  Created while still root, chowned to the drop user;
 * falls back to /tmp (which the unit also allows) if /run is not writable. */
static void state_dir_init(uid_t uid, gid_t gid)
{
    if (mkdir("/run/vc4d", 0755) == 0 || errno == EEXIST) {
        if (chown("/run/vc4d", uid, gid) == 0 ||
            (access("/run/vc4d", W_OK) == 0)) {
            state_dir = "/run/vc4d";
            return;
        }
    }
    state_dir = "/tmp";
}

static void drop_privileges(void)
{
    struct passwd *pw = getpwnam(opt.drop_user);
    uid_t uid = pw ? pw->pw_uid : 65534;
    gid_t dgid = pw ? pw->pw_gid : 65534;
    if (!pw)
        logmsg("user '%s' not found, dropping to 65534", opt.drop_user);
    state_dir_init(uid, dgid);
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
    /* Journald and a redirected log must see each line as it happens: the
     * arena retry/degrade/restore lines are timing information, and a
     * block-buffered stderr makes them appear long after the fact. */
    setvbuf(stderr, NULL, _IONBF, 0);
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
    logmsg("arena %zu MiB%s at phys 0x%08x bus 0x%08x (requested %zu MiB)",
           A.bytes >> 20, A.degraded ? " DEGRADED" : "", A.phys, A.bus,
           A.want >> 20);
    if (!A.degraded)
        arena_stamp_clear(A.want);  /* a real restore resets the backoff */

    if (gpu_open(&G, err, sizeof(err)) != 0) {
        logmsg("gpu: %s", err);
        arena_close(&A);
        return 1;
    }
    logmsg("V3D IDENT0=0x%08x IDENT1=0x%08x errstat=0x%08x",
           G.regs[R_IDENT0 / 4], G.regs[R_IDENT1 / 4], G.errstat_base);

    /* Load the known-good probe program into the reserved last slot and point
       the recovery path at it.  It is uploaded exactly like a client program
       (bytes in the code slot, a zeroed 64-byte uniform block) so the probe
       exercises the same access path a kernel does. */
    {
        uint32_t pcode = A_CODE_BASE + PROBE_SLOT * A_CODE_SLOT;
        uint32_t puni = A_UNI_BASE + PROBE_SLOT * A_UNI_SLOT;
        memcpy(A.map + pcode, probe_words, sizeof(probe_words));
        memset(A.map + pcode + sizeof(probe_words), 0,
               A_CODE_SLOT - sizeof(probe_words));
        memset(A.map + puni, 0, A_UNI_SLOT);
        __sync_synchronize();
        gpu_use_probe(&G, arena_bus(&A, pcode), arena_bus(&A, puni));
        logmsg("probe program in slot %d: code bus 0x%08x uni bus 0x%08x",
               (int)PROBE_SLOT, arena_bus(&A, pcode), arena_bus(&A, puni));
    }

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
    else
        state_dir_init(getuid(), getgid());

    /* close anything inherited that we do not need (0,1,2 are the journal) */
    for (int fd = 3; fd < 256; ++fd) {
        if (fd != A.fd && fd != listen_fd && fd != gpu_lock_fd)
            close(fd);
    }

    struct pollfd pfds[MAX_CLIENTS + 1];
    /* Who is at each poll slot, by *index*, not by pointer: serving one
       client can reap another (HELLO sweeps dead peers, accept_client may
       reclaim a departed slot), and a pointer cached before the sweep would
       then dangle.  -1 is the listening socket. */
    int who[MAX_CLIENTS + 1];
    while (!stop_flag) {
        /* Release arena held by clients whose process has exited, whether or
           not their socket is still open (see peer_alive()).  Once per poll
           iteration is enough: the timeout is 1 s at most. */
        reap_dead_clients(-1, "peer process is gone");
        arena_maybe_restore();
        if (stop_flag || restart_needed)
            break;
        int n = 0;
        pfds[n].fd = listen_fd;
        pfds[n].events = POLLIN;
        pfds[n].revents = 0;
        who[n] = -1;
        n++;
        for (int i = 0; i < MAX_CLIENTS; ++i) {
            if (clients[i]) {
                pfds[n].fd = clients[i]->fd;
                pfds[n].events = POLLIN;
                pfds[n].revents = 0;
                who[n] = i;
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
            if (who[pi] < 0 || !clients[who[pi]]) {
                if (pfds[pi].revents & POLLIN)
                    accept_client();
                continue;
            }
            struct client *c = clients[who[pi]];
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
                h.version < VC4D_VERSION_MIN || h.version > VC4D_VERSION ||
                h.status != 0) {
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
    logmsg("exit%s (reclaimed %llu client(s), %llu GPU reset(s), %llu slot eviction(s))",
           restart_needed ? ": restarting" : "",
           (unsigned long long)st_reclaimed, (unsigned long long)st_recoveries,
           (unsigned long long)st_evicted);
    if (restart_needed)
        logmsg("restart reason: %s", restart_reason);
    return restart_needed ? 75 : 0;
}
