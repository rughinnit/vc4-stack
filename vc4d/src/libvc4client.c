/*
 * libvc4client.c - unprivileged client for vc4d.
 *
 * Request/reply over SOCK_SEQPACKET, one outstanding request at a time.  The
 * only interesting bit is ALLOC: the daemon passes the arena dma-buf fd with
 * SCM_RIGHTS and the library mmaps the requested byte range of it.
 *
 * Coherency: the daemon maps the arena uncached through /dev/mem, and the
 * client's dma-buf mapping is write-combining (cma-heap pgprot), so both see
 * DRAM.  vc4c_sync() still does the architectural cache maintenance by VA
 * (harmless on WC memory) and asks the daemon to flush the GPU's L2.
 */
#define _GNU_SOURCE
#include "vc4client.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

struct vc4c_conn {
    int fd;
    uint32_t seq;
    int last;
    struct vc4d_hello_rsp info;
    uint8_t rbuf[VC4D_MAX_MSG];
};

static void dsb(void)
{
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/* Clean+invalidate by VA.  Note: DC IVAC (invalidate without writeback) is
 * UNDEFINED at EL0 on ARMv8 - it traps as SIGILL - so userspace must use
 * DC CIVAC, which is what the engine's cache_clean() does too.  On a
 * write-combining dma-buf mapping this is architecturally a no-op and simply
 * keeps the code correct if a mapping ever comes back cacheable. */
static void cache_op(void *addr, size_t bytes)
{
#if defined(__aarch64__)
    uintptr_t line = (uintptr_t)addr & ~63ul;
    uintptr_t end = (uintptr_t)addr + bytes;
    while (line < end) {
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
        line += 64;
    }
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)addr; (void)bytes;
    __sync_synchronize();
#endif
}

const char *vc4c_strerror(int rc)
{
    switch (rc < 0 ? -rc : rc) {
    case VC4D_OK:         return "ok";
    case VC4D_E_PROTO:    return "malformed request";
    case VC4D_E_VERSION:  return "protocol version mismatch";
    case VC4D_E_TOOLARGE: return "too large";
    case VC4D_E_INVAL:    return "invalid argument";
    case VC4D_E_NOMEM:    return "out of arena/slots";
    case VC4D_E_NOENT:    return "no such object";
    case VC4D_E_BUSY:     return "daemon busy (launch in flight)";
    case VC4D_E_TIMEOUT:  return "GPU timeout";
    case VC4D_E_GPU:      return "GPU error";
    case VC4D_E_DENIED:   return "denied by daemon policy";
    case VC4D_E_IO:       return "I/O error";
    case VC4D_E_DEAD:     return "daemon is dead (GPU timeout earlier)";
    default:              return "unknown error";
    }
}

/*
 * Send one request and read one reply.  outfd, when non-NULL, receives the
 * SCM_RIGHTS fd from an ALLOC reply.  d0/d1 receive the reply header details.
 * Returns 0 or a negative VC4D_E_* code.
 */
static int xfer(vc4c_conn *c, uint16_t op, const void *payload, size_t plen,
                int pass_fd, int *outfd, void *out, size_t outcap,
                size_t *outlen, uint32_t *d0, uint32_t *d1)
{
    if (sizeof(struct vc4d_hdr) + plen > VC4D_MAX_MSG)
        return -VC4D_E_TOOLARGE;
    if (outlen)
        *outlen = 0;
    if (outfd)
        *outfd = -1;

    struct vc4d_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC;
    h.len = (uint32_t)(sizeof(h) + plen);
    h.op = op;
    h.version = VC4D_VERSION;
    h.seq = ++c->seq;

    struct iovec iov[2] = {
        { .iov_base = &h, .iov_len = sizeof(h) },
        { .iov_base = (void *)payload, .iov_len = plen },
    };
    struct msghdr m;
    memset(&m, 0, sizeof(m));
    m.msg_iov = iov;
    m.msg_iovlen = plen ? 2 : 1;

    char cbuf[CMSG_SPACE(sizeof(int))];
    if (pass_fd >= 0) {
        memset(cbuf, 0, sizeof(cbuf));
        m.msg_control = cbuf;
        m.msg_controllen = sizeof(cbuf);
        struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cm), &pass_fd, sizeof(int));
    }
    for (;;) {
        if (sendmsg(c->fd, &m, MSG_NOSIGNAL) >= 0)
            break;
        if (errno == EINTR)
            continue;
        c->last = -VC4D_E_IO;
        return c->last;
    }

    struct iovec riov = { .iov_base = c->rbuf, .iov_len = sizeof(c->rbuf) };
    char rcbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr rm;
    memset(&rm, 0, sizeof(rm));
    rm.msg_iov = &riov;
    rm.msg_iovlen = 1;
    rm.msg_control = rcbuf;
    rm.msg_controllen = sizeof(rcbuf);
    ssize_t got = recvmsg(c->fd, &rm, MSG_CMSG_CLOEXEC);
    if (got < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
        got = -1;
    if (got < (ssize_t)sizeof(struct vc4d_hdr)) {
        c->last = -VC4D_E_IO;
        return c->last;
    }
    struct vc4d_hdr rh;
    memcpy(&rh, c->rbuf, sizeof(rh));
    if (rh.magic != VC4D_MAGIC || rh.len != (uint32_t)got) {
        c->last = -VC4D_E_PROTO;
        return c->last;
    }
    c->last = rh.status;
    if (d0)
        *d0 = rh.detail0;
    if (d1)
        *d1 = rh.detail1;
    if (outfd) {
        for (struct cmsghdr *cm = CMSG_FIRSTHDR(&rm); cm; cm = CMSG_NXTHDR(&rm, cm)) {
            if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) {
                int f = -1;
                memcpy(&f, CMSG_DATA(cm), sizeof(int));
                *outfd = f;
            }
        }
    }

    size_t n = (size_t)got - sizeof(rh);
    if (out && outcap) {
        size_t cp = n < outcap ? n : outcap;
        memcpy(out, c->rbuf + sizeof(rh), cp);
    }
    if (outlen)
        *outlen = n;

    if (rh.status != VC4D_OK)
        return -rh.status;
    if (rh.seq != h.seq || rh.op != op) {
        c->last = -VC4D_E_PROTO;
        return c->last;
    }
    if (out && n > outcap) {
        c->last = -VC4D_E_TOOLARGE;
        return c->last;
    }
    return 0;
}

int vc4c_info(vc4c_conn *c, struct vc4d_hello_rsp *out)
{
    struct vc4d_hello_req q = { .version = VC4D_VERSION, .flags = 0 };
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_HELLO, &q, sizeof(q), -1, NULL,
                  out, sizeof(*out), &n, NULL, NULL);
    if (rc == 0 && n != sizeof(*out))
        return -VC4D_E_PROTO;
    return rc;
}

int vc4c_connect(vc4c_conn **out, const char *path)
{
    if (!out)
        return -VC4D_E_INVAL;
    *out = NULL;
    struct vc4c_conn *c = calloc(1, sizeof(*c));
    if (!c)
        return -VC4D_E_NOMEM;
    c->fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (c->fd < 0) {
        free(c);
        return -VC4D_E_IO;
    }
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", path);
    if (connect(c->fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        int e = errno;
        close(c->fd);
        free(c);
        return e == EACCES || e == EPERM ? -VC4D_E_DENIED : -VC4D_E_IO;
    }
    struct timeval tv = { .tv_sec = 60, .tv_usec = 0 };
    setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(c->fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int rc = vc4c_info(c, &c->info);
    if (rc != 0) {
        close(c->fd);
        free(c);
        return rc;
    }
    *out = c;
    return 0;
}

void vc4c_close(vc4c_conn *c)
{
    if (!c)
        return;
    (void)xfer(c, VC4D_OP_CLOSE, NULL, 0, -1, NULL, NULL, 0, NULL, NULL, NULL);
    close(c->fd);
    free(c);
}

int vc4c_alloc(vc4c_conn *c, size_t bytes, unsigned flags, vc4c_buf *out)
{
    if (!c || !out || !bytes)
        return -VC4D_E_INVAL;
    struct vc4d_alloc_req q = { .bytes = (uint32_t)bytes, .flags = flags };
    struct vc4d_alloc_rsp r;
    size_t n = 0;
    int gotfd = -1;
    int rc = xfer(c, VC4D_OP_ALLOC, &q, sizeof(q), -1, &gotfd,
                  &r, sizeof(r), &n, NULL, NULL);
    if (rc != 0) {
        if (gotfd >= 0)
            close(gotfd);
        return rc;
    }
    if (n != sizeof(r) || gotfd < 0) {
        if (gotfd >= 0)
            close(gotfd);
        return -VC4D_E_PROTO;
    }
    void *p = mmap(NULL, r.len, PROT_READ | PROT_WRITE, MAP_SHARED, gotfd,
                   (off_t)r.off);
    close(gotfd);
    if (p == MAP_FAILED)
        return -VC4D_E_IO;
    out->handle = r.handle;
    out->off = r.off;
    out->len = r.len;
    out->bus = r.bus;
    out->map = p;
    return 0;
}

int vc4c_free(vc4c_conn *c, vc4c_buf *b)
{
    if (!c || !b || !b->map)
        return -VC4D_E_INVAL;
    munmap(b->map, b->len);
    b->map = NULL;
    struct vc4d_free_req q = { .handle = b->handle };
    return xfer(c, VC4D_OP_FREE, &q, sizeof(q), -1, NULL, NULL, 0, NULL, NULL, NULL);
}

int vc4c_upload(vc4c_conn *c, const void *code, size_t bytes,
                unsigned max_qpu, unsigned nuni, vc4c_prog *out)
{
    if (!c || !code || !out || !bytes ||
        sizeof(struct vc4d_upload_req) + bytes > VC4D_MAX_MSG)
        return -VC4D_E_INVAL;
    uint8_t *pl = malloc(sizeof(struct vc4d_upload_req) + bytes);
    if (!pl)
        return -VC4D_E_NOMEM;
    struct vc4d_upload_req q = { .bytes = (uint32_t)bytes,
                                 .max_qpu = max_qpu, .nuni = nuni, .flags = 0 };
    memcpy(pl, &q, sizeof(q));
    memcpy(pl + sizeof(q), code, bytes);
    struct vc4d_upload_rsp r;
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_UPLOAD, pl, sizeof(q) + bytes, -1, NULL,
                  &r, sizeof(r), &n, NULL, NULL);
    free(pl);
    if (rc != 0)
        return rc;
    if (n != sizeof(r))
        return -VC4D_E_PROTO;
    out->handle = r.prog;
    out->code_off = r.code_off;
    out->words = r.words;
    out->nuni = nuni;
    return 0;
}

int vc4c_upload_file(vc4c_conn *c, const char *path, unsigned max_qpu,
                     unsigned nuni, vc4c_prog *out)
{
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0 ||
        st.st_size > (off_t)VC4D_MAX_PROG)
        return -VC4D_E_INVAL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return -VC4D_E_IO;
    uint8_t *b = malloc((size_t)st.st_size);
    if (!b) {
        fclose(f);
        return -VC4D_E_NOMEM;
    }
    size_t got = fread(b, 1, (size_t)st.st_size, f);
    fclose(f);
    int rc = got == (size_t)st.st_size ? vc4c_upload(c, b, got, max_qpu, nuni, out)
                                       : -VC4D_E_IO;
    free(b);
    return rc;
}

int vc4c_launch(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                unsigned nuni, const vc4c_uni *uni, unsigned timeout_ms)
{
    if (!c || !p || !uni || !nqpu || !nuni || nqpu > VC4D_MAX_QPU ||
        nuni > VC4D_MAX_UNI)
        return -VC4D_E_INVAL;
    size_t plen = sizeof(struct vc4d_launch_req) +
                  (size_t)nqpu * nuni * sizeof(*uni);
    if (plen > VC4D_MAX_MSG)
        return -VC4D_E_TOOLARGE;
    uint8_t *pl = malloc(plen);
    if (!pl)
        return -VC4D_E_NOMEM;
    struct vc4d_launch_req q = { .prog = p->handle, .timeout_ms = timeout_ms,
                                 .nqpu = nqpu, .nuni = nuni, .flags = 0 };
    memcpy(pl, &q, sizeof(q));
    memcpy(pl + sizeof(q), uni, (size_t)nqpu * nuni * sizeof(*uni));
    struct vc4d_launch_rsp r;
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_LAUNCH, pl, plen, -1, NULL,
                  &r, sizeof(r), &n, NULL, NULL);
    free(pl);
    if (rc != 0)
        return rc;
    return n == sizeof(r) ? 0 : -VC4D_E_PROTO;
}

int vc4c_launch_arena(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                      const vc4c_buf *uniforms, unsigned uni_off,
                      unsigned timeout_ms)
{
    if (!c || !p || !uniforms || !uniforms->map || !nqpu ||
        nqpu > VC4D_MAX_QPU || (uni_off & 63u))
        return -VC4D_E_INVAL;
    if (!p->nuni || p->nuni > VC4D_MAX_UNI)
        return -VC4D_E_INVAL;
    if ((uint64_t)uni_off + 64ull * nqpu > uniforms->len)
        return -VC4D_E_INVAL;
    struct vc4d_launch_req q = { .prog = p->handle, .timeout_ms = timeout_ms,
                                 .nqpu = nqpu, .nuni = p->nuni,
                                 .flags = VC4D_LAUNCH_UNI_ARENA,
                                 .uni_handle = uniforms->handle,
                                 .uni_off = uni_off };
    struct vc4d_launch_rsp r;
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_LAUNCH, &q, sizeof(q), -1, NULL,
                  &r, sizeof(r), &n, NULL, NULL);
    if (rc != 0)
        return rc;
    return n == sizeof(r) ? 0 : -VC4D_E_PROTO;
}

int vc4c_launch_wait(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                     const vc4c_buf *uniforms, unsigned uni_off,
                     unsigned timeout_ms, vc4c_result *out)
{
    if (!c || !p || !uniforms || !uniforms->map || !nqpu ||
        nqpu > VC4D_MAX_QPU || (uni_off & 63u))
        return -VC4D_E_INVAL;
    if (!p->nuni || p->nuni > VC4D_MAX_UNI)
        return -VC4D_E_INVAL;
    if ((uint64_t)uni_off + 64ull * nqpu > uniforms->len)
        return -VC4D_E_INVAL;
    struct vc4d_launch_req q = { .prog = p->handle, .timeout_ms = timeout_ms,
                                 .nqpu = nqpu, .nuni = p->nuni,
                                 .flags = VC4D_LAUNCH_UNI_ARENA |
                                          VC4D_LAUNCH_WAIT,
                                 .uni_handle = uniforms->handle,
                                 .uni_off = uni_off };
    struct vc4d_wait_rsp r;
    memset(&r, 0, sizeof(r));
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_LAUNCH, &q, sizeof(q), -1, NULL,
                  &r, sizeof(r), &n, NULL, NULL);
    if (out && n >= sizeof(r)) {
        out->launch = r.launch;
        out->errs = r.errs;
        out->wall_ns = r.wall_ns;
        out->cycles = r.cycles;
    }
    if (rc != 0)
        return rc;
    return n == sizeof(r) ? 0 : -VC4D_E_PROTO;
}

int vc4c_flush(vc4c_conn *c, const vc4c_buf *b, unsigned dir)
{
    if (!c || !b || dir > VC4D_SYNC_G2R)
        return -VC4D_E_INVAL;
    struct vc4d_sync_req q = { .handle = b->handle, .dir = dir };
    return xfer(c, VC4D_OP_SYNC, &q, sizeof(q), -1, NULL, NULL, 0, NULL, NULL, NULL);
}

int vc4c_wait(vc4c_conn *c, vc4c_result *out)
{
    if (!c || !out)
        return -VC4D_E_INVAL;
    struct vc4d_wait_rsp r;
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_WAIT, NULL, 0, -1, NULL,
                  &r, sizeof(r), &n, NULL, NULL);
    if (n >= sizeof(r)) {
        out->launch = r.launch;
        out->errs = r.errs;
        out->wall_ns = r.wall_ns;
        out->cycles = r.cycles;
    }
    return rc;
}

int vc4c_sync(vc4c_conn *c, const vc4c_buf *b, unsigned dir)
{
    if (!c || !b || !b->map)
        return -VC4D_E_INVAL;
    cache_op(b->map, b->len);
    dsb();
    struct vc4d_sync_req q = { .handle = b->handle, .dir = dir };
    return xfer(c, VC4D_OP_SYNC, &q, sizeof(q), -1, NULL, NULL, 0, NULL, NULL, NULL);
}

int vc4c_read_regs(vc4c_conn *c, struct vc4d_regs *out)
{
    struct vc4d_regs_req q = { .off = 0, .count = 0 };
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_READ_REGS, &q, sizeof(q), -1, NULL,
                  out, sizeof(*out), &n, NULL, NULL);
    if (rc != 0)
        return rc;
    return n == sizeof(*out) ? 0 : -VC4D_E_PROTO;
}

int vc4c_read_reg_words(vc4c_conn *c, unsigned off, unsigned count, uint32_t *out)
{
    if (!c || !out || !count || count > 1024 || (off & 3))
        return -VC4D_E_INVAL;
    struct vc4d_regs_req q = { .off = off, .count = count };
    size_t need = sizeof(q) + count * 4;
    uint8_t *buf = malloc(need);
    if (!buf)
        return -VC4D_E_NOMEM;
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_READ_REGS, &q, sizeof(q), -1, NULL,
                  buf, need, &n, NULL, NULL);
    if (rc == 0 && n == need)
        memcpy(out, buf + sizeof(q), count * 4);
    else if (rc == 0)
        rc = -VC4D_E_PROTO;
    free(buf);
    return rc;
}

int vc4c_write_reg(vc4c_conn *c, unsigned off, unsigned val, unsigned *old)
{
    if (!c)
        return -VC4D_E_INVAL;
    struct vc4d_wreg_req q = { .off = off, .val = val };
    uint32_t d0 = 0;
    int rc = xfer(c, VC4D_OP_WRITEREG, &q, sizeof(q), -1, NULL,
                  NULL, 0, NULL, &d0, NULL);
    if (rc == 0 && old)
        *old = d0;
    return rc;
}

int vc4c_get_stats(vc4c_conn *c, vc4c_stats *out)
{
    size_t n = 0;
    int rc = xfer(c, VC4D_OP_STATS, NULL, 0, -1, NULL,
                  out, sizeof(*out), &n, NULL, NULL);
    if (rc != 0)
        return rc;
    return n == sizeof(*out) ? 0 : -VC4D_E_PROTO;
}

int vc4c_ping(vc4c_conn *c)
{
    return xfer(c, VC4D_OP_PING, NULL, 0, -1, NULL, NULL, 0, NULL, NULL, NULL);
}
