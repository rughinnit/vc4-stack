/*
 * hostile.c - protocol abuse against a running vc4d.
 *
 * Every case must be rejected with an error and the daemon must survive: after
 * each group we PING, and at the end we check STATS.  The only real QPU launch
 * here uses the engine's verified pack10_256 program with valid uniforms (so a
 * validation bug in the hostile cases cannot wedge the board: the hostile
 * LAUNCH messages are rejected before the queue is kicked).
 *
 * usage: hostile [--socket PATH] [--blobs DIR] [--fuzz N]
 */
#define _GNU_SOURCE
#include "../proto/vc4d_proto.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static int fd = -1;
static uint32_t seq = 1;
static int pass, fail;
static const char *sockpath = "/run/vc4d.sock";
static const char *blobs = "blobs";

/* UNI_IN_ARENA is a cacheable client mapping: the daemon reads it uncached,
 * so a client that writes a uniform word must clean it before LAUNCH (the
 * engine's remote backend does exactly this).  Without it the daemon sees a
 * stale word and the test would be testing nothing. */
static void clean_line(const void *p)
{
#if defined(__aarch64__)
    uintptr_t line = (uintptr_t)p & ~63ul;
    __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)p;
    __sync_synchronize();
#endif
}

static void ok(int cond, const char *name, const char *why)
{
    if (cond) {
        pass++;
        printf("  PASS  %s\n", name);
    } else {
        fail++;
        printf("  FAIL  %s%s%s\n", name, why ? " - " : "", why ? why : "");
    }
}

static const char *sname(int s)
{
    switch (s) {
    case VC4D_OK: return "OK";
    case VC4D_E_PROTO: return "E_PROTO";
    case VC4D_E_VERSION: return "E_VERSION";
    case VC4D_E_TOOLARGE: return "E_TOOLARGE";
    case VC4D_E_INVAL: return "E_INVAL";
    case VC4D_E_NOMEM: return "E_NOMEM";
    case VC4D_E_NOENT: return "E_NOENT";
    case VC4D_E_BUSY: return "E_BUSY";
    case VC4D_E_TIMEOUT: return "E_TIMEOUT";
    case VC4D_E_GPU: return "E_GPU";
    case VC4D_E_DENIED: return "E_DENIED";
    case VC4D_E_IO: return "E_IO";
    case VC4D_E_DEAD: return "E_DEAD";
    default: return "?";
    }
}

static int conn_open(void)
{
    int s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (s < 0)
        return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", sockpath);
    if (connect(s, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(s);
        return -1;
    }
    return s;
}

static int send_msg(int s, const void *buf, size_t len, int passfd)
{
    struct iovec iov = { .iov_base = (void *)buf, .iov_len = len };
    struct msghdr m;
    memset(&m, 0, sizeof(m));
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    char cbuf[CMSG_SPACE(sizeof(int))];
    if (passfd >= 0) {
        memset(cbuf, 0, sizeof(cbuf));
        m.msg_control = cbuf;
        m.msg_controllen = sizeof(cbuf);
        struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cm), &passfd, sizeof(int));
    }
    for (;;) {
        ssize_t n = sendmsg(s, &m, MSG_NOSIGNAL);
        if (n >= 0)
            return 0;
        if (errno == EINTR)
            continue;
        return -errno;
    }
}

/* returns 0 if a reply arrived, -1 on timeout/EOF */
static int recv_msg(int s, struct vc4d_hdr *h, void *pl, size_t cap,
                    size_t *plen, int *outfd)
{
    struct pollfd p = { .fd = s, .events = POLLIN };
    if (poll(&p, 1, 5000) <= 0)
        return -1;
    static uint8_t buf[VC4D_MAX_MSG];
    struct iovec iov = { .iov_base = buf, .iov_len = sizeof(buf) };
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr m;
    memset(&m, 0, sizeof(m));
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = cbuf;
    m.msg_controllen = sizeof(cbuf);
    ssize_t got = recvmsg(s, &m, MSG_CMSG_CLOEXEC);
    if (got < (ssize_t)sizeof(*h))
        return -1;
    memcpy(h, buf, sizeof(*h));
    if (h->magic != VC4D_MAGIC || h->len != (uint32_t)got)
        return -1;
    if (plen)
        *plen = (size_t)got - sizeof(*h);
    if (pl && cap) {
        size_t n = (size_t)got - sizeof(*h);
        if (n > cap) n = cap;
        memcpy(pl, buf + sizeof(*h), n);
    }
    if (outfd) {
        *outfd = -1;
        for (struct cmsghdr *cm = CMSG_FIRSTHDR(&m); cm; cm = CMSG_NXTHDR(&m, cm))
            if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS)
                memcpy(outfd, CMSG_DATA(cm), sizeof(int));
    }
    return 0;
}

/* send a well-formed request, wait for the reply header */
static int do_req(int s, uint16_t op, const void *pl, size_t plen, int passfd,
                  struct vc4d_hdr *rh, void *rpl, size_t cap, size_t *rlen,
                  int *outfd)
{
    uint8_t buf[VC4D_MAX_MSG];
    if (sizeof(struct vc4d_hdr) + plen > sizeof(buf))
        return -1;
    struct vc4d_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC;
    h.len = (uint32_t)(sizeof(h) + plen);
    h.op = op;
    h.version = VC4D_VERSION;
    h.seq = seq++;
    memcpy(buf, &h, sizeof(h));
    if (plen)
        memcpy(buf + sizeof(h), pl, plen);
    if (send_msg(s, buf, h.len, passfd) != 0)
        return -1;
    return recv_msg(s, rh, rpl, cap, rlen, outfd);
}

static int ping(void)
{
    if (fd < 0)
        fd = conn_open();
    if (fd < 0)
        return 0;
    struct vc4d_hdr h;
    if (do_req(fd, VC4D_OP_PING, NULL, 0, -1, &h, NULL, 0, NULL, NULL) != 0)
        return 0;
    return h.status == VC4D_OK;
}

static void ensure_alive(const char *where)
{
    if (!ping()) {
        printf("  (reconnecting after %s)\n", where);
        if (fd >= 0)
            close(fd);
        fd = conn_open();
        ok(fd >= 0 && ping(), "daemon alive", where);
    }
}

static void expect(const char *name, int want, uint16_t op,
                   const void *pl, size_t plen, int passfd)
{
    struct vc4d_hdr h;
    if (do_req(fd, op, pl, plen, passfd, &h, NULL, 0, NULL, NULL) != 0) {
        ok(0, name, "no reply");
        return;
    }
    if (h.status != want) {
        char why[96];
        snprintf(why, sizeof(why), "got %s, want %s",
                 sname(h.status), sname(want));
        ok(0, name, why);
        return;
    }
    ok(1, name, NULL);
}

static void expect_raw(const char *name, int want, const void *buf, size_t len)
{
    if (send_msg(fd, buf, len, -1) != 0) {
        /* the kernel refused it (e.g. EMSGSIZE) - also a rejection */
        ok(1, name, NULL);
        return;
    }
    struct vc4d_hdr h;
    if (recv_msg(fd, &h, NULL, 0, NULL, NULL) != 0) {
        ok(0, name, "no reply");
        return;
    }
    if (h.status != want) {
        char why[96];
        snprintf(why, sizeof(why), "got %s, want %s", sname(h.status), sname(want));
        ok(0, name, why);
        return;
    }
    ok(1, name, NULL);
}

static uint8_t *read_file(const char *path, size_t *len)
{
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0)
        return NULL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    uint8_t *b = malloc((size_t)st.st_size);
    if (!b) { fclose(f); return NULL; }
    size_t n = fread(b, 1, (size_t)st.st_size, f);
    fclose(f);
    if (n != (size_t)st.st_size) { free(b); return NULL; }
    *len = n;
    return b;
}

int main(int argc, char **argv)
{
    int fuzz = 2000, expect_denied = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--socket") && i + 1 < argc) sockpath = argv[++i];
        else if (!strcmp(argv[i], "--blobs") && i + 1 < argc) blobs = argv[++i];
        else if (!strcmp(argv[i], "--fuzz") && i + 1 < argc) fuzz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expect-denied")) expect_denied = 1;
    }

    fd = conn_open();
    if (fd < 0) {
        fprintf(stderr, "hostile: cannot connect %s: %s\n", sockpath, strerror(errno));
        return 2;
    }

    if (expect_denied) {
        /* strict-mode check: a program whose bytes differ from every
           allowlisted blob must be refused at UPLOAD */
        char path[512];
        snprintf(path, sizeof(path), "%s/pack10_256.bin", blobs);
        size_t n = 0;
        uint8_t *b = read_file(path, &n);
        if (!b) {
            fprintf(stderr, "hostile: need %s\n", path);
            return 2;
        }
        b[n / 2] ^= 0x01;
        struct vc4d_upload_req q = { .bytes = (uint32_t)n, .max_qpu = 4,
                                     .nuni = 8, .flags = 0 };
        uint8_t *pl = malloc(sizeof(q) + n);
        memcpy(pl, &q, sizeof(q));
        memcpy(pl + sizeof(q), b, n);
        struct vc4d_hdr h;
        int rc = do_req(fd, VC4D_OP_UPLOAD, pl, sizeof(q) + n, -1, &h,
                        NULL, 0, NULL, NULL);
        printf("strict upload of a modified program: %s\n",
               rc == 0 ? sname(h.status) : "no reply");
        int okd = rc == 0 && h.status == VC4D_E_DENIED;
        printf("%s allowlist rejects an unlisted program\n", okd ? "PASS" : "FAIL");
        free(pl);
        free(b);
        close(fd);
        return okd ? 0 : 1;
    }

    printf("hostile-input suite against %s\n", sockpath);

    /* ---------------------------------------------------------- header */
    printf("[header]\n");
    struct vc4d_hdr h;
    uint8_t raw[64];

    memset(&h, 0, sizeof(h));
    h.magic = 0xdeadbeef; h.len = 24; h.op = VC4D_OP_PING;
    h.version = VC4D_VERSION; h.seq = seq++;
    memcpy(raw, &h, sizeof(h));
    expect_raw("bad magic", VC4D_E_PROTO, raw, sizeof(h));

    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC; h.len = 24; h.op = VC4D_OP_PING;
    h.version = 99; h.seq = seq++;
    memcpy(raw, &h, sizeof(h));
    expect_raw("bad version", VC4D_E_VERSION, raw, sizeof(h));

    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC; h.len = 32; h.op = VC4D_OP_PING;
    h.version = VC4D_VERSION; h.seq = seq++;
    memcpy(raw, &h, sizeof(h));
    expect_raw("len says 32, sent 24", VC4D_E_PROTO, raw, sizeof(h));

    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC; h.len = 24; h.op = 999;
    h.version = VC4D_VERSION; h.seq = seq++;
    memcpy(raw, &h, sizeof(h));
    expect_raw("unknown opcode", VC4D_E_PROTO, raw, sizeof(h));

    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC; h.len = 24; h.op = VC4D_OP_PING;
    h.version = VC4D_VERSION; h.seq = seq++; h.status = 7;
    memcpy(raw, &h, sizeof(h));
    expect_raw("request with nonzero status", VC4D_E_PROTO, raw, sizeof(h));

    for (int n = 1; n < 24; ++n) {
        char name[64];
        snprintf(name, sizeof(name), "truncated to %d bytes", n);
        expect_raw(name, VC4D_E_PROTO, raw, (size_t)n);
    }

    {
        size_t big = 70 * 1024;
        uint8_t *b = calloc(1, big);
        struct vc4d_hdr *bh = (struct vc4d_hdr *)b;
        bh->magic = VC4D_MAGIC; bh->len = (uint32_t)big;
        bh->op = VC4D_OP_PING; bh->version = VC4D_VERSION; bh->seq = seq++;
        expect_raw("70 KiB datagram (over 64 KiB cap)", VC4D_E_TOOLARGE, b, big);
        free(b);
    }
    {
        size_t big = 2u << 20;
        uint8_t *b = calloc(1, big);
        struct vc4d_hdr *bh = (struct vc4d_hdr *)b;
        bh->magic = VC4D_MAGIC; bh->len = (uint32_t)big;
        bh->op = VC4D_OP_PING; bh->version = VC4D_VERSION; bh->seq = seq++;
        int rc = send_msg(fd, b, big, -1);
        ok(rc != 0 || ping(), "2 MiB datagram refused and daemon alive", NULL);
        free(b);
    }
    ensure_alive("header group");

    /* ---------------------------------------------------------- objects */
    printf("[objects]\n");
    struct vc4d_alloc_req aq;
    struct vc4d_hdr rh;
    size_t rlen = 0;
    int gotfd = -1;
    struct vc4d_alloc_rsp ar;

    aq.bytes = 0; aq.flags = 0;
    expect("ALLOC size 0", VC4D_E_INVAL, VC4D_OP_ALLOC, &aq, sizeof(aq), -1);

    aq.bytes = 0xFFFFFFFF; aq.flags = 0;
    expect("ALLOC 4 GiB", VC4D_E_TOOLARGE, VC4D_OP_ALLOC, &aq, sizeof(aq), -1);

    aq.bytes = 4096; aq.flags = 0x80000000;
    expect("ALLOC unknown flags", VC4D_E_INVAL, VC4D_OP_ALLOC, &aq, sizeof(aq), -1);

    uint8_t badextra[sizeof(struct vc4d_alloc_req) + 4] = { 0 };
    expect("ALLOC with trailing bytes", VC4D_E_PROTO, VC4D_OP_ALLOC,
           badextra, sizeof(badextra), -1);

    struct vc4d_free_req fq = { .handle = 0xFFFF };
    expect("FREE unknown handle", VC4D_E_NOENT, VC4D_OP_FREE, &fq, sizeof(fq), -1);

    /* real regions A (4 KiB) and B (8 KiB) stay alive; C is freed */
    struct region { uint32_t handle, off, len; } ra, rb, rc;
    aq.bytes = 4096; aq.flags = 0;
    if (do_req(fd, VC4D_OP_ALLOC, &aq, sizeof(aq), -1, &rh, &ar, sizeof(ar),
               &rlen, &gotfd) != 0 || rh.status != VC4D_OK) {
        printf("FATAL: cannot allocate test buffers\n");
        return 2;
    }
    if (gotfd >= 0) close(gotfd);
    ra.handle = ar.handle; ra.off = ar.off; ra.len = ar.len;
    aq.bytes = 8192;
    if (do_req(fd, VC4D_OP_ALLOC, &aq, sizeof(aq), -1, &rh, &ar, sizeof(ar),
               &rlen, &gotfd) != 0 || rh.status != VC4D_OK) {
        printf("FATAL: cannot allocate test buffer B\n");
        return 2;
    }
    if (gotfd >= 0) close(gotfd);
    rb.handle = ar.handle; rb.off = ar.off; rb.len = ar.len;
    aq.bytes = 4096;
    if (do_req(fd, VC4D_OP_ALLOC, &aq, sizeof(aq), -1, &rh, &ar, sizeof(ar),
               &rlen, &gotfd) != 0 || rh.status != VC4D_OK) {
        printf("FATAL: cannot allocate test buffer C\n");
        return 2;
    }
    if (gotfd >= 0) close(gotfd);
    rc.handle = ar.handle; rc.off = ar.off; rc.len = ar.len;

    fq.handle = rc.handle;
    expect("FREE valid region", VC4D_OK, VC4D_OP_FREE, &fq, sizeof(fq), -1);
    expect("FREE same region twice", VC4D_E_NOENT, VC4D_OP_FREE, &fq, sizeof(fq), -1);
    expect("FREE a freed region's uniform", VC4D_E_NOENT, VC4D_OP_FREE,
           &fq, sizeof(fq), -1);

    /* ---------------------------------------------------------- upload */
    char ppath[512];
    snprintf(ppath, sizeof(ppath), "%s/pack10_256.bin", blobs);
    size_t codelen = 0;
    uint8_t *code = read_file(ppath, &codelen);
    if (!code) {
        fprintf(stderr, "hostile: need %s\n", ppath);
        return 2;
    }

    struct vc4d_upload_req uq;
    uq.bytes = 0; uq.max_qpu = 4; uq.nuni = 8; uq.flags = 0;
    expect("UPLOAD 0 bytes", VC4D_E_INVAL, VC4D_OP_UPLOAD, &uq, sizeof(uq), -1);

    uq.bytes = 100;   /* not a multiple of 8 */
    uint8_t *p100 = calloc(1, sizeof(uq) + 100);
    memcpy(p100, &uq, sizeof(uq));
    expect("UPLOAD 100 bytes (not 8-aligned)", VC4D_E_INVAL, VC4D_OP_UPLOAD,
           p100, sizeof(uq) + 100, -1);
    free(p100);

    uq.bytes = 8; uq.max_qpu = 99;
    uint8_t *p8 = calloc(1, sizeof(uq) + 8);
    memcpy(p8, &uq, sizeof(uq));
    expect("UPLOAD max_qpu=99", VC4D_E_INVAL, VC4D_OP_UPLOAD, p8, sizeof(uq) + 8, -1);
    free(p8);

    uq.bytes = 1000; uq.max_qpu = 4;
    uint8_t *p100b = calloc(1, sizeof(uq) + 100);
    memcpy(p100b, &uq, sizeof(uq));
    expect("UPLOAD claims 1000, sends 100", VC4D_E_PROTO, VC4D_OP_UPLOAD,
           p100b, sizeof(uq) + 100, -1);
    free(p100b);

    uq.bytes = 20 * 1024; uq.max_qpu = 4;
    uint8_t *p20k = calloc(1, sizeof(uq) + 20 * 1024);
    memcpy(p20k, &uq, sizeof(uq));
    expect("UPLOAD 20 KiB (over 16 KiB)", VC4D_E_INVAL, VC4D_OP_UPLOAD,
           p20k, sizeof(uq) + 20 * 1024, -1);
    free(p20k);

    {
        int memfd = open("/dev/null", O_RDONLY);
        expect("UPLOAD with an unexpected fd", VC4D_E_DENIED, VC4D_OP_UPLOAD,
               code, sizeof(uq) + 0, memfd);
        close(memfd);
        ensure_alive("unexpected fd");
    }

    /* a valid upload of the pack program (this is the only real kernel) */
    struct vc4d_upload_rsp ur;
    uint8_t *up = malloc(sizeof(uq) + codelen);
    uq.bytes = (uint32_t)codelen; uq.max_qpu = 4; uq.nuni = 8;
    memcpy(up, &uq, sizeof(uq));
    memcpy(up + sizeof(uq), code, codelen);
    if (do_req(fd, VC4D_OP_UPLOAD, up, sizeof(uq) + codelen, -1, &rh, &ur,
               sizeof(ur), &rlen, NULL) != 0 || rh.status != VC4D_OK) {
        printf("FATAL: cannot upload the pack program (%s)\n", sname(rh.status));
        return 2;
    }
    free(up);
    ok(1, "UPLOAD a valid program", NULL);
    uint32_t prog = ur.prog;

    /* ---------------------------------------------------------- launch */
    printf("[launch]\n");
    struct vc4d_uni uni[VC4D_MAX_QPU * VC4D_MAX_UNI];
    uint8_t lbuf[VC4D_MAX_MSG];
    struct vc4d_launch_req lq;
    memset(&lq, 0, sizeof(lq));
    memset(uni, 0, sizeof(uni));

    /* helpers to send a modified launch */
/* the program declares 8 uniforms at UPLOAD, so every hostile LAUNCH sends
 * 8 entries: only entry 0 varies, the rest stay zero and the launch must be
 * rejected before the queue is kicked */
#define SEND_LAUNCH(name, want) do {                                            \
        lq.nqpu = 1; lq.nuni = 8; lq.timeout_ms = 2000;                        \
        lq.prog = prog; lq.flags = 0;                                           \
        memcpy(lbuf, &lq, sizeof(lq));                                          \
        memcpy(lbuf + sizeof(lq), uni, 8 * 16);                                 \
        if (do_req(fd, VC4D_OP_LAUNCH, lbuf, sizeof(lq) + 8 * 16, -1, &rh,      \
                   NULL, 0, &rlen, NULL) != 0) {                                \
            ok(0, name, "no reply");                                            \
        } else {                                                                \
            if (rh.status != want) {                                            \
                char why[96];                                                   \
                snprintf(why, sizeof(why), "got %s, want %s",                  \
                         sname(rh.status), sname(want));                        \
                ok(0, name, why);                                               \
            } else ok(1, name, NULL);                                           \
        }                                                                       \
    } while (0)

    lq.prog = 0xFFFFFF; lq.nqpu = 1; lq.nuni = 8; lq.timeout_ms = 1000;
    memcpy(lbuf, &lq, sizeof(lq)); memcpy(lbuf + sizeof(lq), uni, 8 * 16);
    expect("LAUNCH unknown program handle", VC4D_E_NOENT, VC4D_OP_LAUNCH,
           lbuf, sizeof(lq) + 8 * 16, -1);

    lq.prog = prog; lq.nqpu = 13; lq.nuni = 8;
    memcpy(lbuf, &lq, sizeof(lq)); memcpy(lbuf + sizeof(lq), uni, 8 * 16);
    expect("LAUNCH nqpu=13", VC4D_E_INVAL, VC4D_OP_LAUNCH, lbuf, sizeof(lq) + 8 * 16, -1);

    lq.nqpu = 1; lq.nuni = 17;
    memcpy(lbuf, &lq, sizeof(lq)); memcpy(lbuf + sizeof(lq), uni, 8 * 16);
    expect("LAUNCH nuni=17", VC4D_E_INVAL, VC4D_OP_LAUNCH, lbuf, sizeof(lq) + 8 * 16, -1);

    lq.nqpu = 2; lq.nuni = 8;   /* claims 16 entries, sends 8 */
    memcpy(lbuf, &lq, sizeof(lq)); memcpy(lbuf + sizeof(lq), uni, 8 * 16);
    expect("LAUNCH uniform payload too short", VC4D_E_PROTO, VC4D_OP_LAUNCH,
           lbuf, sizeof(lq) + 8 * 16, -1);

    lq.nqpu = 1; lq.nuni = 8;
    uni[0].kind = VC4D_UNI_PTR; uni[0].handle = ra.handle;
    uni[0].off = 0; uni[0].len = 8192;
    SEND_LAUNCH("LAUNCH pointer past the end of its buffer", VC4D_E_INVAL);
    uni[0].off = 4096; uni[0].len = 4096;
    SEND_LAUNCH("LAUNCH pointer starting at buffer end", VC4D_E_INVAL);
    uni[0].off = 4096; uni[0].len = 0x1000;
    SEND_LAUNCH("LAUNCH pointer one page past its buffer", VC4D_E_INVAL);
    uni[0].off = 0xFFFFFFF0u; uni[0].len = 0x20;
    SEND_LAUNCH("LAUNCH pointer with an overflow offset", VC4D_E_INVAL);
    uni[0].off = 0; uni[0].len = 0;
    SEND_LAUNCH("LAUNCH pointer with zero length", VC4D_E_INVAL);
    uni[0].off = 2; uni[0].len = 4;
    SEND_LAUNCH("LAUNCH pointer with an unaligned offset", VC4D_E_INVAL);
    uni[0].handle = 0xFFFFFFFF; uni[0].off = 0; uni[0].len = 4;
    SEND_LAUNCH("LAUNCH pointer with a bogus handle", VC4D_E_NOENT);
    uni[0].handle = rb.handle; uni[0].off = 0; uni[0].len = 4;
    uni[0].kind = VC4D_UNI_LIT; uni[0].len = 0;
    SEND_LAUNCH("LAUNCH literal with a nonzero handle", VC4D_E_INVAL);
    uni[0].handle = 0; uni[0].len = 0; uni[0].off = 0xC0000000u;
    SEND_LAUNCH("LAUNCH literal = GPU bus address", VC4D_E_DENIED);
    uni[0].off = 0x3FC00000u;
    SEND_LAUNCH("LAUNCH literal = V3D MMIO", VC4D_E_DENIED);
    uni[0].off = 0;
    uni[0].kind = 7;
    SEND_LAUNCH("LAUNCH unknown uniform kind", VC4D_E_INVAL);

    /* nuni must equal the count declared at UPLOAD (8) */
    uni[0].kind = VC4D_UNI_LIT; uni[0].handle = 0; uni[0].len = 0; uni[0].off = 0;
    lq.nqpu = 1; lq.nuni = 7;
    memcpy(lbuf, &lq, sizeof(lq));
    memcpy(lbuf + sizeof(lq), uni, 7 * 16);
    if (do_req(fd, VC4D_OP_LAUNCH, lbuf, sizeof(lq) + 7 * 16, -1, &rh,
               NULL, 0, &rlen, NULL) == 0)
        ok(rh.status == VC4D_E_INVAL, "LAUNCH nuni != declared at UPLOAD",
           sname(rh.status));
    else
        ok(0, "LAUNCH nuni != declared at UPLOAD", "no reply");
    lq.nuni = 1;

    /* UNI_IN_ARENA: the uniform words are materialised in the client's own
       region.  Only refusals are exercised here: an accepted launch would
       actually kick the queue. */
    {
        struct vc4d_alloc_rsp ma;
        aq.bytes = 4096; aq.flags = 0;
        int mfd = -1;
        if (do_req(fd, VC4D_OP_ALLOC, &aq, sizeof(aq), -1, &rh, &ma,
                   sizeof(ma), &rlen, &mfd) == 0 && rh.status == VC4D_OK &&
            mfd >= 0) {
            void *m = mmap(NULL, ma.len, PROT_READ | PROT_WRITE, MAP_SHARED,
                           mfd, (off_t)ma.off);
            close(mfd);
            if (m != MAP_FAILED) {
                memset(m, 0, 64);
                struct vc4d_launch_req aq2 = {
                    .prog = prog, .timeout_ms = 2000, .nqpu = 1, .nuni = 8,
                    .flags = VC4D_LAUNCH_UNI_ARENA,
                    .uni_handle = ma.handle, .uni_off = 0 };
                uint8_t abuf[sizeof(aq2) + 64];
#define SEND_ARENA(name, want, extra) do {                                      \
                    memcpy(abuf, &aq2, sizeof(aq2));                            \
                    if (do_req(fd, VC4D_OP_LAUNCH, abuf,                       \
                               sizeof(aq2) + (extra), -1, &rh, NULL, 0,        \
                               &rlen, NULL) != 0) {                            \
                        ok(0, name, "no reply");                               \
                    } else if (rh.status != (want)) {                          \
                        char why[96];                                          \
                        snprintf(why, sizeof(why), "got %s, want %s",          \
                                 sname(rh.status), sname(want));               \
                        ok(0, name, why);                                      \
                    } else ok(1, name, NULL);                                  \
                } while (0)

                aq2.uni_handle = 0xFFFFFF;
                SEND_ARENA("UNI_IN_ARENA unknown region", VC4D_E_NOENT, 0);

                aq2.uni_handle = ma.handle; aq2.uni_off = 4;
                SEND_ARENA("UNI_IN_ARENA unaligned block", VC4D_E_INVAL, 0);

                aq2.uni_off = (uint32_t)ma.len - 32;
                SEND_ARENA("UNI_IN_ARENA block past the region", VC4D_E_INVAL, 0);

                aq2.uni_off = 0; aq2.flags = 0x80;
                SEND_ARENA("UNI_IN_ARENA unknown flag bit", VC4D_E_INVAL, 0);

                aq2.flags = VC4D_LAUNCH_UNI_ARENA;
                SEND_ARENA("UNI_IN_ARENA trailing payload", VC4D_E_PROTO, 16);

                ((uint32_t *)m)[0] = 0xC0000000u;   /* bus window, not our arena */
                clean_line(m);
                aq2.uni_off = 0;
                SEND_ARENA("UNI_IN_ARENA word outside the arena", VC4D_E_DENIED, 0);

                ((uint32_t *)m)[0] = 0x3FC00000u;  /* V3D MMIO */
                clean_line(m);
                SEND_ARENA("UNI_IN_ARENA word = V3D MMIO", VC4D_E_DENIED, 0);

                munmap(m, ma.len);
#undef SEND_ARENA
            } else {
                ok(0, "UNI_IN_ARENA mmap", "mmap failed");
            }
            struct vc4d_free_req mf = { .handle = ma.handle, .pad = 0 };
            expect("FREE the UNI_IN_ARENA region", VC4D_OK, VC4D_OP_FREE,
                   &mf, sizeof(mf), -1);
        } else {
            ok(0, "UNI_IN_ARENA region alloc", "ALLOC failed");
        }
    }

    /* WAIT with nothing in flight */
    expect("WAIT with no launch", VC4D_E_NOENT, VC4D_OP_WAIT, NULL, 0, -1);
    expect("WAIT with a payload", VC4D_E_PROTO, VC4D_OP_WAIT, "x", 1, -1);

    /* ---------------------------------------------------------- registers */
    printf("[registers]\n");
    struct vc4d_regs_req rq;
    struct vc4d_regs regs;
    size_t n = 0;
    if (do_req(fd, VC4D_OP_READ_REGS, &(struct vc4d_regs_req){ 0, 0 },
               sizeof(rq), -1, &rh, &regs, sizeof(regs), &n, NULL) == 0 &&
        rh.status == VC4D_OK) {
        printf("  snapshot: IDENT0=0x%08x IDENT1=0x%08x SRQCS=0x%08x\n",
               regs.ident0, regs.ident1, regs.srqcs);
        ok(regs.ident0 != 0xdeadbeef && n == sizeof(regs),
           "READ_REGS snapshot", NULL);
    } else {
        ok(0, "READ_REGS snapshot", "no reply");
    }

    rq.off = 0x4000; rq.count = 1;
    expect("READ_REGS off=0x4000", VC4D_E_INVAL, VC4D_OP_READ_REGS, &rq, sizeof(rq), -1);
    rq.off = 0; rq.count = 2000;
    expect("READ_REGS count=2000", VC4D_E_INVAL, VC4D_OP_READ_REGS, &rq, sizeof(rq), -1);
    rq.off = 2; rq.count = 1;
    expect("READ_REGS unaligned off", VC4D_E_INVAL, VC4D_OP_READ_REGS, &rq, sizeof(rq), -1);
    rq.off = 0; rq.count = 1024;             /* whole 4 KiB block is fine */
    expect("READ_REGS whole block", VC4D_OK, VC4D_OP_READ_REGS, &rq, sizeof(rq), -1);

    struct vc4d_wreg_req wq = { .off = 0x0000, .val = 1 };  /* IDENT0 */
    expect("WRITEREG IDENT0 (not allowlisted)", VC4D_E_DENIED,
           VC4D_OP_WRITEREG, &wq, sizeof(wq), -1);
    wq.off = 0x0600;
    expect("WRITEREG 0x0600 (in block, not allowlisted)", VC4D_E_DENIED,
           VC4D_OP_WRITEREG, &wq, sizeof(wq), -1);
    wq.off = 0x7FFC;
    expect("WRITEREG outside the register block", VC4D_E_INVAL,
           VC4D_OP_WRITEREG, &wq, sizeof(wq), -1);
    wq.off = 0x25;                          /* unaligned */
    expect("WRITEREG unaligned offset", VC4D_E_INVAL,
           VC4D_OP_WRITEREG, &wq, sizeof(wq), -1);
    wq.off = 0x0024; wq.val = 0xFFFFFFFF;   /* SLCACTL: harmless cache op */
    expect("WRITEREG SLCACTL (allowlisted)", VC4D_OK,
           VC4D_OP_WRITEREG, &wq, sizeof(wq), -1);

    /* ---------------------------------------------------------- busy */
    printf("[lifecycle]\n");
    /* a valid launch of the pack program on one QPU, with a second LAUNCH
       immediately after it: the second must be refused as BUSY */
    {
        struct vc4d_uni u[8];
        memset(u, 0, sizeof(u));
        u[0].kind = VC4D_UNI_LIT; u[0].off = 256;                 /* K */
        u[1].kind = VC4D_UNI_LIT; u[1].off = 0;                   /* unit off */
        u[2].kind = VC4D_UNI_PTR; u[2].handle = rb.handle; u[2].off = 0; u[2].len = 4096;
        u[3].kind = VC4D_UNI_LIT; u[3].off = 8192;                /* unit bytes */
        u[4].kind = VC4D_UNI_PTR; u[4].handle = rb.handle; u[4].off = 0; u[4].len = 4096;
        u[5].kind = VC4D_UNI_LIT; u[5].off = 1;                   /* 1 unit */
        u[6].kind = VC4D_UNI_LIT; u[6].off = 0;
        u[7].kind = VC4D_UNI_LIT; u[7].off = 8192;

        struct vc4d_launch_req l2 = { .prog = prog, .timeout_ms = 2000,
                                      .nqpu = 1, .nuni = 8 };
        uint8_t buf[sizeof(l2) + sizeof(u)];
        memcpy(buf, &l2, sizeof(l2));
        memcpy(buf + sizeof(l2), u, sizeof(u));
        if (do_req(fd, VC4D_OP_LAUNCH, buf, sizeof(buf), -1, &rh, NULL, 0, NULL, NULL) == 0 &&
            rh.status == VC4D_OK) {
            ok(1, "LAUNCH valid pack kernel", NULL);
        } else {
            ok(0, "LAUNCH valid pack kernel", sname(rh.status));
        }
        /* immediately launch again: c->inflight is still set */
        if (do_req(fd, VC4D_OP_LAUNCH, buf, sizeof(buf), -1, &rh, NULL, 0, NULL, NULL) == 0)
            ok(rh.status == VC4D_E_BUSY, "second LAUNCH while one is in flight",
               sname(rh.status));
        else
            ok(0, "second LAUNCH while one is in flight", "no reply");

        struct vc4d_wait_rsp wr;
        if (do_req(fd, VC4D_OP_WAIT, NULL, 0, -1, &rh, &wr, sizeof(wr), &rlen, NULL) == 0)
            ok(rh.status == VC4D_OK, "WAIT for the pack kernel", sname(rh.status));
        else
            ok(0, "WAIT for the pack kernel", "no reply");
    }

    /* ---------------------------------------------------------- isolation */
    printf("[isolation]\n");
    {
        /* a second client must not be able to use the first client's program
           handle or its buffer handles */
        int s3 = conn_open();
        if (s3 < 0) {
            ok(0, "second client connects", strerror(errno));
        } else {
            struct vc4d_uni u3[8];
            memset(u3, 0, sizeof(u3));
            u3[0].kind = VC4D_UNI_LIT; u3[0].off = 256;
            u3[1].kind = VC4D_UNI_LIT; u3[1].off = 0;
            u3[2].kind = VC4D_UNI_PTR; u3[2].handle = rb.handle;
            u3[2].off = 0; u3[2].len = 8192;
            u3[3].kind = VC4D_UNI_LIT; u3[3].off = 8192;
            u3[4].kind = VC4D_UNI_PTR; u3[4].handle = rb.handle;
            u3[4].off = 0; u3[4].len = 8192;
            u3[5].kind = VC4D_UNI_LIT; u3[5].off = 1;
            u3[7].kind = VC4D_UNI_LIT; u3[7].off = 8192;

            struct vc4d_launch_req l3 = { .prog = prog, .timeout_ms = 2000,
                                          .nqpu = 1, .nuni = 8 };
            uint8_t b3[sizeof(l3) + sizeof(u3)];
            memcpy(b3, &l3, sizeof(l3));
            memcpy(b3 + sizeof(l3), u3, sizeof(u3));
            if (do_req(s3, VC4D_OP_LAUNCH, b3, sizeof(b3), -1, &rh, NULL, 0,
                       NULL, NULL) == 0)
                ok(rh.status == VC4D_E_NOENT,
                   "client B cannot launch client A's program", sname(rh.status));
            else
                ok(0, "client B cannot launch client A's program", "no reply");

            /* upload A's kernel as B, then point a uniform at A's buffer */
            struct vc4d_upload_req uq3 = { .bytes = (uint32_t)codelen,
                                           .max_qpu = 4, .nuni = 8, .flags = 0 };
            uint8_t ub3[VC4D_MAX_PROG + 32];
            memcpy(ub3, &uq3, sizeof(uq3));
            memcpy(ub3 + sizeof(uq3), code, codelen);
            struct vc4d_upload_rsp ur3;
            size_t un3 = 0;
            if (do_req(s3, VC4D_OP_UPLOAD, ub3, sizeof(uq3) + codelen, -1, &rh,
                       &ur3, sizeof(ur3), &un3, NULL) == 0 && rh.status == VC4D_OK) {
                l3.prog = ur3.prog;
                memcpy(b3, &l3, sizeof(l3));
                if (do_req(s3, VC4D_OP_LAUNCH, b3, sizeof(b3), -1, &rh, NULL, 0,
                           NULL, NULL) == 0)
                    ok(rh.status == VC4D_E_NOENT,
                       "client B cannot point at client A's buffer", sname(rh.status));
                else
                    ok(0, "client B cannot point at client A's buffer", "no reply");
            } else {
                ok(0, "client B upload", "failed");
            }
            close(s3);
        }
    }

    /* ---------------------------------------------------------- fuzz */
    printf("[fuzz] %d random datagrams\n", fuzz);
    {
        uint32_t x = 0x1234567u;
        int sent = 0, replies = 0;
        for (int i = 0; i < fuzz; ++i) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            size_t len = x % 600;
            static uint8_t b[600];
            for (size_t k = 0; k < len; ++k) {
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                b[k] = (uint8_t)x;
            }
            if (x & 1) {   /* half the time, make it look like a valid header */
                if (len >= sizeof(struct vc4d_hdr)) {
                    struct vc4d_hdr *fh = (struct vc4d_hdr *)b;
                    fh->magic = VC4D_MAGIC;
                    fh->len = (uint32_t)len;
                    fh->version = VC4D_VERSION;
                    fh->seq = seq++;
                    fh->status = 0;
                    fh->op = (uint16_t)(1 + (x % 14));
                }
            }
            if (send_msg(fd, b, len, -1) == 0)
                sent++;
            else {
                /* zero-length datagram is treated as a disconnect */
                if (fd >= 0) close(fd);
                fd = conn_open();
                if (fd < 0) { ok(0, "reconnect during fuzz", strerror(errno)); break; }
                continue;
            }
            struct vc4d_hdr fh;
            if (recv_msg(fd, &fh, NULL, 0, NULL, NULL) == 0)
                replies++;
            else {
                if (fd >= 0) close(fd);
                fd = conn_open();
                if (fd < 0) { ok(0, "reconnect during fuzz", strerror(errno)); break; }
            }
        }
        printf("  sent %d, replies %d\n", sent, replies);
        ensure_alive("fuzz");
        ok(ping(), "daemon alive after fuzzing", NULL);
    }

    /* client disappears with a launch in flight */
    printf("[disconnect mid-launch]\n");
    {
        struct vc4d_uni u[8];
        memset(u, 0, sizeof(u));
        u[0].off = 256;
        u[2].kind = VC4D_UNI_PTR; u[2].handle = rb.handle; u[2].len = 8192;
        u[3].off = 8192;
        u[4].kind = VC4D_UNI_PTR; u[4].handle = rb.handle; u[4].len = 8192;
        u[5].off = 1; u[7].off = 8192;
        struct vc4d_launch_req l2 = { .prog = prog, .timeout_ms = 2000,
                                      .nqpu = 1, .nuni = 8 };
        uint8_t buf[sizeof(l2) + sizeof(u)];
        memcpy(buf, &l2, sizeof(l2));
        memcpy(buf + sizeof(l2), u, sizeof(u));
        int s2 = conn_open();
        uint32_t prog2 = 0;
        struct region rb2 = { 0, 0, 0 };
        if (s2 >= 0) {
            struct vc4d_alloc_req aq2 = { .bytes = 8192, .flags = 0 };
            struct vc4d_alloc_rsp ar2;
            size_t an = 0;
            int afd = -1;
            if (do_req(s2, VC4D_OP_ALLOC, &aq2, sizeof(aq2), -1, &rh, &ar2,
                       sizeof(ar2), &an, &afd) == 0 && rh.status == VC4D_OK) {
                rb2.handle = ar2.handle; rb2.off = ar2.off; rb2.len = ar2.len;
                if (afd >= 0) close(afd);
            }
            struct vc4d_upload_req uq2 = { .bytes = (uint32_t)codelen,
                                           .max_qpu = 4, .nuni = 8, .flags = 0 };
            uint8_t ub[VC4D_MAX_PROG + 32];
            memcpy(ub, &uq2, sizeof(uq2));
            memcpy(ub + sizeof(uq2), code, codelen);
            struct vc4d_upload_rsp ur2;
            size_t un = 0;
            if (do_req(s2, VC4D_OP_UPLOAD, ub, sizeof(uq2) + codelen, -1, &rh,
                       &ur2, sizeof(ur2), &un, NULL) == 0 && rh.status == VC4D_OK)
                prog2 = ur2.prog;
            l2.prog = prog2;
            memcpy(buf, &l2, sizeof(l2));
            /* regions are owned per client too: point at s2's own buffer */
            u[2].handle = rb2.handle;
            u[4].handle = rb2.handle;
            memcpy(buf + sizeof(l2), u, sizeof(u));
        }
        if (s2 >= 0 && prog2 &&
            do_req(s2, VC4D_OP_LAUNCH, buf, sizeof(buf), -1, &rh, NULL, 0, NULL, NULL) == 0 &&
            rh.status == VC4D_OK) {
            close(s2);                    /* vanish without WAIT */
            usleep(20 * 1000);
            ok(ping(), "daemon survives a client that vanishes mid-launch", NULL);
        } else {
            char why[96];
            snprintf(why, sizeof(why), "launch failed (%s)", sname(rh.status));
            ok(0, "disconnect mid-launch setup", why);
            if (s2 >= 0) close(s2);
        }
    }

    /* ---------------------------------------------------------- summary */
    struct vc4d_stats_rsp st;
    size_t sn = 0;
    if (do_req(fd, VC4D_OP_STATS, NULL, 0, -1, &rh, &st, sizeof(st), &sn, NULL) == 0 &&
        rh.status == VC4D_OK) {
        printf("daemon stats: launches=%llu failed=%llu rejects=%llu dead=%u\n",
               (unsigned long long)st.launches,
               (unsigned long long)st.launches_failed,
               (unsigned long long)st.rejects, st.dead);
        ok(st.rejects > 0, "daemon counted rejected requests", NULL);
        ok(st.dead == 0, "daemon is not dead", NULL);
    }
    printf("hostile: %d passed, %d failed\n", pass, fail);
    free(code);
    if (fd >= 0)
        close(fd);
    return fail ? 1 : 0;
}
