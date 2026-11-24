/*
 * test_hello_version.c - protocol 5's half of the HELLO negotiation.
 *
 * The client library descends the HELLO ladder so a new engine can reach an
 * old daemon.  This test is the daemon's side of that contract:
 *
 *   1. HELLO 5 is served and names the map ("slots256") and its data base;
 *   2. HELLO 4 is refused with E_VERSION and leaves the connection usable,
 *      which is what lets the client's ladder try 4, 3, 2 in turn;
 *   3. a non-HELLO request that says version 4 is refused too, so an old
 *      client that ignores the HELLO reply cannot lay its 3 MiB heap out
 *      under this daemon's map;
 *   4. HELLO 5 still works after the refusals.
 *
 * What must not happen is a version-4 ALLOC succeeding: it would place the
 * client's fixed zones where this daemon reserved something else.
 *
 * usage: test_hello_version [SOCKET]
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "vc4d_proto.h"

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static int conn(const char *path)
{
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    snprintf(sa.sun_path, sizeof(sa.sun_path), "%s", path);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* One request, one reply.  `version` is the header's version, which is what
 * the daemon checks before it dispatches. */
static int req(int fd, uint16_t op, uint16_t version, const void *pl,
               size_t plen, struct vc4d_hdr *rh, void *rp, size_t rcap)
{
    struct vc4d_hdr h;
    memset(&h, 0, sizeof(h));
    h.magic = VC4D_MAGIC;
    h.len = (uint32_t)(sizeof(h) + plen);
    h.op = op;
    h.version = version;
    h.seq = 1;
    uint8_t buf[256];
    if (h.len > sizeof(buf))
        return -1;
    memcpy(buf, &h, sizeof(h));
    if (plen)
        memcpy(buf + sizeof(h), pl, plen);
    if (send(fd, buf, h.len, 0) != (ssize_t)h.len)
        return -1;
    ssize_t got = recv(fd, buf, sizeof(buf), 0);
    if (got < (ssize_t)sizeof(*rh))
        return -1;
    memcpy(rh, buf, sizeof(*rh));
    size_t n = (size_t)got - sizeof(*rh);
    if (rp && rcap) {
        if (n > rcap)
            return -1;
        memcpy(rp, buf + sizeof(*rh), n);
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/run/vc4d.sock";
    struct vc4d_hello_req q = { .version = 5, .flags = 0 };
    struct vc4d_hello_rsp r;
    struct vc4d_hdr h;
    int fd = conn(path);
    if (fd < 0) {
        printf("connect %s: %s\n", path, strerror(errno));
        return 2;
    }

    memset(&r, 0, sizeof(r));
    int rc = req(fd, VC4D_OP_HELLO, 5, &q, sizeof(q), &h, &r, sizeof(r));
    CHECK(rc == 0 && h.status == VC4D_OK, "HELLO 5 served (%s)",
          rc == 0 ? "reply" : "no reply");
    CHECK(rc == 0 && r.version == VC4D_VERSION, "reply version is %u (got %u)",
          VC4D_VERSION, rc == 0 ? r.version : 0);
    CHECK(rc == 0 && strncmp(r.layout, VC4D_LAYOUT_NAME, sizeof(r.layout)) == 0,
          "reply names layout '%s' (got '%.12s')", VC4D_LAYOUT_NAME,
          rc == 0 ? r.layout : "");
    /* 0x600000 is the protocol-5 map ('slots256'); the wire carries it in
     * HELLO because a client cannot derive it from the message sizes. */
    CHECK(rc == 0 && r.data_base == 0x600000u,
          "reply data base is 0x600000 (got 0x%x)",
          rc == 0 ? r.data_base : 0);

    /* The ladder's first step: 4 must be refused, not served. */
    q.version = 4;
    rc = req(fd, VC4D_OP_HELLO, 4, &q, sizeof(q), &h, &r, sizeof(r));
    CHECK(rc == 0 && h.status == VC4D_E_VERSION,
          "HELLO 4 refused with E_VERSION (%d)", rc == 0 ? h.status : -1);
    CHECK(rc == 0 && h.detail0 == VC4D_VERSION,
          "the refusal reports the served version (%u)",
          rc == 0 ? h.detail0 : 0);

    /* An old client that skips HELLO must not get an ALLOC through. */
    struct vc4d_alloc_req aq = { .bytes = 4096, .flags = 0 };
    rc = req(fd, VC4D_OP_ALLOC, 4, &aq, sizeof(aq), &h, &r, sizeof(r));
    CHECK(rc == 0 && h.status == VC4D_E_VERSION,
          "version-4 ALLOC refused with E_VERSION (%d)",
          rc == 0 ? h.status : -1);

    /* The connection survives both refusals: the ladder needs that. */
    q.version = 5;
    memset(&r, 0, sizeof(r));
    rc = req(fd, VC4D_OP_HELLO, 5, &q, sizeof(q), &h, &r, sizeof(r));
    CHECK(rc == 0 && h.status == VC4D_OK && r.version == VC4D_VERSION,
          "HELLO 5 still served after the refusals");

    close(fd);
    printf(fails ? "HELLO VERSION FAIL (%d)\n" : "HELLO VERSION PASS\n", fails);
    return fails ? 1 : 0;
}
