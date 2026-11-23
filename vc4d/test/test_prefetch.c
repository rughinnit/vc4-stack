/*
 * test_prefetch.c - the daemon prepares request N+1 while request N runs.
 *
 * What this pins, against a real daemon (two of them, one with --no-prefetch,
 * compared by test_prefetch.sh):
 *
 *   - two SUBMITs outstanding on one connection complete in order, with
 *     per-entry statuses and launch ids as the synchronous path gives them;
 *   - a refusal inside the *prepared* request is reported at its own entry,
 *     every later entry is NOTRUN, and the header carries the first failure -
 *     the same numbers whether the daemon prepared it during the previous
 *     launch or read it afterwards;
 *   - the launches of one request are contiguous in id, i.e. no other request
 *     was executed in the middle of the batch (two clients still serialise);
 *   - a client that dies with a request in flight does not wedge the queue:
 *     its region and program are released and a new client runs immediately.
 *
 * The kernel is the daemon's own probe program: thrend plus padding, no
 * uniforms, one QPU.  It touches no memory, so the test is safe to run next to
 * anything, and it finishes before the QPU's completion counter can lie.
 *
 * The counters that prove the preparation actually ran under a launch are in
 * the daemon's exit line ("prefetch: N datagram(s) read early, M prepared under
 * a launch"); test_prefetch.sh checks them, because a client cannot observe
 * them without a protocol change.
 *
 * usage: test_prefetch SOCKET
 */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "vc4client.h"

static const char *sock = "/run/vc4d.sock";
static int fails;

#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

/* thrend followed by padding: the daemon's probe program (vc4d.c, probe_words),
 * a valid one-QPU program that reads no uniform and writes no memory. */
static const uint32_t probe_words[8] = {
    0x9e70009eu, 0x30000000u,
    0x9e70009eu, 0x10000000u,
    0x9e70009eu, 0x10000000u,
    0x9e70009eu, 0x10000000u,
};

static struct vc4d_batch_item items[VC4D_BATCH_MAX];

/* A region for the uniform blocks and the no-op program. */
static int prepare(vc4c_conn *c, vc4c_buf *uni, vc4c_prog *prog)
{
    int rc = vc4c_alloc(c, 4096, 0, uni);
    if (rc != 0)
        return rc;
    memset(uni->map, 0, 64);
    return vc4c_upload(c, probe_words, sizeof(probe_words), 1, 1, prog);
}

static void ent_launch(vc4c_batch_ent *e, uint32_t prog, uint32_t off)
{
    memset(e, 0, sizeof(*e));
    e->kind = VC4D_BENT_LAUNCH;
    e->prog = prog;
    e->nqpu = 1;
    e->nuni = 1;
    e->uni_off = off;
    e->flags = VC4D_LAUNCH_UNI_MASK;
    e->uni_addr = 0;              /* the single word is a value, not an address */
}

/* Two SUBMITs outstanding, polled in order. */
static void case_two_deep(vc4c_conn *c, const vc4c_buf *uni, const vc4c_prog *p)
{
    vc4c_batch_ent a[1], b[1];
    vc4c_token ta, tb, tc;
    ent_launch(&a[0], p->handle, 0);
    ent_launch(&b[0], p->handle, 64);

    int rc = vc4c_submit_batch(c, uni, a, 1, &ta);
    CHECK(rc == 0, "SUBMIT A queued: %s", vc4c_strerror(rc));
    rc = vc4c_submit_batch(c, uni, b, 1, &tb);
    CHECK(rc == 0 && vc4c_submits_outstanding(c) == 2,
          "SUBMIT B queued, two outstanding: %s outstanding=%d",
          vc4c_strerror(rc), vc4c_submits_outstanding(c));

    rc = vc4c_submit_batch(c, uni, b, 1, &tc);
    CHECK(rc == -VC4D_E_BUSY, "a third SUBMIT is refused: %s", vc4c_strerror(rc));

    unsigned done = 0;
    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, ta, items, 1, &done);
    uint32_t ida = items[0].launch;
    CHECK(rc == 0 && done == 1 && items[0].status == VC4D_OK,
          "poll A: rc=%s done=%u item=%u launch=%u", vc4c_strerror(rc), done,
          items[0].status, ida);

    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, tb, items, 1, &done);
    CHECK(rc == 0 && done == 1 && items[0].status == VC4D_OK &&
          items[0].launch > ida,
          "poll B: rc=%s done=%u item=%u launch=%u (after A's %u)",
          vc4c_strerror(rc), done, items[0].status, items[0].launch, ida);
    CHECK(vc4c_submits_outstanding(c) == 0, "nothing outstanding afterwards");
}

/* A refusal inside the prepared request, reported in the right entry. */
static void case_refusal(vc4c_conn *c, const vc4c_buf *uni, const vc4c_prog *p)
{
    vc4c_batch_ent a[1], b[2];
    vc4c_token ta, tb;
    unsigned done = 0;

    ent_launch(&a[0], p->handle, 0);
    ent_launch(&b[0], p->handle, 64);
    ent_launch(&b[1], 0xdead, 128);      /* unknown program handle */

    int rc = vc4c_submit_batch(c, uni, a, 1, &ta);
    CHECK(rc == 0, "SUBMIT A queued: %s", vc4c_strerror(rc));
    rc = vc4c_submit_batch(c, uni, b, 2, &tb);
    CHECK(rc == 0, "SUBMIT B (second entry bad) queued: %s", vc4c_strerror(rc));

    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, ta, items, 1, &done);
    CHECK(rc == 0 && items[0].status == VC4D_OK, "A completed: %s",
          vc4c_strerror(rc));

    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, tb, items, 2, &done);
    CHECK(rc == -VC4D_E_NOENT && done == 2 &&
          items[0].status == VC4D_OK && items[1].status == VC4D_E_NOENT,
          "B: header=%s done=%u item0=%u item1=%u", vc4c_strerror(rc), done,
          items[0].status, items[1].status);

    /* The same, with the failure in entry 0: nothing after it runs. */
    memset(b, 0, sizeof(b));
    b[0].kind = VC4D_BENT_FENCE;
    b[0].handle = 0xdead;
    b[0].dir = VC4D_SYNC_G2R;
    ent_launch(&b[1], p->handle, 0);

    rc = vc4c_submit_batch(c, uni, a, 1, &ta);
    CHECK(rc == 0, "SUBMIT A queued: %s", vc4c_strerror(rc));
    rc = vc4c_submit_batch(c, uni, b, 2, &tb);
    CHECK(rc == 0, "SUBMIT B (bad fence first) queued: %s", vc4c_strerror(rc));

    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, ta, items, 1, &done);
    CHECK(rc == 0 && items[0].status == VC4D_OK, "A completed: %s",
          vc4c_strerror(rc));

    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, tb, items, 2, &done);
    CHECK(rc == -VC4D_E_INVAL && done == 1 &&
          items[0].status == VC4D_E_INVAL &&
          items[1].status == VC4D_E_NOTRUN,
          "B: header=%s done=%u item0=%u item1=%u", vc4c_strerror(rc), done,
          items[0].status, items[1].status);
}

/* One request's launches are contiguous in id: no other request ran inside it. */
static void case_contiguous(vc4c_conn *c, const vc4c_buf *uni, const vc4c_prog *p)
{
    vc4c_batch_ent e[8];
    vc4c_token t;
    unsigned done = 0;
    for (unsigned i = 0; i < 8; ++i)
        ent_launch(&e[i], p->handle, (i & 7) * 64);
    int rc = vc4c_submit_batch(c, uni, e, 8, &t);
    CHECK(rc == 0, "SUBMIT of 8 queued: %s", vc4c_strerror(rc));
    memset(items, 0, sizeof(items));
    rc = vc4c_poll_batch(c, t, items, 8, &done);
    int contiguous = 1;
    for (unsigned i = 1; i < 8; ++i)
        if (items[i].launch != items[i - 1].launch + 1)
            contiguous = 0;
    CHECK(rc == 0 && done == 8 && contiguous,
          "8 launches ran back to back and in order: rc=%s ids %u..%u",
          vc4c_strerror(rc), items[0].launch, items[7].launch);
}

/* A client that queues a request and dies must not wedge the daemon or its
 * arena.  The child exits without reading its reply, so the daemon meets EOF
 * with the launch completed and one unreaped reply in the socket. */
static void case_dying_client(vc4c_conn *c)
{
    pid_t pid = fork();
    if (pid == 0) {
        vc4c_conn *cc = NULL;
        vc4c_buf uni;
        vc4c_prog prog;
        if (vc4c_connect(&cc, sock) != 0 || prepare(cc, &uni, &prog) != 0)
            _exit(1);
        vc4c_batch_ent e[2];
        vc4c_token t;
        ent_launch(&e[0], prog.handle, 0);
        ent_launch(&e[1], prog.handle, 64);
        if (vc4c_submit_batch(cc, &uni, e, 2, &t) != 0)
            _exit(1);
        _exit(0);                       /* no poll, no CLOSE, no FREE */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    usleep(200 * 1000);

    int rc = vc4c_ping(c);
    CHECK(rc == 0, "daemon still answers after a client died mid-request: %s",
          vc4c_strerror(rc));

    vc4c_conn *c2 = NULL;
    vc4c_buf uni;
    vc4c_prog prog;
    rc = vc4c_connect(&c2, sock);
    if (rc == 0)
        rc = prepare(c2, &uni, &prog);
    CHECK(rc == 0, "a new client gets a region and a program: %s",
          vc4c_strerror(rc));
    if (rc == 0) {
        vc4c_batch_ent e[1];
        vc4c_token t;
        unsigned done = 0;
        ent_launch(&e[0], prog.handle, 0);
        rc = vc4c_submit_batch(c2, &uni, e, 1, &t);
        memset(items, 0, sizeof(items));
        if (rc == 0)
            rc = vc4c_poll_batch(c2, t, items, 1, &done);
        CHECK(rc == 0 && items[0].status == VC4D_OK,
              "and runs a launch: rc=%s item=%u", vc4c_strerror(rc),
              items[0].status);
        vc4c_free(c2, &uni);
        vc4c_close(c2);
    }
}

int main(int argc, char **argv)
{
    if (argc > 1)
        sock = argv[1];
    vc4c_conn *c = NULL;
    vc4c_buf uni;
    vc4c_prog prog;
    memset(&uni, 0, sizeof(uni));
    memset(&prog, 0, sizeof(prog));

    int rc = vc4c_connect(&c, sock);
    if (rc != 0) {
        printf("connect %s: %s\n", sock, vc4c_strerror(rc));
        return 2;
    }
    struct vc4d_hello_rsp info;
    rc = vc4c_info(c, &info);
    CHECK(rc == 0 && (info.flags & VC4D_F_ASYNC_SUBMIT),
          "HELLO advertises VC4D_F_ASYNC_SUBMIT (flags=0x%x, arena %u bytes)",
          info.flags, info.arena_bytes);
    if (rc != 0 || !(info.flags & VC4D_F_ASYNC_SUBMIT)) {
        vc4c_close(c);
        return 1;
    }

    rc = prepare(c, &uni, &prog);
    CHECK(rc == 0, "region and program ready: %s", vc4c_strerror(rc));
    if (rc != 0) {
        vc4c_close(c);
        return 1;
    }

    case_two_deep(c, &uni, &prog);
    case_refusal(c, &uni, &prog);
    case_contiguous(c, &uni, &prog);
    case_dying_client(c);

    vc4c_free(c, &uni);
    vc4c_close(c);
    printf("%s (%d failure(s))\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
