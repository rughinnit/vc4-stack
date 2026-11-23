/*
 * test_submit.c - protocol 4: SUBMIT without waiting, poll later.
 *
 * No GPU and no daemon install: the test starts test/fake_daemon.py, which
 * speaks the wire protocol, hands out a memfd arena and delays each request so
 * the client has time to do work while it is "executing".  What is pinned
 * here is the connection contract the engine's depth-2 pipeline depends on:
 *
 *   - HELLO advertises VC4D_F_ASYNC_SUBMIT;
 *   - two submits can be outstanding, and the second returns while the first
 *     is still executing (that is the overlap);
 *   - the third is refused with E_BUSY, and so is a synchronous batch while
 *     any submit is outstanding (the reply arrows would no longer line up);
 *   - polls are in submit order; a poll for the wrong token is refused;
 *   - a submit whose entries are malformed is refused before it is sent;
 *   - a close with one outstanding submit does not hang or misread CLOSE's
 *     reply.
 *
 * usage: test_submit [FAKE_DAEMON] [SOCKET]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "vc4client.h"

static int fails;

#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

int main(int argc, char **argv)
{
    const char *script = argc > 1 ? argv[1] : "test/fake_daemon.py";
    char sock[96];
    if (argc > 2) {
        snprintf(sock, sizeof(sock), "%s", argv[2]);
    } else {
        const char *tmp = getenv("TMPDIR");
        snprintf(sock, sizeof(sock), "%s/fake-vc4d-submit.sock",
                 tmp && *tmp ? tmp : "/run/user");
    }
    unlink(sock);
    pid_t daemon = fork();
    if (daemon == 0) {
        execlp("python3", "python3", script, sock, "--delay-ms", "300", NULL);
        _exit(127);
    }
    if (daemon < 0) {
        printf("fork: %s\n", strerror(errno));
        return 2;
    }

    vc4c_conn *c = NULL;
    int rc = -1;
    for (int i = 0; i < 100 && rc != 0; ++i) {
        rc = vc4c_connect(&c, sock);
        if (rc != 0)
            usleep(50000);
    }
    if (rc != 0) {
        printf("connect %s: %s\n", sock, vc4c_strerror(rc));
        kill(daemon, SIGTERM);
        waitpid(daemon, NULL, 0);
        return 2;
    }

    struct vc4d_hello_rsp info;
    rc = vc4c_info(c, &info);
    CHECK(rc == 0 && (info.flags & VC4D_F_ASYNC_SUBMIT),
          "HELLO advertises VC4D_F_ASYNC_SUBMIT (v%u flags=0x%x)",
          info.version, info.flags);
    CHECK(info.version == 4, "negotiated protocol 4 (got %u)", info.version);

    vc4c_buf region;
    rc = vc4c_alloc(c, 1 << 20, 0, &region);
    CHECK(rc == 0 && region.map, "ALLOC 1 MiB: %s", vc4c_strerror(rc));
    if (rc != 0) {
        if (c)
            vc4c_close(c);
        kill(daemon, SIGTERM);
        waitpid(daemon, NULL, 0);
        return 1;
    }

    vc4c_prog prog;
    uint8_t code[16] = { 0 };
    rc = vc4c_upload(c, code, sizeof(code), 1, 4, &prog);
    CHECK(rc == 0 && prog.handle, "UPLOAD a stand-in program: %s",
          vc4c_strerror(rc));

    vc4c_batch_ent e[2];
    memset(e, 0, sizeof(e));
    for (int i = 0; i < 2; ++i) {
        e[i].kind = VC4D_BENT_LAUNCH;
        e[i].prog = prog.handle;
        e[i].nqpu = 1;
        e[i].nuni = 4;
        e[i].flags = VC4D_LAUNCH_UNI_MASK;
        e[i].uni_off = (uint32_t)i * 64u;
        e[i].uni_addr = 0;
    }
    struct vc4d_batch_item items[VC4D_BATCH_MAX];
    unsigned done = 0;

    /* 1. validation happens before the wire. */
    rc = vc4c_submit_batch(c, &region, e, 0, NULL);
    CHECK(rc == -VC4D_E_INVAL, "submit with no token refused: %s",
          vc4c_strerror(rc));
    vc4c_token t0, t1, t2;
    rc = vc4c_submit_batch(c, &region, e, 0, &t0);
    CHECK(rc == -VC4D_E_INVAL, "submit count 0 refused: %s", vc4c_strerror(rc));
    vc4c_batch_ent bad = e[0];
    bad.nuni = 0;
    rc = vc4c_submit_batch(c, &region, &bad, 1, &t0);
    CHECK(rc == -VC4D_E_INVAL, "submit nuni 0 refused: %s", vc4c_strerror(rc));
    CHECK(vc4c_submits_outstanding(c) == 0, "nothing outstanding after refusals");

    /* 2. the overlap: two in flight, the second returns while the first runs. */
    uint64_t t_begin = now_ms();
    rc = vc4c_submit_batch(c, &region, e, 2, &t0);
    CHECK(rc == 0, "submit #1 sent: %s", vc4c_strerror(rc));
    rc = vc4c_submit_batch(c, &region, e, 2, &t1);
    uint64_t t_two = now_ms() - t_begin;
    CHECK(rc == 0 && vc4c_submits_outstanding(c) == 2,
          "two submits outstanding (rc=%s outstanding=%d)",
          vc4c_strerror(rc), vc4c_submits_outstanding(c));
    CHECK(t_two < 300,
          "second submit returned in %llu ms, before the 300 ms request ended",
          (unsigned long long)t_two);

    /* 3. a third is refused while two are outstanding. */
    rc = vc4c_submit_batch(c, &region, e, 2, &t2);
    CHECK(rc == -VC4D_E_BUSY, "third submit refused with BUSY: %s",
          vc4c_strerror(rc));

    /* 4. the synchronous form cannot be mixed with an outstanding submit. */
    rc = vc4c_launch_batch(c, &region, e, 2, items, &done);
    CHECK(rc == -VC4D_E_BUSY, "sync batch refused while a submit is in flight: %s",
          vc4c_strerror(rc));

    /* 5. polling out of order is refused, not answered with the wrong reply. */
    vc4c_token wrong = t1;
    wrong.seq ^= 0x80000000u;
    rc = vc4c_poll_batch(c, wrong, items, 2, &done);
    CHECK(rc == -VC4D_E_INVAL, "poll with an unknown token refused: %s",
          vc4c_strerror(rc));
    CHECK(vc4c_submits_outstanding(c) == 2, "a refused poll consumes nothing");

    /* 6. polls in order get their own completions. */
    rc = vc4c_poll_batch(c, t0, items, 2, &done);
    CHECK(rc == 0 && done == 2 && items[0].status == VC4D_OK &&
          items[1].status == VC4D_OK,
          "poll #1: %s done=%u item0=%u item1=%u", vc4c_strerror(rc), done,
          items[0].status, items[1].status);
    rc = vc4c_poll_batch(c, t1, items, 2, &done);
    CHECK(rc == 0 && done == 2, "poll #2: %s done=%u", vc4c_strerror(rc), done);
    CHECK(vc4c_submits_outstanding(c) == 0, "nothing outstanding after polls");

    /* 7. the slot is free again: another submit is accepted. */
    rc = vc4c_submit_batch(c, &region, e, 2, &t0);
    CHECK(rc == 0, "a further submit is accepted: %s", vc4c_strerror(rc));

    /* 8. closing with an outstanding submit reads its reply first. */
    vc4c_close(c);

    kill(daemon, SIGTERM);
    waitpid(daemon, NULL, 0);
    printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails,
           fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
