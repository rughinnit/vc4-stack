/*
 * test_direct_lock.c - G11: a direct-mode client and a running vc4d must not
 * drive the QPUs at the same time.
 *
 * Before the fix the direct path (VC4INFER_DAEMON=0, root, /dev/mem) never
 * touched /tmp/vc4-gpu.lock, which vc4d holds for each launch.  The two
 * programs therefore wrote SRQCS/SRQPC/VPMBASE over each other: a direct
 * launch next to an active daemon timed out (`conv_nhwc_dil: GPU timeout`)
 * while the daemon stayed healthy.
 *
 * What this test checks, in order:
 *
 *   1. the lock is honoured: another process holds /tmp/vc4-gpu.lock for
 *      HOLD_MS, then this process launches a direct-mode matmul.  Its wall
 *      time must include that wait (it may not start while the lock is held).
 *      Before the fix it started and finished inside the hold window.
 *   2. a stale/long holder produces a named error, not a hang: with
 *      VC4INFER_GPU_LOCK_MS=200 and the lock held for 1 s the launch must
 *      fail with a message naming /tmp/vc4-gpu.lock.
 *   3. end to end: with a real vc4d running (given on the command line), a
 *      direct-mode launch and a daemon launch alternated N times both succeed
 *      with no timeouts on either side.
 *
 * It needs root (direct mode maps /dev/mem) and a vc4d socket for step 3.
 *
 * usage: test_direct_lock [SOCKET]            (default /run/vc4d.sock)
 *        run as root:  sudo ./build/test_direct_lock /run/vc4d.sock
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "vc4infer.h"
#include "vc4d_proto.h"
#include "vc4client.h"

#define LOCK_PATH "/tmp/vc4-gpu.lock"
#define HOLD_MS   600

static int fails;
#define CHECK(cond, ...) do {                                          \
        if (cond) { printf("  PASS  "); }                              \
        else { printf("  FAIL  "); fails++; }                          \
        printf(__VA_ARGS__);                                           \
        printf("\n");                                                  \
    } while (0)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Hold the lock in a child for hold_ms: the parent cannot be inside a launch
 * during that window. */
static pid_t hold_lock(int hold_ms)
{
    pid_t p = fork();
    if (p != 0)
        return p;
    int fd = open(LOCK_PATH, O_RDWR | O_CREAT, 0666);
    if (fd < 0)
        fd = open(LOCK_PATH, O_RDONLY);
    if (fd < 0 || flock(fd, LOCK_EX) != 0)
        _exit(2);
    usleep(hold_ms * 1000);
    _exit(0);
}

int main(int argc, char **argv)
{
    const char *sock = argc > 1 ? argv[1] : "/run/vc4d.sock";

    if (getenv("VC4INFER_DAEMON")) {
        printf("FAIL: run this with VC4INFER_DAEMON unset (it tests the direct "
               "path)\n");
        return 1;
    }

    /* ---- 1. the direct launch waits for the lock --------------------- */
    vc4_ctx *ctx = NULL;
    vc4_status st = vc4_init(&ctx, NULL, 0);
    if (st != VC4_OK) {
        printf("FAIL: direct vc4_init: %s (run as root)\n",
               vc4_strerror(st));
        return 1;
    }
    printf("direct context: arena %.1f MiB\n", vc4_arena_bytes(ctx) / 1048576.0);

    vc4_buf a, b, c;
    if (vc4_alloc(ctx, 256 * 256 * 4, &a) ||
        vc4_alloc(ctx, 256 * 256 * 4, &b) ||
        vc4_alloc(ctx, 256 * 256 * 4, &c)) {
        printf("FAIL: alloc\n");
        return 1;
    }
    float *pa = vc4_host_ptr(ctx, &a), *pb = vc4_host_ptr(ctx, &b);
    for (int i = 0; i < 256 * 256; ++i) {
        pa[i] = 1.0f;
        pb[i] = 1.0f;
    }
    vc4_upload(ctx, &a, pa, 256 * 256 * 4);
    vc4_upload(ctx, &b, pb, 256 * 256 * 4);

    /* warm up the kernel cache and prove the launch works at all */
    st = vc4_matmul(ctx, &a, &b, &c, 256, 256, 256, 0);
    CHECK(st == VC4_OK, "direct matmul works on its own: %s",
          vc4_strerror(st));
    if (st != VC4_OK)
        return 1;

    pid_t holder = hold_lock(HOLD_MS);
    usleep(50 * 1000);                /* let the child take it */
    double t0 = now_ms();
    st = vc4_matmul(ctx, &a, &b, &c, 256, 256, 256, 0);
    double took = now_ms() - t0;
    int ws = 0;
    waitpid(holder, &ws, 0);
    CHECK(st == VC4_OK && took > HOLD_MS * 0.8,
          "direct launch waited for the held GPU lock: %.0f ms for a %.0f ms "
          "hold (must be >= %.0f)", took, (double)HOLD_MS,
          HOLD_MS * 0.8);

    /* ---- 2. a stuck holder is an error, not a hang ------------------- */
    holder = hold_lock(1500);
    usleep(50 * 1000);
    setenv("VC4INFER_GPU_LOCK_MS", "250", 1);
    t0 = now_ms();
    st = vc4_matmul(ctx, &a, &b, &c, 256, 256, 256, 0);
    took = now_ms() - t0;
    waitpid(holder, &ws, 0);
    unsetenv("VC4INFER_GPU_LOCK_MS");
    CHECK(st != VC4_OK && took < 1200 &&
          strstr(vc4_last_error(ctx), LOCK_PATH) != NULL,
          "a stuck lock holder fails in %.0f ms with a message naming %s "
          "(\"%s\")", took, LOCK_PATH, vc4_last_error(ctx));

    /* ---- 3. interleaved with a real daemon --------------------------- */
    vc4c_conn *dc = NULL;
    int rc = vc4c_connect(&dc, sock);
    if (rc) {
        printf("  SKIP  no daemon on %s (%s): step 3 needs one\n", sock,
               vc4c_strerror(rc));
    } else {
        /* the daemon's known-good 4-instruction probe: cheap and harmless */
        static const uint32_t good[8] = {
            0x9e70009eu, 0x30000000u, 0x9e70009eu, 0x10000000u,
            0x9e70009eu, 0x10000000u, 0x9e70009eu, 0x10000000u,
        };
        vc4c_buf uni;
        vc4c_prog prog;
        vc4c_result r;
        int bad = 0;
        if (vc4c_alloc(dc, 4096, VC4D_ALLOC_ZERO, &uni) ||
            vc4c_upload(dc, good, sizeof(good), 1, 1, &prog)) {
            printf("FAIL: daemon setup\n");
            fails++;
        } else {
            for (int i = 0; i < 8; ++i) {
                memset(&r, 0, sizeof(r));
                if (vc4c_launch_wait(dc, &prog, 1, &uni, 0, 5000, &r) != 0)
                    bad++;
                if (vc4_matmul(ctx, &a, &b, &c, 256, 256, 256, 0) != VC4_OK)
                    bad++;
            }
            CHECK(bad == 0, "16 interleaved launches (8 direct + 8 daemon) "
                  "with no timeout or error (%d bad)", bad);
        }
        vc4c_close(dc);
    }

    vc4_shutdown(ctx);
    printf(fails ? "G11 FAIL (%d)\n" : "G11 PASS\n", fails);
    return fails ? 1 : 0;
}
