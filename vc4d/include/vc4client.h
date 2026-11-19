/*
 * vc4client.h - client library for vc4d.
 *
 * Everything here runs unprivileged: the library talks to /run/vc4d.sock and
 * mmaps the dma-buf fds the daemon passes back.  It never opens /dev/mem.
 */
#ifndef VC4CLIENT_H
#define VC4CLIENT_H

#include <stddef.h>
#include <stdint.h>
#include "../proto/vc4d_proto.h"

typedef struct vc4c_conn vc4c_conn;

typedef struct {
    uint32_t handle;      /* daemon-side region handle */
    uint32_t off, len;    /* byte offset/length inside the arena */
    uint32_t bus;         /* GPU bus address of map[0] (from ALLOC) */
    void    *map;         /* mmap of this region (write-combining) */
} vc4c_buf;

typedef struct {
    uint32_t handle;      /* program handle */
    uint32_t code_off, words;
    unsigned nuni;        /* uniform words declared at UPLOAD */
} vc4c_prog;

typedef struct {
    uint32_t kind, handle, off, len;
} vc4c_uni;

#define VC4C_LIT(v)        ((vc4c_uni){ .kind = VC4D_UNI_LIT, .off = (uint32_t)(v) })
#define VC4C_PTR(b, o, l)  ((vc4c_uni){ .kind = VC4D_UNI_PTR, \
        .handle = (b)->handle, .off = (uint32_t)(o), .len = (uint32_t)(l) })

typedef struct {
    uint32_t launch, errs;
    uint64_t wall_ns;
    uint32_t cycles;
} vc4c_result;

typedef struct vc4d_stats_rsp vc4c_stats;

int  vc4c_connect(vc4c_conn **out, const char *path);
void vc4c_close(vc4c_conn *c);
const char *vc4c_strerror(int rc);
int  vc4c_info(vc4c_conn *c, struct vc4d_hello_rsp *out);

int  vc4c_alloc(vc4c_conn *c, size_t bytes, unsigned flags, vc4c_buf *out);
int  vc4c_free(vc4c_conn *c, vc4c_buf *b);
int  vc4c_upload(vc4c_conn *c, const void *code, size_t bytes,
                 unsigned max_qpu, unsigned nuni, vc4c_prog *out);
int  vc4c_upload_file(vc4c_conn *c, const char *path, unsigned max_qpu,
                      unsigned nuni, vc4c_prog *out);

/* uni[] holds nqpu*nuni entries, QPU-major. */
int  vc4c_launch(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                 unsigned nuni, const vc4c_uni *uni, unsigned timeout_ms);

/* UNI_IN_ARENA launch: the client has already written nqpu*64 bytes of
 * uniform words at uniforms->map + uni_off (64-byte aligned) and only sends
 * the block's location.  The daemon validates every word as "in this arena"
 * or "a safe literal" and points the QPUs at it, so callers that materialise
 * uniforms through a mapped pointer need no per-word tagging.
 * Call vc4c_flush(c, uniforms, VC4D_SYNC_W2G) before launching. */
int  vc4c_launch_arena(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                       const vc4c_buf *uniforms, unsigned uni_off,
                       unsigned timeout_ms);

/* UNI_IN_ARENA + WAIT: kick, block until the queue is idle and return the
 * cycles/errs in one round trip.  The daemon flushes L2 write-back before the
 * reply, so results are readable immediately. */
int  vc4c_launch_wait(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                      const vc4c_buf *uniforms, unsigned uni_off,
                      unsigned timeout_ms, vc4c_result *out);

/* Like vc4c_sync(), but only asks the daemon to flush the GPU L2: the
 * dma-buf mapping is write-combining, so there is no CPU cache to maintain
 * and walking a 60 MiB mapping per launch would dominate the launch. */
int  vc4c_flush(vc4c_conn *c, const vc4c_buf *b, unsigned dir);
int  vc4c_wait(vc4c_conn *c, vc4c_result *out);

/* Flush/invalidate a buffer: dir 0 = CPU wrote it, 1 = GPU wrote it.
 * Does the client-side cache maintenance and asks the daemon to flush L2. */
int  vc4c_sync(vc4c_conn *c, const vc4c_buf *b, unsigned dir);

int  vc4c_read_regs(vc4c_conn *c, struct vc4d_regs *out);
int  vc4c_read_reg_words(vc4c_conn *c, unsigned off, unsigned count, uint32_t *out);
int  vc4c_write_reg(vc4c_conn *c, unsigned off, unsigned val, unsigned *old);
int  vc4c_get_stats(vc4c_conn *c, vc4c_stats *out);
int  vc4c_ping(vc4c_conn *c);

#endif /* VC4CLIENT_H */
