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
/* Named upload (G3): `name` is the program-slot LRU key, 1..63 printable
 * bytes.  A daemon that predates it (no VC4D_F_NAMED_PROGS) gets a plain
 * upload; the caller can check vc4c_info()->flags first if it cares whether
 * eviction will key by name. */
int  vc4c_upload_named(vc4c_conn *c, const char *name, const void *code,
                       size_t bytes, unsigned max_qpu, unsigned nuni,
                       vc4c_prog *out);
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

/* Same two, with the per-slot address mask of G4: bit u says uniform word u
 * holds a GPU bus address.  Words without the bit are values, so a float
 * <= -2.0f (bit pattern >= 0xC0000000) is legal.  The address words are still
 * range-checked against this client's arena.  Passed as a separate function so
 * a caller that does not use it never sends the flag, and the plain forms keep
 * the daemon's old heuristic. */
int  vc4c_launch_arena_u(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                         const vc4c_buf *uniforms, unsigned uni_off,
                         unsigned timeout_ms, uint32_t uni_addr);
int  vc4c_launch_wait_u(vc4c_conn *c, const vc4c_prog *p, unsigned nqpu,
                        const vc4c_buf *uniforms, unsigned uni_off,
                        unsigned timeout_ms, vc4c_result *out,
                        uint32_t uni_addr);

/* Ordered batch (protocol 3).  Every entry is executed to completion before
 * the next is kicked, exactly as one LAUNCH_WAIT per entry would be, but one
 * request and one completion cover the list.  Uniforms live in `uniforms`
 * (the client region), at uni_off, nqpu blocks of 64 B; a batch entry cannot
 * carry the per-word uniform array.  `out`, when non-NULL, receives one
 * struct vc4d_batch_item per entry in request order; `done` receives how many
 * entries ran.  Returns 0 when every entry ran, otherwise the first failure
 * (the entries after it are VC4D_E_NOTRUN).  Check vc4c_info()->flags for
 * VC4D_F_LAUNCH_BATCH before calling this on a daemon that may be older. */
typedef struct {
    uint32_t kind;        /* VC4D_BENT_LAUNCH or VC4D_BENT_FENCE          */
    uint32_t prog;        /* LAUNCH: program handle from vc4c_upload*     */
    uint32_t timeout_ms;  /* LAUNCH: 0 = daemon default                   */
    uint32_t nqpu;        /* LAUNCH                                       */
    uint32_t nuni;        /* LAUNCH: the program's declared uniform count  */
    uint32_t flags;       /* LAUNCH: VC4D_LAUNCH_UNI_MASK when exact      */
    uint32_t uni_off;     /* LAUNCH: 64-byte aligned offset in uniforms   */
    uint32_t uni_addr;    /* LAUNCH: bit u = word u is a bus address      */
    uint32_t handle;      /* FENCE: client region handle                  */
    uint32_t dir;         /* FENCE: VC4D_SYNC_W2G / VC4D_SYNC_G2R         */
} vc4c_batch_ent;

int  vc4c_launch_batch(vc4c_conn *c, const vc4c_buf *uniforms,
                       const vc4c_batch_ent *ent, unsigned count,
                       struct vc4d_batch_item *out, unsigned *done);

/*
 * SUBMIT (protocol 4): the same request as vc4c_launch_batch, sent without
 * waiting for its completion.  Up to VC4D_SUBMIT_MAX may be outstanding on one
 * connection, and they must be polled in the order they were submitted: the
 * daemon serves requests in arrival order, so the oldest token owns the next
 * reply.
 *
 *   vc4c_token t;
 *   vc4c_submit_batch(c, &region, ent, n, &t);   // queued, not run yet
 *   ... prepare the next request, in different buffers ...
 *   vc4c_poll_batch(c, t, items, n, &done);      // completion + per-entry status
 *
 * Only the daemon's VC4D_F_ASYNC_SUBMIT says it will serve more than one
 * outstanding request; a client that submits to a daemon without it is
 * talking to something that will read the second request only after the first
 * reply, which is correct but pointless.  Mixing an outstanding SUBMIT with a
 * synchronous call (launch_batch, flush, upload, ...) is refused: the reply
 * arrows no longer line up.  `count` in the poll is the size of `out` in
 * entries and must match the submitted count.
 */
typedef struct { uint32_t seq; } vc4c_token;

int  vc4c_submit_batch(vc4c_conn *c, const vc4c_buf *uniforms,
                       const vc4c_batch_ent *ent, unsigned count,
                       vc4c_token *tok);
int  vc4c_poll_batch(vc4c_conn *c, vc4c_token tok,
                     struct vc4d_batch_item *out, unsigned count,
                     unsigned *done);
/* Number of submitted-but-unpolled batches; 0..VC4D_SUBMIT_MAX. */
int  vc4c_submits_outstanding(const vc4c_conn *c);

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
