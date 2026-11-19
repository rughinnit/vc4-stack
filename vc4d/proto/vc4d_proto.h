/*
 * vc4d_proto.h - wire protocol shared by vc4d and libvc4client.
 *
 * Transport: AF_UNIX SOCK_SEQPACKET (message boundaries preserved, SCM_RIGHTS
 * for the dma-buf fd).  Every message starts with struct vc4d_hdr; hdr.len is
 * the total message length in bytes and MUST equal the datagram size.  All
 * integers are host-endian (client and daemon are the same machine).
 *
 * Requests carry status == 0 and the op's payload after the header.  Replies
 * echo magic/op/version/seq, set status to a VC4D_* code and put op-specific
 * data in detail0/detail1 and/or a payload.
 */
#ifndef VC4D_PROTO_H
#define VC4D_PROTO_H

#include <stdint.h>

#define VC4D_MAGIC      0x44344356u   /* "VC4D" */
/* 2: launch_req grew flags/uni_handle/uni_off (UNI_IN_ARENA mode). */
#define VC4D_VERSION    2u

#define VC4D_MAX_MSG    (64u * 1024u) /* hard cap on one message             */
#define VC4D_MAX_QPU    12u
#define VC4D_MAX_UNI    16u           /* uniform words per QPU (64 B / 4)    */
#define VC4D_MAX_PROG   16u * 1024u   /* max QPU program bytes (one slot)    */

/* ---- status codes ------------------------------------------------------- */
enum {
    VC4D_OK       = 0,
    VC4D_E_PROTO  = 1,   /* malformed: bad magic/len/reserved/payload size */
    VC4D_E_VERSION= 2,   /* unsupported protocol version                   */
    VC4D_E_TOOLARGE=3,   /* message or object exceeds a limit              */
    VC4D_E_INVAL  = 4,   /* bad argument / out of range                    */
    VC4D_E_NOMEM  = 5,   /* arena or slot table exhausted                  */
    VC4D_E_NOENT  = 6,   /* unknown handle / no launch in flight           */
    VC4D_E_BUSY   = 7,   /* another launch is in flight                    */
    VC4D_E_TIMEOUT= 8,   /* GPU did not finish; daemon is dead             */
    VC4D_E_GPU    = 9,   /* V3D error status                               */
    VC4D_E_DENIED = 10,  /* policy refusal (allowlist, raw reg, ...)       */
    VC4D_E_IO     = 11,  /* socket / fd transfer failure                   */
    VC4D_E_DEAD   = 12,  /* daemon marked dead after a GPU timeout         */
};

/* HELLO flags */
#define VC4D_F_RAW_SRQ      (1u << 0)  /* raw SRQ / VPMBASE writes allowed */
#define VC4D_F_ALLOWLIST    (1u << 1)  /* UPLOAD hashes are pinned         */

/* ALLOC flags */
#define VC4D_ALLOC_ZERO     (1u << 0)  /* daemon zeroes the region         */

/* SYNC direction */
#define VC4D_SYNC_W2G       0u         /* CPU wrote: clean L2 for the GPU  */
#define VC4D_SYNC_G2R       1u         /* GPU wrote: flush+invalidate L2   */

/* uniform entry kinds */
#define VC4D_UNI_LIT        0u         /* off = literal 32-bit word        */
#define VC4D_UNI_PTR        1u         /* gpu_bus + buf + off, len checked */

enum {
    VC4D_OP_HELLO     = 1,
    VC4D_OP_ALLOC     = 2,
    VC4D_OP_FREE      = 3,
    VC4D_OP_UPLOAD    = 4,
    VC4D_OP_LAUNCH    = 5,
    VC4D_OP_WAIT      = 6,
    VC4D_OP_READ_REGS = 7,
    VC4D_OP_WRITEREG  = 8,
    VC4D_OP_STATS     = 9,
    VC4D_OP_SYNC      = 10,
    VC4D_OP_CLOSE     = 11,
    VC4D_OP_PING      = 12,
};

struct vc4d_hdr {
    uint32_t magic;
    uint32_t len;      /* total bytes (header + payload) */
    uint16_t op;
    uint16_t version;
    uint32_t seq;      /* echoed in the reply */
    int32_t  status;   /* request: 0 */
    uint32_t detail0;  /* reply: op-specific */
    uint32_t detail1;
} __attribute__((packed));

struct vc4d_hello_req  { uint32_t version; uint32_t flags; };
struct vc4d_hello_rsp {
    uint32_t version, arena_bytes, data_base, page_size;
    uint32_t max_msg, max_qpu, max_uni, max_prog;
    uint32_t flags;
    uint32_t ident0, ident1;      /* read-only V3D identification */
    uint32_t reserved[4];
};

struct vc4d_alloc_req  { uint32_t bytes; uint32_t flags; };
/* pad carries the GPU bus address of byte 0 of the region (arena_bus + off),
 * so a client can build pointer uniforms without knowing the arena base. */
struct vc4d_alloc_rsp  { uint32_t handle; uint32_t off; uint32_t len; uint32_t bus; };

struct vc4d_free_req   { uint32_t handle; uint32_t pad; };

/* nuni is the number of uniform words the program consumes; LAUNCH must
 * supply exactly that many, so a program can never run on a short uniform
 * block. */
struct vc4d_upload_req { uint32_t bytes; uint32_t max_qpu; uint32_t nuni; uint32_t flags; };
struct vc4d_upload_rsp { uint32_t prog; uint32_t code_off; uint32_t words; uint32_t pad; };

/* One uniform slot.  LIT: off is the value, handle/len must be 0.
 * PTR: the daemon resolves gpu_bus + buf.off + off after checking that
 * [off, off+len) lies inside the client's buffer. */
struct vc4d_uni { uint32_t kind; uint32_t handle; uint32_t off; uint32_t len; };

/*
 * LAUNCH modes.
 *
 * flags == 0: the request is followed by nqpu*nuni struct vc4d_uni entries
 *             (QPU-major); the daemon resolves them into the program's own
 *             uniform slots.
 *
 * flags & VC4D_LAUNCH_UNI_ARENA: the uniform words are already materialised
 *             by the client in its own mapped region.  uni_handle names the
 *             region and uni_off is the byte offset of the first 64-byte
 *             QPU block; QPU qpu uses [uni_off + 64*qpu, +64).  The request
 *             ends after this struct (no vc4d_uni entries).  The daemon
 *             validates every word as "inside this arena" or "a literal
 *             outside the bus/arena/MMIO windows" and launches with the
 *             arena addresses directly.  This is what lets an engine that
 *             writes uniforms through a mapped pointer keep its op code.
 *
 * flags & VC4D_LAUNCH_WAIT: kick and block until the queue is idle, all in
 *             one round trip.  The reply is struct vc4d_wait_rsp instead of
 *             struct vc4d_launch_rsp, and the L2 flush that makes GPU results
 *             visible to the CPU has already happened.  This is the mode the
 *             engine uses: it removes the LAUNCH/WAIT pair and the explicit
 *             SYNC round trips (the client mapping is write-combining, so
 *             there is nothing to clean on the CPU side).
 */
#define VC4D_LAUNCH_UNI_ARENA  (1u << 0)
#define VC4D_LAUNCH_WAIT       (1u << 1)
#define VC4D_LAUNCH_FLAGS_ALL  (VC4D_LAUNCH_UNI_ARENA | VC4D_LAUNCH_WAIT)

struct vc4d_launch_req {
    uint32_t prog;
    uint32_t timeout_ms;
    uint32_t nqpu;
    uint32_t nuni;
    uint32_t flags;        /* VC4D_LAUNCH_* */
    uint32_t uni_handle;   /* UNI_ARENA: client region holding the uniforms */
    uint32_t uni_off;      /* UNI_ARENA: byte offset, 64-byte aligned */
    uint32_t pad;
};
struct vc4d_launch_rsp { uint32_t launch; uint32_t nqpu; uint32_t pad0; uint32_t pad1; };

struct vc4d_wait_rsp {
    uint32_t launch;
    uint32_t errs;        /* newly raised V3D_ERR_STAT bits */
    uint64_t wall_ns;     /* kick -> queue idle */
    uint32_t cycles;      /* PCTR0 exec cycles */
    uint32_t pad;
};

struct vc4d_sync_req   { uint32_t handle; uint32_t dir; };
struct vc4d_wreg_req   { uint32_t off; uint32_t val; };

/* READ_REGS: count == 0 -> snapshot (payload = struct vc4d_regs),
 * count >  0 -> raw words [off, off+count), payload = vc4d_regs_req + words */
struct vc4d_regs_req   { uint32_t off; uint32_t count; };
struct vc4d_regs {
    uint32_t ident0, ident1, ident2, ident3;
    uint32_t l2cactl, slcactl, vpmbase, vpacntl;
    uint32_t intctl, intena, intdis, errors;
    uint32_t srqcs, srqpc, srqua, srqul;
    uint32_t cycles, pad0;
};

struct vc4d_stats_rsp {
    uint64_t launches, launches_failed, bytes_uploaded, reg_writes;
    uint64_t rejects, arena_bytes, arena_used, gpu_ns;
    uint32_t clients, max_clients, programs, dead;
};

_Static_assert(sizeof(struct vc4d_hdr) == 28, "hdr");
_Static_assert(sizeof(struct vc4d_uni) == 16, "uni");
_Static_assert(sizeof(struct vc4d_launch_req) == 32, "launch_req");
_Static_assert(sizeof(struct vc4d_regs) == 72, "regs");

#endif /* VC4D_PROTO_H */
