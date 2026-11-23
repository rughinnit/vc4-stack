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
/* 2: launch_req grew flags/uni_handle/uni_off (UNI_IN_ARENA mode).
 *
 * Version 3 adds the batch launch below.  HELLO accepts both 2 and 3: a
 * version-2 client gets the same daemon it always had (the batch op is simply
 * never sent, and the HELLO flag says so), and a version-3 client checks
 * VC4D_F_LAUNCH_BATCH before using it.  The single-launch message, its
 * validation and its replies are byte-for-byte unchanged.
 *
 * Version 2 is also where the G3/G4 additions live: they are opt-in, so a
 * client that sends neither a program name nor a uniform address mask sees
 * exactly the old behaviour. */
#define VC4D_VERSION    3u
#define VC4D_VERSION_MIN 2u           /* oldest HELLO version still served  */

#define VC4D_MAX_MSG    (64u * 1024u) /* hard cap on one message             */
#define VC4D_MAX_QPU    12u
#define VC4D_MAX_UNI    16u           /* uniform words per QPU (64 B / 4)    */
#define VC4D_MAX_PROG   16u * 1024u   /* max QPU program bytes (one slot)    */
/* One batch request carries at most this many entries.  The limit is what
 * keeps one request inside VC4D_MAX_MSG and gives the client a fixed shadow
 * slot for every launch's uniform block: 64 * 12 * 64 = 48 KiB. */
#define VC4D_BATCH_MAX  64u

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
    VC4D_E_NOTRUN = 13,  /* batch entry was not executed: an earlier one
                          * failed and execution stopped                   */
};

/* HELLO flags */
#define VC4D_F_RAW_SRQ      (1u << 0)  /* raw SRQ / VPMBASE writes allowed */
#define VC4D_F_ALLOWLIST    (1u << 1)  /* UPLOAD hashes are pinned         */
#define VC4D_F_NAMED_PROGS  (1u << 2)  /* UPLOAD may carry a program name  */
#define VC4D_F_UNI_MASK     (1u << 3)  /* LAUNCH may declare address slots */
#define VC4D_F_ARENA_RESTORE (1u << 4) /* a degraded arena is restored when
                                        * CMA recovers and the daemon is idle */
#define VC4D_F_LAUNCH_BATCH  (1u << 5) /* VC4D_OP_LAUNCH_BATCH is supported */

/* UPLOAD flags.  NAMED: the payload is
 *     struct vc4d_upload_req, code bytes, name bytes (no NUL)
 * where the name is 1..VC4D_NAME_MAX-1 printable bytes.  The name is the key
 * of the program-slot LRU: re-uploading the same name reuses the slot, and an
 * eviction frees the least recently used one.  Without the flag the daemon
 * falls back to the code sha256 as the key. */
#define VC4D_UPLOAD_NAMED   (1u << 0)
#define VC4D_NAME_MAX       64u

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
    VC4D_OP_LAUNCH_BATCH = 13,
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
 *
 * flags & VC4D_LAUNCH_UNI_MASK (UNI_IN_ARENA only): launch_req.uni_addr is
 *             authoritative instead of the daemon's heuristic.  Bit u means
 *             "uniform word u is a GPU bus address the program consumes", and
 *             the daemon requires it to lie inside this client's arena; a word
 *             whose bit is clear is a *value* and is not compared with the bus
 *             window at all.  This is what makes a float whose bit pattern is
 *             >= 0xC0000000 (-2.0f and everything below it) legal: the old
 *             per-word guess could not tell such a value from an address.  It
 *             is a per-slot mask rather than a per-word kind because every QPU
 *             block has the same layout, so 16 bits cover all 12 QPUs.
 *             Bits >= nuni must be zero.
 */
#define VC4D_LAUNCH_UNI_ARENA  (1u << 0)
#define VC4D_LAUNCH_WAIT       (1u << 1)
#define VC4D_LAUNCH_UNI_MASK   (1u << 2)
#define VC4D_LAUNCH_FLAGS_ALL  (VC4D_LAUNCH_UNI_ARENA | VC4D_LAUNCH_WAIT | \
                                VC4D_LAUNCH_UNI_MASK)

struct vc4d_launch_req {
    uint32_t prog;
    uint32_t timeout_ms;
    uint32_t nqpu;
    uint32_t nuni;
    uint32_t flags;        /* VC4D_LAUNCH_* */
    uint32_t uni_handle;   /* UNI_ARENA: client region holding the uniforms */
    uint32_t uni_off;      /* UNI_ARENA: byte offset, 64-byte aligned */
    uint32_t uni_addr;     /* UNI_MASK: bit u = uniform word u is an address */
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

/*
 * LAUNCH_BATCH: one request, an ordered list of launches, one completion.
 *
 * The goal is the per-launch host cost of a graph: an inference that issues
 * hundreds of launches otherwise pays a sendmsg/recvmsg pair and a daemon
 * request dispatch for every one.  A batch is executed strictly in order and
 * each launch waits for its own completion before the next is kicked, so the
 * GPU sees exactly the sequence it would have seen from separate LAUNCH
 * messages, including kernel-to-kernel memory ordering (the daemon's L2
 * write-back after each kick).
 *
 * One entry is a launch or a fence:
 *
 *   VC4D_BENT_LAUNCH  prog/timeout_ms/nqpu/flags/uni_handle/uni_off/uni_addr
 *                     mean exactly what they mean in struct vc4d_launch_req.
 *                     VC4D_LAUNCH_UNI_ARENA is required (a batch carries no
 *                     per-word uniform array; uniforms live in the arena),
 *                     VC4D_LAUNCH_WAIT is implied (every entry completes
 *                     before the next starts), and VC4D_LAUNCH_UNI_MASK is
 *                     optional.  handle/dir must be 0.
 *
 *   VC4D_BENT_FENCE   handle/dir are a struct vc4d_sync_req: the daemon does
 *                     the same L2 write-back/invalidate a SYNC request does,
 *                     at this point in the order.  This is a *device* fence.
 *                     A client-side fence (the CPU is about to read a result
 *                     or rewrite a buffer an earlier entry reads) is a batch
 *                     boundary by construction: no entry may depend on CPU
 *                     work between two entries.
 *
 * Validation is per entry and identical to LAUNCH's.  Execution stops at the
 * first entry that fails; every entry after it is reported VC4D_E_NOTRUN, and
 * entries before it have already run and completed.  The reply header status
 * is the first failure (VC4D_OK when all ran), detail0 is the number of
 * entries that ran (a fence counts as run), and detail1 is the failing index
 * (0xffffffff when none).  The payload is `count` entries of
 * struct vc4d_batch_item, in request order.
 */
#define VC4D_BENT_LAUNCH 0u
#define VC4D_BENT_FENCE  1u

struct vc4d_batch_ent {
    uint32_t kind;         /* VC4D_BENT_*                                  */
    uint32_t prog;         /* LAUNCH: program handle                       */
    uint32_t timeout_ms;   /* LAUNCH: 0 = daemon default                   */
    uint32_t nqpu;         /* LAUNCH                                       */
    uint32_t nuni;         /* LAUNCH: must equal the UPLOAD declaration    */
    uint32_t flags;        /* LAUNCH: VC4D_LAUNCH_* (ARENA required)       */
    uint32_t uni_handle;   /* LAUNCH: client region holding the uniforms   */
    uint32_t uni_off;      /* LAUNCH: byte offset, 64-byte aligned         */
    uint32_t uni_addr;     /* LAUNCH: UNI_MASK slot mask                   */
    uint32_t handle;       /* FENCE: client region                         */
    uint32_t dir;          /* FENCE: VC4D_SYNC_W2G / G2R                   */
};
struct vc4d_batch_req {
    uint32_t count;        /* 1..VC4D_BATCH_MAX entries follow             */
    uint32_t flags;        /* reserved, must be 0                          */
    uint32_t reserved[2];
};
struct vc4d_batch_item {
    uint32_t status;       /* VC4D_* for this entry                        */
    uint32_t launch;       /* LAUNCH: launch id; FENCE: 0                  */
    uint32_t errs;         /* LAUNCH: newly raised V3D_ERR_STAT bits       */
    uint32_t pad;
    uint64_t wall_ns;      /* LAUNCH: kick -> idle; FENCE: 0               */
    uint32_t cycles;       /* LAUNCH: PCTR0 exec cycles                    */
    uint32_t pad2;
};

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
    uint64_t evictions;     /* program slots freed by the LRU             */
    uint32_t arena_want;    /* size requested at startup                  */
    uint32_t arena_degraded;/* 1: CMA refused arena_want, using less      */
};

_Static_assert(sizeof(struct vc4d_hdr) == 28, "hdr");
_Static_assert(sizeof(struct vc4d_uni) == 16, "uni");
_Static_assert(sizeof(struct vc4d_launch_req) == 32, "launch_req");
_Static_assert(sizeof(struct vc4d_batch_ent) == 44, "batch_ent");
_Static_assert(sizeof(struct vc4d_batch_req) == 16, "batch_req");
_Static_assert(sizeof(struct vc4d_batch_item) == 32, "batch_item");
_Static_assert(sizeof(struct vc4d_stats_rsp) == 96, "stats_rsp");
_Static_assert(sizeof(struct vc4d_regs) == 72, "regs");

#endif /* VC4D_PROTO_H */
