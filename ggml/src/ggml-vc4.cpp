// ggml-vc4.cpp - a ggml backend that runs graphs on the VideoCore IV QPUs.
//
// Design (see docs/GGML.md):
//   * one ggml_backend_buffer is one region of the engine's dma-buf arena
//     (allocated from a free-list the backend keeps over a reserved pool),
//     so a tensor's data pointer is directly the QPU-visible address and no
//     per-op upload happens;
//   * set_tensor/get_tensor do the cache maintenance (dc civac) the cacheable
//     mapping needs, exactly like the engine's own vc4_upload/vc4_download;
//   * graph_compute walks the graph, lowers supported nodes to engine calls
//     and returns GGML_STATUS_FAILED for anything else.  supports_op is the
//     honest list, so ggml_backend_sched partitions the rest to the CPU.
//
// Every reduction is a GEMM against a constant matrix and every broadcast is
// a copy2d_x lane run: vc4_reduce, vc4_gather, vc4_cumsum and vc4_select3 are
// refused/wrong on this board (infer/python/vc4infer/_dispatch_norm.py).

#include "ggml-vc4.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include "vc4infer.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <unistd.h>

#define VC4_ALIGN 64u
#define VC4_DEFAULT_POOL_MB 48u
#define VC4_DEFAULT_SCRATCH_MB 8u

// --------------------------------------------------------------------------
// small helpers
// --------------------------------------------------------------------------
static bool vc4_trace_enabled() {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("VC4GGML_TRACE");
        on = (e && *e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return on != 0;
}

static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

static int env_int(const char * name, int dflt) {
    const char * e = getenv(name);
    if (!e || !*e) return dflt;
    return atoi(e);
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

// --------------------------------------------------------------------------
// arena pool: a first-fit free list over a reserved slice of the arena
// --------------------------------------------------------------------------
struct vc4_pool_blk {
    uint32_t off;   // relative to the pool base
    size_t   size;
    bool     used;
};

struct vc4_backend_ctx {
    vc4_ctx * ctx = nullptr;
    size_t    arena_bytes = 0;
    uint8_t * arena = nullptr;     // host mapping of the arena base
    vc4_buf   pool = {};           // the reserved slice
    size_t    pool_usable = 0;     // pool.size - scratch
    std::vector<vc4_pool_blk> blocks;
    size_t    free_bytes = 0;
    uint32_t  scratch_off = 0;     // absolute arena offset
    size_t    scratch_size = 0;
    size_t    scratch_top = 0;     // relative to scratch_off

    std::mutex mu;
    // statistics for the report
    uint64_t nodes_on_vc4 = 0;
    uint64_t nodes_failed = 0;
    double   compute_ms = 0.0;
};

struct vc4_buffer_ctx {
    vc4_backend_ctx * be = nullptr;
    uint32_t off = 0;
    size_t   size = 0;
};

// One arena per process (the engine maps a dma-buf once and vc4d hands out one
// arena per client), so one VC4 backend instance.  The buffer type's
// alloc_buffer has no backend argument, hence this registry.
static std::atomic<vc4_backend_ctx *> g_be{nullptr};
static std::atomic<size_t> g_arena_bytes{0};
static std::atomic<size_t> g_pool_bytes{0};

// --------------------------------------------------------------------------
// buffer type / device / reg forward declarations
// --------------------------------------------------------------------------
static ggml_backend_buffer_type_t vc4_buft(void);
static ggml_backend_dev_t         vc4_device(void);
static ggml_backend_reg_t         vc4_reg(void);

// --------------------------------------------------------------------------
// pool allocator
// --------------------------------------------------------------------------
static bool pool_alloc(vc4_backend_ctx * be, size_t bytes, uint32_t * out_off) {
    bytes = align_up(bytes, VC4_ALIGN);
    if (bytes == 0 || bytes > be->free_bytes) return false;

    for (size_t i = 0; i < be->blocks.size(); ++i) {
        if (be->blocks[i].used || be->blocks[i].size < bytes) continue;

        // copy the fields out before insert(): it can reallocate the vector
        const uint32_t boff  = be->blocks[i].off;
        const size_t   bsize = be->blocks[i].size;

        be->blocks[i].size = bytes;
        be->blocks[i].used = true;
        if (bsize > bytes + VC4_ALIGN) {
            vc4_pool_blk tail;
            tail.off  = boff + (uint32_t)bytes;
            tail.size = bsize - bytes;
            tail.used = false;
            be->blocks.insert(be->blocks.begin() + (long)i + 1, tail);
        }
        be->free_bytes -= bytes;
        *out_off = boff;
        return true;
    }
    return false;
}

static void pool_free(vc4_backend_ctx * be, uint32_t off) {
    for (size_t i = 0; i < be->blocks.size(); ++i) {
        if (be->blocks[i].off != off || !be->blocks[i].used) continue;
        be->blocks[i].used = false;
        be->free_bytes += be->blocks[i].size;
        // coalesce with the following and preceding free blocks
        if (i + 1 < be->blocks.size() && !be->blocks[i + 1].used) {
            be->blocks[i].size += be->blocks[i + 1].size;
            be->blocks.erase(be->blocks.begin() + (long)i + 1);
        }
        if (i > 0 && !be->blocks[i - 1].used) {
            be->blocks[i - 1].size += be->blocks[i].size;
            be->blocks.erase(be->blocks.begin() + (long)i);
        }
        return;
    }
}

static bool tmp_alloc(vc4_backend_ctx * be, size_t bytes, vc4_buf * out) {
    bytes = align_up(bytes, VC4_ALIGN);
    if (be->scratch_top + bytes > be->scratch_size) return false;
    out->off   = be->scratch_off + (uint32_t)be->scratch_top;
    out->bytes = bytes;
    be->scratch_top += bytes;
    return true;
}

static void tmp_reset(vc4_backend_ctx * be) { be->scratch_top = 0; }

// --------------------------------------------------------------------------
// arena <-> vc4_buf helpers
// --------------------------------------------------------------------------
static inline vc4_buf tensor_buf(vc4_backend_ctx * be, const ggml_tensor * t,
                                 size_t off = 0, size_t bytes = 0) {
    vc4_buf b;
    b.off   = (uint32_t)((uintptr_t)t->data + off - (uintptr_t)be->arena);
    b.bytes = bytes ? bytes : (size_t)ggml_nbytes(t) - off;
    return b;
}

// make a vc4_buf from an arbitrary arena pointer (used for offset views)
static inline vc4_buf ptr_buf(vc4_backend_ctx * be, const void * p,
                              size_t bytes) {
    vc4_buf b;
    b.off   = (uint32_t)((uintptr_t)p - (uintptr_t)be->arena);
    b.bytes = bytes;
    return b;
}

// dc civac a range: clean what the CPU wrote, drop stale lines before reading.
static void civac(vc4_backend_ctx * be, void * p, size_t bytes) {
    if (!bytes) return;
    vc4_buf b = ptr_buf(be, p, bytes);
    vc4_invalidate_buf(be->ctx, &b);
}

// --------------------------------------------------------------------------
// buffer type
// --------------------------------------------------------------------------
static const char * vc4_buft_name(ggml_backend_buffer_type_t) {
    return "VC4";
}

static ggml_backend_buffer_t vc4_buft_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                   size_t size) {
    vc4_backend_ctx * be = g_be.load();
    if (!be) {
        fprintf(stderr, "vc4: alloc_buffer before a backend exists\n");
        return nullptr;
    }
    size = align_up(size, VC4_ALIGN);

    std::lock_guard<std::mutex> lock(be->mu);
    uint32_t off = 0;
    if (!pool_alloc(be, size, &off)) {
        fprintf(stderr, "vc4: alloc_buffer(%zu B) failed: %zu B free of %zu B pool\n",
                size, be->free_bytes, be->pool_usable);
        return nullptr;
    }

    vc4_buffer_ctx * bc = new (std::nothrow) vc4_buffer_ctx;
    if (!bc) {
        pool_free(be, off);
        return nullptr;
    }
    bc->be   = be;
    bc->off  = off;
    bc->size = size;

    static const struct ggml_backend_buffer_i iface = {
        /* .free_buffer  = */ [](ggml_backend_buffer_t buffer) {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            std::lock_guard<std::mutex> lk(c->be->mu);
            pool_free(c->be, c->off);
            delete c;
        },
        /* .get_base     = */ [](ggml_backend_buffer_t buffer) -> void * {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            return c->be->arena + c->be->pool.off + c->off;
        },
        /* .init_tensor  = */ nullptr,
        /* .memset_tensor= */ [](ggml_backend_buffer_t buffer, ggml_tensor * t,
                                uint8_t value, size_t offset, size_t size) {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            uint8_t * p = (uint8_t *)t->data + offset;
            civac(c->be, p, size);
            memset(p, value, size);
            civac(c->be, p, size);
        },
        /* .set_tensor   = */ [](ggml_backend_buffer_t buffer, ggml_tensor * t,
                                const void * data, size_t offset, size_t size) {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            uint8_t * p = (uint8_t *)t->data + offset;
            civac(c->be, p, size);       // drop stale lines before overwriting
            memcpy(p, data, size);
            civac(c->be, p, size);       // clean the new bytes to DRAM
        },
        /* .get_tensor   = */ [](ggml_backend_buffer_t buffer,
                                const ggml_tensor * t, void * data,
                                size_t offset, size_t size) {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            uint8_t * p = (uint8_t *)t->data + offset;
            civac(c->be, p, size);       // QPU-written bytes reach DRAM
            memcpy(data, p, size);
        },
        /* .set_tensor_2d= */ nullptr,
        /* .get_tensor_2d= */ nullptr,
        /* .cpy_tensor   = */ nullptr,
        /* .clear        = */ [](ggml_backend_buffer_t buffer, uint8_t value) {
            vc4_buffer_ctx * c = (vc4_buffer_ctx *)buffer->context;
            uint8_t * p = c->be->arena + c->be->pool.off + c->off;
            civac(c->be, p, c->size);
            memset(p, value, c->size);
            civac(c->be, p, c->size);
        },
        /* .reset        = */ nullptr,
        /* .set_tensor_n = */ nullptr,
        /* .get_tensor_n = */ nullptr,
        /* .cpy_tensor_n = */ nullptr,
    };

    return ggml_backend_buffer_init(buft, iface, bc, size);
}

static size_t vc4_buft_get_alignment(ggml_backend_buffer_type_t) {
    return VC4_ALIGN;
}

static size_t vc4_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return g_pool_bytes.load();
}

static bool vc4_buft_is_host(ggml_backend_buffer_type_t) {
    // data is host-accessible, but the mapping is cacheable and coherency is
    // the backend's job.  Reporting "not host" routes every access through
    // set_tensor/get_tensor and keeps the scheduler's device copies explicit.
    return false;
}

static ggml_backend_buffer_type_t vc4_buft(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name        = */ vc4_buft_name,
            /* .alloc_buffer    = */ vc4_buft_alloc_buffer,
            /* .alloc_buffer_n  = */ nullptr,
            /* .get_alignment   = */ vc4_buft_get_alignment,
            /* .get_max_size    = */ vc4_buft_get_max_size,
            /* .get_alloc_size  = */ nullptr,
            /* .get_alloc_size_n= */ nullptr,
            /* .is_host         = */ vc4_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &buft;
}

// --------------------------------------------------------------------------
// lowering helpers
// --------------------------------------------------------------------------
static bool tensor_is_f32(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32;
}
static bool tensor_cont(const ggml_tensor * t) {
    return t && ggml_is_contiguous(t);
}

// rows x n words: destination rows are contiguous, the source is read with a
// lane stride (slane == 0 means "broadcast the source word").  This is the
// same two-call copy2d_x composition the ONNX runner uses; the tail is a
// depth<16 store, so no word past the row is written.
static vc4_status lane_runs(vc4_backend_ctx * be, vc4_buf src, vc4_buf dst,
                            uint32_t rows, uint32_t n, uint32_t srow,
                            uint32_t drow, uint32_t slane) {
    if (!rows || !n) return VC4_OK;

    if (slane == 0) {
        return vc4_copy2d_x(be->ctx, &src, &dst, rows, n, srow, drow,
                            0, 4, 4, 1, 1, 64, 64, 1, 1);
    }

    uint32_t nv = n / 16u, rem = n % 16u;
    vc4_status st = VC4_OK;
    if (nv) {
        uint32_t units = (nv % 4u == 0) ? 4u : (nv % 2u == 0 ? 2u : 1u);
        st = vc4_copy2d_x(be->ctx, &src, &dst, rows, nv, srow, drow,
                          16u * slane, 64, slane, 16, 1, 64, 64, units, 16);
    }
    if (st == VC4_OK && rem) {
        vc4_buf s2 = src, d2 = dst;
        s2.off  += nv * 16u * slane;
        d2.off  += nv * 16u * 4u;
        s2.bytes -= nv * 16u * slane;
        d2.bytes -= nv * 16u * 4u;
        st = vc4_copy2d_x(be->ctx, &s2, &d2, rows, 1, srow, drow,
                          64, 64, slane, rem, 1, 64, 64, 1, rem);
    }
    return st;
}

// byte copy of nelements floats using the QPU (no host memcpy of activations)
static vc4_status copy_words(vc4_backend_ctx * be, vc4_buf src, vc4_buf dst,
                             size_t words) {
    if (words > 0xFFFFFFFFull) return VC4_ERR_UNSUPPORTED;
    return lane_runs(be, src, dst, 1, (uint32_t)words, 0, 0, 4);
}

// repeat src into dst (dst shape = dst_ne, src contiguous, src->ne[0] ==
// dst_ne[0]).  Byte copies only; used for ggml's broadcast rule.
static bool expand_repeat(vc4_backend_ctx * be, const ggml_tensor * src,
                          vc4_buf dst, const int64_t dst_ne[4]) {
    if (!tensor_cont(src)) return false;
    if (src->ne[0] != dst_ne[0]) return false;
    for (int k = 1; k < 4; ++k) {
        if (src->ne[k] != 1 && src->ne[k] != dst_ne[k]) return false;
        if (dst_ne[k] % src->ne[k] != 0) return false;
    }

    civac(be, src->data, (size_t)ggml_nbytes(src));   // src may be QPU-written

    const size_t rowb = (size_t)src->ne[0] * 4u;
    const uint8_t * s = (const uint8_t *)src->data;
    uint8_t * d = be->arena + dst.off;

    for (int64_t i3 = 0; i3 < dst_ne[3]; ++i3) {
        const int64_t s3 = i3 % src->ne[3];
        for (int64_t i2 = 0; i2 < dst_ne[2]; ++i2) {
            const int64_t s2 = i2 % src->ne[2];
            for (int64_t i1 = 0; i1 < dst_ne[1]; ++i1) {
                const int64_t s1 = i1 % src->ne[1];
                const uint8_t * sp = s + (size_t)(((s3 * src->ne[2] + s2) *
                                                  src->ne[1] + s1) * rowb);
                uint8_t * dp = d + (size_t)(((i3 * dst_ne[2] + i2) *
                                             dst_ne[1] + i1) * rowb);
                memcpy(dp, sp, rowb);
            }
        }
    }
    civac(be, d, (size_t)(dst_ne[0] * dst_ne[1] * dst_ne[2] * dst_ne[3]) * 4u);
    return true;
}

// --------------------------------------------------------------------------
// op lowering
// --------------------------------------------------------------------------
static enum ggml_status fail(const char * what, const ggml_tensor * node,
                             const char * why) {
    fprintf(stderr, "vc4: %s (%s) not executable here: %s\n",
            what, node && node->name[0] ? node->name : "?", why);
    return GGML_STATUS_FAILED;
}

static enum ggml_status run_bin(vc4_backend_ctx * be, ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (!a || !b) return fail("binary", node, "missing operand");

    const size_t n = (size_t)ggml_nelements(node);
    vc4_buf ab = tensor_buf(be, a);
    vc4_buf bb;

    if (ggml_are_same_shape(a, b)) {
        bb = tensor_buf(be, b);
    } else {
        if (!ggml_can_repeat(b, a) || b->ne[0] != a->ne[0])
            return fail("binary", node, "unsupported broadcast shape");
        if (!tmp_alloc(be, n * 4, &bb))
            return GGML_STATUS_ALLOC_FAILED;
        int64_t ne[4] = { a->ne[0], a->ne[1], a->ne[2], a->ne[3] };
        if (!expand_repeat(be, b, bb, ne))
            return fail("binary", node, "broadcast copy failed");
    }

    vc4_buf db = tensor_buf(be, node);
    vc4_status st = VC4_ERR_UNSUPPORTED;

    switch (node->op) {
        case GGML_OP_ADD:
            st = vc4_elementwise(be->ctx, VC4_OP_ADD, &ab, &bb, &db, n);
            break;
        case GGML_OP_MUL:
            st = vc4_elementwise(be->ctx, VC4_OP_MUL, &ab, &bb, &db, n);
            break;
        case GGML_OP_SUB: {
            // no public sub kernel: b' = -b, then add
            vc4_buf nb;
            if (!tmp_alloc(be, n * 4, &nb))
                return GGML_STATUS_ALLOC_FAILED;
            st = vc4_unary(be->ctx, VC4_UOP_NEG, &bb, &nb, n, 0, 0, 0);
            if (st == VC4_OK)
                st = vc4_elementwise(be->ctx, VC4_OP_ADD, &ab, &nb, &db, n);
        } break;
        case GGML_OP_DIV: {
            // no public div kernel: b' = 1/b, then multiply
            vc4_buf rb;
            if (!tmp_alloc(be, n * 4, &rb))
                return GGML_STATUS_ALLOC_FAILED;
            st = vc4_unary(be->ctx, VC4_UOP_RECIPROCAL, &bb, &rb, n, 0, 0, 0);
            if (st == VC4_OK)
                st = vc4_elementwise(be->ctx, VC4_OP_MUL, &ab, &rb, &db, n);
        } break;
        case GGML_OP_SQR:
            st = vc4_elementwise(be->ctx, VC4_OP_MUL, &ab, &ab, &db, n);
            break;
        default:
            return fail("binary", node, "unknown op");
    }

    if (st != VC4_OK) {
        fprintf(stderr, "vc4: %s: %s\n", ggml_op_name(node->op),
                vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status run_unary(vc4_backend_ctx * be, ggml_tensor * node) {
    const ggml_tensor * a = node->src[0];
    if (!a) return fail("unary", node, "missing operand");
    const size_t n = (size_t)ggml_nelements(node);
    vc4_buf ab = tensor_buf(be, a);
    vc4_buf db = tensor_buf(be, node);

    vc4_status st = VC4_ERR_UNSUPPORTED;
    switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_ABS:
            st = vc4_unary(be->ctx, VC4_UOP_ABS, &ab, &db, n, 0, 0, 0); break;
        case GGML_UNARY_OP_NEG:
            st = vc4_unary(be->ctx, VC4_UOP_NEG, &ab, &db, n, 0, 0, 0); break;
        case GGML_UNARY_OP_TANH:
            st = vc4_unary(be->ctx, VC4_UOP_TANH, &ab, &db, n, 0, 0, 0); break;
        case GGML_UNARY_OP_GELU:
            st = vc4_unary(be->ctx, VC4_UOP_GELU, &ab, &db, n, 0, 0, 0); break;
        case GGML_UNARY_OP_GELU_ERF:
            st = vc4_unary(be->ctx, VC4_UOP_GELU_ERF, &ab, &db, n, 0, 0, 0); break;
        case GGML_UNARY_OP_RELU:
            st = vc4_relu(be->ctx, &ab, &db, n); break;
        case GGML_UNARY_OP_SIGMOID:
            st = vc4_sigmoid(be->ctx, &ab, &db, n); break;
        case GGML_UNARY_OP_SILU:
            st = vc4_silu(be->ctx, &ab, &db, n); break;
        default:
            return fail("unary", node, "unary op not implemented on VC4");
    }
    if (st != VC4_OK) {
        fprintf(stderr, "vc4: unary: %s\n", vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status run_mul_mat(vc4_backend_ctx * be, ggml_tensor * node) {
    // ggml_mul_mat(a, b): dst[i0 + i1*ne0] = sum_k a[k,i0] * b[k,i1],
    //   ne0 = a->ne[1], ne1 = b->ne[1], K = a->ne[0] = b->ne[0].
    const ggml_tensor * A = node->src[0];   // usually the weight
    const ggml_tensor * B = node->src[1];   // usually the activations
    if (!A || !B) return fail("mul_mat", node, "missing operand");
    if (A->ne[2] != 1 || A->ne[3] != 1 || B->ne[2] != 1 || B->ne[3] != 1)
        return fail("mul_mat", node, "batched mul_mat falls back");

    const uint32_t K  = (uint32_t)A->ne[0];
    const uint32_t M0 = (uint32_t)A->ne[1];   // dst ne0
    const uint32_t N1 = (uint32_t)B->ne[1];   // dst ne1

    vc4_buf ab = tensor_buf(be, A);
    vc4_buf bb = tensor_buf(be, B);
    vc4_buf db = tensor_buf(be, node);

    // layout 1: B is contiguous, rows of K floats (ggml's required layout)
    const bool b_cont = B->nb[0] == sizeof(float) &&
                        B->nb[1] == (size_t)K * sizeof(float);
    // layout 2: B is a transposed view of a contiguous [N1][K] tensor, i.e.
    //           K x N1 row-major memory (nb0 = N1*4, nb1 = 4)
    const bool b_view = B->nb[0] == (size_t)N1 * sizeof(float) &&
                        B->nb[1] == sizeof(float);
    const bool a_cont = A->nb[0] == sizeof(float) &&
                        A->nb[1] == (size_t)K * sizeof(float);

    vc4_status st = VC4_ERR_UNSUPPORTED;

    if (b_cont) {
        // dst memory is N1 rows of M0 = B_rows x A_rows^T: exactly
        // matmul_bt(engine A = B, engine B = A).
        if (!a_cont)
            return fail("mul_mat", node, "non-contiguous src0 falls back");
        if (M0 % 16 || K % 16)
            return fail("mul_mat", node, "src0 needs ne0%16==0 and ne1%16==0");

        uint32_t Mp = (N1 + 15u) & ~15u;
        if (Mp != N1) {
            vc4_buf ap, cp;
            if (!tmp_alloc(be, (size_t)Mp * K * 4, &ap) ||
                !tmp_alloc(be, (size_t)Mp * M0 * 4, &cp))
                return GGML_STATUS_ALLOC_FAILED;
            civac(be, B->data, (size_t)ggml_nbytes(B));   // B may be QPU-written
            uint8_t * ap_p = be->arena + ap.off;
            uint8_t * cp_p = be->arena + cp.off;
            memset(ap_p, 0, (size_t)Mp * K * 4);
            for (uint32_t r = 0; r < N1; ++r)
                memcpy(ap_p + (size_t)r * K * 4,
                       (const uint8_t *)B->data + (size_t)r * K * 4, (size_t)K * 4);
            civac(be, ap_p, (size_t)Mp * K * 4);
            st = vc4_matmul_bt(be->ctx, &ap, &ab, &cp, Mp, M0, K);
            if (st == VC4_OK) {
                civac(be, cp_p, (size_t)Mp * M0 * 4);
                for (uint32_t r = 0; r < N1; ++r)
                    memcpy((uint8_t *)node->data + (size_t)r * M0 * 4,
                           cp_p + (size_t)r * M0 * 4, (size_t)M0 * 4);
                civac(be, node->data, (size_t)N1 * M0 * 4);
            }
        } else {
            st = vc4_matmul_bt(be->ctx, &bb, &ab, &db, N1, M0, K);
        }
    } else if (b_view) {
        // engine C = A * B is M0 x N1 (rows of i0), ggml's dst is the
        // transpose (N1 rows of M0).  Compute then transpose on the QPUs.
        if (!a_cont)
            return fail("mul_mat", node, "non-contiguous src0 falls back");
        if (M0 % 16 || N1 % 16 || K % 16 || N1 % 64)
            return fail("mul_mat", node, "transposed-B shape needs 16/64 alignment");
        vc4_buf cp;
        if (!tmp_alloc(be, (size_t)M0 * N1 * 4, &cp))
            return GGML_STATUS_ALLOC_FAILED;
        st = vc4_matmul(be->ctx, &ab, &bb, &cp, M0, N1, K,
                        VC4_LAYOUT_ROW_MAJOR);
        if (st == VC4_OK)
            st = vc4_transpose(be->ctx, &cp, &db, M0, N1);
    } else {
        return fail("mul_mat", node, "unknown B layout");
    }

    if (st != VC4_OK) {
        fprintf(stderr, "vc4: mul_mat K=%u M0=%u N1=%u: %s\n", K, M0, N1,
                vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// row sum of squares / sum via a GEMM against a constant ones matrix.
// returns [rows][16] where every column holds the sum.
static vc4_status row_sums(vc4_backend_ctx * be, vc4_buf x, uint32_t rows,
                           uint32_t cols, vc4_buf * out) {
    vc4_buf ones;
    if (!tmp_alloc(be, (size_t)cols * 16u * 4u, &ones))
        return VC4_ERR_NOMEM;
    float * p = (float *)(be->arena + ones.off);
    for (size_t i = 0; i < (size_t)cols * 16u; ++i) p[i] = 1.0f;
    civac(be, p, (size_t)cols * 16u * 4u);

    vc4_buf sum;
    if (!tmp_alloc(be, (size_t)rows * 16u * 4u, &sum))
        return VC4_ERR_NOMEM;
    vc4_status st = vc4_matmul(be->ctx, &x, &ones, &sum, rows, 16, cols,
                               VC4_LAYOUT_ROW_MAJOR);
    if (st == VC4_OK) *out = sum;
    return st;
}

static enum ggml_status run_norm(vc4_backend_ctx * be, ggml_tensor * node,
                                 bool rms) {
    const ggml_tensor * x = node->src[0];
    if (!x) return fail("norm", node, "missing operand");
    const uint32_t cols = (uint32_t)x->ne[0];
    const uint32_t rows = (uint32_t)(x->ne[1] * x->ne[2] * x->ne[3]);
    float eps = 0.0f;
    memcpy(&eps, node->op_params, sizeof(eps));

    vc4_buf xb = tensor_buf(be, x);
    vc4_buf yb = tensor_buf(be, node);

    vc4_buf sq;
    if (!tmp_alloc(be, (size_t)rows * cols * 4u, &sq))
        return GGML_STATUS_ALLOC_FAILED;

    vc4_status st = VC4_OK;
    vc4_buf src = xb;
    if (rms) {
        st = vc4_elementwise(be->ctx, VC4_OP_MUL, &xb, &xb, &sq, (size_t)rows * cols);
        if (st == VC4_OK) src = sq;
    } else {
        // layernorm: mean and variance through the fused engine kernel, with
        // constant gamma=1, beta=0 (the weight is applied by the caller).
        vc4_buf gamma, beta, mean, istd;
        if (!tmp_alloc(be, (size_t)cols * 4u, &gamma) ||
            !tmp_alloc(be, (size_t)cols * 4u, &beta) ||
            !tmp_alloc(be, (size_t)rows * 4u, &mean) ||
            !tmp_alloc(be, (size_t)rows * 4u, &istd))
            return GGML_STATUS_ALLOC_FAILED;
        float * g = (float *)(be->arena + gamma.off);
        float * bt = (float *)(be->arena + beta.off);
        for (uint32_t i = 0; i < cols; ++i) { g[i] = 1.0f; bt[i] = 0.0f; }
        civac(be, g, (size_t)cols * 4u);
        civac(be, bt, (size_t)cols * 4u);
        st = vc4_layernorm_fwd(be->ctx, &xb, &gamma, &beta, &yb, &mean, &istd,
                               rows, cols, eps);
        if (st != VC4_OK) {
            fprintf(stderr, "vc4: layernorm: %s\n", vc4_strerror(st));
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    vc4_buf sums;
    if (st == VC4_OK) st = row_sums(be, src, rows, cols, &sums);
    vc4_buf inv;
    if (st == VC4_OK) {
        if (!tmp_alloc(be, (size_t)rows * 16u * 4u, &inv)) return GGML_STATUS_ALLOC_FAILED;
        st = vc4_affine(be->ctx, &sums, &inv, (size_t)rows * 16u,
                        1.0f / (float)cols, eps);
    }
    if (st == VC4_OK)
        st = vc4_rsqrt(be->ctx, &inv, &inv, (size_t)rows * 16u);
    vc4_buf invb;
    if (st == VC4_OK) {
        if (!tmp_alloc(be, (size_t)rows * cols * 4u, &invb))
            return GGML_STATUS_ALLOC_FAILED;
        // inv row r is 16 identical words; each output row reads word 0
        st = lane_runs(be, inv, invb, rows, cols, 16u * 4u, cols * 4u, 0);
    }
    if (st == VC4_OK)
        st = vc4_elementwise(be->ctx, VC4_OP_MUL, &xb, &invb, &yb,
                             (size_t)rows * cols);

    if (st != VC4_OK) {
        fprintf(stderr, "vc4: %s: %s\n", rms ? "rms_norm" : "norm",
                vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status run_soft_max(vc4_backend_ctx * be, ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];
    if (!x) return fail("soft_max", node, "missing operand");
    if (node->src[1]) return fail("soft_max", node, "attention mask falls back");
    if (node->src[2]) return fail("soft_max", node, "attention sinks fall back");

    float scale = 1.0f, max_bias = 0.0f;
    memcpy(&scale, node->op_params, sizeof(scale));
    memcpy(&max_bias, (const char *)node->op_params + sizeof(scale), sizeof(max_bias));
    if (max_bias != 0.0f) return fail("soft_max", node, "ALiBi max_bias falls back");

    const uint32_t cols = (uint32_t)node->ne[0];
    const uint32_t rows = (uint32_t)(node->ne[1] * node->ne[2] * node->ne[3]);
    vc4_buf xb = tensor_buf(be, x);
    vc4_buf db = tensor_buf(be, node);

    vc4_status st = VC4_OK;
    if (scale != 1.0f) {
        vc4_buf t;
        if (!tmp_alloc(be, (size_t)rows * cols * 4u, &t))
            return GGML_STATUS_ALLOC_FAILED;
        st = vc4_affine(be->ctx, &xb, &t, (size_t)rows * cols, scale, 0.0f);
        if (st == VC4_OK) xb = t;
    }
    if (st == VC4_OK)
        st = vc4_softmax(be->ctx, &xb, rows, cols, &db);

    if (st != VC4_OK) {
        fprintf(stderr, "vc4: soft_max: %s\n", vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status run_get_rows(vc4_backend_ctx * be, ggml_tensor * node) {
    const ggml_tensor * t = node->src[0];
    const ggml_tensor * idx = node->src[1];
    if (!t || !idx || idx->type != GGML_TYPE_I32)
        return fail("get_rows", node, "need an i32 index tensor");

    const uint32_t cols = (uint32_t)t->ne[0];
    const int64_t nrows = t->ne[1];
    const int64_t n = ggml_nelements(idx);

    // indices are integer geometry: read them, then route rows with QPU copies
    std::vector<int32_t> ids((size_t)n);
    civac(be, idx->data, (size_t)ggml_nbytes(idx));
    memcpy(ids.data(), idx->data, (size_t)n * sizeof(int32_t));

    vc4_status st = VC4_OK;
    for (int64_t i = 0; i < n && st == VC4_OK; ++i) {
        const int64_t r = ids[(size_t)i];
        vc4_buf db = tensor_buf(be, node, (size_t)i * cols * 4u);
        if (r < 0 || r >= nrows) {
            // ggml guarantees valid indices; a fill keeps the contract visible
            memset(be->arena + db.off, 0, (size_t)cols * 4u);
            civac(be, be->arena + db.off, (size_t)cols * 4u);
            continue;
        }
        vc4_buf sb = tensor_buf(be, t, (size_t)r * cols * 4u);
        st = copy_words(be, sb, db, cols);
    }
    if (st != VC4_OK) {
        fprintf(stderr, "vc4: get_rows: %s\n", vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status run_copy(vc4_backend_ctx * be, ggml_tensor * node) {
    const ggml_tensor * s = node->src[0];
    if (!s) return fail("cpy", node, "missing operand");
    if (s->type != node->type) return fail("cpy", node, "dtype conversion falls back");
    if (ggml_nelements(s) != ggml_nelements(node))
        return fail("cpy", node, "element count differs");
    if (!tensor_cont(s) || !tensor_cont(node))
        return fail("cpy", node, "strided copy falls back");
    if (s->data == node->data) return GGML_STATUS_SUCCESS;

    vc4_buf sb = tensor_buf(be, s);
    vc4_buf db = tensor_buf(be, node);
    civac(be, s->data, (size_t)ggml_nbytes(s));
    vc4_status st = copy_words(be, sb, db, (size_t)ggml_nelements(s));
    if (st != VC4_OK) {
        fprintf(stderr, "vc4: cpy: %s\n", vc4_strerror(st));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// --------------------------------------------------------------------------
// supports_op - the honest list (everything else is a CPU fallback)
// --------------------------------------------------------------------------
static bool vc4_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // metadata / leaf: the scheduler must accept any dtype here
            return true;
        default:
            break;
    }

    if (!tensor_is_f32(op) || !tensor_cont(op)) return false;

    switch (op->op) {

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            if (!a || !b || !tensor_is_f32(a) || !tensor_is_f32(b)) return false;
            if (!tensor_cont(a)) return false;
            if (ggml_are_same_shape(a, b)) return true;
            return ggml_can_repeat(b, a) && b->ne[0] == a->ne[0] && tensor_cont(b);
        }
        case GGML_OP_SQR:
            return a && tensor_is_f32(a) && tensor_cont(a);

        case GGML_OP_UNARY:
            if (!a || !tensor_is_f32(a) || !tensor_cont(a)) return false;
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SILU:
                    return true;
                default:
                    return false;
            }

        case GGML_OP_MUL_MAT: {
            if (!a || !b || !tensor_is_f32(a) || !tensor_is_f32(b)) return false;
            if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1)
                return false;
            const int64_t K  = a->ne[0];
            const int64_t M0 = a->ne[1];
            const int64_t N1 = b->ne[1];
            const bool a_cont = a->nb[0] == sizeof(float) &&
                                a->nb[1] == (size_t)K * sizeof(float);
            if (!a_cont) return false;
            if (b->nb[0] == sizeof(float) &&
                b->nb[1] == (size_t)K * sizeof(float)) {
                // standard layout: rows of K; matmul_bt needs ne1 and K 16-aligned
                return M0 % 16 == 0 && K % 16 == 0;
            }
            if (b->nb[0] == (size_t)N1 * sizeof(float) &&
                b->nb[1] == sizeof(float)) {
                return M0 % 16 == 0 && N1 % 64 == 0 && K % 16 == 0;
            }
            return false;
        }

        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM: {
            if (!a || !tensor_is_f32(a) || !tensor_cont(a)) return false;
            // rows % 8 and cols % 16 keep vc4_matmul off its host-padding path
            const int64_t rows = a->ne[1] * a->ne[2] * a->ne[3];
            return a->ne[0] % 16 == 0 && rows % 8 == 0;
        }

        case GGML_OP_SOFT_MAX:
            if (!a || !tensor_is_f32(a) || !tensor_cont(a)) return false;
            if (op->src[1] || op->src[2]) return false;
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const char *)op->op_params + sizeof(float),
                       sizeof(max_bias));
                return max_bias == 0.0f;
            }

        case GGML_OP_GET_ROWS:
            return a && b && tensor_is_f32(a) && tensor_cont(a) &&
                   b->type == GGML_TYPE_I32 && tensor_cont(b);

        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return a && tensor_is_f32(a) && tensor_cont(a) && tensor_is_f32(op) &&
                   tensor_cont(op) && ggml_nelements(a) == ggml_nelements(op);

        default:
            return false;
    }
}

// --------------------------------------------------------------------------
// graph compute
// --------------------------------------------------------------------------
// A node the scheduler put on this backend must live in this backend's pool.
// Anything else is a partition bug, and letting it through would program a
// bus address outside the arena; fail loudly instead.
static bool owned_tensor(vc4_backend_ctx * be, const ggml_tensor * t) {
    if (!t || !t->data) return false;
    const uintptr_t p  = (uintptr_t)t->data;
    const uintptr_t lo = (uintptr_t)be->arena + be->pool.off;
    const uintptr_t hi = lo + be->pool_usable;
    const size_t    nb = (size_t)ggml_nbytes(t);
    return p >= lo && p + nb <= hi;
}

static enum ggml_status vc4_graph_compute(ggml_backend_t backend,
                                          ggml_cgraph * cgraph) {
    vc4_backend_ctx * be = (vc4_backend_ctx *)backend->context;
    const double t0 = now_ms();

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        tmp_reset(be);

        if (!owned_tensor(be, node))
            return fail(ggml_op_name(node->op), node,
                        "destination is not in the VC4 pool");
        if (node->op != GGML_OP_NONE && node->op != GGML_OP_RESHAPE &&
            node->op != GGML_OP_VIEW && node->op != GGML_OP_PERMUTE &&
            node->op != GGML_OP_TRANSPOSE) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] && !owned_tensor(be, node->src[s]))
                    return fail(ggml_op_name(node->op), node,
                                "a source tensor is not in the VC4 pool");
            }
        }

        const double n0 = now_ms();
        enum ggml_status st = GGML_STATUS_SUCCESS;

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_SQR:
                st = run_bin(be, node);
                break;

            case GGML_OP_UNARY:
                st = run_unary(be, node);
                break;

            case GGML_OP_MUL_MAT:
                st = run_mul_mat(be, node);
                break;

            case GGML_OP_RMS_NORM:
                st = run_norm(be, node, true);
                break;

            case GGML_OP_NORM:
                st = run_norm(be, node, false);
                break;

            case GGML_OP_SOFT_MAX:
                st = run_soft_max(be, node);
                break;

            case GGML_OP_GET_ROWS:
                st = run_get_rows(be, node);
                break;

            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_DUP:
                st = run_copy(be, node);
                break;

            default:
                st = fail(ggml_op_name(node->op), node, "no VC4 lowering");
                break;
        }

        if (st != GGML_STATUS_SUCCESS) {
            be->nodes_failed++;
            return st;
        }
        be->nodes_on_vc4++;

        if (vc4_trace_enabled()) {
            fprintf(stderr, "vc4: [%3d] %-14s %-24s %7.3f ms\n", i,
                    ggml_op_name(node->op), node->name, now_ms() - n0);
        }
    }

    be->compute_ms += now_ms() - t0;
    return GGML_STATUS_SUCCESS;
}

// --------------------------------------------------------------------------
// backend / device / reg
// --------------------------------------------------------------------------
static const char * vc4_backend_get_name(ggml_backend_t) { return "VC4"; }

static void vc4_backend_free(ggml_backend_t backend) {
    vc4_backend_ctx * be = (vc4_backend_ctx *)backend->context;
    if (be) {
        if (g_be.load() == be) {
            g_be.store(nullptr);
            g_arena_bytes.store(0);
            g_pool_bytes.store(0);
        }
        if (be->ctx) vc4_shutdown(be->ctx);
        delete be;
    }
    delete backend;
}

static const struct ggml_backend_i vc4_backend_iface = {
    /* .get_name          = */ vc4_backend_get_name,
    /* .free              = */ vc4_backend_free,
    /* .set_tensor_async  = */ nullptr,
    /* .get_tensor_async  = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async  = */ nullptr,
    /* .synchronize       = */ nullptr,
    /* .graph_plan_create = */ nullptr,
    /* .graph_plan_free   = */ nullptr,
    /* .graph_plan_update = */ nullptr,
    /* .graph_plan_compute= */ nullptr,
    /* .graph_compute     = */ vc4_graph_compute,
    /* .event_record      = */ nullptr,
    /* .event_wait        = */ nullptr,
    /* .graph_optimize    = */ nullptr,
};

static ggml_guid vc4_guid = {
    0x56, 0x43, 0x34, 0x2d, 0x67, 0x67, 0x6d, 0x6c,
    0x2d, 0x76, 0x63, 0x34, 0x2d, 0x30, 0x30, 0x31,
};

static ggml_backend_t vc4_device_init_backend(ggml_backend_dev_t dev,
                                              const char * params) {
    (void)params;

    vc4_backend_ctx * be = new (std::nothrow) vc4_backend_ctx;
    if (!be) return nullptr;

    if (g_be.load() != nullptr) {
        fprintf(stderr, "vc4: only one VC4 backend per process (one arena); "
                        "reuse the existing backend\n");
        delete be;
        return nullptr;
    }

    const char * cache = getenv("VC4INFER_CACHE");
    vc4_status st = vc4_init(&be->ctx, cache && *cache ? cache : nullptr, 0);
    if (st != VC4_OK) {
        fprintf(stderr, "vc4: vc4_init failed: %s (is vc4d running? set "
                        "VC4INFER_DAEMON=1 for the unprivileged path)\n",
                vc4_strerror(st));
        delete be;
        return nullptr;
    }
    be->arena_bytes = vc4_arena_bytes(be->ctx);

    size_t pool_mb = (size_t)std::max(4, env_int("VC4GGML_ARENA_MB",
                                                 (int)VC4_DEFAULT_POOL_MB));
    size_t pool_bytes = pool_mb << 20;
    const size_t cap = (be->arena_bytes * 3u) / 5u;
    if (pool_bytes > cap) pool_bytes = cap & ~(size_t)(VC4_ALIGN - 1);
    if (pool_bytes < (2u << 20)) {
        fprintf(stderr, "vc4: arena is only %zu B; not enough for a backend\n",
                be->arena_bytes);
        vc4_shutdown(be->ctx);
        delete be;
        return nullptr;
    }

    st = vc4_alloc(be->ctx, pool_bytes, &be->pool);
    if (st != VC4_OK) {
        fprintf(stderr, "vc4: cannot reserve %zu B of the %zu B arena: %s\n",
                pool_bytes, be->arena_bytes, vc4_strerror(st));
        vc4_shutdown(be->ctx);
        delete be;
        return nullptr;
    }

    be->arena = (uint8_t *)vc4_host_ptr(be->ctx, &be->pool) - be->pool.off;

    size_t scratch = (size_t)env_int("VC4GGML_SCRATCH_MB",
                                     (int)VC4_DEFAULT_SCRATCH_MB) << 20;
    scratch = std::min(scratch, pool_bytes / 4);
    scratch = align_up(scratch, VC4_ALIGN);
    if (scratch < (1u << 20)) scratch = (1u << 20);

    be->pool_usable = pool_bytes - scratch;
    be->blocks.push_back({ 0, be->pool_usable, false });
    be->free_bytes  = be->pool_usable;
    be->scratch_off = be->pool.off + (uint32_t)be->pool_usable;
    be->scratch_size = scratch;

    g_arena_bytes.store(be->arena_bytes);
    g_pool_bytes.store(be->pool_usable);
    g_be.store(be);

    fprintf(stderr,
            "vc4: arena %zu MiB, tensor pool %zu MiB, op scratch %zu MiB, "
            "ggml backend API %d\n",
            be->arena_bytes >> 20, be->pool_usable >> 20, scratch >> 20,
            GGML_BACKEND_API_VERSION);

    ggml_backend_t backend = new (std::nothrow) ggml_backend;
    if (!backend) {
        vc4_shutdown(be->ctx);
        delete be;
        return nullptr;
    }
    backend->guid    = &vc4_guid;
    backend->iface   = vc4_backend_iface;
    backend->device  = dev;
    backend->context = be;
    return backend;
}

static const char * vc4_device_get_name(ggml_backend_dev_t) { return "VC4"; }
static const char * vc4_device_get_description(ggml_backend_dev_t) {
    return "VideoCore IV QPUs (V3D, Raspberry Pi 3B) via vc4infer";
}

static void vc4_device_get_memory(ggml_backend_dev_t, size_t * free,
                                  size_t * total) {
    // report the reserved pool (what ggml can actually allocate), not the
    // whole arena: the engine's operation scratch lives above it.
    size_t arena = g_arena_bytes.load();
    size_t pool  = g_pool_bytes.load();
    if (arena == 0) {
        const int mb = env_int("VC4INFER_ARENA_MB", 64);
        arena = (size_t)mb << 20;
        pool  = std::min<size_t>(VC4_DEFAULT_POOL_MB << 20, (arena * 3u) / 5u);
    }
    if (free)  *free  = pool;
    if (total) *total = arena;
}

static enum ggml_backend_dev_type vc4_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void vc4_device_get_props(ggml_backend_dev_t dev,
                                 ggml_backend_dev_props * props) {
    props->name        = vc4_device_get_name(dev);
    props->description = vc4_device_get_description(dev);
    vc4_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type        = vc4_device_get_type(dev);
    props->device_id   = nullptr;
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_buffer_type_t vc4_device_get_buffer_type(ggml_backend_dev_t) {
    return vc4_buft();
}

static bool vc4_device_supports_buft(ggml_backend_dev_t,
                                     ggml_backend_buffer_type_t buft) {
    return buft == vc4_buft();
}

static const struct ggml_backend_device_i vc4_device_iface = {
    /* .get_name             = */ vc4_device_get_name,
    /* .get_description      = */ vc4_device_get_description,
    /* .get_memory           = */ vc4_device_get_memory,
    /* .get_type             = */ vc4_device_get_type,
    /* .get_props            = */ vc4_device_get_props,
    /* .init_backend         = */ vc4_device_init_backend,
    /* .get_buffer_type      = */ vc4_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ vc4_supports_op,
    /* .supports_buft        = */ vc4_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * vc4_reg_get_name(ggml_backend_reg_t) { return "VC4"; }
static size_t vc4_reg_get_device_count(ggml_backend_reg_t) { return 1; }

static ggml_backend_dev_t vc4_device(void) {
    static struct ggml_backend_device dev = {
        /* .iface   = */ {},   // filled below on first use
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    static bool init = false;
    if (!init) {
        dev.iface = vc4_device_iface;
        dev.reg   = vc4_reg();
        init = true;
    }
    return &dev;
}

static ggml_backend_dev_t vc4_reg_get_device(ggml_backend_reg_t, size_t index) {
    return index == 0 ? vc4_device() : nullptr;
}

static void * vc4_reg_get_proc_address(ggml_backend_reg_t, const char *) {
    return nullptr;
}

static const struct ggml_backend_reg_i vc4_reg_iface = {
    /* .get_name         = */ vc4_reg_get_name,
    /* .get_device_count = */ vc4_reg_get_device_count,
    /* .get_device       = */ vc4_reg_get_device,
    /* .get_proc_address = */ vc4_reg_get_proc_address,
};

static ggml_backend_reg_t vc4_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {},   // filled below on first use
        /* .context     = */ nullptr,
    };
    static bool init = false;
    if (!init) {
        reg.iface = vc4_reg_iface;
        init = true;
    }
    return &reg;
}

ggml_backend_reg_t ggml_backend_vc4_reg(void) {
    ggml_backend_reg_t reg = vc4_reg();
    // glue the singleton graph: buft -> device -> reg
    ggml_backend_buffer_type_t buft = vc4_buft();
    buft->device  = vc4_device();
    buft->context = nullptr;
    return reg;
}

void ggml_vc4_register(void) {
    ggml_backend_register(ggml_backend_vc4_reg());
}

int ggml_backend_vc4_score(void) {
    // cheap probe: the daemon socket, or /dev/mem for the direct path
    const bool daemon = access("/run/vc4d.sock", F_OK) == 0;
    const bool direct = access("/dev/mem", R_OK | W_OK) == 0;
    if (!daemon && !direct) return 0;
    // do not score higher than the CPU backend (it must stay the fallback)
    return 40;
}
