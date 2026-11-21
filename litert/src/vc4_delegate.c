/*
 * vc4_delegate.c - a LiteRT / TensorFlow Lite external delegate for the
 * VideoCore IV QPUs.
 *
 * ABI.  The installed runtime is ai-edge-litert 2.3.0 built from LiteRT
 * v2.3.0.  Its Python `Delegate` (ai_edge_litert/interpreter.py) dlopen()s
 * the library and calls the classic external-delegate plugin ABI:
 *
 *   TfLiteDelegate *tflite_plugin_create_delegate(
 *       const char *const *keys, const char *const *values,
 *       size_t num_options, void (*report_error)(const char *));
 *   void tflite_plugin_destroy_delegate(TfLiteDelegate *);
 *
 * The returned TfLiteDelegate uses the internal `Prepare` hook and the
 * documented `TfLiteContext::ReplaceNodeSubsetsWithDelegateKernels` path.
 * The newer dispatch API (LiteRtDispatchDelegateGetHooks /
 * LiteRtSetDelegateFunction, litert/c/internal/litert_dispatch_delegate.h)
 * is present in the same shared object, but its header is marked internal,
 * it expects a vendor bytecode executable and a tensor-buffer registration
 * model, and the plugin ABI above hands the runtime a TfLiteDelegate anyway.
 * The classic path reaches every op the engine implements without a compiler
 * backend, so that is what this delegate implements.
 *
 * Partitioning.  `Prepare` walks the execution plan, asks vc4_plan_node()
 * about each node and replaces maximal runs of claimed nodes with delegate
 * kernel nodes.  A node is either claimed and executed by the engine, or it
 * is left untouched for LiteRT.  There is no host fallback inside a claimed
 * partition: if an engine call fails, Invoke returns kTfLiteError.
 *
 * Memory.  Every claimed tensor gets a persistent arena buffer that is a
 * byte-for-byte image of the TFLite tensor (dense fp32).  Only partition
 * boundary tensors cross the PCIe-less bus: they are uploaded before the
 * first step and downloaded after the last.  Intermediates never reach the
 * ARM, which is what makes "no host compute" checkable.
 *
 * Layouts.  The engine's NHWC conv writes `align16(wo)` pixels per row; the
 * delegate compacts that pitched row into the dense TFLite tensor with
 * vc4_copy2d_n (a QPU layout copy, not an arithmetic fallback).  TFLite
 * weights are tap-major ([Cout][KH][KW][Cin]); the engine packer wants
 * [Cout][Cin][KH][KW], so constants are transposed once at prepare time on
 * the host.  That is a layout transform of constants, never of activations.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4infer.h"
#include "vc4_plan.h"

#include "tflite/builtin_ops.h"
#include "tflite/core/c/builtin_op_data.h"
#include "tflite/core/c/common.h"
#include "tflite/delegates/external/external_delegate_interface.h"

#define VC4_DELEGATE_VERSION "0.1.0"
#define VC4_MAX_PARTITION_NODES 4096

/* ------------------------------------------------------------------ state */

typedef struct {
  TfLiteDelegate base;
  char *mode;      /* auto | plan | off */
  char *cache_dir;
  size_t arena_bytes;
  int verbose;
  int dry; /* claim and replace, but never execute: structural check only */
  int claimed;
  int declined;
  int partitions;
  /* Plans are built in delegate context, where GetNodeAndRegistration is
   * legal, and looked up by node index from the kernel context, where it is
   * not (tflite/core/subgraph.cc: SwitchToKernelContext). */
  int num_nodes;
  vc4_node_plan *node_plans;
  unsigned char *node_claimed;
} Vc4Delegate;

typedef struct {
  int present;
  int is_const;
  int boundary_in;
  int boundary_out;
  size_t bytes;
  void *host; /* mode=dry only: a host shadow, never a QPU buffer */
  vc4_buf buf;
  vc4_buf cur; /* the buffer holding the current value (== buf, or an
                  activation temporary) */
} Vc4Slot;

typedef struct {
  vc4_node_plan plan;
  vc4_buf wpanel;
  vc4_buf act;
} Vc4Step;

typedef struct {
  Vc4Delegate *owner;
  int ntensors;
  Vc4Slot *slots; /* indexed by interpreter tensor index */
  int nsteps;
  Vc4Step *steps;
  int prepared;
  int npitch;
  vc4_buf pitch;
} Vc4Kernel;

/* One engine context per process.  vc4_ctx is documented as not thread
 * safe; LiteRT invokes one interpreter at a time here, and the delegate
 * serialises nothing else. */
static vc4_ctx *g_ctx;
static int g_init_failed;

static void report(TfLiteContext *ctx, const char *fmt, ...)
{
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (ctx && ctx->ReportError)
        ctx->ReportError(ctx, "vc4-delegate: %s", msg);
    else
        fprintf(stderr, "vc4-delegate: %s\n", msg);
}

static void plugin_error(void (*report_error)(const char *), const char *fmt,
                         ...)
{
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (report_error)
        report_error(msg);
}

/* ------------------------------------------------------------- TfLiteIntArray */

static TfLiteIntArray *intarray_new(int n)
{
    TfLiteIntArray *a;
    if (n < 0)
        return NULL;
    a = (TfLiteIntArray *)malloc(sizeof(TfLiteIntArray) + (size_t)n * sizeof(int));
    if (!a)
        return NULL;
    a->size = n;
    return a;
}

/* ---------------------------------------------------------------- options */

static char *dup_opt(const char *s)
{
    size_t n;
    char *p;
    if (!s)
        return NULL;
    n = strlen(s) + 1;
    p = (char *)malloc(n);
    if (p)
        memcpy(p, s, n);
    return p;
}

/* ------------------------------------------------------------------ engine */

static vc4_ctx *engine_get(Vc4Delegate *d, TfLiteContext *ctx)
{
    vc4_status st;
    if (g_ctx)
        return g_ctx;
    if (g_init_failed)
        return NULL;
    st = vc4_init(&g_ctx, d->cache_dir, d->arena_bytes);
    if (st != VC4_OK || !g_ctx) {
        g_ctx = NULL;
        g_init_failed = 1;
        report(ctx, "vc4_init failed: %s", vc4_strerror(st));
        return NULL;
    }
    return g_ctx;
}

/* ----------------------------------------------------------- slot handling */

static Vc4Slot *slot_get(Vc4Kernel *k, TfLiteContext *ctx, int tensor_index)
{
    Vc4Slot *s;
    if (tensor_index == kTfLiteOptionalTensor)
        return NULL;
    if (tensor_index < 0 || tensor_index >= k->ntensors)
        return NULL;
    s = &k->slots[tensor_index];
    if (!s->present) {
        s->present = 1;
        s->bytes = ctx->tensors[tensor_index].bytes;
    }
    return s;
}

/* ------------------------------------------------------ constant repacking */

/* TFLite conv filters are [Cout][KH][KW][Cin]; the engine packer reads
 * [Cout][Cin][KH][KW].  Both are dense fp32 constants of the same size. */
static void repack_conv_weights(const TfLiteTensor *t, int Cin, int Cout,
                                int KH, int KW, float *dst)
{
    const float *src = (const float *)t->data.data;
    const int taps = KH * KW;
    int co, ci, tap;
    if (!src)
        return;
    for (co = 0; co < Cout; ++co)
        for (ci = 0; ci < Cin; ++ci)
            for (tap = 0; tap < taps; ++tap)
                dst[((size_t)co * Cin + ci) * taps + tap] =
                    src[((size_t)co * taps + tap) * Cin + ci];
}

/* TFLite depthwise filters are [1][3][3][C]; vc4_depthwise_pack_w reads
 * [C][9]. */
static void repack_depthwise_weights(const TfLiteTensor *t, int C, float *dst)
{
    const float *src = (const float *)t->data.data;
    int c, tap;
    if (!src)
        return;
    for (c = 0; c < C; ++c)
        for (tap = 0; tap < 9; ++tap)
            dst[(size_t)c * 9 + tap] = src[(size_t)tap * C + c];
}

/* ------------------------------------------------------------- step execute */

static TfLiteStatus run_conv(Vc4Kernel *k, Vc4Step *st, TfLiteContext *ctx)
{
    const vc4_node_plan *p = &st->plan;
    vc4_ctx *g = engine_get(k->owner, ctx);
    vc4_buf *in;
    vc4_buf *out;
    vc4_status s;
    int ho = 0, wo = 0, pitch = 0;
    size_t rows, vpr;

    if (!g)
        return kTfLiteError;
    in = &k->slots[p->in0].cur;
    out = &k->slots[p->out].buf;

    if (p->is_depthwise)
        s = vc4_depthwise_nhwc(g, in, p->nimg, p->Cin, p->H, p->W, p->W,
                               &st->wpanel, &k->pitch, &ho, &wo, &pitch);
    else
        s = vc4_conv_nhwc(g, in, p->nimg, p->Cin, p->H, p->W, p->W,
                          &st->wpanel, p->Cout, p->KH, p->KW, p->stride,
                          p->pad, &k->pitch, &ho, &wo, &pitch);
    if (s != VC4_OK) {
        report(ctx, "conv node %s failed: %s", vc4_plan_builtin_name(p->builtin),
               vc4_strerror(s));
        return kTfLiteError;
    }
    if (ho != p->ho || wo != p->wo) {
        report(ctx, "conv geometry drift: engine %dx%d, plan %dx%d", ho, wo,
               p->ho, p->wo);
        return kTfLiteError;
    }

    /* rows of [wo][Cout] floats; the source row is align16(wo)*Cout wide. */
    rows = (size_t)p->nimg * (size_t)ho;
    vpr = (size_t)wo * (size_t)p->Cout / 16u;
    if (!vpr) {
        report(ctx, "conv output row cannot be compacted (wo=%d Cout=%d)", wo,
               p->Cout);
        return kTfLiteError;
    }
    s = vc4_copy2d_n(g, &k->pitch, out, (uint32_t)rows, (uint32_t)vpr,
                     (uint32_t)p->Wg * p->Cout * 4u,
                     (uint32_t)wo * p->Cout * 4u, 64, 64, 4, 16, 1, 16);
    if (s != VC4_OK) {
        report(ctx, "conv output compaction failed: %s", vc4_strerror(s));
        return kTfLiteError;
    }

    if (p->activation != kTfLiteActNone) {
        const size_t n = (size_t)p->nimg * ho * wo * p->Cout;
        switch (p->activation) {
        case kTfLiteActRelu:
            s = vc4_relu(g, out, &st->act, n);
            break;
        case kTfLiteActRelu6:
            s = vc4_unary(g, VC4_UOP_CLIP, out, &st->act, n, 0.0f, 6.0f, 0.0f);
            break;
        default:
            report(ctx, "unsupported fused activation %d", p->activation);
            return kTfLiteError;
        }
        if (s != VC4_OK) {
            report(ctx, "fused activation failed: %s", vc4_strerror(s));
            return kTfLiteError;
        }
        k->slots[p->out].cur = st->act;
    }
    return kTfLiteOk;
}

static TfLiteStatus run_unary(Vc4Kernel *k, Vc4Step *st, TfLiteContext *ctx)
{
    const vc4_node_plan *p = &st->plan;
    vc4_ctx *g = engine_get(k->owner, ctx);
    const vc4_buf *in;
    vc4_buf *out;
    vc4_status s;

    if (!g)
        return kTfLiteError;
    in = &k->slots[p->in0].cur;
    out = &k->slots[p->out].buf;
    switch (p->unop) {
    case VC4_PLAN_U_RELU:
        s = vc4_relu(g, in, out, p->n);
        break;
    case VC4_PLAN_U_SIGMOID:
        s = vc4_sigmoid(g, in, out, p->n);
        break;
    case VC4_PLAN_U_RELU6:
        s = vc4_unary(g, VC4_UOP_CLIP, in, out, p->n, 0.0f, 6.0f, 0.0f);
        break;
    case VC4_PLAN_U_RELU_N1TO1:
        s = vc4_unary(g, VC4_UOP_CLIP, in, out, p->n, -1.0f, 1.0f, 0.0f);
        break;
    case VC4_PLAN_U_TANH:
        s = vc4_unary(g, VC4_UOP_TANH, in, out, p->n, 0.0f, 0.0f, 0.0f);
        break;
    default:
        report(ctx, "unary plan has no operation");
        return kTfLiteError;
    }
    if (s != VC4_OK) {
        report(ctx, "%s failed: %s", vc4_plan_builtin_name(p->builtin),
               vc4_strerror(s));
        return kTfLiteError;
    }
    return kTfLiteOk;
}

static TfLiteStatus run_binary(Vc4Kernel *k, Vc4Step *st, TfLiteContext *ctx)
{
    const vc4_node_plan *p = &st->plan;
    vc4_ctx *g = engine_get(k->owner, ctx);
    const vc4_buf *a, *b;
    vc4_buf *out;
    vc4_op op;
    vc4_status s;

    if (!g)
        return kTfLiteError;
    a = &k->slots[p->in0].cur;
    b = &k->slots[p->in1].cur;
    out = &k->slots[p->out].buf;
    switch (p->binop) {
    case VC4_PLAN_B_ADD: op = VC4_OP_ADD; break;
    case VC4_PLAN_B_MUL: op = VC4_OP_MUL; break;
    case VC4_PLAN_B_SUB: op = VC4_OP_SUB; break;
    default: op = VC4_OP_DIV; break;
    }
    s = vc4_elementwise(g, op, a, b, out, p->n);
    if (s != VC4_OK) {
        report(ctx, "%s failed: %s", vc4_plan_builtin_name(p->builtin),
               vc4_strerror(s));
        return kTfLiteError;
    }
    return kTfLiteOk;
}

static TfLiteStatus run_softmax(Vc4Kernel *k, Vc4Step *st, TfLiteContext *ctx)
{
    const vc4_node_plan *p = &st->plan;
    vc4_ctx *g = engine_get(k->owner, ctx);
    vc4_status s;

    if (!g)
        return kTfLiteError;
    s = vc4_softmax(g, &k->slots[p->in0].cur, (uint32_t)p->rows,
                    (uint32_t)p->cols, &k->slots[p->out].buf);
    if (s != VC4_OK) {
        report(ctx, "softmax failed: %s", vc4_strerror(s));
        return kTfLiteError;
    }
    return kTfLiteOk;
}

/* ------------------------------------------------------------ kernel hooks */

static void *kernel_init(TfLiteContext *ctx, const char *buffer, size_t length)
{
    Vc4Kernel *k;
    (void)ctx;
    (void)buffer;
    (void)length;
    k = (Vc4Kernel *)calloc(1, sizeof(*k));
    return k;
}

static void kernel_free(TfLiteContext *ctx, void *buffer)
{
    Vc4Kernel *k = (Vc4Kernel *)buffer;
    (void)ctx;
    if (!k)
        return;
    if (k->slots) {
        int i;
        for (i = 0; i < k->ntensors; ++i)
            free(k->slots[i].host);
    }
    free(k->slots);
    free(k->steps);
    free(k);
}

static TfLiteStatus kernel_prepare(TfLiteContext *ctx, TfLiteNode *node)
{
    TfLiteDelegateParams *params = (TfLiteDelegateParams *)node->builtin_data;
    Vc4Kernel *k = (Vc4Kernel *)node->user_data;
    vc4_ctx *g;
    int i, j;
    size_t pitch = 0;

    if (!k || !params || !params->delegate)
        return kTfLiteError;
    /* Static shapes only: a second AllocateTensors with the same geometry
     * needs no new device buffers, and the arena has no free(). */
    if (k->prepared)
        return kTfLiteOk;
    k->owner = (Vc4Delegate *)params->delegate->data_;
    k->ntensors = (int)ctx->tensors_size;
    k->slots = (Vc4Slot *)calloc((size_t)k->ntensors, sizeof(Vc4Slot));
    k->steps = (Vc4Step *)calloc((size_t)params->nodes_to_replace->size,
                                 sizeof(Vc4Step));
    if (!k->slots || !k->steps) {
        report(ctx, "out of memory while planning a partition");
        return kTfLiteError;
    }

    /* 1. Look up the plan built in delegate context.  GetNodeAndRegistration
     *    is forbidden here: the runtime switches the context to kernel mode
     *    before prepare runs. */
    for (i = 0; i < params->nodes_to_replace->size; ++i) {
        const int node_index = params->nodes_to_replace->data[i];
        Vc4Step *stp = &k->steps[i];
        const vc4_node_plan *p;
        int idx[4];

        if (node_index < 0 || node_index >= k->owner->num_nodes ||
            !k->owner->node_claimed[node_index]) {
            report(ctx, "node %d was replaced but has no cached plan",
                   node_index);
            return kTfLiteError;
        }
        stp->plan = k->owner->node_plans[node_index];
        p = &stp->plan;
        k->nsteps = i + 1;
        idx[0] = p->in0;
        idx[1] = p->in1;
        idx[2] = p->in2;
        idx[3] = p->out;
        for (j = 0; j < 4; ++j)
            if (!slot_get(k, ctx, idx[j]) && idx[j] != kTfLiteOptionalTensor) {
                report(ctx, "node %d references tensor %d out of range",
                       node_index, idx[j]);
                return kTfLiteError;
            }
    }

    /* 2. Boundary flags from the partition description. */
    for (i = 0; i < params->input_tensors->size; ++i) {
        const int ti = params->input_tensors->data[i];
        Vc4Slot *s = slot_get(k, ctx, ti);
        if (!s) {
            report(ctx, "partition input tensor %d out of range", ti);
            return kTfLiteError;
        }
        s->boundary_in = 1;
    }
    for (i = 0; i < params->output_tensors->size; ++i) {
        const int ti = params->output_tensors->data[i];
        Vc4Slot *s = slot_get(k, ctx, ti);
        if (!s) {
            report(ctx, "partition output tensor %d out of range", ti);
            return kTfLiteError;
        }
        s->boundary_out = 1;
    }

    /* 3. The engine.  mode=dry stops here: the replacement, the kernel
     *    plumbing and the plan re-derivation have all been exercised, but no
     *    device is opened and no buffer is resident on it. */
    if (k->owner->dry) {
        for (i = 0; i < k->ntensors; ++i) {
            Vc4Slot *s = &k->slots[i];
            if (!s->present)
                continue;
            s->host = calloc(1, s->bytes ? s->bytes : 1);
            if (!s->host) {
                report(ctx, "out of host memory for a dry shadow");
                return kTfLiteError;
            }
            s->buf.bytes = s->bytes;
            s->cur = s->buf;
        }
        if (k->owner->verbose)
            fprintf(stderr, "VC4_DELEGATE dry nodes=%d tensors=%d\n",
                    k->nsteps, k->ntensors);
        k->prepared = 1;
        return kTfLiteOk;
    }
    g = engine_get(k->owner, ctx);
    if (!g)
        return kTfLiteError;

    /* 4. Persistent buffers, one per tensor used by the partition.  A
     *    read-only-mapped tensor is a constant and is uploaded once here. */
    for (i = 0; i < k->ntensors; ++i) {
        Vc4Slot *s = &k->slots[i];
        const TfLiteTensor *t = &ctx->tensors[i];
        if (!s->present)
            continue;
        if (s->bytes == 0) {
            report(ctx, "tensor %d has zero bytes", i);
            return kTfLiteError;
        }
        if (vc4_alloc(g, s->bytes, &s->buf) != VC4_OK) {
            report(ctx, "arena exhausted at tensor %d (%zu B)", i, s->bytes);
            return kTfLiteError;
        }
        s->cur = s->buf;
        if (t->allocation_type == kTfLiteMmapRo) {
            if (!t->data.data) {
                report(ctx, "constant tensor %d has no data", i);
                return kTfLiteError;
            }
            s->is_const = 1;
            if (vc4_upload(g, &s->buf, t->data.data, s->bytes) != VC4_OK) {
                report(ctx, "upload of constant tensor %d failed", i);
                return kTfLiteError;
            }
        }
    }

    /* 5. Step-owned buffers: weight panel, pitched conv output, fused
     *    activation temporary. */
    for (i = 0; i < k->nsteps; ++i) {
        Vc4Step *stp = &k->steps[i];
        const vc4_node_plan *p = &stp->plan;
        if (p->kind != VC4_STEP_CONV)
            continue;

        if (p->is_depthwise) {
            const size_t wb = vc4_depthwise_wpanel_bytes(p->Cin);
            float *tmp = (float *)malloc(ctx->tensors[p->in1].bytes);
            if (!wb || !tmp) {
                free(tmp);
                report(ctx, "depthwise panel size is zero");
                return kTfLiteError;
            }
            if (vc4_alloc(g, wb, &stp->wpanel) != VC4_OK) {
                free(tmp);
                report(ctx, "arena exhausted for a depthwise panel");
                return kTfLiteError;
            }
            repack_depthwise_weights(&ctx->tensors[p->in1], p->Cin, tmp);
            if (vc4_upload(g, &k->slots[p->in1].buf, tmp, ctx->tensors[p->in1].bytes) !=
                VC4_OK) {
                free(tmp);
                report(ctx, "depthwise weight upload failed");
                return kTfLiteError;
            }
            free(tmp);
            if (vc4_depthwise_pack_w(g, &k->slots[p->in1].buf, p->Cin,
                                     &stp->wpanel) != VC4_OK) {
                report(ctx, "depthwise weight packing failed");
                return kTfLiteError;
            }
        } else {
            const size_t wb = vc4_conv_nhwc_wpanel_bytes(
                p->Cin, p->Cout, p->KH, p->KW, p->in2 >= 0);
            const int taps = p->KH * p->KW;
            if (!wb) {
                report(ctx, "conv panel size is zero");
                return kTfLiteError;
            }
            if (vc4_alloc(g, wb, &stp->wpanel) != VC4_OK) {
                report(ctx, "arena exhausted for a conv panel");
                return kTfLiteError;
            }
            if (taps != 1) {
                const size_t nb = ctx->tensors[p->in1].bytes;
                float *tmp = (float *)malloc(nb);
                if (!tmp) {
                    report(ctx, "out of host memory for a weight repack");
                    return kTfLiteError;
                }
                repack_conv_weights(&ctx->tensors[p->in1], p->Cin, p->Cout,
                                    p->KH, p->KW, tmp);
                if (vc4_upload(g, &k->slots[p->in1].buf, tmp, nb) != VC4_OK) {
                    free(tmp);
                    report(ctx, "conv weight upload failed");
                    return kTfLiteError;
                }
                free(tmp);
            }
            if (vc4_conv_nhwc_pack_w(g, &k->slots[p->in1].buf, p->Cin,
                                     p->Cout, p->KH, p->KW,
                                     p->in2 >= 0 ? &k->slots[p->in2].buf : NULL,
                                     &stp->wpanel) != VC4_OK) {
                report(ctx, "conv weight packing failed");
                return kTfLiteError;
            }
        }
        {
            const size_t need = (size_t)p->nimg * p->ho * p->Wg * p->Cout * 4u;
            if (need > pitch)
                pitch = need;
        }
        if (p->activation != kTfLiteActNone) {
            if (vc4_alloc(g, p->out_bytes, &stp->act) != VC4_OK) {
                report(ctx, "arena exhausted for an activation temporary");
                return kTfLiteError;
            }
        }
    }

    if (pitch) {
        if (vc4_alloc(g, pitch, &k->pitch) != VC4_OK) {
            report(ctx, "arena exhausted for a conv output pitch (%zu B)",
                   pitch);
            return kTfLiteError;
        }
    }
    if (k->owner->verbose)
        fprintf(stderr,
                "VC4_DELEGATE partition nodes=%d tensors=%d arena_used=%zu/%zu\n",
                k->nsteps, k->ntensors, vc4_arena_used(g), vc4_arena_bytes(g));
    k->prepared = 1;
    return kTfLiteOk;
}

static TfLiteStatus kernel_invoke(TfLiteContext *ctx, TfLiteNode *node)
{
    Vc4Kernel *k = (Vc4Kernel *)node->user_data;
    vc4_ctx *g;
    int i;

    if (!k)
        return kTfLiteError;
    if (k->owner->dry) {
        report(ctx, "mode=dry claimed this partition but does not execute it");
        return kTfLiteError;
    }
    g = engine_get(k->owner, ctx);
    if (!g)
        return kTfLiteError;

    /* Boundary inputs only.  Intermediate tensors stay in the arena. */
    for (i = 0; i < k->ntensors; ++i) {
        Vc4Slot *s = &k->slots[i];
        const TfLiteTensor *t = &ctx->tensors[i];
        if (!s->present || !s->boundary_in || s->is_const)
            continue;
        if (!t->data.data) {
            report(ctx, "input tensor %d has no host buffer", i);
            return kTfLiteError;
        }
        if (vc4_upload(g, &s->buf, t->data.data, s->bytes) != VC4_OK) {
            report(ctx, "upload of input tensor %d failed", i);
            return kTfLiteError;
        }
        s->cur = s->buf;
    }

    for (i = 0; i < k->nsteps; ++i) {
        Vc4Step *stp = &k->steps[i];
        TfLiteStatus st;
        switch (stp->plan.kind) {
        case VC4_STEP_CONV: st = run_conv(k, stp, ctx); break;
        case VC4_STEP_UNARY: st = run_unary(k, stp, ctx); break;
        case VC4_STEP_BINARY: st = run_binary(k, stp, ctx); break;
        case VC4_STEP_SOFTMAX: st = run_softmax(k, stp, ctx); break;
        default: st = kTfLiteError; break;
        }
        if (st != kTfLiteOk)
            return st;
    }

    for (i = 0; i < k->ntensors; ++i) {
        Vc4Slot *s = &k->slots[i];
        const TfLiteTensor *t = &ctx->tensors[i];
        if (!s->present || !s->boundary_out)
            continue;
        if (!t->data.data) {
            report(ctx, "output tensor %d has no host buffer", i);
            return kTfLiteError;
        }
        if (vc4_download(g, &s->cur, t->data.data, s->bytes) != VC4_OK) {
            report(ctx, "download of output tensor %d failed", i);
            return kTfLiteError;
        }
    }
    return kTfLiteOk;
}

static TfLiteRegistration make_registration(void)
{
    TfLiteRegistration r;
    memset(&r, 0, sizeof r);
    r.init = kernel_init;
    r.free = kernel_free;
    r.prepare = kernel_prepare;
    r.invoke = kernel_invoke;
    r.builtin_code = kTfLiteBuiltinDelegate;
    r.custom_name = "vc4";
    r.version = 1;
    return r;
}

/* --------------------------------------------------------- delegate Prepare */

typedef struct {
  int claimed;
  vc4_node_plan plan;
} NodeClaim;

static size_t estimate_run(const NodeClaim *run, int n,
                           const TfLiteTensor *tensors, size_t ntensors,
                           unsigned char *seen)
{
    size_t total = 0, max_pitch = 0, max_scratch = 0, max_tensor = 0;
    int i, j;

    memset(seen, 0, ntensors);
    for (i = 0; i < n; ++i) {
        const vc4_node_plan *p = &run[i].plan;
        vc4_plan_estimate est;
        int idx[4];
        idx[0] = p->in0;
        idx[1] = p->in1;
        idx[2] = p->in2;
        idx[3] = p->out;
        for (j = 0; j < 4; ++j) {
            const int ti = idx[j];
            if (ti == kTfLiteOptionalTensor || (size_t)ti >= ntensors ||
                seen[ti])
                continue;
            seen[ti] = 1;
            total += tensors[ti].bytes;
            if (tensors[ti].bytes > max_tensor)
                max_tensor = tensors[ti].bytes;
        }
        vc4_plan_estimate_node(p, &est);
        total += est.persistent;
        if (est.pitch > max_pitch)
            max_pitch = est.pitch;
        if (est.scratch > max_scratch)
            max_scratch = est.scratch;
    }
    /* + max_tensor: vc4_conv_nhwc_pack_w builds its repacked copy in scratch
     * above the persistent top, so one weight-sized transient is live while
     * panels are still being allocated. */
    return total + max_pitch + max_scratch + max_tensor + (1u << 20);
}

static TfLiteStatus delegate_prepare(TfLiteContext *ctx, TfLiteDelegate *del)
{
    Vc4Delegate *d = (Vc4Delegate *)del->data_;
    TfLiteIntArray *plan = NULL;
    NodeClaim *claims = NULL;
    unsigned char *seen = NULL;
    const int plan_mode = strcmp(d->mode, "plan") == 0;
    const int off_mode = strcmp(d->mode, "off") == 0;
    size_t budget, total = 0;
    int i, run_start = -1;
    TfLiteStatus rc = kTfLiteError;

    if (ctx->GetExecutionPlan(ctx, &plan) != kTfLiteOk || !plan) {
        report(ctx, "GetExecutionPlan failed");
        return kTfLiteError;
    }
    d->claimed = d->declined = d->partitions = 0;
    claims = (NodeClaim *)calloc((size_t)plan->size, sizeof(NodeClaim));
    seen = (unsigned char *)calloc(ctx->tensors_size ? ctx->tensors_size : 1, 1);
    if (!claims || !seen) {
        report(ctx, "out of memory in Prepare");
        goto out;
    }

    budget = (d->arena_bytes ? d->arena_bytes : VC4_ARENA_DEFAULT);
    if (budget > (1u << 20))
        budget -= (1u << 20);

    /* Pass 1: the claim table. */
    for (i = 0; i < plan->size; ++i) {
        const int node_index = plan->data[i];
        TfLiteNode *node = NULL;
        TfLiteRegistration *reg = NULL;
        vc4_claim_status st;

        if (ctx->GetNodeAndRegistration(ctx, node_index, &node, &reg) !=
            kTfLiteOk) {
            report(ctx, "GetNodeAndRegistration(%d) failed", node_index);
            goto out;
        }
        st = vc4_plan_node(reg, node, ctx->tensors, ctx->tensors_size,
                           &claims[i].plan);
        claims[i].claimed = (st == VC4_CLAIM_OK) && !off_mode;
        if (d->verbose || plan_mode)
            fprintf(stderr,
                    "VC4_DELEGATE node=%d op=%s claim=%s reason=%s\n", node_index,
                    vc4_plan_builtin_name(reg->builtin_code),
                    claims[i].claimed ? "yes" : "no", vc4_plan_reason(st));
        if (!claims[i].claimed)
            d->declined++;
    }

    /* Pass 2: the arena estimate.  A claimed node that would not fit is a
     * reason to leave the whole graph to LiteRT; aborting inside Invoke
     * would be worse.  The estimate is per maximal run because each run
     * becomes one delegate kernel with its own persistent buffers. */
    total = 0;
    {
        int start = -1;
        for (i = 0; i <= plan->size; ++i) {
            const int on = i < plan->size && claims[i].claimed;
            if (on && start < 0)
                start = i;
            if (!on && start >= 0) {
                total += estimate_run(&claims[start], i - start, ctx->tensors,
                                      ctx->tensors_size, seen);
                start = -1;
            }
        }
    }
    if (total > budget) {
        report(ctx,
               "planned partitions need %zu B of a %zu B budget; declining every "
               "node so LiteRT keeps the graph", total, budget);
        for (i = 0; i < plan->size; ++i)
            claims[i].claimed = 0;
    }

    /* Cache the accepted plans by node index for the kernel hooks. */
    free(d->node_plans);
    free(d->node_claimed);
    d->node_plans = NULL;
    d->node_claimed = NULL;
    d->num_nodes = 0;
    for (i = 0; i < plan->size; ++i)
        if (plan->data[i] + 1 > d->num_nodes)
            d->num_nodes = plan->data[i] + 1;
    if (d->num_nodes) {
        d->node_plans = (vc4_node_plan *)calloc((size_t)d->num_nodes,
                                                sizeof(vc4_node_plan));
        d->node_claimed = (unsigned char *)calloc((size_t)d->num_nodes, 1);
        if (!d->node_plans || !d->node_claimed) {
            report(ctx, "out of memory for the node plan table");
            goto out;
        }
    }
    for (i = 0; i < plan->size; ++i) {
        if (claims[i].claimed) {
            d->node_plans[plan->data[i]] = claims[i].plan;
            d->node_claimed[plan->data[i]] = 1;
        }
    }

    /* Pass 3: replace maximal runs of claimed nodes. */
    run_start = -1;
    for (i = 0; i <= plan->size; ++i) {
        const int on = i < plan->size && claims[i].claimed;
        if (on && run_start < 0)
            run_start = i;
        if (!on && run_start >= 0) {
            const int n = i - run_start;
            TfLiteIntArray *nodes = intarray_new(n);
            TfLiteRegistration reg = make_registration();
            int j;
            if (!nodes) {
                report(ctx, "out of memory for a partition list");
                goto out;
            }
            for (j = 0; j < n; ++j) {
                nodes->data[j] = plan->data[run_start + j];
                d->claimed++;
            }
            if (d->verbose || plan_mode)
                fprintf(stderr, "VC4_DELEGATE partition=%d nodes=%d\n",
                        d->partitions, n);
            d->partitions++;
            if (!plan_mode &&
                ctx->ReplaceNodeSubsetsWithDelegateKernels(ctx, reg, nodes,
                                                           del) != kTfLiteOk) {
                free(nodes);
                report(ctx, "ReplaceNodeSubsetsWithDelegateKernels failed");
                goto out;
            }
            free(nodes);
            run_start = -1;
        }
    }
    if (d->verbose || plan_mode)
        fprintf(stderr,
                "VC4_DELEGATE summary claimed=%d declined=%d partitions=%d "
                "mode=%s api=%s version=%s\n",
                d->claimed, d->declined, d->partitions, d->mode,
                "external-delegate/TfLiteDelegate", VC4_DELEGATE_VERSION);

    rc = kTfLiteOk;
out:
    free(claims);
    free(seen);
    return rc;
}

/* ------------------------------------------------------------ plugin entry */

TfLiteDelegate *tflite_plugin_create_delegate(const char *const *options_keys,
                                             const char *const *options_values,
                                             size_t num_options,
                                             void (*report_error)(const char *))
{
    Vc4Delegate *d;
    size_t i;

    d = (Vc4Delegate *)calloc(1, sizeof(*d));
    if (!d) {
        plugin_error(report_error, "out of memory");
        return NULL;
    }
    d->mode = dup_opt("auto");
    d->arena_bytes = VC4_ARENA_DEFAULT;
    d->base.data_ = d;
    d->base.Prepare = delegate_prepare;
    /* flags = kTfLiteDelegateFlagsNone (0): the delegate supports only the
     * static shapes the engine can plan for. */
    d->base.flags = 0;

    for (i = 0; i < num_options; ++i) {
        const char *key = options_keys[i];
        const char *val = options_values[i];
        if (!key || !val) {
            plugin_error(report_error, "null option");
            goto fail;
        }
        if (strcmp(key, "mode") == 0) {
            free(d->mode);
            d->mode = dup_opt(val);
            if (strcmp(val, "auto") != 0 && strcmp(val, "plan") != 0 &&
                strcmp(val, "off") != 0 && strcmp(val, "dry") != 0) {
                plugin_error(report_error,
                             "mode must be auto, plan, dry or off (got '%s')",
                             val);
                goto fail;
            }
            d->dry = strcmp(val, "dry") == 0;
        } else if (strcmp(key, "arena_mb") == 0) {
            long mb = strtol(val, NULL, 10);
            if (mb < 8 || mb > 512) {
                plugin_error(report_error, "arena_mb out of range: %s", val);
                goto fail;
            }
            d->arena_bytes = (size_t)mb << 20;
        } else if (strcmp(key, "cache_dir") == 0) {
            free(d->cache_dir);
            d->cache_dir = dup_opt(val);
        } else if (strcmp(key, "verbose") == 0) {
            d->verbose = atoi(val) != 0;
        } else {
            plugin_error(report_error, "unknown option '%s'", key);
            goto fail;
        }
    }
    if (!d->mode) {
        plugin_error(report_error, "out of memory");
        goto fail;
    }
    return &d->base;

fail:
    free(d->mode);
    free(d->cache_dir);
    free(d);
    return NULL;
}

void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate)
{
    Vc4Delegate *d;
    if (!delegate)
        return;
    d = (Vc4Delegate *)delegate->data_;
    if (d) {
        free(d->mode);
        free(d->cache_dir);
        free(d->node_plans);
        free(d->node_claimed);
        free(d);
    }
    /* The engine context is shared by every kernel of every interpreter in
     * the process and is released by process exit. */
}
