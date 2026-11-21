/*
 * vc4_plan.c - the claim table of the VC4 LiteRT delegate.
 *
 * Every predicate here mirrors a constraint in infer/include/vc4infer.h or in
 * the kernel source it names.  When a bound is not provable from the TFLite
 * descriptors alone (SAME padding that the engine's single `pad` cannot
 * express, an asymmetric quantized tensor, a filter that is not a constant)
 * the node is refused rather than approximated.
 */
#include "vc4_plan.h"

#include <limits.h>
#include <string.h>

#include "tflite/builtin_ops.h"
#include "tflite/core/c/builtin_op_data.h"

#define VC4_MAX_DIMS 8

static size_t al16(size_t v) { return (v + 15u) & ~(size_t)15u; }

static size_t al256(size_t v) { return (v + 255u) & ~(size_t)255u; }

static const char *k_reasons[] = {
    "ok",
    "op-not-claimed",
    "dtype",
    "dynamic-shape",
    "shape",
    "layout",
    "param",
    "activation",
    "broadcast",
    "arena-size",
};

const char *vc4_plan_reason(vc4_claim_status st)
{
    if ((unsigned)st < sizeof(k_reasons) / sizeof(k_reasons[0]))
        return k_reasons[st];
    return "unknown";
}

const char *vc4_plan_builtin_name(int builtin)
{
    switch (builtin) {
    case kTfLiteBuiltinAdd: return "ADD";
    case kTfLiteBuiltinConv2d: return "CONV_2D";
    case kTfLiteBuiltinDepthwiseConv2d: return "DEPTHWISE_CONV_2D";
    case kTfLiteBuiltinDiv: return "DIV";
    case kTfLiteBuiltinFullyConnected: return "FULLY_CONNECTED";
    case kTfLiteBuiltinLogistic: return "LOGISTIC";
    case kTfLiteBuiltinMul: return "MUL";
    case kTfLiteBuiltinRelu: return "RELU";
    case kTfLiteBuiltinRelu6: return "RELU6";
    case kTfLiteBuiltinReluN1To1: return "RELU_N1_TO_1";
    case kTfLiteBuiltinSoftmax: return "SOFTMAX";
    case kTfLiteBuiltinSub: return "SUB";
    case kTfLiteBuiltinTanh: return "TANH";
    case kTfLiteBuiltinAveragePool2d: return "AVERAGE_POOL_2D";
    case kTfLiteBuiltinMaxPool2d: return "MAX_POOL_2D";
    case kTfLiteBuiltinReshape: return "RESHAPE";
    case kTfLiteBuiltinSin: return "SIN";
    case kTfLiteBuiltinLogSoftmax: return "LOG_SOFTMAX";
    case kTfLiteBuiltinConcatenation: return "CONCATENATION";
    default: return "OTHER";
    }
}

/* A tensor is usable only if it is dense fp32 without any quantization and
 * every extent is a positive constant.  `bytes == 4 * numel` is what makes
 * the delegate's arena buffer a byte-for-byte image of the TFLite tensor: a
 * padded or strided tensor is refused, not guessed at. */
static vc4_claim_status check_tensor(const TfLiteTensor *t)
{
    int64_t n = 1;
    int i;

    if (!t || !t->dims)
        return VC4_CLAIM_SHAPE;
    if (t->type != kTfLiteFloat32)
        return VC4_CLAIM_DTYPE;
    if (t->quantization.type != kTfLiteNoQuantization)
        return VC4_CLAIM_DTYPE;
    if (t->dims->size < 1 || t->dims->size > VC4_MAX_DIMS)
        return VC4_CLAIM_SHAPE;
    for (i = 0; i < t->dims->size; ++i) {
        const int d = t->dims->data[i];
        if (d <= 0)
            return VC4_CLAIM_DYNAMIC;
        if (n > (int64_t)(INT_MAX / 4) / d)
            return VC4_CLAIM_SHAPE;
        n *= d;
    }
    if (t->bytes != (size_t)n * sizeof(float))
        return VC4_CLAIM_LAYOUT;
    /* A kTfLiteDynamic tensor is resized/reallocated by the runtime; the
     * delegate's arena image would go stale. */
    if (t->allocation_type == kTfLiteDynamic)
        return VC4_CLAIM_DYNAMIC;
    if (t->dims_signature && t->dims_signature->size > 0) {
        for (i = 0; i < t->dims_signature->size; ++i)
            if (t->dims_signature->data[i] <= 0)
                return VC4_CLAIM_DYNAMIC;
    }
    return VC4_CLAIM_OK;
}

static int64_t tensor_elems(const TfLiteTensor *t)
{
    int64_t n = 1;
    int i;
    for (i = 0; i < t->dims->size; ++i)
        n *= t->dims->data[i];
    return n;
}

static vc4_claim_status get_tensor(const TfLiteTensor *tensors, size_t ntensors,
                                   int index, const TfLiteTensor **out)
{
    if (index == kTfLiteOptionalTensor) {
        *out = NULL;
        return VC4_CLAIM_OK;
    }
    if (index < 0 || (size_t)index >= ntensors)
        return VC4_CLAIM_SHAPE;
    *out = &tensors[index];
    return check_tensor(*out);
}

static int same_shape(const TfLiteTensor *a, const TfLiteTensor *b)
{
    int i;
    if (!a || !b || a->dims->size != b->dims->size)
        return 0;
    for (i = 0; i < a->dims->size; ++i)
        if (a->dims->data[i] != b->dims->data[i])
            return 0;
    return 1;
}

static vc4_claim_status check_activation(int act)
{
    switch (act) {
    case kTfLiteActNone:
    case kTfLiteActRelu:
    case kTfLiteActRelu6:
        return VC4_CLAIM_OK;
    default:
        return VC4_CLAIM_ACTIVATION;
    }
}

/* The engine takes one symmetric `pad`, so a SAME window whose total padding
 * is odd (3x3 stride 2 on an even extent, for example) is refused. */
static vc4_claim_status same_pad(int in, int k, int stride, int out,
                                 int *pad_out)
{
    int total = (out - 1) * stride + k - in;
    if (total < 0)
        total = 0;
    if (total % 2)
        return VC4_CLAIM_PARAM;
    *pad_out = total / 2;
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_conv(const TfLiteRegistration *reg,
                                  const TfLiteNode *node,
                                  const TfLiteTensor *tensors, size_t ntensors,
                                  vc4_node_plan *p)
{
    const TfLiteConvParams *params = (const TfLiteConvParams *)node->builtin_data;
    const TfLiteTensor *in, *flt, *bias, *out;
    vc4_claim_status st;
    int ho, wo;

    (void)reg;
    if (!params)
        return VC4_CLAIM_PARAM;
    if (node->inputs->size != 3 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &in)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[1], &flt)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[2], &bias)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!in || !flt || !out || in->dims->size != 4 || flt->dims->size != 4 ||
        out->dims->size != 4)
        return VC4_CLAIM_SHAPE;
    if (params->dilation_width_factor != 1 || params->dilation_height_factor != 1)
        return VC4_CLAIM_PARAM;
    if (params->stride_width != params->stride_height ||
        (params->stride_width != 1 && params->stride_width != 2))
        return VC4_CLAIM_PARAM;
    if ((st = check_activation(params->activation)))
        return st;

    p->kind = VC4_STEP_CONV;
    p->builtin = kTfLiteBuiltinConv2d;
    p->in0 = node->inputs->data[0];
    p->in1 = node->inputs->data[1];
    p->in2 = node->inputs->data[2];
    p->out = node->outputs->data[0];
    p->activation = params->activation;
    p->is_depthwise = 0;
    p->is_fc = 0;
    p->stride = params->stride_width;
    p->nimg = in->dims->data[0];
    p->H = in->dims->data[1];
    p->W = in->dims->data[2];
    p->Cin = in->dims->data[3];
    p->KH = flt->dims->data[1];
    p->KW = flt->dims->data[2];
    p->Cout = flt->dims->data[0];

    if (flt->dims->data[3] != p->Cin)
        return VC4_CLAIM_SHAPE;
    if (p->Cin % 16 || p->Cout % 16)
        return VC4_CLAIM_SHAPE;
    if (!((p->KH == 1 || p->KH == 3 || p->KH == 5) &&
          (p->KW == 1 || p->KW == 3 || p->KW == 5)))
        return VC4_CLAIM_PARAM;
    /* VDR pitch of one 16-pixel im2col row. */
    if ((size_t)p->stride * (size_t)p->Cin * 4u > 8191u)
        return VC4_CLAIM_PARAM;
    if (flt->allocation_type != kTfLiteMmapRo)
        return VC4_CLAIM_PARAM;

    if (params->padding == kTfLitePaddingValid) {
        p->pad = 0;
        ho = (p->H - p->KH) / p->stride + 1;
        wo = (p->W - p->KW) / p->stride + 1;
    } else if (params->padding == kTfLitePaddingSame) {
        ho = (p->H + p->stride - 1) / p->stride;
        wo = (p->W + p->stride - 1) / p->stride;
        if ((st = same_pad(p->H, p->KH, p->stride, ho, &p->pad)))
            return st;
        {
            int padw = 0;
            if ((st = same_pad(p->W, p->KW, p->stride, wo, &padw)))
                return st;
            if (padw != p->pad)
                return VC4_CLAIM_PARAM;
        }
    } else {
        return VC4_CLAIM_PARAM;
    }
    if (ho <= 0 || wo <= 0)
        return VC4_CLAIM_SHAPE;
    if (out->dims->data[0] != p->nimg || out->dims->data[1] != ho ||
        out->dims->data[2] != wo || out->dims->data[3] != p->Cout)
        return VC4_CLAIM_SHAPE;

    if (bias) {
        if (bias->dims->size != 1 || bias->dims->data[0] != p->Cout)
            return VC4_CLAIM_SHAPE;
        if (bias->allocation_type != kTfLiteMmapRo)
            return VC4_CLAIM_PARAM;
    }
    p->ho = ho;
    p->wo = wo;
    p->Wg = (int)al16((size_t)wo);
    p->out_bytes = (size_t)p->nimg * ho * wo * p->Cout * 4u;
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_depthwise(const TfLiteRegistration *reg,
                                       const TfLiteNode *node,
                                       const TfLiteTensor *tensors,
                                       size_t ntensors, vc4_node_plan *p)
{
    const TfLiteDepthwiseConvParams *params =
        (const TfLiteDepthwiseConvParams *)node->builtin_data;
    const TfLiteTensor *in, *flt, *bias, *out;
    vc4_claim_status st;

    (void)reg;
    if (!params)
        return VC4_CLAIM_PARAM;
    if (node->inputs->size != 3 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &in)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[1], &flt)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[2], &bias)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!in || !flt || !out || in->dims->size != 4 || flt->dims->size != 4 ||
        out->dims->size != 4)
        return VC4_CLAIM_SHAPE;
    if (params->padding != kTfLitePaddingSame ||
        params->stride_width != 1 || params->stride_height != 1 ||
        params->dilation_width_factor != 1 ||
        params->dilation_height_factor != 1)
        return VC4_CLAIM_PARAM;
    if ((st = check_activation(params->activation)))
        return st;

    p->kind = VC4_STEP_CONV;
    p->builtin = kTfLiteBuiltinDepthwiseConv2d;
    p->in0 = node->inputs->data[0];
    p->in1 = node->inputs->data[1];
    p->in2 = node->inputs->data[2];
    p->out = node->outputs->data[0];
    p->activation = params->activation;
    p->is_depthwise = 1;
    p->is_fc = 0;
    p->stride = 1;
    p->pad = 1;
    p->nimg = in->dims->data[0];
    p->H = in->dims->data[1];
    p->W = in->dims->data[2];
    p->Cin = in->dims->data[3];
    p->Cout = p->Cin;
    p->KH = flt->dims->data[1];
    p->KW = flt->dims->data[2];

    /* TFLite depthwise filter: [1][KH][KW][C].  The engine is 3x3/pad 1 only. */
    if (flt->dims->data[0] != 1 || p->KH != 3 || p->KW != 3 ||
        flt->dims->data[3] != p->Cin)
        return VC4_CLAIM_PARAM;
    if (params->depth_multiplier != 1)
        return VC4_CLAIM_PARAM;
    if (p->Cin % 16)
        return VC4_CLAIM_SHAPE;
    if (flt->allocation_type != kTfLiteMmapRo)
        return VC4_CLAIM_PARAM;
    if (out->dims->data[0] != p->nimg || out->dims->data[1] != p->H ||
        out->dims->data[2] != p->W || out->dims->data[3] != p->Cin)
        return VC4_CLAIM_SHAPE;
    /* vc4_depthwise_pack_w has no bias slot.  An all-zero bias is the same
     * arithmetic as no bias; anything else would silently drop a term. */
    if (bias) {
        const float *b = (const float *)bias->data.data;
        int i;
        if (bias->dims->size != 1 || bias->dims->data[0] != p->Cin)
            return VC4_CLAIM_SHAPE;
        if (!b || bias->allocation_type != kTfLiteMmapRo)
            return VC4_CLAIM_PARAM;
        for (i = 0; i < p->Cin; ++i)
            if (b[i] != 0.0f)
                return VC4_CLAIM_PARAM;
    }
    p->ho = p->H;
    p->wo = p->W;
    p->Wg = (int)al16((size_t)p->W);
    p->out_bytes = (size_t)p->nimg * p->H * p->W * p->Cin * 4u;
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_fc(const TfLiteRegistration *reg,
                                const TfLiteNode *node,
                                const TfLiteTensor *tensors, size_t ntensors,
                                vc4_node_plan *p)
{
    const TfLiteFullyConnectedParams *params =
        (const TfLiteFullyConnectedParams *)node->builtin_data;
    const TfLiteTensor *in, *flt, *bias, *out;
    vc4_claim_status st;

    (void)reg;
    if (!params)
        return VC4_CLAIM_PARAM;
    if (node->inputs->size != 3 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &in)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[1], &flt)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[2], &bias)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!in || !flt || !out || in->dims->size != 2 || flt->dims->size != 2 ||
        out->dims->size != 2)
        return VC4_CLAIM_SHAPE;
    if (params->keep_num_dims ||
        params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault)
        return VC4_CLAIM_PARAM;
    if ((st = check_activation(params->activation)))
        return st;
    if (flt->allocation_type != kTfLiteMmapRo)
        return VC4_CLAIM_PARAM;

    p->kind = VC4_STEP_CONV;
    p->builtin = kTfLiteBuiltinFullyConnected;
    p->in0 = node->inputs->data[0];
    p->in1 = node->inputs->data[1];
    p->in2 = node->inputs->data[2];
    p->out = node->outputs->data[0];
    p->activation = params->activation;
    p->is_depthwise = 0;
    p->is_fc = 1;
    p->nimg = in->dims->data[0];
    p->H = 1;
    p->W = 1;
    p->Cin = in->dims->data[1];
    p->Cout = flt->dims->data[0];
    p->KH = 1;
    p->KW = 1;
    p->stride = 1;
    p->pad = 0;

    if (flt->dims->data[1] != p->Cin)
        return VC4_CLAIM_SHAPE;
    if (p->Cin % 16 || p->Cout % 16)
        return VC4_CLAIM_SHAPE;
    if (out->dims->data[0] != p->nimg || out->dims->data[1] != p->Cout)
        return VC4_CLAIM_SHAPE;
    if (bias) {
        if (bias->dims->size != 1 || bias->dims->data[0] != p->Cout)
            return VC4_CLAIM_SHAPE;
        if (bias->allocation_type != kTfLiteMmapRo)
            return VC4_CLAIM_PARAM;
    }
    p->ho = 1;
    p->wo = 1;
    p->Wg = 16;
    p->out_bytes = (size_t)p->nimg * p->Cout * 4u;
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_binary(const TfLiteNode *node,
                                    const TfLiteTensor *tensors,
                                    size_t ntensors, vc4_node_plan *p,
                                    int builtin)
{
    const TfLiteTensor *a, *b, *out;
    vc4_claim_status st;
    int act = kTfLiteActNone;

    if (node->inputs->size != 2 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &a)) ||
        (st = get_tensor(tensors, ntensors, node->inputs->data[1], &b)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!a || !b || !out)
        return VC4_CLAIM_SHAPE;
    if (!same_shape(a, out) || !same_shape(b, out))
        return VC4_CLAIM_BROADCAST;

    if (node->builtin_data) {
        switch (builtin) {
        case kTfLiteBuiltinAdd:
            act = ((const TfLiteAddParams *)node->builtin_data)->activation;
            break;
        case kTfLiteBuiltinMul:
            act = ((const TfLiteMulParams *)node->builtin_data)->activation;
            break;
        case kTfLiteBuiltinSub:
            act = ((const TfLiteSubParams *)node->builtin_data)->activation;
            break;
        case kTfLiteBuiltinDiv:
            act = ((const TfLiteDivParams *)node->builtin_data)->activation;
            break;
        default:
            return VC4_CLAIM_UNSUPPORTED_OP;
        }
    }
    /* The engine's elementwise kernels have no fused-activation variant, and
     * a claimed node may not leave a term to the host. */
    if (act != kTfLiteActNone)
        return VC4_CLAIM_ACTIVATION;

    p->kind = VC4_STEP_BINARY;
    p->builtin = builtin;
    p->in0 = node->inputs->data[0];
    p->in1 = node->inputs->data[1];
    p->in2 = -1;
    p->out = node->outputs->data[0];
    p->n = (size_t)tensor_elems(out);
    p->out_bytes = out->bytes;
    switch (builtin) {
    case kTfLiteBuiltinAdd: p->binop = VC4_PLAN_B_ADD; break;
    case kTfLiteBuiltinMul: p->binop = VC4_PLAN_B_MUL; break;
    case kTfLiteBuiltinSub: p->binop = VC4_PLAN_B_SUB; break;
    default: p->binop = VC4_PLAN_B_DIV; break;
    }
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_unary(const TfLiteNode *node,
                                   const TfLiteTensor *tensors,
                                   size_t ntensors, vc4_node_plan *p,
                                   int builtin)
{
    const TfLiteTensor *in, *out;
    vc4_claim_status st;

    if (node->inputs->size != 1 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &in)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!in || !out || !same_shape(in, out))
        return VC4_CLAIM_SHAPE;

    p->kind = VC4_STEP_UNARY;
    p->builtin = builtin;
    p->in0 = node->inputs->data[0];
    p->in1 = -1;
    p->in2 = -1;
    p->out = node->outputs->data[0];
    p->n = (size_t)tensor_elems(out);
    p->out_bytes = out->bytes;
    switch (builtin) {
    case kTfLiteBuiltinRelu: p->unop = VC4_PLAN_U_RELU; break;
    case kTfLiteBuiltinRelu6: p->unop = VC4_PLAN_U_RELU6; break;
    case kTfLiteBuiltinReluN1To1: p->unop = VC4_PLAN_U_RELU_N1TO1; break;
    case kTfLiteBuiltinLogistic: p->unop = VC4_PLAN_U_SIGMOID; break;
    case kTfLiteBuiltinTanh: p->unop = VC4_PLAN_U_TANH; break;
    default: return VC4_CLAIM_UNSUPPORTED_OP;
    }
    return VC4_CLAIM_OK;
}

static vc4_claim_status plan_softmax(const TfLiteNode *node,
                                     const TfLiteTensor *tensors,
                                     size_t ntensors, vc4_node_plan *p)
{
    const TfLiteSoftmaxParams *params =
        (const TfLiteSoftmaxParams *)node->builtin_data;
    const TfLiteTensor *in, *out;
    vc4_claim_status st;
    int64_t n;

    if (node->inputs->size != 1 || node->outputs->size != 1)
        return VC4_CLAIM_SHAPE;
    if ((st = get_tensor(tensors, ntensors, node->inputs->data[0], &in)) ||
        (st = get_tensor(tensors, ntensors, node->outputs->data[0], &out)))
        return st;
    if (!in || !out || !same_shape(in, out) || in->dims->size < 2)
        return VC4_CLAIM_SHAPE;
    if (!params || params->beta != 1.0f)
        return VC4_CLAIM_PARAM;

    p->cols = in->dims->data[in->dims->size - 1];
    n = tensor_elems(in);
    p->rows = (int)(n / p->cols);
    /* vc4_softmax uses the numerically stable fused row kernel only for
     * <= 16 classes with rows % 8 == 0; the generic path does not subtract
     * the row maximum, so it is not a drop-in for TFLite's softmax. */
    if (p->cols < 1 || p->cols > 16)
        return VC4_CLAIM_PARAM;
    if (p->rows < 1 || p->rows % 8 != 0)
        return VC4_CLAIM_PARAM;

    p->kind = VC4_STEP_SOFTMAX;
    p->builtin = kTfLiteBuiltinSoftmax;
    p->in0 = node->inputs->data[0];
    p->in1 = -1;
    p->in2 = -1;
    p->out = node->outputs->data[0];
    p->n = (size_t)n;
    p->out_bytes = out->bytes;
    return VC4_CLAIM_OK;
}

vc4_claim_status vc4_plan_node(const TfLiteRegistration *reg,
                               const TfLiteNode *node,
                               const TfLiteTensor *tensors, size_t ntensors,
                               vc4_node_plan *plan)
{
    vc4_claim_status st;

    if (!reg || !node || !node->inputs || !node->outputs || !plan)
        return VC4_CLAIM_SHAPE;
    memset(plan, 0, sizeof(*plan));
    plan->in0 = plan->in1 = plan->in2 = -1;
    plan->out = -1;

    switch (reg->builtin_code) {
    case kTfLiteBuiltinConv2d:
        st = plan_conv(reg, node, tensors, ntensors, plan);
        break;
    case kTfLiteBuiltinDepthwiseConv2d:
        st = plan_depthwise(reg, node, tensors, ntensors, plan);
        break;
    case kTfLiteBuiltinFullyConnected:
        st = plan_fc(reg, node, tensors, ntensors, plan);
        break;
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinDiv:
        st = plan_binary(node, tensors, ntensors, plan, reg->builtin_code);
        break;
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinTanh:
        st = plan_unary(node, tensors, ntensors, plan, reg->builtin_code);
        break;
    case kTfLiteBuiltinSoftmax:
        st = plan_softmax(node, tensors, ntensors, plan);
        break;
    default:
        st = VC4_CLAIM_UNSUPPORTED_OP;
        break;
    }
    if (st != VC4_CLAIM_OK)
        memset(plan, 0, sizeof(*plan));
    return st;
}

void vc4_plan_estimate_node(const vc4_node_plan *p, vc4_plan_estimate *est)
{
    size_t kp, panel, padimg;

    memset(est, 0, sizeof(*est));
    if (!p)
        return;
    switch (p->kind) {
    case VC4_STEP_CONV:
        kp = p->is_depthwise ? al16((size_t)9 * p->Cin)
                             : al16((size_t)p->Cin * p->KH * p->KW);
        est->persistent = p->is_depthwise
                              ? (size_t)p->Cin * kp * 4u
                              : (size_t)p->Cout * (kp + (p->in2 >= 0 ? 16u : 0u)) * 4u;
        if (p->activation != kTfLiteActNone)
            est->persistent += p->out_bytes;
        est->pitch = (size_t)p->nimg * p->ho * p->Wg * p->Cout * 4u;
        padimg = (size_t)p->nimg * (p->H + 2 * p->pad) *
                 (p->W + 2 * p->pad) * p->Cin * 4u;
        panel = (size_t)p->nimg * p->ho * p->Wg * kp * 4u;
        /* padded image + activation panel + kernel slack */
        est->scratch = padimg + panel + 256u * 1024u;
        break;
    case VC4_STEP_UNARY:
    case VC4_STEP_BINARY:
        est->scratch = (p->n % 256u) ? 4u * al256(p->n) * 4u : 4096u;
        break;
    case VC4_STEP_SOFTMAX:
        est->scratch = 2u * p->n * 4u + 4096u;
        break;
    }
}
