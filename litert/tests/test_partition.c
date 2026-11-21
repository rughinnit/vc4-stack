/*
 * test_partition.c - the delegate's claim table, exercised without LiteRT or
 * a GPU.
 *
 * The test builds the same TfLiteTensor / TfLiteNode / TfLiteRegistration
 * structures the interpreter hands to a delegate and checks, for every op and
 * for the boundaries that matter, that the node is claimed only when the
 * engine can execute it exactly.  It links only vc4_plan.o: no interpreter,
 * no engine, no QPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vc4_plan.h"

#include "tflite/builtin_ops.h"
#include "tflite/core/c/builtin_op_data.h"

static int failures;
static int checks;

static void expect(int cond, const char *name)
{
    ++checks;
    if (cond) {
        printf("ok   %s\n", name);
    } else {
        printf("FAIL %s\n", name);
        ++failures;
    }
}

static TfLiteIntArray *ia(const int *d, int n)
{
    TfLiteIntArray *a =
        (TfLiteIntArray *)malloc(sizeof(TfLiteIntArray) + (size_t)n * sizeof(int));
    int i;
    a->size = n;
    for (i = 0; i < n; ++i)
        a->data[i] = d[i];
    return a;
}

typedef struct {
    TfLiteTensor t[8];
    TfLiteIntArray *dims[8];
    int nt;
    TfLiteNode node;
    TfLiteIntArray *ins;
    TfLiteIntArray *outs;
    TfLiteRegistration reg;
    TfLiteConvParams conv;
    TfLiteDepthwiseConvParams dw;
    TfLiteFullyConnectedParams fc;
    TfLiteAddParams add;
    TfLiteMulParams mul;
    TfLiteSoftmaxParams soft;
} Case;

static void case_init(Case *c)
{
    memset(c, 0, sizeof(*c));
}

static void case_free(Case *c)
{
    int i;
    for (i = 0; i < c->nt; ++i)
        free(c->dims[i]);
    free(c->ins);
    free(c->outs);
    if (c->t[1].data.data && c->t[1].allocation_type == kTfLiteMmapRo)
        free(c->t[1].data.data);
    if (c->t[2].data.data && c->t[2].allocation_type == kTfLiteMmapRo)
        free(c->t[2].data.data);
    memset(c, 0, sizeof(*c));
}

/* Adds a tensor and returns its index.  `bytes_extra` perturbs the byte count
 * to exercise the dense-layout rejection. */
static int add_tensor(Case *c, const int *d, int nd, TfLiteType ty,
                      TfLiteAllocationType at, size_t extra)
{
    const int i = c->nt++;
    int64_t n = 1;
    int k;
    for (k = 0; k < nd; ++k)
        n *= d[k];
    c->dims[i] = ia(d, nd);
    c->t[i].dims = c->dims[i];
    c->t[i].type = ty;
    c->t[i].allocation_type = at;
    c->t[i].quantization.type = kTfLiteNoQuantization;
    c->t[i].bytes = (size_t)n * 4u + extra;
    if (at == kTfLiteMmapRo) {
        c->t[i].data.data = calloc((size_t)n + 4u, 4u);
    }
    return i;
}

static void set_node(Case *c, const int *ins, int nin, int out)
{
    c->ins = ia(ins, nin);
    c->outs = ia(&out, 1);
    c->node.inputs = c->ins;
    c->node.outputs = c->outs;
}

static vc4_claim_status claim_plan(Case *c, int builtin, vc4_node_plan *plan)
{
    vc4_node_plan local;
    c->reg.builtin_code = builtin;
    return vc4_plan_node(&c->reg, &c->node, c->t, (size_t)c->nt,
                         plan ? plan : &local);
}

static vc4_claim_status claim(Case *c, int builtin)
{
    return claim_plan(c, builtin, NULL);
}

static void conv_case(Case *c, int n, int h, int w, int cin, int cout, int kh,
                      int kw, int stride, TfLitePadding pad, int act)
{
    int in_d[4] = {n, h, w, cin};
    int f_d[4] = {cout, kh, kw, cin};
    int b_d[1] = {cout};
    int out_d[4] = {n, 0, 0, cout};
    int ins[3], out;
    int ho = pad == kTfLitePaddingSame ? (h + stride - 1) / stride
                                       : (h - kh) / stride + 1;
    int wo = pad == kTfLitePaddingSame ? (w + stride - 1) / stride
                                       : (w - kw) / stride + 1;
    out_d[1] = ho;
    out_d[2] = wo;
    case_init(c);
    ins[0] = add_tensor(c, in_d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
    ins[1] = add_tensor(c, f_d, 4, kTfLiteFloat32, kTfLiteMmapRo, 0);
    ins[2] = add_tensor(c, b_d, 1, kTfLiteFloat32, kTfLiteMmapRo, 0);
    out = add_tensor(c, out_d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
    set_node(c, ins, 3, out);
    memset(&c->conv, 0, sizeof(c->conv));
    c->conv.padding = pad;
    c->conv.stride_width = c->conv.stride_height = stride;
    c->conv.dilation_width_factor = c->conv.dilation_height_factor = 1;
    c->conv.activation = act;
    c->node.builtin_data = &c->conv;
}

int main(void)
{
    Case c;
    vc4_node_plan plan;
    vc4_claim_status st;

    /* ---- CONV_2D ---------------------------------------------------- */
    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    st = claim_plan(&c, kTfLiteBuiltinConv2d, &plan);
    expect(st == VC4_CLAIM_OK, "conv 3x3 SAME stride 1 fp32 is claimed");
    expect(plan.pad == 1 && plan.ho == 8 && plan.wo == 8 && plan.Wg == 16,
           "conv 3x3 SAME geometry (pad 1, 8x8, align16(8))");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 5, 5, 1, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_OK,
           "conv 5x5 SAME is claimed");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 2, 2, 1, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_PARAM,
           "conv 2x2 is refused (engine is 1/3/5 only)");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingValid, kTfLiteActRelu);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_OK,
           "conv VALID with fused RELU is claimed");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActTanh);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_ACTIVATION,
           "conv with fused TANH is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 3, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_SHAPE,
           "conv with Cin=3 is refused (engine needs 16-aligned channels)");
    case_free(&c);

    conv_case(&c, 1, 10, 10, 16, 16, 3, 3, 2, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_PARAM,
           "conv 3x3 stride 2 SAME on an even extent is refused (asymmetric pad)");
    case_free(&c);

    conv_case(&c, 1, 9, 9, 16, 16, 3, 3, 2, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_OK,
           "conv 3x3 stride 2 SAME on an odd extent is claimed (pad 1)");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 2, kTfLitePaddingValid, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_OK,
           "conv 3x3 stride 2 VALID is claimed");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 1, 1, kTfLitePaddingValid, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_OK,
           "conv 3x1 VALID is claimed (one pad serves both axes)");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 1, 1, kTfLitePaddingSame, kTfLiteActNone);
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_PARAM,
           "conv 3x1 SAME is refused (pad 1 vs 0 cannot share one pad)");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.conv.dilation_width_factor = 2;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_PARAM,
           "conv with dilation 2 is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.t[0].type = kTfLiteInt8;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_DTYPE,
           "int8 input is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.t[1].allocation_type = kTfLiteArenaRw;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_PARAM,
           "non-constant filter is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.t[3].dims->data[3] = 32;
    c.t[3].bytes = (size_t)1 * 8 * 8 * 32 * 4;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_SHAPE,
           "output channel mismatch is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.t[0].bytes += 4;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_LAYOUT,
           "a tensor whose bytes are not numel*4 is refused");
    case_free(&c);

    conv_case(&c, 1, 8, 8, 16, 16, 3, 3, 1, kTfLitePaddingSame, kTfLiteActNone);
    c.t[0].dims->data[1] = -1;
    expect(claim(&c, kTfLiteBuiltinConv2d) == VC4_CLAIM_DYNAMIC,
           "a dynamic input extent is refused");
    case_free(&c);

    /* ---- DEPTHWISE_CONV_2D ------------------------------------------ */
    {
        int in_d[4] = {1, 8, 8, 16};
        int f_d[4] = {1, 3, 3, 16};
        int b_d[1] = {16};
        int o_d[4] = {1, 8, 8, 16};
        int ins[3], out;
        case_init(&c);
        ins[0] = add_tensor(&c, in_d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        ins[1] = add_tensor(&c, f_d, 4, kTfLiteFloat32, kTfLiteMmapRo, 0);
        ins[2] = add_tensor(&c, b_d, 1, kTfLiteFloat32, kTfLiteMmapRo, 0);
        out = add_tensor(&c, o_d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        set_node(&c, ins, 3, out);
        memset(&c.dw, 0, sizeof(c.dw));
        c.dw.padding = kTfLitePaddingSame;
        c.dw.stride_width = c.dw.stride_height = 1;
        c.dw.depth_multiplier = 1;
        c.dw.dilation_width_factor = c.dw.dilation_height_factor = 1;
        c.dw.activation = kTfLiteActNone;
        c.node.builtin_data = &c.dw;
        expect(claim(&c, kTfLiteBuiltinDepthwiseConv2d) == VC4_CLAIM_OK,
               "depthwise 3x3 SAME stride 1 with a zero bias is claimed");
        ((float *)c.t[2].data.data)[3] = 1.0f;
        expect(claim(&c, kTfLiteBuiltinDepthwiseConv2d) == VC4_CLAIM_PARAM,
               "depthwise with a non-zero bias is refused");
        ((float *)c.t[2].data.data)[3] = 0.0f;
        c.dw.depth_multiplier = 2;
        expect(claim(&c, kTfLiteBuiltinDepthwiseConv2d) == VC4_CLAIM_PARAM,
               "depthwise with depth_multiplier 2 is refused");
        case_free(&c);
    }

    /* ---- FULLY_CONNECTED -------------------------------------------- */
    {
        int in_d[2] = {4, 16};
        int f_d[2] = {16, 16};
        int b_d[1] = {16};
        int o_d[2] = {4, 16};
        int ins[3], out;
        case_init(&c);
        ins[0] = add_tensor(&c, in_d, 2, kTfLiteFloat32, kTfLiteArenaRw, 0);
        ins[1] = add_tensor(&c, f_d, 2, kTfLiteFloat32, kTfLiteMmapRo, 0);
        ins[2] = add_tensor(&c, b_d, 1, kTfLiteFloat32, kTfLiteMmapRo, 0);
        out = add_tensor(&c, o_d, 2, kTfLiteFloat32, kTfLiteArenaRw, 0);
        set_node(&c, ins, 3, out);
        memset(&c.fc, 0, sizeof(c.fc));
        c.fc.activation = kTfLiteActNone;
        c.fc.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
        c.node.builtin_data = &c.fc;
        expect(claim(&c, kTfLiteBuiltinFullyConnected) == VC4_CLAIM_OK,
               "FC [4,16]x[16,16] is claimed");
        c.fc.keep_num_dims = 1;
        expect(claim(&c, kTfLiteBuiltinFullyConnected) == VC4_CLAIM_PARAM,
               "FC with keep_num_dims is refused");
        c.fc.keep_num_dims = 0;
        c.t[0].dims->data[1] = 15;
        c.t[0].bytes = (size_t)4 * 15 * 4;
        c.t[1].dims->data[1] = 15;
        c.t[1].bytes = (size_t)16 * 15 * 4;
        expect(claim(&c, kTfLiteBuiltinFullyConnected) == VC4_CLAIM_SHAPE,
               "FC with K=15 is refused");
        case_free(&c);
    }

    /* ---- elementwise ------------------------------------------------- */
    {
        int d[4] = {1, 8, 8, 16};
        int small[4] = {1, 1, 1, 16};
        int ins[2], out;
        case_init(&c);
        ins[0] = add_tensor(&c, d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        ins[1] = add_tensor(&c, d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        out = add_tensor(&c, d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        set_node(&c, ins, 2, out);
        memset(&c.add, 0, sizeof(c.add));
        c.add.activation = kTfLiteActNone;
        c.node.builtin_data = &c.add;
        expect(claim(&c, kTfLiteBuiltinAdd) == VC4_CLAIM_OK,
               "same-shape ADD is claimed");
        memset(&c.mul, 0, sizeof(c.mul));
        c.mul.activation = kTfLiteActNone;
        c.node.builtin_data = &c.mul;
        expect(claim(&c, kTfLiteBuiltinMul) == VC4_CLAIM_OK,
               "same-shape MUL is claimed");
        c.mul.activation = kTfLiteActRelu;
        expect(claim(&c, kTfLiteBuiltinMul) == VC4_CLAIM_ACTIVATION,
               "MUL with a fused activation is refused");
        c.mul.activation = kTfLiteActNone;
        free(c.dims[1]);
        c.dims[1] = ia(small, 4);
        c.t[1].dims = c.dims[1];
        c.t[1].bytes = (size_t)16 * 4;
        expect(claim(&c, kTfLiteBuiltinAdd) == VC4_CLAIM_BROADCAST,
               "broadcast ADD is refused");
        case_free(&c);
    }

    /* ---- unary and softmax ------------------------------------------- */
    {
        int d[4] = {1, 8, 8, 16};
        int ins[1], out;
        case_init(&c);
        ins[0] = add_tensor(&c, d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        out = add_tensor(&c, d, 4, kTfLiteFloat32, kTfLiteArenaRw, 0);
        set_node(&c, ins, 1, out);
        expect(claim(&c, kTfLiteBuiltinRelu) == VC4_CLAIM_OK, "RELU is claimed");
        expect(claim(&c, kTfLiteBuiltinRelu6) == VC4_CLAIM_OK,
               "RELU6 is claimed");
        expect(claim(&c, kTfLiteBuiltinLogistic) == VC4_CLAIM_OK,
               "LOGISTIC is claimed");
        expect(claim(&c, kTfLiteBuiltinTanh) == VC4_CLAIM_OK, "TANH is claimed");
        expect(claim(&c, kTfLiteBuiltinMaxPool2d) == VC4_CLAIM_UNSUPPORTED_OP,
               "MAX_POOL_2D is not claimed");
        case_free(&c);
    }

    {
        int d[2] = {64, 16};
        int bad[2] = {12, 16};
        int wide[2] = {8, 32};
        int ins[1], out;
        case_init(&c);
        ins[0] = add_tensor(&c, d, 2, kTfLiteFloat32, kTfLiteArenaRw, 0);
        out = add_tensor(&c, d, 2, kTfLiteFloat32, kTfLiteArenaRw, 0);
        set_node(&c, ins, 1, out);
        memset(&c.soft, 0, sizeof(c.soft));
        c.soft.beta = 1.0f;
        c.node.builtin_data = &c.soft;
        expect(claim(&c, kTfLiteBuiltinSoftmax) == VC4_CLAIM_OK,
               "softmax 64x16 (rows % 8 == 0) is claimed");
        c.soft.beta = 2.0f;
        expect(claim(&c, kTfLiteBuiltinSoftmax) == VC4_CLAIM_PARAM,
               "softmax with beta != 1 is refused");
        c.soft.beta = 1.0f;
        free(c.dims[0]);
        free(c.dims[1]);
        c.dims[0] = ia(bad, 2);
        c.dims[1] = ia(bad, 2);
        c.t[0].dims = c.dims[0];
        c.t[1].dims = c.dims[1];
        c.t[0].bytes = c.t[1].bytes = (size_t)12 * 16 * 4;
        expect(claim(&c, kTfLiteBuiltinSoftmax) == VC4_CLAIM_PARAM,
               "softmax with rows % 8 != 0 is refused");
        free(c.dims[0]);
        free(c.dims[1]);
        c.dims[0] = ia(wide, 2);
        c.dims[1] = ia(wide, 2);
        c.t[0].dims = c.dims[0];
        c.t[1].dims = c.dims[1];
        c.t[0].bytes = c.t[1].bytes = (size_t)8 * 32 * 4;
        expect(claim(&c, kTfLiteBuiltinSoftmax) == VC4_CLAIM_PARAM,
               "softmax over 32 classes is refused (stable kernel is <= 16)");
        case_free(&c);
    }

    printf("\n%d checks, %d failure(s)\n", checks, failures);
    return failures ? 1 : 0;
}
