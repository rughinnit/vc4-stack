/*
 * vc4_api.c - the op registry and the coverage table.
 *
 * STRICT DISPATCH: this table is the single source of truth.  An op that is
 * not implemented on the GPU is listed with supported = 0 and the entry point
 * returns VC4_ERR_UNSUPPORTED.  Nothing falls back to the CPU.
 *
 * `differentiable` says whether a backward entry point exists in this
 * library.  The two flags are independent on purpose:
 *   - a forward can be supported without a gradient ("batchnorm" is the
 *     opposite case: training only, no inference kernel here),
 *   - an op the ONNX runner composes on the host (pad, concat, split,
 *     avgpool) has no single forward kernel but does have a GPU backward.
 * vc4_op_supported() answers "can I run this forward on the QPU"; use
 * vc4_op_differentiable() to answer "can I train through it".
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <string.h>

static const vc4_op_info OPS[] = {
    /* implemented forwards */
    { "matmul", "qpu gemm(asm10) + qpu pack",
      1, "f32 row-major; M,N,K any size (8/64/16 tiles internally)", 1 },
    { "conv2d/3x3", "qpu im2col + qpu gemm(asm3)",
      1, "f32 NCHW; stride 1; pad 0..; Cin%16==0; Cout%16==0; Hp*Wp*4<=8191", 1 },
    { "conv2d/1x1", "qpu im2col + qpu gemm(asm3)",
      1, "f32 NCHW; stride 1; pad 0; Cin%16==0; Cout%16==0; H*Wp*4<=8191", 1 },
    { "add", "qpu elementwise", 1, "f32, exact IEEE; n rounded to 256 internally", 1 },
    { "sub", "qpu ewb(sub)", 1, "f32; n rounded to a 256-float group", 1 },
    { "mul", "qpu elementwise", 1, "f32, exact IEEE; n rounded to 256 internally", 1 },
    { "div", "qpu ewb(div)", 1, "f32; reciprocal + multiply (|err| < 3e-7)", 1 },
    { "max", "qpu ewb(max)", 1,
      "f32; one FPU FMAX per element, bit-exact selection; NaN follows FMAX", 1 },
    { "min", "qpu ewb(min)", 1,
      "f32; one FPU FMIN per element, bit-exact selection; NaN follows FMIN", 1 },
    { "relu", "qpu elementwise", 1, "f32, exact IEEE", 1 },
    { "sigmoid", "qpu elementwise", 1,
      "f32; 2^f degree-5 poly + 3x Newton rcp; |err| < 1e-6 (measured)", 1 },
    { "silu", "qpu elementwise", 1,
      "f32; x*sigmoid(x); |err| < 1e-6 (measured)", 1 },
    { "tanh", "qpu uop(tanh)", 1, "f32; exp2-based, |err| < 2e-6", 1 },
    { "exp", "qpu uop(exp)", 1, "f32; 2^(x log2 e), |err| < 1e-6", 1 },
    { "log", "qpu uop(log)", 1, "f32; exponent split + atanh series, |err| < 1e-6", 1 },
    { "sqrt", "qpu uop(sqrt)", 1, "f32; rsqrt seed + Newton", 1 },
    { "abs", "qpu uop(abs)", 1, "f32, bit mask", 1 },
    { "clip", "qpu uop(clip)", 1, "f32; lo/hi are op attributes", 1 },
    { "leakyrelu", "qpu uop(leakyrelu)", 1, "f32; alpha is an op attribute", 1 },
    { "hardswish", "qpu uop(hardswish)", 1, "f32; ONNX alpha=1/6 beta=0.5", 1 },
    { "hardsigmoid", "qpu uop(hardsigmoid)", 1, "f32; ONNX alpha=1/6 beta=0.5", 1 },
    { "gelu", "qpu uop(gelu)", 1, "f32; tanh approximation", 1 },
    { "gelu_erf", "qpu uop(gelu_erf)", 1, "f32; exact erf form", 1 },
    { "erf", "qpu uop(erf)", 1, "f32; A&S 7.1.26, |err| < 2e-7", 1 },
    { "relu_bwd", "qpu elementwise", 1,
      "f32; bit-exact mask: out = dy if x>0 else 0", 1 },
    { "rsqrt", "qpu elementwise", 1,
      "f32; 1/sqrt(x), bit-trick seed + 4 Newton (Adam); rel err < 2e-6", 0 },
    { "transpose", "qpu pack (VDW strided)", 1,
      "dense NxM transpose; rows,cols % 16 == 0; col*4 <= 8191", 1 },
    { "matmul_bt", "qpu pack + qpu gemm3(bmode=1)", 1,
      "C = A*B^T, f32; M,N,K % 16 == 0 (training dW / dX path)", 1 },
    { "softmax_ce_bwd", "qpu fused row kernel / generic composition", 1,
      "rows%8==0 for the 16-class fused kernel; vc4_softmax_ce_bwd takes any class count", 1 },
    { "im2col", "qpu im2col (row-major)", 1,
      "f32 NCHW; stride 1; KH=KW in {1,3}; Cin%16==0; Hp*Wp*4<=8191", 1 },
    { "col2im", "qpu scatter-add (fixed tap order)", 1,
      "dA [HO*Wp][Cin*KH*KW] -> padded dX; Cin%16==0; deterministic fp32", 1 },
    { "conv2d/bwd_dx", "qpu im2col_r + gemm3 + qpu col2im", 1,
      "dy+W -> dX; stride 1; KH=KW in {1,3}; Cin,Cout%16==0", 1 },
    { "conv2d/bwd_dw", "qpu im2col_r + qpu transpose + gemm3", 1,
      "dy+A -> dW (ci-major); stride 1; KH=KW in {1,3}; Cin,Cout%16==0", 1 },
    { "maxpool/2x2s2", "qpu max + store", 1,
      "f32 NCHW; H,W even; W%16==0; out row pitch padded to 16", 1 },
    { "maxpool_bwd", "qpu argmax mask", 1,
      "dy routed to the group maximum; ties share the gradient", 1 },
    { "maxpool/NxNsS", "qpu max + 4-row store", 1,
      "any N>=1, stride>=1, pad>=0; host pads the border with -inf (bytes only)", 1 },
    { "maxpool/NxNsS_bwd", "reduce+eqmask+gather composition", 1,
      "any N/stride/pad; ties share the gradient; input needs 64 B slack", 1 },
    { "conv2d/nhwc", "qpu im2col_nhwc + qpu gemm3 (relu epilogue)", 1,
      "f32 NHWC; stride 1/2; KH=KW in {1,3}; Cin,Cout%16==0; stride*Cin*4<=8191", 1 },
    { "conv2d/stride2", "qpu im2col_nhwc (pitch = 2*Cin*4)", 1,
      "NHWC path only (vc4_conv_nhwc); NCHW stride 2 stays unsupported", 0 },
    { "conv2d/depthwise", "qpu im2col_nhwc + gemm3 (block-diagonal B)", 1,
      "3x3 pad 1 stride 1, C%16==0; costs C/16x a true depthwise", 0 },
    { "maxpool/3x3", "qpu max + 4-row store (NHWC)", 1,
      "kxk stride 1, pad=(k-1)/2, k in {3,5}; the -inf border is a host fill", 1 },
    { "maxpool/5x5s1", "qpu max + 4-row store (NHWC)", 1,
      "5x5 stride 1, pad 2 (SPPF); 3x3 also generated", 1 },
    { "upsample/nearest2x", "qpu gather table", 1,
      "2x nearest, NHWC (vc4_upsample2x_nhwc); any scale via vc4_resize_nearest", 1 },

    /* training-only entries: a forward exists nowhere in this layer, the
       backward is a first-class entry point */
    { "softmax", "reduce+bcast+exp+div composition", 1,
      "any class count on the last axis; numerically stable (max subtracted)", 1 },
    { "softmax_bwd", "reduce+mul+sub composition", 1,
      "dx = y*(g - sum_c g*y), any class count", 1 },
    { "batchnorm", "reduce+transpose+affine composition", 1,
      "train mode: batch mean/var, running stats updated on the QPU", 1 },
    { "batchnorm_bwd", "reduce+transpose+affine composition", 1,
      "dx, dgamma, dbeta from saved mean/invstd", 1 },
    { "batchnorm_infer", "affine composition", 1,
      "y = (x-rm)/sqrt(rv+eps)*gamma+beta", 0 },
    { "layernorm", "reduce+bcast+affine composition", 1,
      "last axis, any width; eps is an attribute", 1 },
    { "layernorm_bwd", "reduce+bcast+affine composition", 1,
      "dx, dgamma, dbeta", 1 },
    { "avgpool", "-", 0,
      "no single forward kernel: the ONNX runner composes sum+scale "
      "(vc4_reduce + vc4_affine); the backward is implemented", 1 },
    { "avgpool_bwd", "gather-list scatter composition", 1,
      "any N/stride/pad, count_include_pad both ways", 1 },
    { "adaptive_avgpool", "-", 0,
      "no forward kernel; composed by the runner", 1 },
    { "adaptive_avgpool_bwd", "gather-list scatter composition", 1,
      "ONNX window bounds; out dims must divide the input extent or smaller", 1 },
    { "global_avgpool", "-", 0, "composed from reduce+affine by the runner", 1 },
    { "global_avgpool_bwd", "affine+bcast composition", 1,
      "dy/(H*W) broadcast over the plane", 1 },
    { "bwop", "qpu ewb family", 1,
      "elementwise derivatives: sigmoid silu tanh exp log sqrt abs clip "
      "leakyrelu hardswish hardsigmoid gelu gelu_erf erf", 1 },
    { "binop_bwd", "gather+reduce composition", 1,
      "add/sub/mul/div with numpy broadcasting, both inputs", 1 },
    { "concat", "-", 0, "host byte concat (NCHW) / vc4_concat_nhwc (NHWC)", 1 },
    { "split", "-", 0, "host byte split (NCHW)", 1 },
    { "resize/nearest", "qpu gather table", 1,
      "any scale, asymmetric mapping; [planes][H][W]", 1 },
    { "resize/nearest_bwd", "inverted-table scatter composition", 1,
      "accumulates every output pixel that maps to a source cell", 1 },
    { "resize/bilinear", "4-tap gather composition", 1,
      "half-pixel centres, edge clamp (torch align_corners=False)", 1 },
    { "resize/bilinear_bwd", "inverted-tap scatter composition", 1,
      "the adjoint of the 4-tap interpolation", 1 },
    { "transpose_bwd", "qpu pack (VDW strided) / gather", 1,
      "= transpose; falls back to a gather table when rows/cols % 16 != 0", 1 },
    { "pad", "-", 0, "host byte pad (zero fill)", 1 },
    { "pad_bwd", "qpu gather table", 1, "the crop of the padded region", 1 },
    { "sum", "qpu reduce", 0, "use vc4_reduce/vc4_sum (needs a shape); bwd is sum_bwd", 1 },
    { "mean", "qpu reduce", 0, "use vc4_reduce/vc4_mean; bwd is sum_bwd(mean=1)", 1 },
    { "sum_bwd", "gather broadcast composition", 1,
      "broadcasts the gradient back over the reduced axis", 1 },
    { "gather", "qpu gather table", 1, "out[o] = in[idx[o]] (arena byte offsets)", 1 },
    { "gather_bwd", "inverted-table scatter composition", 1,
      "additive scatter; a target hit more than 64 times is refused", 1 },
    { "scatter_bwd", "qpu gather table", 1, "out[idx[i]] = x[i] -> dx[i] = g[idx[i]]", 1 },
    { "affine", "qpu uop(affine)", 1, "a*p0 + p1 (the scalar scale/offset)", 0 },
    { "cmp/eq", "qpu ew bit-compare", 1,
      "a==b; exact for every fp32, NaN gives 0, -0.0==+0.0", 1 },
    { "cmp/gt", "qpu ew bit-compare", 1, "a>b; exact, NaN gives 0", 1 },
    { "cmp/ge", "qpu ew bit-compare", 1, "a>=b; exact, NaN gives 0", 1 },
    { "cmp/lt", "qpu ew bit-compare", 1, "a<b; exact, NaN gives 0", 1 },
    { "cmp/le", "qpu ew bit-compare", 1, "a<=b; exact, NaN gives 0", 1 },
    { "isnan", "qpu ew bit-test", 1, "|bits| > 0x7F800000", 1 },
    { "isinf", "qpu ew bit-test", 1, "|bits| == 0x7F800000 (both signs)", 1 },

    /* not implemented -- listed so the table is an honest coverage statement */
    { "conv2d/5x5", "-", 0, "no kernel; would need another im2col tap schedule", 0 },
    { "conv2d/grouped", "-", 0, "the NHWC grouped path exists; NCHW does not", 0 },
    { "matmul/f16", "-", 0, "kernels are fp32 only", 0 },
    { "matmul/f32x4-tile", "-", 0, "4-column register tile: the next speed step", 0 },
    { "batched-matmul", "-", 0, "N launches; not fused", 0 },
    { "transpose/permute", "qpu pack (VDW strided)", 1,
      "2-D dense transpose only; arbitrary permutations use gather_bwd/split", 1 },
    { "quantize/dequantize", "-", 0, "not written", 0 },
};

void vc4_invalidate_buf(vc4_ctx *ctx, const vc4_buf *b)
{
    if (!ctx || !b || !ctx->arena)
        return;
    vc4__invalidate(ctx->arena + b->off, b->bytes);
}

const vc4_op_info *vc4_supported_ops(size_t *count)
{
    if (count)
        *count = sizeof(OPS) / sizeof(OPS[0]);
    return OPS;
}

const char *vc4_op_name(vc4_op op)
{
    switch (op) {
    case VC4_OP_MATMUL:  return "matmul";
    case VC4_OP_CONV2D:  return "conv2d/3x3";
    case VC4_OP_ADD:     return "add";
    case VC4_OP_MUL:     return "mul";
    case VC4_OP_RELU:    return "relu";
    case VC4_OP_SIGMOID: return "sigmoid";
    case VC4_OP_SILU:    return "silu";
    case VC4_OP_RELU_BWD:return "relu_bwd";
    case VC4_OP_RSQRT:   return "rsqrt";
    case VC4_OP_CONV2D_BWD: return "conv2d/bwd_dx";
    case VC4_OP_COL2IM:  return "col2im";
    case VC4_OP_IM2COL:  return "im2col";
    case VC4_OP_MAXPOOL: return "maxpool/2x2s2";
    case VC4_OP_MAXPOOL_BWD: return "maxpool_bwd";
    case VC4_OP_SUB:     return "sub";
    case VC4_OP_DIV:     return "div";
    case VC4_OP_MAX:     return "max";
    case VC4_OP_MIN:     return "min";
    case VC4_OP_SOFTMAX: return "softmax";
    case VC4_OP_SOFTMAX_BWD: return "softmax_bwd";
    case VC4_OP_SOFTMAX_CE_BWD: return "softmax_ce_bwd";
    case VC4_OP_BATCHNORM: return "batchnorm";
    case VC4_OP_BATCHNORM_BWD: return "batchnorm_bwd";
    case VC4_OP_LAYERNORM: return "layernorm";
    case VC4_OP_LAYERNORM_BWD: return "layernorm_bwd";
    case VC4_OP_AVGPOOL: return "avgpool";
    case VC4_OP_AVGPOOL_BWD: return "avgpool_bwd";
    case VC4_OP_ADAPTIVE_AVGPOOL: return "adaptive_avgpool";
    case VC4_OP_ADAPTIVE_AVGPOOL_BWD: return "adaptive_avgpool_bwd";
    case VC4_OP_MAXPOOLN_BWD: return "maxpool/NxNsS_bwd";
    case VC4_OP_BWOP:    return "bwop";
    case VC4_OP_BINOP_BWD: return "binop_bwd";
    case VC4_OP_CAT:     return "concat";
    case VC4_OP_SPLIT:   return "split";
    case VC4_OP_RESIZE_NEAREST: return "resize/nearest";
    case VC4_OP_RESIZE_BILINEAR: return "resize/bilinear";
    case VC4_OP_TRANSPOSE_BWD: return "transpose_bwd";
    case VC4_OP_PAD:     return "pad";
    case VC4_OP_PAD_BWD: return "pad_bwd";
    case VC4_OP_SUM:     return "sum";
    case VC4_OP_MEAN:    return "mean";
    case VC4_OP_SUM_BWD: return "sum_bwd";
    case VC4_OP_GATHER:  return "gather";
    case VC4_OP_GATHER_BWD: return "gather_bwd";
    case VC4_OP_SCATTER_BWD: return "scatter_bwd";
    case VC4_OP_GLOBAL_AVGPOOL: return "global_avgpool";
    case VC4_OP_GLOBAL_AVGPOOL_BWD: return "global_avgpool_bwd";
    case VC4_OP_CMP_EQ:  return "cmp/eq";
    case VC4_OP_CMP_GT:  return "cmp/gt";
    case VC4_OP_CMP_GE:  return "cmp/ge";
    case VC4_OP_CMP_LT:  return "cmp/lt";
    case VC4_OP_CMP_LE:  return "cmp/le";
    case VC4_OP_ISNAN:   return "isnan";
    case VC4_OP_ISINF:   return "isinf";
    default:             return "?";
    }
}

static const vc4_op_info *find(const char *n)
{
    size_t cnt = 0;
    const vc4_op_info *t = vc4_supported_ops(&cnt);
    for (size_t i = 0; i < cnt; ++i)
        if (strcmp(t[i].name, n) == 0)
            return &t[i];
    return NULL;
}

int vc4_op_supported(vc4_op op)
{
    const vc4_op_info *e = find(vc4_op_name(op));
    return e ? e->supported : 0;
}

int vc4_op_differentiable(vc4_op op)
{
    const vc4_op_info *e = find(vc4_op_name(op));
    return e ? e->differentiable : 0;
}

void vc4_print_supported_ops(FILE *f)
{
    size_t cnt = 0;
    const vc4_op_info *t = vc4_supported_ops(&cnt);
    fprintf(f, "\nlibvc4infer %s -- op registry (strict dispatch, no CPU fallback)\n",
            VC4INFER_VERSION_STRING);
    fprintf(f, "%-22s %-10s %-30s %s\n", "op", "fwd  bwd", "backend",
            "constraints / reason");
    fprintf(f, "%-22s %-10s %-30s %s\n", "----------------------",
            "---------", "------------------------------",
            "------------------------------");
    int nsup = 0, ndif = 0;
    for (size_t i = 0; i < cnt; ++i) {
        char fb[16];
        snprintf(fb, sizeof(fb), "%s  %s", t[i].supported ? "yes" : "-",
                 t[i].differentiable ? "yes" : "-");
        fprintf(f, "%-22s %-10s %-30s %s\n", t[i].name, fb, t[i].backend,
                t[i].note);
        nsup += t[i].supported ? 1 : 0;
        ndif += t[i].differentiable ? 1 : 0;
    }
    fprintf(f, "-- %d of %zu forwards dispatchable, %d of %zu differentiable\n",
            nsup, cnt, ndif, cnt);
}
