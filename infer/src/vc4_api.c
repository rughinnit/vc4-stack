/*
 * vc4_api.c - the op registry and the coverage table.
 *
 * STRICT DISPATCH: this table is the single source of truth.  An op that is
 * not implemented on the GPU is listed with supported = 0 and the entry point
 * returns VC4_ERR_UNSUPPORTED.  Nothing falls back to the CPU.
 */
#define _GNU_SOURCE
#include "vc4_int.h"

#include <string.h>

static const vc4_op_info OPS[] = {
    /* implemented */
    { "matmul", "qpu gemm(asm10) + qpu pack",
      1, "f32 row-major; M,N,K any size (8/64/16 tiles internally)" },
    { "conv2d/3x3", "qpu im2col + qpu gemm(asm3)",
      1, "f32 NCHW; stride 1; pad 0..; Cin%16==0; Cout%16==0; Hp*Wp*4<=8191" },
    { "conv2d/1x1", "qpu im2col + qpu gemm(asm3)",
      1, "f32 NCHW; stride 1; pad 0; Cin%16==0; Cout%16==0; H*Wp*4<=8191" },
    { "add", "qpu elementwise", 1, "f32, exact IEEE; n rounded to 256 internally" },
    { "mul", "qpu elementwise", 1, "f32, exact IEEE; n rounded to 256 internally" },
    { "relu", "qpu elementwise", 1, "f32, exact IEEE" },
    { "sigmoid", "qpu elementwise", 1,
      "f32; 2^f degree-5 poly + 3x Newton rcp; |err| < 1e-6 (measured)" },
    { "silu", "qpu elementwise", 1,
      "f32; x*sigmoid(x); |err| < 1e-6 (measured)" },
    { "relu_bwd", "qpu elementwise", 1,
      "f32; bit-exact mask: out = dy if x>0 else 0" },
    { "rsqrt", "qpu elementwise", 1,
      "f32; 1/sqrt(x), bit-trick seed + 4 Newton (Adam); rel err < 2e-6" },
    { "transpose", "qpu pack (VDW strided)", 1,
      "dense NxM transpose; rows,cols % 16 == 0; col*4 <= 8191" },
    { "matmul_bt", "qpu pack + qpu gemm3(bmode=1)", 1,
      "C = A*B^T, f32; M,N,K % 16 == 0 (training dW / dX path)" },
    { "softmax_ce_bwd", "qpu fused row kernel", 1,
      "rows % 8 == 0; 16 padded classes; returns dlogits=(p-onehot)/B, loss, argmax count" },
    { "im2col", "qpu im2col (row-major)", 1,
      "f32 NCHW; stride 1; KH=KW in {1,3}; Cin%16==0; Hp*Wp*4<=8191" },
    { "col2im", "qpu scatter-add (fixed tap order)", 1,
      "dA [HO*Wp][Cin*KH*KW] -> padded dX; Cin%16==0; deterministic fp32" },
    { "conv2d/bwd_dx", "qpu im2col_r + gemm3 + qpu col2im", 1,
      "dy+W -> dX; stride 1; KH=KW in {1,3}; Cin,Cout%16==0; no bias grad" },
    { "conv2d/bwd_dw", "qpu im2col_r + qpu transpose + gemm3", 1,
      "dy+A -> dW (ci-major); stride 1; KH=KW in {1,3}; Cin,Cout%16==0" },
    { "maxpool/2x2s2", "qpu max + store", 1,
      "f32 NCHW; H,W even; W%16==0; out row pitch padded to 16" },
    { "maxpool_bwd", "qpu argmax mask", 1,
      "dy routed to the group maximum; ties share the gradient" },
    { "maxpool/NxNsS", "qpu max + 4-row store", 1,
      "any N>=1, stride>=1, pad>=0; host pads the border with -inf (bytes only)" },
    /* the NHWC channel-tiled plane (src/vc4_nhwc.c): the im2col VDR pitch is
       stride*Cin*4, so the map size is no longer limited by the 8191 B field */
    { "conv2d/nhwc", "qpu im2col_nhwc + qpu gemm3", 1,
      "f32 NHWC; stride 1/2; KH=KW in {1,3}; Cin,Cout%16==0; stride*Cin*4<=8191" },
    { "conv2d/stride2", "qpu im2col_nhwc (pitch = 2*Cin*4)", 1,
      "NHWC path only (vc4_conv_nhwc); NCHW stride 2 stays unsupported" },
    { "conv2d/depthwise", "qpu im2col_nhwc + gemm3 (block-diagonal B)", 1,
      "3x3 pad 1 stride 1, C%16==0; costs C/16x a true depthwise" },
    { "maxpool/3x3", "qpu max + 4-row store (NHWC)", 1,
      "kxk stride 1, pad=(k-1)/2, k in {3,5}; the -inf border is a host fill" },
    { "maxpool/5x5s1", "qpu max + 4-row store (NHWC)", 1,
      "5x5 stride 1, pad 2 (SPPF); 3x3 also generated" },
    { "concat/channel", "-", 0, "host byte concat (NCHW or NHWC), no GPU kernel" },
    { "upsample/nearest2x", "-", 0, "host byte gather" },
    { "avgpool", "-", 0, "not written" },
    /* not implemented -- listed so the table is an honest coverage statement */
    { "conv2d/5x5", "-", 0, "no kernel; would need another im2col tap schedule" },
    { "conv2d/grouped", "-", 0, "needs one GEMM per group (trivial extension)" },
    { "matmul/f16", "-", 0, "kernels are fp32 only" },
    { "matmul/f32x4-tile", "-", 0, "4-column register tile: the documented next speed step" },
    { "batched-matmul", "-", 0, "N launches; not yet fused" },
    { "softmax", "qpu fused row kernel ('p' probe)", 1,
      "forward probabilities; cols<=16, rows%8==0; CPU pads/compacts bytes only" },
    { "layernorm", "-", 0, "needs reduction + rsqrt kernel" },
    { "slice/split", "-", 0, "pure data movement (host), no GPU kernel" },
    { "transpose/permute", "qpu pack (VDW strided)", 1,
      "2-D dense transpose only; no arbitrary permutation" },
    { "quantize/dequantize", "-", 0, "not written" },
};

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
    default:             return "?";
    }
}

int vc4_op_supported(vc4_op op)
{
    const char *n = vc4_op_name(op);
    size_t cnt = 0;
    const vc4_op_info *t = vc4_supported_ops(&cnt);
    for (size_t i = 0; i < cnt; ++i)
        if (strcmp(t[i].name, n) == 0)
            return t[i].supported;
    return 0;
}

void vc4_print_supported_ops(FILE *f)
{
    size_t cnt = 0;
    const vc4_op_info *t = vc4_supported_ops(&cnt);
    fprintf(f, "\nlibvc4infer %s -- op registry (strict dispatch, no CPU fallback)\n",
            VC4INFER_VERSION_STRING);
    fprintf(f, "%-20s %-6s %-24s %s\n", "op", "gpu", "backend", "constraints / reason");
    fprintf(f, "%-20s %-6s %-24s %s\n", "--------------------", "------",
            "------------------------", "------------------------------");
    int nsup = 0;
    for (size_t i = 0; i < cnt; ++i) {
        fprintf(f, "%-20s %-6s %-24s %s\n", t[i].name,
                t[i].supported ? "yes" : "NO", t[i].backend, t[i].note);
        nsup += t[i].supported ? 1 : 0;
    }
    fprintf(f, "-- %d of %zu entries dispatchable on the GPU\n", nsup, cnt);
}
