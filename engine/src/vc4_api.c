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
    { "maxpool/3x3", "-", 0, "only 2x2 stride 2 has a kernel" },
    { "avgpool", "-", 0, "not written" },
    /* not implemented -- listed so the table is an honest coverage statement */
    { "conv2d/5x5", "-", 0, "no kernel; would need another im2col tap schedule" },
    { "conv2d/stride2", "-", 0,
      "the VDR loads 16 consecutive pixels; stride 2 needs a decimation pass" },
    { "conv2d/depthwise", "-", 0, "needs a per-channel schedule, not a GEMM" },
    { "conv2d/grouped", "-", 0, "needs one GEMM per group (trivial extension)" },
    { "matmul/f16", "-", 0, "kernels are fp32 only" },
    { "matmul/f32x4-tile", "-", 0, "4-column register tile: the documented next speed step" },
    { "batched-matmul", "-", 0, "N launches; not yet fused" },
    { "softmax", "qpu fused row kernel", 1,
      "forward probabilities are an intermediate of softmax_ce_bwd; 16 padded classes" },
    { "layernorm", "-", 0, "needs reduction + rsqrt kernel" },
    { "concat/slice", "-", 0, "pure data movement; no GPU kernel yet" },
    { "transpose/permute", "qpu pack (VDW strided)", 1,
      "2-D dense transpose only; no arbitrary permutation" },
    { "upsample", "-", 0, "not written" },
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
