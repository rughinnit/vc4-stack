/*
 * vc4_plan.h - the node table of the VC4 LiteRT delegate.
 *
 * This file is deliberately free of engine calls and of any global state: it
 * takes a TFLite node plus the interpreter's tensor array and answers one
 * question, "can the QPU path execute this node exactly?", returning the
 * geometry the engine needs when the answer is yes.  The answer is
 * conservative: anything the engine cannot express bit-for-bit is refused,
 * never emulated on the host.
 *
 * tests/test_partition.c exercises this table with synthetic TFLite
 * structures, so the claimed set can be checked without a GPU.
 */
#ifndef VC4_PLAN_H
#define VC4_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "tflite/core/c/common.h"

typedef enum {
  VC4_CLAIM_OK = 0,
  VC4_CLAIM_UNSUPPORTED_OP, /* opcode is not in the claimed table          */
  VC4_CLAIM_DTYPE,          /* not fp32, or carries quantization           */
  VC4_CLAIM_DYNAMIC,        /* dynamic dimension / dims signature          */
  VC4_CLAIM_SHAPE,          /* rank or extent the engine cannot express    */
  VC4_CLAIM_LAYOUT,         /* not dense row-major (bytes != 4 * numel)    */
  VC4_CLAIM_PARAM,          /* stride/pad/dilation/beta outside the table  */
  VC4_CLAIM_ACTIVATION,     /* fused activation not implemented            */
  VC4_CLAIM_BROADCAST,      /* elementwise node with broadcasting          */
  VC4_CLAIM_SIZE            /* arena estimate over the configured budget   */
} vc4_claim_status;

typedef enum {
  VC4_STEP_CONV = 0, /* CONV_2D, DEPTHWISE_CONV_2D, or FC as a 1x1 conv */
  VC4_STEP_UNARY,
  VC4_STEP_BINARY,
  VC4_STEP_SOFTMAX
} vc4_step_kind;

/* Unary operators the plan can express.  Kept separate from vc4_uop so the
 * partition table compiles without the engine header. */
typedef enum {
  VC4_PLAN_U_NONE = 0,
  VC4_PLAN_U_RELU,
  VC4_PLAN_U_RELU6,
  VC4_PLAN_U_RELU_N1TO1,
  VC4_PLAN_U_SIGMOID,
  VC4_PLAN_U_TANH
} vc4_plan_unop;

typedef enum {
  VC4_PLAN_B_ADD = 0,
  VC4_PLAN_B_MUL,
  VC4_PLAN_B_SUB,
  VC4_PLAN_B_DIV
} vc4_plan_binop;

typedef struct {
  vc4_step_kind kind;
  int builtin;         /* TfLiteBuiltinOperator                         */
  int in0, in1, in2;   /* tensor indices; -1 when the slot is absent    */
  int out;             /* tensor index                                  */
  int activation;      /* TfLiteFusedActivation for conv/FC steps       */
  int is_depthwise;    /* CONV_2D family flavour                        */
  int is_fc;           /* fully connected expressed as a 1x1 conv       */

  /* conv geometry (engine's NHWC convention: [nimg][H][W][Cin]) */
  int stride, pad;
  int nimg, H, W, Cin, Cout, KH, KW;
  int ho, wo, Wg;      /* engine output rows/cols, align16(wo)          */

  /* elementwise / softmax geometry */
  int rows, cols;
  size_t n;            /* element count                                 */

  size_t out_bytes;    /* dense bytes of the output tensor              */
  vc4_plan_unop unop;
  vc4_plan_binop binop;
} vc4_node_plan;

/* Arena cost of one claimed node, excluding the tensor slots (which the
 * delegate adds once per partition). */
typedef struct {
  size_t persistent; /* weight panel + fused-activation temporary          */
  size_t pitch;      /* pitched conv output, max over a partition          */
  size_t scratch;    /* peak engine scratch (padded image, im2col panel)   */
} vc4_plan_estimate;

/* Returns VC4_CLAIM_OK and fills *plan when the QPU path can execute the node
 * exactly.  Never reads tensor data contents except to test a depthwise bias
 * for all-zero (a constant, not an activation). */
vc4_claim_status vc4_plan_node(const TfLiteRegistration *reg,
                               const TfLiteNode *node,
                               const TfLiteTensor *tensors, size_t ntensors,
                               vc4_node_plan *plan);

const char *vc4_plan_reason(vc4_claim_status st);
const char *vc4_plan_builtin_name(int builtin);

void vc4_plan_estimate_node(const vc4_node_plan *plan, vc4_plan_estimate *est);

#endif /* VC4_PLAN_H */
