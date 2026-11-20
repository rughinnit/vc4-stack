/*
 * Lowering of the VHLO subset JAX emits for small fp32 programs into a
 * linear instruction list over arena tensors.  See stablehlo.c for the
 * bytecode reader and docs/PJRT.md for the coverage table.
 */
#ifndef VC4_STABLEHLO_H
#define VC4_STABLEHLO_H

#include <stddef.h>
#include <stdint.h>

#define VC4SL_MAXRANK 8
#define VC4SL_MAXTENSORS 256
#define VC4SL_MAXINSNS 512

typedef enum {
    VC4SL_DT_F32 = 0,
    VC4SL_DT_I1,
    VC4SL_DT_I64,
    VC4SL_DT_I32,
    VC4SL_DT_OTHER,
} vc4sl_dtype;

typedef struct {
    int rank;
    int64_t dims[VC4SL_MAXRANK];
    vc4sl_dtype dtype;
} vc4sl_shape;

typedef enum {
    VC4SL_ADD = 0,
    VC4SL_SUB,
    VC4SL_MUL,
    VC4SL_MAXIMUM,
    VC4SL_ABS,
    VC4SL_AFFINE,
    VC4SL_GATHER,
    VC4SL_NEGATE,
    VC4SL_RELU,
    VC4SL_COMPARE,
    VC4SL_SELECT,
    VC4SL_DOT,
    VC4SL_DOT_BT,
    VC4SL_BROADCAST,
    VC4SL_TRANSPOSE,
    VC4SL_REDUCE_SUM,
    VC4SL_CONSTANT,
} vc4sl_opcode;

typedef struct {
    vc4sl_opcode op;
    const char *name;       /* vhlo mnemonic, for error messages */
    int dst;
    int a, b, c;
    vc4sl_shape shape;      /* result shape */
    int64_t aux[8];         /* cmp direction, outer/red, ... */
    int table;              /* index into program tables, or -1 */
    int table_slot;         /* arena tensor holding the int32 table */
} vc4sl_insn;

typedef struct {
    int32_t *idx;           /* source element index per output element */
    int n;
} vc4sl_table;

typedef struct {
    vc4sl_shape shape;
    int alias_of;           /* logical alias, -1 when this owns storage */
    int is_constant;
    uint8_t *const_data;
    size_t const_len;
} vc4sl_tensor;

typedef struct {
    int num_tensors;
    vc4sl_tensor tensors[VC4SL_MAXTENSORS];
    int num_insns;
    vc4sl_insn insns[VC4SL_MAXINSNS];
    int num_tables;
    vc4sl_table tables[VC4SL_MAXINSNS];
    int num_params;
    int param_slots[16];
    int num_outputs;
    int output_slots[16];
    vc4sl_shape output_shapes[16];
    int arena_bytes;
} vc4sl_program;

/* Parse a serialized StableHLO portable artifact and lower it.  On failure
 * returns -1 and writes a message naming the operation or encoding that is
 * outside the subset. */
int vc4sl_parse(const uint8_t *data, size_t len, vc4sl_program *prog,
                char *err, size_t errlen);

void vc4sl_free(vc4sl_program *prog);

/* Element count of a tensor slot (always stored as f32 in the arena). */
size_t vc4sl_num_elements(const vc4sl_program *prog, int slot);

#endif
