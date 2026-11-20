/* Decode a dumped StableHLO portable artifact and print the lowered program.
 * Development aid for the PJRT executor; not part of the plugin. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/stablehlo.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s module.mlir\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) return 2;
    fclose(f);

    vc4sl_program *p = calloc(1, sizeof(*p));
    char err[512] = {0};
    if (vc4sl_parse(buf, (size_t)n, p, err, sizeof(err))) {
        printf("ERROR: %s\n", err);
        return 1;
    }
    printf("tensors=%d insns=%d params=%d outputs=%d\n", p->num_tensors,
           p->num_insns, p->num_params, p->num_outputs);
    for (int i = 0; i < p->num_tensors; i++) {
        printf("  t%-3d %s dtype=%d shape=[", i,
               p->tensors[i].is_constant ? "const" : "     ",
               p->tensors[i].shape.dtype);
        for (int d = 0; d < p->tensors[i].shape.rank; d++)
            printf("%s%lld", d ? "," : "",
                   (long long)p->tensors[i].shape.dims[d]);
        printf("]\n");
    }
    static const char *ops[] = {"add", "sub", "mul", "maximum", "abs",
                                "affine", "host_gather", "negate", "relu",
                                "compare", "select", "dot", "dot_bt",
                                "bcast", "transpose", "reduce_sum",
                                "constant"};
    for (int i = 0; i < p->num_insns; i++) {
        vc4sl_insn *in = &p->insns[i];
        printf("  %-10s a=%d b=%d c=%d -> t%d", ops[in->op], in->a, in->b,
               in->c, in->dst);
        if (in->op == VC4SL_COMPARE) printf(" cmp=%lld", (long long)in->aux[0]);
        if (in->op == VC4SL_DOT || in->op == VC4SL_DOT_BT)
            printf(" M=%lld N=%lld K=%lld", (long long)in->aux[0],
                   (long long)in->aux[1], (long long)in->aux[2]);
        if (in->op == VC4SL_REDUCE_SUM)
            printf(" outer=%lld red=%lld", (long long)in->aux[0],
                   (long long)in->aux[1]);
        if (in->table >= 0) printf(" table=%d", in->table);
        printf("\n");
    }
    for (int i = 0; i < p->num_tables; i++) {
        printf("  table%d n=%d:", i, p->tables[i].n);
        for (int k = 0; k < p->tables[i].n && k < 32; k++)
            printf(" %d", p->tables[i].idx[k]);
        printf("\n");
    }
    printf("params:");
    for (int i = 0; i < p->num_params; i++) printf(" %d", p->param_slots[i]);
    printf("\noutputs:");
    for (int i = 0; i < p->num_outputs; i++) printf(" %d", p->output_slots[i]);
    printf("\n");
    vc4sl_free(p);
    free(p);
    free(buf);
    return 0;
}
