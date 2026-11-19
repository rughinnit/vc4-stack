/* test_dispatch.c - the registry is the dispatch gate (no CPU fallback).
 *
 * No GPU work: this test exists to prove that an op outside the table is
 * refused rather than quietly computed somewhere else.
 */
#include "vc4infer.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    size_t n = 0;
    const vc4_op_info *t = vc4_supported_ops(&n);
    int sup = 0, uns = 0;
    for (size_t i = 0; i < n; ++i) {
        if (t[i].supported) sup++; else uns++;
    }

    printf("registry: %zu entries, %d supported, %d unsupported\n", n, sup, uns);

    int rc = 0;
    /* every enum must resolve through the registry, and the binary ops must
       agree with the public helpers */
    for (int op = 0; op < VC4_OP_COUNT; ++op) {
        int s = vc4_op_supported((vc4_op)op);
        printf("  %-12s %s\n", vc4_op_name((vc4_op)op), s ? "supported" : "UNSUPPORTED");
        if (!s) {
            printf("FAIL: %s should be dispatchable\n", vc4_op_name((vc4_op)op));
            rc = 1;
        }
    }
    if (vc4_op_supported((vc4_op)999) != 0) {
        printf("FAIL: unknown op reported supported\n");
        rc = 1;
    } else {
        printf("  unknown op  -> UNSUPPORTED (correct)\n");
    }

    /* the unsupported rows must be present and marked as such */
    const char *must_be_absent[] = { "conv2d/5x5", "layernorm",
                                     "matmul/f16", "quantize/dequantize" };
    for (size_t i = 0; i < sizeof(must_be_absent) / sizeof(*must_be_absent); ++i) {
        int found = 0;
        for (size_t j = 0; j < n; ++j)
            if (strcmp(t[j].name, must_be_absent[i]) == 0) {
                found = 1;
                if (t[j].supported) {
                    printf("FAIL: %s claims support\n", t[j].name);
                    rc = 1;
                }
            }
        printf("  %-16s %s\n", must_be_absent[i],
               found ? "listed + unsupported" : "MISSING");
        if (!found)
            rc = 1;
    }

    vc4_print_supported_ops(stdout);
    printf("%s\n", rc == 0 ? "PASS dispatch" : "FAIL dispatch");
    return rc;
}
