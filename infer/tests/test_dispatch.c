/* test_dispatch.c - the registry is the dispatch gate (no CPU fallback).
 *
 * No GPU work: this test proves that the registry is the single source of
 * truth for what may run on the QPU.
 *
 * The invariant is NOT "every op is implemented" - the registry deliberately
 * carries ops that are not, so a caller can ask and get a reason.  It is:
 *
 *   1. every value of the vc4_op enum appears in the table (no silent gap);
 *   2. `vc4_op_supported`/`vc4_op_differentiable` agree with that row;
 *   3. a row that is not supported carries a non-empty `note` explaining why;
 *   4. ops known to be unimplemented are present and marked unsupported.
 *
 * Rule 3 is what stops the table from degrading into a list of "false" with
 * no explanation, and an unknown op still has to be refused.
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
    /* every enum must resolve through the registry, agree with the public
       helpers, and - when unsupported - say why */
    for (int op = 0; op < VC4_OP_COUNT; ++op) {
        const char *name = vc4_op_name((vc4_op)op);
        int sup = vc4_op_supported((vc4_op)op);
        const vc4_op_info *row = NULL;
        for (size_t j = 0; j < n; ++j)
            if (strcmp(t[j].name, name) == 0) { row = &t[j]; break; }
        if (!row) {
            printf("FAIL: %s is in the enum but missing from the registry\n", name);
            rc = 1;
            continue;
        }
        if (sup != row->supported) {
            printf("FAIL: %s: vc4_op_supported=%d but the row says %d\n",
                   name, sup, row->supported);
            rc = 1;
        }
        if (!sup) {
            if (!row->note || !*row->note) {
                printf("FAIL: %s is unsupported with no reason in the table\n", name);
                rc = 1;
            } else {
                printf("  %-20s unimplemented (%s)\n", name, row->note);
            }
            continue;
        }
        printf("  %-20s supported%s\n", name,
               vc4_op_differentiable((vc4_op)op) ? ", backward" : "");
    }
    if (vc4_op_supported((vc4_op)999) != 0) {
        printf("FAIL: unknown op reported supported\n");
        rc = 1;
    } else {
        printf("  unknown op  -> UNSUPPORTED (correct)\n");
    }

    /* the unsupported rows must be present and marked as such */
    const char *must_be_absent[] = { "conv2d/5x5", "batched-matmul",
                                     "matmul/f16", "quantize/dequantize",
                                     "conv2d/grouped" };
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
