/* Minimal driver for videocoreiv-qpu/qpu-sniff/qpudis.c.
 * Set FIELDS=1 to dump every numeric field (used for the field cross-check).
 *   gcc -O2 -std=c99 -I videocoreiv-qpu/qpu-sniff videocoreiv_main.c qpudis.o -o qpu_disasm_vc4
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "qpudis.h"
extern int showfields;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s blob.bin [first] [count]\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    long nw = n / 8;
    uint64_t *w = calloc(nw ? (size_t)nw : 1, 8);
    if (!w || fread(w, 8, (size_t)nw, f) != (size_t)nw) { perror("read"); return 1; }
    fclose(f);
    long first = argc > 2 ? strtol(argv[2], 0, 0) : 0;
    long count = argc > 3 ? strtol(argv[3], 0, 0) : nw - first;
    if (first < 0 || first > nw) return 1;
    if (count > nw - first) count = nw - first;
    showfields = getenv("FIELDS") != NULL;
    for (long i = 0; i < count; i++) {
        printf("%4ld: %08x %08x  ", first + i,
               (uint32_t)(w[first+i] & 0xffffffff), (uint32_t)(w[first+i] >> 32));
        show_qpu_inst((uint32_t *)&w[first+i]);
    }
    return 0;
}
