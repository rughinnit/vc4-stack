/* tests/gpu_gate.h - opt-in gate for a C test that opens the engine.
 *
 * A test that calls vc4_init is a GPU client.  Two clients do not share the
 * QPUs: one starts a launch while the other believes it has the queue, and
 * the board can hang.  So every such test refuses to run unless
 * VC4_REQUIRE_GPU=1 is exported, and reports 77 (the conventional "skipped"
 * status) so run_tests.sh can name it instead of counting it as a pass.
 *
 * This mirrors tests/gpu_gate.py, which does the same for the Python tests.
 */
#ifndef TEST_GPU_GATE_H
#define TEST_GPU_GATE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_SKIP_STATUS 77

static int gpu_gate_skip(const char *name)
{
    const char *v = getenv("VC4_REQUIRE_GPU");

    if (v && *v && strcmp(v, "0") != 0 && strcmp(v, "no") != 0 &&
        strcmp(v, "false") != 0 && strcmp(v, "off") != 0)
        return 0;
    printf("SKIP %s: needs the QPUs; set VC4_REQUIRE_GPU=1 to run it\n", name);
    return 1;
}

#endif /* TEST_GPU_GATE_H */
