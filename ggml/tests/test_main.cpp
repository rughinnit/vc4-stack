// test_ggml_vc4 - run the three acceptance tests and print one block each.

#include <cstdio>
#include <cstring>

extern "C" {
int vc4ggml_test_tiny(char * out, size_t cap);
int vc4ggml_test_matmul(char * out, size_t cap);
int vc4ggml_test_llama(char * out, size_t cap);
}

int main(int argc, char ** argv) {
    const bool all = !(argc > 1 && strcmp(argv[1], "tiny") == 0) &&
                     !(argc > 1 && strcmp(argv[1], "matmul") == 0) &&
                     !(argc > 1 && strcmp(argv[1], "llama") == 0);

    int rc = 0;
    char buf[8192];

    if (all || strcmp(argv[1], "tiny") == 0) {
        buf[0] = 0;
        int r = vc4ggml_test_tiny(buf, sizeof(buf));
        printf("=== (a) tiny graph ===\n%s", buf);
        rc |= r;
    }
    if (all || strcmp(argv[1], "matmul") == 0) {
        buf[0] = 0;
        int r = vc4ggml_test_matmul(buf, sizeof(buf));
        printf("=== (b) MUL_MAT ===\n%s", buf);
        rc |= r;
    }
    if (all || strcmp(argv[1], "llama") == 0) {
        buf[0] = 0;
        int r = vc4ggml_test_llama(buf, sizeof(buf));
        printf("=== (c) llama-family block ===\n%s", buf);
        rc |= r;
    }
    printf("\n%s\n", rc == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return rc;
}
