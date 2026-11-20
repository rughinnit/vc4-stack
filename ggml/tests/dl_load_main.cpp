// test_dl_load - prove that libggml-vc4.so loads as a ggml dynamic backend.
//
//   GGML_BACKEND_PATH=$PWD/libggml-vc4.so ./test_dl_load
//
// This binary links libggml.so only; it never references a VC4 symbol.  It
// runs a tiny add graph on whatever "VC4" device ggml_backend_load_all found,
// and checks the result against a host loop.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const char * path = getenv("GGML_BACKEND_PATH");
    if (!path || !*path) {
        fprintf(stderr, "set GGML_BACKEND_PATH to libggml-vc4.so\n");
        return 2;
    }

    ggml_backend_load_all();

    printf("devices after ggml_backend_load_all():\n");
    const size_t ndev = ggml_backend_dev_count();
    for (size_t i = 0; i < ndev; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(d, &fr, &tot);
        printf("  [%zu] %-8s type=%d  free=%zu MiB total=%zu MiB  %s\n", i,
               ggml_backend_dev_name(d), (int)ggml_backend_dev_type(d),
               fr >> 20, tot >> 20, ggml_backend_dev_description(d));
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("VC4");
    if (!dev) {
        printf("FAIL: no VC4 device was registered\n");
        return 1;
    }

    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        printf("FAIL: VC4 backend init\n");
        return 1;
    }

    const int n = 1024;
    std::vector<float> a(n), b(n), want(n), got(n);
    for (int i = 0; i < n; ++i) { a[i] = (float)i * 0.001f; b[i] = 1.0f; }

    ggml_init_params p = { 4u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * to = ggml_add(ctx, ta, tb);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, to);

    size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,
                                   ggml_backend_dev_buffer_type(dev));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
                                   ctx, ggml_backend_dev_buffer_type(dev));
    if (need == 0 || !buf) {
        printf("FAIL: buffer allocation (%zu B)\n", need);
        return 1;
    }

    ggml_backend_tensor_set(ta, a.data(), 0, sizeof(float) * n);
    ggml_backend_tensor_set(tb, b.data(), 0, sizeof(float) * n);

    const enum ggml_status st = ggml_backend_graph_compute(be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute status %d\n", (int)st);
        return 1;
    }
    ggml_backend_tensor_get(to, got.data(), 0, sizeof(float) * n);

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        want[i] = a[i] + b[i];
        if (std::fabs(got[i] - want[i]) > 1e-6f) bad++;
    }

    printf("add graph on %s: mismatch=%d (n=%d), buffer %zu B\n",
           ggml_backend_name(be), bad, n, need);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    printf("%s\n", bad == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return bad == 0 ? 0 : 1;
}
