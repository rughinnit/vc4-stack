// ggml_vc4_tests.cpp - the verification suite for the VC4 ggml backend.
//
// Exported with C linkage so the same code runs from the executable and from
// the ctypes harness (tests/ctypes_test.py).  Each function returns 0 on pass.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-vc4.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int vc4ggml_test_tiny(char * out, size_t cap);
int vc4ggml_test_matmul(char * out, size_t cap);
int vc4ggml_test_llama(char * out, size_t cap);
}

// --------------------------------------------------------------------------
static bool g_registered = false;

static void ensure_registered() {
    if (!g_registered) {
        ggml_vc4_register();
        g_registered = true;
    }
}

static void say(char * out, size_t cap, const char * fmt, ...) {
    if (!out || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    const size_t used = strlen(out);
    if (used + 1 < cap) vsnprintf(out + used, cap - used, fmt, ap);
    va_end(ap);
}

static uint32_t rng_state = 0x12345678u;
static void rng_seed(uint32_t s) { rng_state = s ? s : 1; }
static float rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((rng_state >> 8) & 0xFFFFFF) / 16777216.0f * 2.0f - 1.0f;
}

struct stats { float max_abs; float max_rel; int mismatch; };

static stats compare(const std::vector<float> & a, const std::vector<float> & b,
                     float tol) {
    stats s{0, 0, 0};
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        // relative error only where the reference is not near zero
        if (std::fabs(b[i]) > 1e-3f)
            s.max_rel = std::max(s.max_rel, d / std::fabs(b[i]));
        if (!(d <= tol + 1e-3f * std::fabs(b[i]))) s.mismatch++;
    }
    if (a.size() != b.size()) s.mismatch += 1000000;
    return s;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

// A graph allocated in one backend's buffer, ready to run.
struct session {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;

    void destroy() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr; gf = nullptr;
    }
};

static bool alloc_ctx(ggml_context * ctx, ggml_backend_buffer_type_t buft,
                      ggml_backend_buffer_t * out) {
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    if (need == 0) return false;
    *out = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    return *out != nullptr;
}

// ==========================================================================
// (a) tiny elementwise graph: relu(add(a,b) * a)
// ==========================================================================
static ggml_tensor * build_tiny(ggml_context * ctx, ggml_tensor ** a,
                                ggml_tensor ** b) {
    const int n = 4096;
    *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(*a, "a");
    ggml_set_name(*b, "b");
    ggml_tensor * t = ggml_add(ctx, *a, *b);
    ggml_set_name(t, "add");
    ggml_tensor * u = ggml_mul(ctx, t, *a);
    ggml_set_name(u, "mul");
    ggml_tensor * r = ggml_relu(ctx, u);
    ggml_set_name(r, "relu");
    return r;
}

extern "C" int vc4ggml_test_tiny(char * out, size_t cap) {
    ensure_registered();
    const int n = 4096;
    std::vector<float> va(n), vb(n);
    rng_seed(7);
    for (int i = 0; i < n; ++i) { va[i] = rnd(); vb[i] = rnd(); }

    ggml_backend_t vc4 = ggml_backend_dev_init(ggml_backend_dev_by_name("VC4"), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!vc4 || !cpu) {
        say(out, cap, "FAIL: backend init (vc4=%p cpu=%p)\n", (void *)vc4, (void *)cpu);
        return 1;
    }

    auto run = [&](ggml_backend_t be, ggml_backend_buffer_type_t buft,
                   std::vector<float> & res, double * ms) {
        session s;
        s.backend = be;
        ggml_init_params p = { 64u << 20, nullptr, true };
        s.ctx = ggml_init(p);
        ggml_tensor *a = nullptr, *b = nullptr;
        ggml_tensor * r = build_tiny(s.ctx, &a, &b);
        s.gf = ggml_new_graph(s.ctx);
        ggml_build_forward_expand(s.gf, r);
        if (!alloc_ctx(s.ctx, buft, &s.buf)) {
            fprintf(stderr, "vc4: tiny: buffer allocation failed\n");
            s.destroy();
            return false;
        }
        ggml_backend_tensor_set(a, va.data(), 0, va.size() * 4);
        ggml_backend_tensor_set(b, vb.data(), 0, vb.size() * 4);
        ggml_backend_graph_compute(be, s.gf);  // warm-up / JIT
        std::vector<double> t;
        for (int i = 0; i < 15; ++i) {
            const double t0 = now_ms();
            if (ggml_backend_graph_compute(be, s.gf) != GGML_STATUS_SUCCESS) {
                s.destroy();
                return false;
            }
            t.push_back(now_ms() - t0);
        }
        *ms = median(t);
        res.resize(ggml_nelements(r));
        ggml_backend_tensor_get(r, res.data(), 0, res.size() * 4);
        s.destroy();
        return true;
    };

    std::vector<float> rv, rc;
    double tv = 0, tc = 0;
    const bool okv = run(vc4, ggml_backend_get_default_buffer_type(vc4), rv, &tv);
    const bool okc = run(cpu, ggml_backend_cpu_buffer_type(), rc, &tc);
    ggml_backend_free(vc4);
    ggml_backend_free(cpu);

    if (!okv || !okc) {
        say(out, cap, "FAIL: tiny graph did not run (vc4=%d cpu=%d)\n", okv, okc);
        return 1;
    }
    const stats st = compare(rv, rc, 1e-5f);
    say(out, cap,
        "tiny graph relu(add(a,b)*a), n=%d: mismatch=%d max_abs=%.3g "
        "max_rel=%.3g | VC4 %.3f ms, CPU %.3f ms (%.2fx)\n",
        n, st.mismatch, st.max_abs, st.max_rel, tv, tc,
        tv > 0 ? tc / tv : 0.0);
    return st.mismatch == 0 ? 0 : 1;
}

// ==========================================================================
// (b) MUL_MAT at 64^3 / 128^3 / 256^3, both ggml layouts
// ==========================================================================
static ggml_tensor * build_matmul(ggml_context * ctx, int M0, int N1, int K,
                                  bool cont_b, ggml_tensor ** w,
                                  ggml_tensor ** x) {
    // dst[i0 + i1*M0] = sum_k A[k,i0] * B[k,i1]
    //   cont_b == true : B = src1 contiguous [N1][K] (what llama.cpp builds)
    //   cont_b == false: B = src1, a transposed view of [N1][K] (KxN1 memory)
    *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M0);
    ggml_set_name(*x, "x");
    if (cont_b) {
        ggml_tensor * wa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N1);
        ggml_set_name(wa, "w");
        *w = wa;
        return ggml_mul_mat(ctx, wa, *x);
    }
    ggml_tensor * raw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N1, K);
    ggml_set_name(raw, "w_raw");
    ggml_tensor * wt = ggml_transpose(ctx, raw);
    ggml_set_name(wt, "w_view");
    *w = wt;
    return ggml_mul_mat(ctx, *x, wt);
}

extern "C" int vc4ggml_test_matmul(char * out, size_t cap) {
    ensure_registered();
    ggml_backend_t vc4 = ggml_backend_dev_init(ggml_backend_dev_by_name("VC4"), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!vc4 || !cpu) {
        say(out, cap, "FAIL: backend init\n");
        return 1;
    }

    int failures = 0;
    struct shape { int M0, N1, K; };
    for (bool cont_b : { true, false }) {
        say(out, cap, "MUL_MAT %s layout | shape (M0,N1,K) | err | VC4 ms | CPU ms | GFLOP/s\n",
            cont_b ? "contiguous-B [N][K]" : "transposed-view B [K][N]");
        std::vector<shape> shapes = {
            {64, 64, 64}, {128, 128, 128}, {256, 256, 256},
        };
        // one non-square shape: the roles of the two operands cannot be
        // swapped without changing the answer
        shapes.push_back(cont_b ? shape{48, 96, 80} : shape{48, 128, 80});

        for (const shape & sh : shapes) {
            const int M0 = sh.M0, N1 = sh.N1, K = sh.K;
            std::vector<float> vw((size_t)N1 * K), vx((size_t)K * M0);
            rng_seed(1000u + (uint32_t)(M0 + N1 + K) + (cont_b ? 0u : 77u));
            for (auto & v : vw) v = rnd() * 0.05f;
            for (auto & v : vx) v = rnd();

            // host reference: dst[i0 + i1*M0] = sum_k A[k,i0] B[k,i1]
            std::vector<float> ref((size_t)M0 * N1, 0.0f);
            for (int m = 0; m < M0; ++m) {
                for (int n = 0; n < N1; ++n) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) {
                        // cont_b: A=w (k,n) at k + n*K; B=x (k,m) at k + m*K
                        // view  : A=x (k,m) at k + m*K; B=w (k,n) at k*N1 + n
                        const double av = cont_b ? vw[(size_t)k + (size_t)n * K]
                                                 : vx[(size_t)k + (size_t)m * K];
                        const double bv = cont_b ? vx[(size_t)k + (size_t)m * K]
                                                 : vw[(size_t)k * N1 + n];
                        acc += av * bv;
                    }
                    ref[(size_t)m + (size_t)n * M0] = (float)acc;
                }
            }

            auto run = [&](ggml_backend_t be, ggml_backend_buffer_type_t buft,
                           std::vector<float> & res, double * ms) {
                session s;
                s.backend = be;
                ggml_init_params p = { 32u << 20, nullptr, true };
                s.ctx = ggml_init(p);
                ggml_tensor *w = nullptr, *x = nullptr;
                ggml_tensor * r = build_matmul(s.ctx, M0, N1, K, cont_b, &w, &x);
                s.gf = ggml_new_graph(s.ctx);
                ggml_build_forward_expand(s.gf, r);
                if (!alloc_ctx(s.ctx, buft, &s.buf)) { s.destroy(); return false; }
                ggml_tensor * wd = w->view_src ? w->view_src : w;
                ggml_backend_tensor_set(wd, vw.data(), 0, vw.size() * 4);
                ggml_backend_tensor_set(x, vx.data(), 0, vx.size() * 4);
                if (ggml_backend_graph_compute(be, s.gf) != GGML_STATUS_SUCCESS) {
                    s.destroy();
                    return false;
                }
                std::vector<double> t;
                for (int i = 0; i < 7; ++i) {
                    const double t0 = now_ms();
                    ggml_backend_graph_compute(be, s.gf);
                    t.push_back(now_ms() - t0);
                }
                *ms = median(t);
                res.resize(ggml_nelements(r));
                ggml_backend_tensor_get(r, res.data(), 0, res.size() * 4);
                s.destroy();
                return true;
            };

            std::vector<float> rv;
            double tv = 0, tc = -1;
            const bool okv = run(vc4, ggml_backend_get_default_buffer_type(vc4), rv, &tv);
            if (cont_b) {
                // ggml's own CPU kernel only accepts the contiguous layout
                std::vector<float> rc;
                run(cpu, ggml_backend_cpu_buffer_type(), rc, &tc);
                if (!rc.empty()) ref = rc;
            }
            if (!okv) {
                say(out, cap, "  (%d,%d,%d): FAILED to run\n", M0, N1, K);
                failures++;
                continue;
            }
            const stats st = compare(rv, ref, 3e-4f);
            const double flops = 2.0 * M0 * N1 * K;
            if (tc >= 0) {
                say(out, cap,
                    "  (%3d,%3d,%3d): mismatch=%d max_abs=%.3g max_rel=%.3g | "
                    "%8.3f | %8.3f | VC4 %.2f CPU %.2f GFLOP/s\n",
                    M0, N1, K, st.mismatch, st.max_abs, st.max_rel, tv, tc,
                    flops / (tv * 1e6), flops / (tc * 1e6));
            } else {
                say(out, cap,
                    "  (%3d,%3d,%3d): mismatch=%d max_abs=%.3g max_rel=%.3g | "
                    "%8.3f | %8s | VC4 %.2f GFLOP/s (host reference)\n",
                    M0, N1, K, st.mismatch, st.max_abs, st.max_rel, tv, "-",
                    flops / (tv * 1e6));
            }
            if (st.mismatch) failures++;
        }
    }
    ggml_backend_free(vc4);
    ggml_backend_free(cpu);
    return failures == 0 ? 0 : 1;
}

// ==========================================================================
// (c) a llama-family block, end to end through ggml_backend_sched
// ==========================================================================
struct llama_model {
    std::vector<ggml_tensor *> weights;
    ggml_tensor * ids = nullptr;
    ggml_tensor * pos = nullptr;

    ggml_tensor * emb, *wq, *wk, *wv, *wo, *w1, *w2, *w3, *wout;
};

static llama_model build_model(ggml_context * ctx, int tok, int embd, int ff,
                               int vocab) {
    llama_model m;
    auto W = [&](int a, int b, const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b);
        ggml_set_name(t, name);
        m.weights.push_back(t);
        return t;
    };
    m.emb  = W(embd, vocab, "token_embd");
    m.wq   = W(embd, embd, "wq");
    m.wk   = W(embd, embd, "wk");
    m.wv   = W(embd, embd, "wv");
    m.wo   = W(embd, embd, "wo");
    m.w1   = W(embd, ff, "w1");
    m.w3   = W(embd, ff, "w3");
    m.w2   = W(ff, embd, "w2");
    m.wout = W(embd, vocab, "output");

    m.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tok);
    ggml_set_name(m.ids, "ids");
    m.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tok);
    ggml_set_name(m.pos, "pos");
    return m;
}

static ggml_tensor * build_graph(ggml_context * ctx, const llama_model & m,
                                 int embd) {
    ggml_tensor * h = ggml_get_rows(ctx, m.emb, m.ids);
    ggml_set_name(h, "h");

    ggml_tensor * n1 = ggml_rms_norm(ctx, h, 1e-5f);
    ggml_tensor * q  = ggml_mul_mat(ctx, m.wq, n1);
    ggml_tensor * k  = ggml_mul_mat(ctx, m.wk, n1);
    ggml_tensor * v  = ggml_mul_mat(ctx, m.wv, n1);
    // a real llama rotates q/k here; the backend does not implement ROPE, so
    // this is the node the scheduler must keep on the CPU.  ROPE wants the
    // token axis in ne[2], as in llama.cpp's [head_dim, n_head, n_tokens].
    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, embd, 1, (int)m.pos->ne[0]);
    q3 = ggml_rope_ext(ctx, q3, m.pos, nullptr, embd, GGML_ROPE_TYPE_NORMAL, 0,
                       10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_set_name(q3, "q_rope");
    q = ggml_reshape_2d(ctx, q3, embd, (int)m.pos->ne[0]);

    ggml_tensor * s   = ggml_mul_mat(ctx, k, q);   // [tok, tok]
    ggml_tensor * p   = ggml_soft_max(ctx, s);
    ggml_tensor * att = ggml_mul_mat(ctx, p, v);   // [embd, tok]
    ggml_tensor * o   = ggml_mul_mat(ctx, m.wo, att);
    ggml_tensor * x   = ggml_add(ctx, h, o);
    ggml_set_name(x, "resid1");

    ggml_tensor * n2   = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_tensor * gate = ggml_mul_mat(ctx, m.w1, n2);
    ggml_tensor * up   = ggml_mul_mat(ctx, m.w3, n2);
    ggml_tensor * act  = ggml_silu(ctx, gate);
    ggml_tensor * mm   = ggml_mul(ctx, act, up);
    ggml_tensor * ffw  = ggml_mul_mat(ctx, m.w2, mm);
    ggml_tensor * x2   = ggml_add(ctx, x, ffw);

    ggml_tensor * n3 = ggml_rms_norm(ctx, x2, 1e-5f);
    ggml_tensor * lg = ggml_mul_mat(ctx, m.wout, n3);
    ggml_tensor * sm = ggml_soft_max(ctx, lg);
    ggml_set_name(sm, "probs");
    return sm;
}

extern "C" int vc4ggml_test_llama(char * out, size_t cap) {
    ensure_registered();
    const int tok = 64, embd = 64, ff = 128, vocab = 256;

    ggml_backend_t vc4 = ggml_backend_dev_init(ggml_backend_dev_by_name("VC4"), nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!vc4 || !cpu) { say(out, cap, "FAIL: backend init\n"); return 1; }

    // identical random weights for both runs
    std::vector<std::vector<float>> wdata;
    std::vector<int32_t> ids(tok), pos(tok);
    for (int i = 0; i < tok; ++i) { ids[i] = (i * 37) % vocab; pos[i] = i; }
    {
        ggml_init_params p = { 32u << 20, nullptr, true };
        ggml_context * c = ggml_init(p);
        llama_model m = build_model(c, tok, embd, ff, vocab);
        rng_seed(4242);
        wdata.resize(m.weights.size());
        for (size_t i = 0; i < m.weights.size(); ++i) {
            const size_t n = (size_t)ggml_nelements(m.weights[i]);
            wdata[i].resize(n);
            for (auto & v : wdata[i]) v = rnd() * 0.08f;
        }
        ggml_free(c);
    }

    auto set_model = [&](const llama_model & m) {
        for (size_t i = 0; i < m.weights.size(); ++i)
            ggml_backend_tensor_set(m.weights[i], wdata[i].data(), 0,
                                    wdata[i].size() * 4);
        ggml_backend_tensor_set(m.ids, ids.data(), 0, ids.size() * 4);
        ggml_backend_tensor_set(m.pos, pos.data(), 0, pos.size() * 4);
    };

    // ---- CPU reference: weights + graph both in CPU buffers ----
    std::vector<float> ref;
    {
        ggml_context * wctx = ggml_init({ 16u << 20, nullptr, true });
        llama_model m = build_model(wctx, tok, embd, ff, vocab);
        ggml_backend_buffer_t wbuf = nullptr;
        if (!alloc_ctx(wctx, ggml_backend_cpu_buffer_type(), &wbuf)) {
            say(out, cap, "FAIL: CPU weight buffer\n");
            return 1;
        }
        set_model(m);

        session s; s.backend = cpu;
        s.ctx = ggml_init({ 64u << 20, nullptr, true });
        ggml_tensor * o = build_graph(s.ctx, m, embd);
        s.gf = ggml_new_graph(s.ctx);
        ggml_build_forward_expand(s.gf, o);
        if (!alloc_ctx(s.ctx, ggml_backend_cpu_buffer_type(), &s.buf)) {
            say(out, cap, "FAIL: CPU graph buffer\n");
            return 1;
        }
        const double t0 = now_ms();
        const enum ggml_status st = ggml_backend_graph_compute(cpu, s.gf);
        const double tcpu = now_ms() - t0;
        if (st != GGML_STATUS_SUCCESS) {
            say(out, cap, "FAIL: CPU reference graph\n");
            s.destroy();
            return 1;
        }
        ref.resize(ggml_nelements(o));
        ggml_backend_tensor_get(o, ref.data(), 0, ref.size() * 4);
        double wb = 0;
        for (auto & w : wdata) wb += (double)w.size() * 4;
        say(out, cap, "llama block: tok=%d embd=%d ff=%d vocab=%d, "
                      "weights %.2f MiB, CPU reference %.2f ms\n",
            tok, embd, ff, vocab, wb / 1048576.0, tcpu);
        s.destroy();
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }

    // ---- VC4 + scheduler (CPU fallback), weights on the device ----
    int rc = 1;
    {
        ggml_context * wctx = ggml_init({ 16u << 20, nullptr, true });
        llama_model m = build_model(wctx, tok, embd, ff, vocab);
        ggml_backend_buffer_t wbuf = nullptr;
        if (!alloc_ctx(wctx, ggml_backend_get_default_buffer_type(vc4), &wbuf)) {
            say(out, cap, "FAIL: VC4 weight buffer\n");
            return 1;
        }
        set_model(m);

        session s; s.backend = vc4;
        s.ctx = ggml_init({ 64u << 20, nullptr, true });
        ggml_tensor * o = build_graph(s.ctx, m, embd);
        s.gf = ggml_new_graph(s.ctx);
        ggml_build_forward_expand(s.gf, o);

        ggml_backend_t bes[2] = { vc4, cpu };
        ggml_backend_buffer_type_t bufts[2] = {
            ggml_backend_get_default_buffer_type(vc4),
            ggml_backend_cpu_buffer_type(),
        };
        ggml_backend_sched_t sched =
            ggml_backend_sched_new(bes, bufts, 2, 4096, false, false);

        double best = 1e30, first = -1;
        enum ggml_status st = GGML_STATUS_FAILED;
        for (int i = 0; i < 3; ++i) {
            const double t0 = now_ms();
            st = ggml_backend_sched_graph_compute(sched, s.gf);
            const double dt = now_ms() - t0;
            if (i == 0) first = dt;
            best = std::min(best, dt);
            if (st != GGML_STATUS_SUCCESS) break;
        }
        if (st != GGML_STATUS_SUCCESS) {
            say(out, cap, "FAIL: scheduled graph status %d\n", (int)st);
            ggml_backend_sched_free(sched);
            s.destroy();
            ggml_backend_free(vc4);
            ggml_backend_free(cpu);
            return 1;
        }

        std::vector<float> got(ggml_nelements(o));
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * 4);
        const stats cmp = compare(got, ref, 2e-3f);

        int n_vc4 = 0, n_cpu = 0, splits = ggml_backend_sched_get_n_splits(sched);
        std::string cpu_ops, vc4_ops;
        double flops = 0;
        for (int i = 0; i < s.gf->n_nodes; ++i) {
            ggml_tensor * t = s.gf->nodes[i];
            if (t->op == GGML_OP_MUL_MAT)
                flops += 2.0 * t->ne[0] * t->ne[1] * t->src[0]->ne[1];
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
            const char * name = ggml_op_name(t->op);
            std::string & list = (b && strcmp(ggml_backend_name(b), "VC4") == 0)
                                     ? vc4_ops : cpu_ops;
            if (b && strcmp(ggml_backend_name(b), "VC4") == 0) n_vc4++; else n_cpu++;
            if (list.find(name) == std::string::npos) {
                list += list.empty() ? "" : ",";
                list += name;
            }
        }

        say(out, cap,
            "  VC4+sched: %.2f ms first, %.2f ms best (%.3f GFLOP/s of MUL_MAT), "
            "%d splits; nodes VC4=%d CPU=%d\n"
            "  on VC4: %s\n  fell back: %s\n"
            "  accuracy vs CPU reference: mismatch=%d max_abs=%.3g max_rel=%.3g\n",
            first, best, flops / (best * 1e6), splits, n_vc4, n_cpu,
            vc4_ops.c_str(), cpu_ops.c_str(), cmp.mismatch, cmp.max_abs,
            cmp.max_rel);
        rc = cmp.mismatch == 0 ? 0 : 1;

        ggml_backend_sched_free(sched);
        s.destroy();
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    ggml_backend_free(vc4);
    ggml_backend_free(cpu);
    return rc;
}
