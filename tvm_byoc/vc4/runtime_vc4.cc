/*
 * libvc4_tvm.so - the JSON runtime module of the VC4 Relax BYOC backend.
 *
 * The Relax codegen (tvm_byoc/vc4/codegen.py) writes one descriptor per fused
 * subgraph and constructs this module through the global function
 * runtime.VC4JSONRuntimeCreate.  The module exposes the subgraph symbol and,
 * when invoked, either
 *
 *   record  writes one log line per call and fills the output with NaN,
 *   host    computes the fused op on the ARM as a plumbing reference,
 *   vc4     drives libvc4infer: vc4_matmul then vc4_elementwise(ADD).
 *
 * record and host never touch the device; vc4 is the only mode that opens the
 * engine.  VC4_TVM_MODE overrides the mode stored in the descriptor, so the
 * same artifact can be checked on a CPU-only host and then executed on the
 * board without regenerating it.
 *
 * The module derives from ffi::ModuleObj directly rather than from the
 * TVM-internal JSONRuntimeBase: the 0.27 wheel ships
 * src/runtime/extra/contrib/json/json_runtime.h but not the headers it
 * includes (support/io.h, support/bytes_io.h), so the base class cannot be
 * compiled against an installed wheel.  The descriptor keeps the same shape:
 * a symbol, an input list and a node list with a shape per tensor.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#ifdef VC4_TVM_ENGINE
#include <vc4infer.h>
#endif

namespace vc4tvm {

namespace ffi = tvm::ffi;
namespace json = tvm::ffi::json;

/* ------------------------------------------------------------------ logging */

static void LogLine(const std::string& line) {
  static FILE* f = [] {
    const char* p = getenv("VC4_TVM_LOG");
    if (p && *p) return fopen(p, "a");
    return stderr;
  }();
  if (!f) return;
  fprintf(f, "%s\n", line.c_str());
  fflush(f);
}

static uint64_t Fnv1a(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

/* -------------------------------------------------------------- descriptor */

struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};

struct FusedNode {
  std::string op;
  std::string name;
  int64_t m = 0, n = 0, k = 0;
  /* 0 none, 1 scalar, 2 row (N), 3 full (M x N) */
  int bias_mode = 0;
  std::vector<float> bias;
};

struct Program {
  std::string symbol;
  std::string mode = "record";
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::vector<FusedNode> nodes;
};

static int64_t JInt(const json::Object& o, const char* key, int64_t dflt = 0) {
  if (!o.count(ffi::String(key))) return dflt;
  return o.at(ffi::String(key)).cast<int64_t>();
}

static std::string JStr(const json::Object& o, const char* key,
                        const char* dflt = "") {
  if (!o.count(ffi::String(key))) return std::string(dflt);
  return std::string(o.at(ffi::String(key)).cast<ffi::String>());
}

static std::vector<int64_t> JShape(const json::Object& o, const char* key) {
  std::vector<int64_t> out;
  if (!o.count(ffi::String(key))) return out;
  for (const auto& e : o.at(ffi::String(key)).cast<json::Array>())
    out.push_back(e.cast<int64_t>());
  return out;
}

static std::vector<float> JFloats(const json::Object& o, const char* key) {
  std::vector<float> out;
  if (!o.count(ffi::String(key))) return out;
  for (const auto& e : o.at(ffi::String(key)).cast<json::Array>())
    out.push_back(static_cast<float>(e.cast<double>()));
  return out;
}

static Program ParseProgram(const std::string& text) {
  json::Object o = json::Parse(text).cast<json::Object>();
  Program p;
  p.symbol = JStr(o, "symbol");
  p.mode = JStr(o, "mode", "record");
  for (const auto& e : o.at(ffi::String("inputs")).cast<json::Array>()) {
    json::Object t = e.cast<json::Object>();
    p.inputs.push_back({JStr(t, "name"), JShape(t, "shape")});
  }
  for (const auto& e : o.at(ffi::String("outputs")).cast<json::Array>()) {
    json::Object t = e.cast<json::Object>();
    p.outputs.push_back({JStr(t, "name", "out"), JShape(t, "shape")});
  }
  for (const auto& e : o.at(ffi::String("nodes")).cast<json::Array>()) {
    json::Object nd = e.cast<json::Object>();
    FusedNode n;
    n.op = JStr(nd, "op");
    n.name = JStr(nd, "name");
    json::Object a = nd.at(ffi::String("attrs")).cast<json::Object>();
    n.m = JInt(a, "M");
    n.n = JInt(a, "N");
    n.k = JInt(a, "K");
    n.bias_mode = static_cast<int>(JInt(a, "bias_mode", 0));
    n.bias = JFloats(a, "bias_data");
    p.nodes.push_back(std::move(n));
  }
  TVM_FFI_ICHECK(!p.symbol.empty()) << "vc4 descriptor has no symbol";
  return p;
}

/* ------------------------------------------------------------- host tensors */

/* A borrowed host view of an argument.  The ffi Tensor owns the storage; this
 * struct only carries what the engine path needs to copy in and out. */
struct HostTensor {
  uint8_t* base = nullptr;
  size_t bytes = 0;
  int ndim = 0;
  std::vector<int64_t> shape;
  DLDataType dtype{};
  DLDevice device{};
};

static HostTensor AsHost(const ffi::Tensor& t, const char* what) {
  HostTensor h;
  h.dtype = t.dtype();
  h.device = t.device();
  h.ndim = t.ndim();
  ffi::ShapeView sv = t.shape();
  h.shape.assign(sv.data(), sv.data() + sv.size());
  TVM_FFI_ICHECK_EQ(static_cast<int>(h.dtype.code), static_cast<int>(kDLFloat))
      << what << ": only float32 is supported on VC4";
  TVM_FFI_ICHECK_EQ(h.dtype.bits, 32) << what << ": only float32 is supported on VC4";
  TVM_FFI_ICHECK_EQ(h.dtype.lanes, 1) << what << ": vector dtypes are not supported";
  TVM_FFI_ICHECK_EQ(static_cast<int>(h.device.device_type), static_cast<int>(kDLCPU))
      << what << ": the module consumes host tensors and owns the device copy";
  TVM_FFI_ICHECK(t.IsContiguous()) << what << ": non-contiguous tensor";
  h.base = static_cast<uint8_t*>(t.data_ptr()) + t.byte_offset();
  h.bytes = static_cast<size_t>(t.numel()) * sizeof(float);
  return h;
}

static void CheckShape(const HostTensor& h, const TensorSpec& spec) {
  TVM_FFI_ICHECK_EQ(h.ndim, static_cast<int>(spec.shape.size())) << spec.name << ": rank";
  for (size_t d = 0; d < spec.shape.size(); ++d) {
    TVM_FFI_ICHECK_EQ(h.shape[d], spec.shape[d]) << spec.name << ": shape";
  }
}

/* ------------------------------------------------------------------ module */

#ifdef VC4_TVM_ENGINE
struct EngineBufs {
  vc4_buf a, b, c, d, bias;
  bool ready = false;
};

struct EngineState {
  vc4_ctx* ctx = nullptr;
  std::mutex mu;
  std::string cache_dir;
  size_t arena = 0;

  vc4_ctx* Get() {
    if (ctx != nullptr) return ctx;
    const char* c = getenv("VC4_TVM_CACHE");
    if (c && *c) cache_dir = c;
    const char* mb = getenv("VC4_TVM_ARENA_MB");
    arena = mb ? static_cast<size_t>(atoi(mb)) << 20 : 0;
    vc4_status st = vc4_init(&ctx, cache_dir.empty() ? nullptr : cache_dir.c_str(), arena);
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_init failed: " << vc4_strerror(st);
    LogLine("{\"event\":\"engine_init\",\"backend\":\"vc4\",\"arena_bytes\":" +
            std::to_string(vc4_arena_bytes(ctx)) + "}");
    return ctx;
  }
};

static EngineState* Engine() {
  static EngineState s;
  return &s;
}
#endif

class VC4JSONModule : public ffi::ModuleObj {
 public:
  VC4JSONModule(std::string symbol, std::string text, std::vector<std::string> const_names)
      : symbol_(std::move(symbol)), text_(std::move(text)), const_names_(std::move(const_names)),
        prog_(ParseProgram(text_)) {
    TVM_FFI_ICHECK_EQ(prog_.symbol, symbol_) << "descriptor symbol mismatch";
  }

  const char* kind() const override { return "vc4_json"; }

  int GetPropertyMask() const override {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::String InspectSource(const ffi::String& format) const override {
    (void)format;
    return text_;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
    if (name == "get_symbol") {
      return ffi::Function([self, this](ffi::PackedArgs, ffi::Any* rv) { *rv = symbol_; });
    }
    if (name == "get_const_vars") {
      return ffi::Function([self, this](ffi::PackedArgs, ffi::Any* rv) {
        ffi::Array<ffi::String> out;
        for (const auto& n : const_names_) out.push_back(ffi::String(n));
        *rv = out;
      });
    }
    if (name == "__init_" + symbol_) {
      return ffi::Function([self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(args.size(), 1U) << "__init_ takes the constant array";
        initialized_ = true;
        *rv = 0;
      });
    }
    if (name == symbol_) {
      return ffi::Function([self, this](ffi::PackedArgs args, ffi::Any* rv) {
        TVM_FFI_ICHECK(initialized_) << "module has not been initialised";
        Run(args);
        *rv = 0;
      });
    }
    return std::nullopt;
  }

  ffi::Bytes SaveToBytes() const override {
    std::string s;
    WriteStr(s, symbol_);
    WriteStr(s, text_);
    const char init = initialized_ ? 1 : 0;
    s.append(&init, 1);
    return ffi::Bytes(std::move(s));
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    std::string s(bytes.data(), bytes.size());
    size_t pos = 0;
    std::string symbol = ReadStr(s, pos);
    std::string text = ReadStr(s, pos);
    auto m = ffi::make_object<VC4JSONModule>(symbol, text, std::vector<std::string>{});
    m->initialized_ = pos < s.size() && s[pos] == 1;
    return ffi::Module(m);
  }

 private:
  static void WriteStr(std::string& s, const std::string& v) {
    uint64_t n = v.size();
    s.append(reinterpret_cast<const char*>(&n), sizeof(n));
    s.append(v);
  }
  static std::string ReadStr(const std::string& s, size_t& pos) {
    uint64_t n = 0;
    std::memcpy(&n, s.data() + pos, sizeof(n));
    pos += sizeof(n);
    std::string v(s.data() + pos, n);
    pos += n;
    return v;
  }

  const char* Mode() const {
    const char* env = getenv("VC4_TVM_MODE");
    if (env && *env) return env;
    return prog_.mode.c_str();
  }

  void Run(const ffi::PackedArgs& args) {
    TVM_FFI_ICHECK_EQ(args.size(), prog_.inputs.size() + prog_.outputs.size())
        << prog_.symbol << ": expected " << prog_.inputs.size() << " inputs and "
        << prog_.outputs.size() << " outputs";
    std::vector<HostTensor> in(prog_.inputs.size());
    for (size_t i = 0; i < in.size(); ++i) {
      ffi::Tensor t = args[i].cast<ffi::Tensor>();
      in[i] = AsHost(t, prog_.inputs[i].name.c_str());
      CheckShape(in[i], prog_.inputs[i]);
    }
    std::vector<HostTensor> out(prog_.outputs.size());
    for (size_t i = 0; i < out.size(); ++i) {
      ffi::Tensor t = args[prog_.inputs.size() + i].cast<ffi::Tensor>();
      out[i] = AsHost(t, "output");
      CheckShape(out[i], prog_.outputs[i]);
    }

    const std::string mode = Mode();
    for (size_t i = 0; i < prog_.nodes.size(); ++i) {
      const FusedNode& nd = prog_.nodes[i];
      TVM_FFI_ICHECK_EQ(nd.op, std::string("vc4.fused_matmul_add"))
          << prog_.symbol << ": unsupported node op " << nd.op;
      LogLine(CallRecord(nd, in[0], in[1], out[0], mode));
      if (mode == "record") {
        float* p = reinterpret_cast<float*>(out[0].base);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (size_t e = 0; e < out[0].bytes / 4; ++e) p[e] = nan;
      } else if (mode == "host") {
        HostMatMulAdd(nd, in[0], in[1], out[0]);
      } else if (mode == "vc4") {
#ifdef VC4_TVM_ENGINE
        EngineMatMulAdd(nd, i, in[0], in[1], out[0]);
#else
        TVM_FFI_THROW(InternalError)
            << "this libvc4_tvm.so was built without the engine "
               "(-DVC4_TVM_ENGINE); rebuild with tvm_byoc/vc4/build.sh";
#endif
      } else {
        TVM_FFI_THROW(ValueError)
            << "unknown VC4 BYOC mode " << mode << " (record, host, vc4)";
      }
    }
  }

  static std::string CallRecord(const FusedNode& nd, const HostTensor& x,
                                const HostTensor& w, const HostTensor& y,
                                const std::string& mode) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "{\"event\":\"call\",\"backend\":\"vc4\",\"mode\":\"%s\","
             "\"op\":\"%s\",\"node\":\"%s\",\"M\":%lld,\"N\":%lld,\"K\":%lld,"
             "\"bias_mode\":%d,\"bias_elems\":%zu,"
             "\"in_bytes\":[%zu,%zu],\"out_bytes\":%zu,"
             "\"x_fnv1a\":\"0x%016llx\",\"w_fnv1a\":\"0x%016llx\"}",
             mode.c_str(), nd.op.c_str(), nd.name.c_str(), (long long)nd.m,
             (long long)nd.n, (long long)nd.k, nd.bias_mode, nd.bias.size(),
             x.bytes, w.bytes, y.bytes,
             (unsigned long long)Fnv1a(x.base, x.bytes),
             (unsigned long long)Fnv1a(w.base, w.bytes));
    return std::string(buf);
  }

  static void HostMatMulAdd(const FusedNode& nd, const HostTensor& x,
                            const HostTensor& w, const HostTensor& y) {
    const float* A = reinterpret_cast<const float*>(x.base);
    const float* B = reinterpret_cast<const float*>(w.base);
    float* C = reinterpret_cast<float*>(y.base);
    for (int64_t m = 0; m < nd.m; ++m) {
      for (int64_t n = 0; n < nd.n; ++n) {
        float acc = 0.f;
        if (nd.bias_mode == 1) acc = nd.bias[0];
        else if (nd.bias_mode == 2) acc = nd.bias[n];
        else if (nd.bias_mode == 3) acc = nd.bias[m * nd.n + n];
        for (int64_t k = 0; k < nd.k; ++k) acc += A[m * nd.k + k] * B[k * nd.n + n];
        C[m * nd.n + n] = acc;
      }
    }
  }

#ifdef VC4_TVM_ENGINE
  /* The fused op is two engine launches: the GEMM writes an intermediate
   * tile, the elementwise ADD folds the bias.  Bytes never change on the ARM:
   * the bias is expanded to [M][N] once, at first use, and then lives in the
   * arena.  The per-node buffers are allocated once per module instance. */
  void EngineMatMulAdd(const FusedNode& nd, size_t node, const HostTensor& x,
                       const HostTensor& w, const HostTensor& y) {
    EngineState* eng = Engine();
    std::lock_guard<std::mutex> guard(eng->mu);
    vc4_ctx* ctx = eng->Get();
    if (bufs_.size() <= node) bufs_.resize(node + 1);
    EngineBufs& b = bufs_[node];
    vc4_status st = VC4_OK;
    const size_t a_bytes = (size_t)nd.m * nd.k * 4;
    const size_t b_bytes = (size_t)nd.k * nd.n * 4;
    const size_t c_bytes = (size_t)nd.m * nd.n * 4;
    if (!b.ready) {
      st = vc4_alloc(ctx, a_bytes, &b.a);
      if (st == VC4_OK) st = vc4_alloc(ctx, b_bytes, &b.b);
      if (st == VC4_OK) st = vc4_alloc(ctx, c_bytes, &b.c);
      if (st == VC4_OK) st = vc4_alloc(ctx, c_bytes, &b.d);
      if (st == VC4_OK && nd.bias_mode != 0) st = vc4_alloc(ctx, c_bytes, &b.bias);
      TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_alloc: " << vc4_strerror(st) << " for "
                                    << nd.m << "x" << nd.n << "x" << nd.k;
      if (nd.bias_mode != 0) {
        std::vector<float> tiled(c_bytes / 4, 0.f);
        for (int64_t m = 0; m < nd.m; ++m)
          for (int64_t n = 0; n < nd.n; ++n) {
            if (nd.bias_mode == 1) tiled[m * nd.n + n] = nd.bias[0];
            else if (nd.bias_mode == 2) tiled[m * nd.n + n] = nd.bias[n];
            else tiled[m * nd.n + n] = nd.bias[m * nd.n + n];
          }
        st = vc4_upload(ctx, &b.bias, tiled.data(), c_bytes);
        TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_upload(bias): " << vc4_strerror(st);
      }
      b.ready = true;
    }
    st = vc4_upload(ctx, &b.a, x.base, a_bytes);
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_upload(x): " << vc4_strerror(st);
    st = vc4_upload(ctx, &b.b, w.base, b_bytes);
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_upload(w): " << vc4_strerror(st);

    vc4_reset_stats(ctx);
    st = vc4_matmul(ctx, &b.a, &b.b, &b.c, (uint32_t)nd.m, (uint32_t)nd.n,
                    (uint32_t)nd.k, VC4_LAYOUT_ROW_MAJOR);
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_matmul: " << vc4_strerror(st) << " ("
                                  << vc4_last_error(ctx) << ")";
    if (nd.bias_mode != 0) {
      st = vc4_elementwise(ctx, VC4_OP_ADD, &b.c, &b.bias, &b.d, (size_t)nd.m * nd.n);
    } else {
      st = vc4_blit(ctx, &b.c, &b.d, c_bytes);
    }
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4 add: " << vc4_strerror(st) << " ("
                                  << vc4_last_error(ctx) << ")";
    st = vc4_download(ctx, &b.d, y.base, c_bytes);
    TVM_FFI_ICHECK_EQ(st, VC4_OK) << "vc4_download: " << vc4_strerror(st);
    vc4_stats stats;
    vc4_get_stats(ctx, &stats);
    LogLine("{\"event\":\"engine\",\"backend\":\"vc4\",\"node\":\"" + nd.name +
            "\",\"launches\":" + std::to_string(stats.launches) + ",\"flops\":" +
            std::to_string(stats.flops) + "}");
  }
#endif

  std::string symbol_;
  std::string text_;
  std::vector<std::string> const_names_;
  Program prog_;
  bool initialized_ = true; /* this codegen folds constants into the descriptor */
#ifdef VC4_TVM_ENGINE
  std::vector<EngineBufs> bufs_;
#endif
};

static ffi::Module Create(ffi::String symbol, ffi::String text,
                          const ffi::Array<ffi::String>& const_names) {
  std::vector<std::string> names;
  for (const auto& n : const_names) names.push_back(std::string(n));
  auto mod = ffi::make_object<VC4JSONModule>(std::string(symbol), std::string(text),
                                             std::move(names));
  return ffi::Module(mod);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.VC4JSONRuntimeCreate", Create)
      .def("ffi.Module.load_from_bytes.vc4_json", VC4JSONModule::LoadFromBytes);
}

}  // namespace vc4tvm
