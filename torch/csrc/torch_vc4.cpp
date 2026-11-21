/*
 * torch_vc4 - a deliberately tiny PyTorch PrivateUse1 backend for the
 * Raspberry Pi VideoCore IV, running on top of libvc4infer over vc4d.
 *
 * Design rules (kept deliberately narrow and honest):
 *   - device "vc4" == c10::DeviceType::PrivateUse1, one device, index 0;
 *   - every tensor lives inside one region of the engine's dma-buf arena,
 *     obtained through one vc4_alloc() and sub-allocated here;
 *   - only these aten ops exist on vc4:
 *       empty.memory_format, empty_strided, copy_, mm, matmul (2-D),
 *       add.Tensor, mul.Tensor, relu, normal_
 *     everything else hits a catch-all PrivateUse1 fallback that throws.
 *     There is no CPU fallback path anywhere in this file.
 *   - all engine calls are synchronous (the daemon's LAUNCH_WAIT round trip),
 *     so there is no stream/queue and no async error surface.
 *
 * This file deliberately avoids torch/extension.h (the C++ frontend plus
 * pybind11): a 1 GB board cannot compile it.  Only ATen + torch/library.h
 * are used, and the Python-visible helpers are registered as a small custom
 * operator namespace (torch.ops.torch_vc4.*).
 */
#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "vc4infer.h"

namespace torch_vc4 {

constexpr c10::DeviceType kDev = c10::DeviceType::PrivateUse1;
constexpr c10::DeviceIndex kDevIndex = 0;
constexpr size_t kAlign = 64;      /* TMU/DMA lines */
constexpr size_t kSlack = 64;      /* kernel over-read/write guard */
constexpr size_t kReserveWant = 24u << 20;   /* of the 64 MiB client arena */
constexpr size_t kReserveMin = 4u << 20;

/* ------------------------------------------------------------------ state */

struct Block {
    size_t off;      /* offset inside the torch region */
    size_t size;
    bool used;
};

struct State {
    vc4_ctx *ctx = nullptr;
    uint8_t *base = nullptr;     /* host mapping of the torch region */
    uint32_t region_off = 0;     /* region offset inside the arena */
    size_t cap = 0;              /* region size in bytes */
    std::vector<Block> blocks;
    std::mutex gpu_mu;           /* vc4_ctx is not thread safe */
    std::mutex alloc_mu;
    std::mt19937 rng{std::random_device{}()};
};

/* Leaked on purpose: tensors may outlive static destruction at exit. */
static State &state()
{
    static State *s = new State();
    return *s;
}

/* The engine tree this build was configured against (setup.py passes
 * -DVC4_ENGINE_DIR=...).  Both the kernel-blob cache and the JIT generator
 * default to that tree, so a build never mixes blobs from another kgen. */
#ifndef VC4_ENGINE_DIR
#define VC4_ENGINE_DIR "/home/raspi/work/vc4-stack/engine"
#endif

static std::string engine_path(const char *rel)
{
    return std::string(VC4_ENGINE_DIR) + "/" + rel;
}

static std::string default_cache_dir()
{
    const char *e = getenv("VC4INFER_CACHE");
    if (e && *e)
        return e;
    return engine_path("kcache");
}

static std::string init_failure;

static void ensure_init()
{
    static std::once_flag once;
    std::call_once(once, [] {
        State &s = state();
        /* Never touch /dev/mem from this process: force the vc4d backend
         * unless the caller explicitly chose otherwise. */
        if (!getenv("VC4INFER_DAEMON"))
            setenv("VC4INFER_DAEMON", "1", 0);
        if (!getenv("VC4INFER_KGEN")) {
            std::string kg = engine_path("tools/kgen.py");
            setenv("VC4INFER_KGEN", kg.c_str(), 0);
        }
        /* The engine's kgen.py imports the QPU encoder modules (qenc.py,
         * gen.py, vpmwords.py) from $ASM3_DIR, which it does NOT default to
         * the engine tree -- without it the JIT of a shape that is not in the
         * kernel cache fails.  Point it at the first tree that has qenc.py. */
        if (!getenv("ASM3_DIR")) {
            std::vector<std::string> cand;
            cand.push_back(engine_path("tools/qenc.py"));
            cand.push_back(engine_path("../asm3/qenc.py"));
            cand.push_back(std::string(getenv("HOME") ? getenv("HOME")
                                                      : "/home/raspi") +
                           "/work/asm3/qenc.py");
            cand.push_back("/home/raspi/work/asm3/qenc.py");
            for (const auto &c : cand) {
                if (access(c.c_str(), R_OK) != 0)
                    continue;
                std::string dir = c;
                size_t slash = dir.rfind('/');
                if (slash != std::string::npos) {
                    dir.resize(slash);
                    setenv("ASM3_DIR", dir.c_str(), 0);
                    break;
                }
            }
        }
        std::string cache = default_cache_dir();

        vc4_ctx *ctx = nullptr;
        vc4_status st = vc4_init(&ctx, cache.c_str(), 0);
        if (st != VC4_OK) {
            init_failure = std::string("vc4_init failed: ") + vc4_strerror(st);
            return;
        }
        s.ctx = ctx;

        size_t want = kReserveWant;
        if (const char *mb = getenv("TORCH_VC4_ARENA_MB")) {
            long v = strtol(mb, nullptr, 0);
            if (v > 0)
                want = (size_t)v << 20;
        }
        /* The daemon's arena is shared with every other client on the board,
         * and it does not reclaim a dead client's region promptly, so a
         * transient "arena exhausted" is normal: retry for ~10 s. */
        vc4_buf region{0, 0};
        for (int attempt = 0; attempt < 40; ++attempt) {
            for (;;) {
                st = vc4_alloc(ctx, want, &region);
                if (st == VC4_OK || want <= kReserveMin)
                    break;
                want /= 2;
            }
            if (st == VC4_OK)
                break;
            if (attempt == 39)
                break;
            if (getenv("TORCH_VC4_DEBUG"))
                fprintf(stderr, "[vc4] arena busy (%s), retrying\n",
                        vc4_strerror(st));
            usleep(250000);
        }
        if (st != VC4_OK) {
            init_failure = std::string("vc4_alloc(") +
                           std::to_string(kReserveMin) + " B) failed: " +
                           vc4_strerror(st) + " (arena bytes " +
                           std::to_string((unsigned long long)
                                              vc4_arena_bytes(ctx)) + ")";
            return;
        }
        s.base = (uint8_t *)vc4_host_ptr(ctx, &region);
        s.region_off = region.off;
        s.cap = region.bytes;
        s.blocks.push_back(Block{0, s.cap, false});
    });
    if (!state().ctx)
        TORCH_CHECK(false, "torch_vc4: ", init_failure);
}

static void throw_st(vc4_status st, const char *what)
{
    std::string msg = std::string("torch_vc4: ") + what + ": " +
                      vc4_strerror(st);
    const char *detail = vc4_last_error(state().ctx);
    if (detail && *detail)
        msg += std::string(" (") + detail + ")";
    TORCH_CHECK(false, msg);
}

/* ----------------------------------------------------------------- arena */

static size_t arena_alloc(size_t bytes)
{
    ensure_init();
    State &s = state();
    std::lock_guard<std::mutex> g(s.alloc_mu);
    size_t want = (bytes + kSlack + kAlign - 1) & ~(kAlign - 1);
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        Block &b = s.blocks[i];
        if (b.used || b.size < want)
            continue;
        /* NB: capture the offset before insert() - it can reallocate the
         * vector and invalidate `b`. */
        const size_t off = b.off;
        if (b.size == want) {
            b.used = true;
        } else {
            Block rest{off + want, b.size - want, false};
            b.size = want;
            b.used = true;
            s.blocks.insert(s.blocks.begin() + i + 1, rest);
        }
        if (getenv("TORCH_VC4_DEBUG"))
            fprintf(stderr, "[vc4] alloc %zu -> off %zu (base %p)\n", bytes,
                    off, (void *)(s.base + off));
        return off;
    }
    TORCH_CHECK(false, "torch_vc4: out of arena: requested ", want,
                " B, region ", (unsigned long long)s.cap,
                " B is full (torch region only; the rest belongs to the "
                "engine's scratch)");
}

static void arena_free(size_t off)
{
    State &s = state();
    std::lock_guard<std::mutex> g(s.alloc_mu);
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        if (s.blocks[i].off != off)
            continue;
        TORCH_INTERNAL_ASSERT(s.blocks[i].used);
        s.blocks[i].used = false;
        while (i + 1 < s.blocks.size() && !s.blocks[i + 1].used) {
            s.blocks[i].size += s.blocks[i + 1].size;
            s.blocks.erase(s.blocks.begin() + i + 1);
        }
        if (i > 0 && !s.blocks[i - 1].used) {
            s.blocks[i - 1].size += s.blocks[i].size;
            s.blocks.erase(s.blocks.begin() + i);
        }
        return;
    }
    TORCH_INTERNAL_ASSERT(false, "torch_vc4: freeing unknown arena offset");
}

/* c10 only calls a DataPtr's deleter when its *context* is non-null
 * (c10/util/UniqueVoidPtr.h: "the deleter is guaranteed to be called when the
 * unique pointer is destructed and the context is non-null").  The context is
 * therefore the data pointer itself: with a null context every arena
 * allocation leaked until the 24 MiB region was full (measured: a loop of
 * 4096-float tensors died after 1531 iterations).  The deleter recovers the
 * arena offset by subtraction. */
static void arena_deleter(void *p)
{
    State &s = state();
    ptrdiff_t off = static_cast<char *>(p) - reinterpret_cast<char *>(s.base);
    TORCH_INTERNAL_ASSERT(off >= 0 && (size_t)off < s.cap,
                          "torch_vc4: free of a pointer outside the arena");
    arena_free((size_t)off);
}

struct Vc4Allocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t n) override
    {
        size_t off = arena_alloc(n ? n : 1);
        State &s = state();
        void *p = s.base + off;
        return c10::DataPtr(p, p, &arena_deleter,
                            c10::Device(kDev, kDevIndex));
    }
    c10::DeleterFnPtr raw_deleter() const override { return &arena_deleter; }
    void copy_data(void *dest, const void *src, size_t count) const override
    {
        /* Byte copy of two arena addresses; never used to compute an op. */
        memmove(dest, src, count);
    }
};

/* A vc4_buf describing a tensor that lives in the arena. */
static vc4_buf buf_of(const at::Tensor &t)
{
    State &s = state();
    const char *p = static_cast<const char *>(t.const_data_ptr());
    ptrdiff_t local = p - reinterpret_cast<const char *>(s.base);
    size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(local >= 0 && (size_t)local + bytes <= s.cap,
                "torch_vc4: tensor storage is not inside the vc4 arena "
                "(ptr ", (void *)p, " base ", (void *)s.base, " local ",
                (long long)local, " bytes ", bytes, " cap ", s.cap, ")");
    return vc4_buf{(uint32_t)(s.region_off + (size_t)local), bytes};
}

/* ------------------------------------------------------------------ guard */

struct Vc4GuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;

    Vc4GuardImpl() = default;
    explicit Vc4GuardImpl(c10::DeviceType t)
    {
        TORCH_INTERNAL_ASSERT(t == kType);
    }

    c10::DeviceType type() const override { return kType; }

    c10::Device exchangeDevice(c10::Device d) const override
    {
        TORCH_CHECK(d.type() == kType && d.index() == kDevIndex,
                    "torch_vc4: invalid vc4 device ", d);
        return d;
    }
    c10::Device getDevice() const override
    {
        return c10::Device(kType, kDevIndex);
    }
    void setDevice(c10::Device d) const override
    {
        TORCH_CHECK(d.type() == kType && d.index() == kDevIndex,
                    "torch_vc4: invalid vc4 device ", d);
    }
    void uncheckedSetDevice(c10::Device) const noexcept override {}

    c10::Stream getStream(c10::Device) const noexcept override
    {
        return c10::Stream(c10::Stream::UNSAFE,
                           c10::Device(kType, kDevIndex), 0);
    }
    c10::Stream exchangeStream(c10::Stream) const noexcept override
    {
        return c10::Stream(c10::Stream::UNSAFE,
                           c10::Device(kType, kDevIndex), 0);
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }

    /* Every launch is a synchronous LAUNCH_WAIT round trip. */
    void synchronizeDevice(c10::DeviceIndex) const override {}
    void synchronizeStream(const c10::Stream &) const override {}
    bool queryStream(const c10::Stream &) const override { return true; }
};

/* ------------------------------------------------------------------ empty */

static void check_vc4_f32(const at::Tensor &t, const char *what)
{
    TORCH_CHECK(t.device().type() == kDev, "torch_vc4: ", what,
                ": expected a vc4 tensor, got ", t.device());
    TORCH_CHECK(t.scalar_type() == at::kFloat, "torch_vc4: ", what,
                ": only float32 is supported, got ", t.scalar_type());
}

static at::Tensor make_arena_tensor_dtype(c10::IntArrayRef sizes,
                                          at::ScalarType dtype)
{
    int64_t numel = 1;
    for (auto s : sizes) {
        TORCH_CHECK(s >= 0, "torch_vc4: negative size");
        numel *= s;
    }
    size_t bytes = (size_t)numel * c10::elementSize(dtype);
    auto dp = c10::GetAllocator(kDev)->allocate(bytes ? bytes : 1);
    auto *holder = new c10::DataPtr(std::move(dp));
    auto opts = at::TensorOptions()
                    .dtype(dtype)
                    .device(c10::Device(kDev, kDevIndex));
    return at::from_blob(holder->get(), sizes,
                         [holder](void *) { delete holder; }, opts);
}

static at::Tensor make_arena_tensor(c10::IntArrayRef sizes)
{
    return make_arena_tensor_dtype(sizes, at::kFloat);
}

static at::Tensor vc4_empty(c10::IntArrayRef size,
                            std::optional<c10::ScalarType> dtype,
                            std::optional<c10::Layout> layout,
                            std::optional<c10::Device> device,
                            std::optional<bool> pin_memory,
                            std::optional<c10::MemoryFormat> memory_format)
{
    TORCH_CHECK(!device || device->type() == kDev,
                "torch_vc4: empty called for device ",
                device ? device->str() : std::string("<none>"));
    const at::ScalarType want = dtype ? *dtype : at::kFloat;
    TORCH_CHECK(want == at::kFloat || want == at::kLong,
                "torch_vc4: ", c10::toString(want), " is unsupported by "
                "design: the VideoCore IV QPU and every engine kernel are "
                "float32-only (int64 is accepted for index/bookkeeping "
                "buffers only).  Emulating float64 would double storage and "
                "add conversion launches instead of computing in fp64, so it "
                "is rejected rather than silently downgraded.");
    TORCH_CHECK(!layout || *layout == at::kStrided,
                "torch_vc4: only strided layout is supported");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "torch_vc4: pinned memory is not supported");
    TORCH_CHECK(!memory_format ||
                    *memory_format == at::MemoryFormat::Contiguous,
                "torch_vc4: only contiguous memory format is supported");
    return make_arena_tensor_dtype(size, want);
}

static at::Tensor vc4_empty_strided(c10::IntArrayRef size,
                                    c10::IntArrayRef stride,
                                    std::optional<c10::ScalarType> dtype,
                                    std::optional<c10::Layout> layout,
                                    std::optional<c10::Device> device,
                                    std::optional<bool> pin_memory)
{
    /* Only accept strides that describe a contiguous tensor: our tensors are
     * packed, and pretending to support arbitrary strides would silently
     * produce wrong results. */
    /* torch's contiguity rule: unit dims (size <= 1) impose no stride, and a
     * zero-size tensor is contiguous whatever its strides. */
    int64_t expect = 1;
    bool contiguous = true;
    for (int64_t i = (int64_t)size.size() - 1; i >= 0; --i) {
        if (size[i] <= 1)
            continue;
        if (stride[i] != expect) {
            contiguous = false;
            break;
        }
        expect *= size[i];
    }
    TORCH_CHECK(contiguous,
                "torch_vc4: empty_strided: only contiguous strides are "
                "supported on vc4");
    return vc4_empty(size, dtype, layout, device, pin_memory,
                     at::MemoryFormat::Contiguous);
}

/* ------------------------------------------------------------------- copy */

static at::Tensor &vc4_copy_(at::Tensor &self, const at::Tensor &src,
                             bool non_blocking)
{
    (void)non_blocking;
    ensure_init();
    const bool self_vc4 = self.device().type() == kDev;
    const bool src_vc4 = src.device().type() == kDev;
    TORCH_CHECK(self_vc4 || src_vc4,
                "torch_vc4: copy_ called with neither side on vc4");
    TORCH_CHECK(self.sizes() == src.sizes(),
                "torch_vc4: copy_ needs identical shapes (got ", self.sizes(),
                " and ", src.sizes(), "); broadcasting is not implemented");
    TORCH_CHECK(self.scalar_type() == src.scalar_type(),
                "torch_vc4: copy_ does not convert dtypes (",
                self.scalar_type(), " vs ", src.scalar_type(), ")");
    TORCH_CHECK(self.scalar_type() == at::kFloat ||
                    self.scalar_type() == at::kLong,
                "torch_vc4: copy_ supports float32 and int64 only, got ",
                self.scalar_type());

    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    size_t bytes = (size_t)self.numel() * self.element_size();

    if (src_vc4 && !self_vc4) {
        TORCH_CHECK(self.is_contiguous(),
                    "torch_vc4: copy_ to a non-contiguous CPU tensor is not "
                    "supported");
        vc4_buf b = buf_of(src);
        vc4_status st = vc4_download(s.ctx, &b, self.mutable_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ vc4->cpu");
        return self;
    }
    if (!src_vc4 && self_vc4) {
        at::Tensor c = src.is_contiguous() ? src : src.contiguous();
        vc4_buf b = buf_of(self);
        vc4_status st = vc4_upload(s.ctx, &b, c.const_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ cpu->vc4");
        return self;
    }
    /* vc4 -> vc4: stage through the host so the engine's dirty tracking and
     * cache maintenance stay in charge (byte copy, no arithmetic). */
    {
        at::Tensor c = src.is_contiguous() ? src : src.contiguous();
        vc4_buf b = buf_of(self);
        vc4_status st = vc4_upload(s.ctx, &b, c.const_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ vc4->vc4");
    }
    return self;
}

/* --------------------------------------------------------------------- mm */

static at::Tensor vc4_mm(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    check_vc4_f32(a, "mm(A)");
    check_vc4_f32(b, "mm(B)");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: mm needs 2-D tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: mm needs contiguous tensors "
                "(call .contiguous() first)");
    int64_t M = a.size(0), K = a.size(1), K2 = b.size(0), N = b.size(1);
    TORCH_CHECK(K == K2, "torch_vc4: mm shape mismatch ", a.sizes(), " x ",
                b.sizes());

    auto out = make_arena_tensor({M, N});
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bc = buf_of(out);
    vc4_status st = vc4_matmul(s.ctx, &ba, &bb, &bc, (uint32_t)M, (uint32_t)N,
                               (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
    if (st != VC4_OK)
        throw_st(st, "matmul");
    return out;
}

static at::Tensor vc4_matmul(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: matmul is implemented for 2-D x 2-D only (got ",
                a.dim(), "-D x ", b.dim(), "-D)");
    return vc4_mm(a, b);
}

/* ----------------------------------------------------------- binary/unary */

static void check_same_shape(const at::Tensor &a, const at::Tensor &b,
                             const char *what)
{
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vc4: ", what,
                " requires identical shapes, got ", a.sizes(), " and ",
                b.sizes(), "; broadcasting is not implemented");
}

/* Defined later in this file; the aten kernels below accept a CPU 0-dim
 * tensor operand (torch wraps Python scalars into one, e.g. ``0.5 * x``). */
static at::Tensor vc4_mul_scalar(const at::Tensor &a, double s);
static at::Tensor vc4_add_scalar(const at::Tensor &a, double s);

static bool cpu_scalar(const at::Tensor &t)
{
    return t.device().type() == at::kCPU && t.numel() == 1;
}

static double cpu_scalar_value(const at::Tensor &t)
{
    return t.item().toDouble();
}

static at::Tensor vc4_add(const at::Tensor &a, const at::Tensor &b,
                          const at::Scalar &alpha)
{
    ensure_init();
    if (cpu_scalar(a) && b.device().type() == kDev)
        return vc4_add_scalar(b, cpu_scalar_value(a) * alpha.toDouble());
    if (cpu_scalar(b) && a.device().type() == kDev)
        return vc4_add_scalar(a, cpu_scalar_value(b) * alpha.toDouble());
    check_vc4_f32(a, "add");
    check_vc4_f32(b, "add");
    check_same_shape(a, b, "add");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: add needs contiguous tensors");
    TORCH_CHECK(alpha.toDouble() == 1.0,
                "torch_vc4: add with alpha != 1 is not implemented (got ",
                alpha.toDouble(), ")");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s.ctx, VC4_OP_ADD, &ba, &bb, &bo,
                                    (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "add");
    return out;
}

static at::Tensor vc4_mul(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    if (cpu_scalar(a) && b.device().type() == kDev)
        return vc4_mul_scalar(b, cpu_scalar_value(a));
    if (cpu_scalar(b) && a.device().type() == kDev)
        return vc4_mul_scalar(a, cpu_scalar_value(b));
    check_vc4_f32(a, "mul");
    check_vc4_f32(b, "mul");
    check_same_shape(a, b, "mul");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: mul needs contiguous tensors");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s.ctx, VC4_OP_MUL, &ba, &bb, &bo,
                                    (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "mul");
    return out;
}

static at::Tensor vc4_relu(const at::Tensor &a)
{
    ensure_init();
    check_vc4_f32(a, "relu");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: relu needs a contiguous tensor");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_relu(s.ctx, &ba, &bo, (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "relu");
    return out;
}

/* ----------------------------------------------------------- normal_ (RNG) */
/* Random numbers are generated on the host (std::mt19937) and uploaded.  The
 * GPU has no RNG kernel; this is stated in the README rather than hidden. */
static at::Tensor &vc4_normal_(at::Tensor &self, double mean, double std,
                               std::optional<at::Generator> generator)
{
    (void)generator;
    ensure_init();
    check_vc4_f32(self, "normal_");
    TORCH_CHECK(self.is_contiguous(),
                "torch_vc4: normal_ needs a contiguous tensor");
    size_t n = (size_t)self.numel();
    std::vector<float> host(n);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.alloc_mu);
        const char *seed = getenv("TORCH_VC4_SEED");
        if (seed && *seed) {
            s.rng.seed((uint32_t)strtoul(seed, nullptr, 0));
            unsetenv("TORCH_VC4_SEED");
        }
        std::normal_distribution<float> dist((float)mean, (float)std);
        for (size_t i = 0; i < n; ++i)
            host[i] = dist(s.rng);
    }
    std::lock_guard<std::mutex> g(state().gpu_mu);
    vc4_buf b = buf_of(self);
    vc4_status st = vc4_upload(state().ctx, &b, host.data(),
                               n * sizeof(float));
    if (st != VC4_OK)
        throw_st(st, "normal_");
    return self;
}

/* ==========================================================================
 * Subset backend, part 2: the ops a small CNN needs.
 *
 * Everything that computes runs on the QPU.  The CPU only
 *   - moves bytes (channel padding, phase splits, nearest resampling,
 *     weight permutes, concat, compaction), and
 *   - materialises constants (fill_, a ones vector for sum, the adaptive-pool
 *     averaging matrix, per-channel scale/shift for batch_norm).
 * No multiply-accumulate happens on the ARM in this file.
 *
 * Coherency rule.  The arena mapping is cacheable, so:
 *   - every CPU write into the arena goes through vc4_upload() (it marks the
 *     range dirty, the engine cleans+invalidates it before the next launch);
 *   - every CPU read of arena memory that the GPU may have written goes
 *     through vc4_download() (clean+invalidate, then copy).
 * `host_get`/`host_put` below are the only two paths; `invalidate_for_cpu`
 * is used for the engine ops (conv/pool/softmax) that themselves memcpy
 * their input on the ARM: without it the ARM could read a stale, clean cache
 * line left over from a previous tenant of the same arena block.
 * ======================================================================== */

static inline int64_t al4(int64_t v) { return (v + 3) & ~int64_t(3); }
static inline int64_t al8(int64_t v) { return (v + 7) & ~int64_t(7); }
static inline int64_t al16(int64_t v) { return (v + 15) & ~int64_t(15); }

static int64_t numel_of(c10::IntArrayRef sizes)
{
    int64_t n = 1;
    for (auto s : sizes)
        n *= s;
    return n;
}

/* Read a vc4 tensor into host memory (invalidating the CPU's view first). */
static std::vector<float> host_get(const at::Tensor &t)
{
    ensure_init();
    State &s = state();
    std::vector<float> v((size_t)t.numel());
    if (v.empty())
        return v;
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf b = buf_of(t);
    vc4_status st = vc4_download(s.ctx, &b, v.data(), v.size() * sizeof(float));
    if (st != VC4_OK)
        throw_st(st, "download");
    return v;
}

/* Materialise host memory as a new vc4 tensor (marks the range dirty). */
static at::Tensor host_put(c10::IntArrayRef sizes, const float *src, size_t n)
{
    auto t = make_arena_tensor(sizes);
    if (n) {
        TORCH_CHECK(n == (size_t)t.numel(), "torch_vc4: internal size mismatch");
        ensure_init();
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf b = buf_of(t);
        vc4_status st = vc4_upload(s.ctx, &b, src, n * sizeof(float));
        if (st != VC4_OK)
            throw_st(st, "upload");
    }
    return t;
}

/* Drop the CPU's (possibly stale) cache lines for a GPU-written buffer.  The
 * copy is thrown away; the clean+invalidate in vc4_download is the point. */
static void invalidate_for_cpu(const at::Tensor &t)
{
    ensure_init();
    State &s = state();
    size_t n = (size_t)t.numel();
    if (!n)
        return;
    std::vector<float> scratch(n);
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf b = buf_of(t);
    vc4_status st = vc4_download(s.ctx, &b, scratch.data(), n * sizeof(float));
    if (st != VC4_OK)
        throw_st(st, "touch");
}

/* ------------------------------------------------------------------ views */

static at::Tensor vc4_view(const at::Tensor &t, c10::IntArrayRef sizes)
{
    ensure_init();
    check_vc4_f32(t, "view");
    /* torch's view semantics: one dim may be -1 (inferred) */
    std::vector<int64_t> sz(sizes.begin(), sizes.end());
    int64_t known = 1, minus_one = -1;
    for (size_t i = 0; i < sz.size(); ++i) {
        if (sz[i] == -1) {
            TORCH_CHECK(minus_one < 0, "torch_vc4: view: more than one -1 dim");
            minus_one = (int64_t)i;
        } else {
            TORCH_CHECK(sz[i] >= 0, "torch_vc4: view: negative size ", sz[i]);
            known *= sz[i];
        }
    }
    if (minus_one >= 0) {
        TORCH_CHECK(known > 0 && t.numel() % known == 0,
                    "torch_vc4: view: cannot infer -1 for ", t.sizes(),
                    " as ", sizes);
        sz[(size_t)minus_one] = t.numel() / known;
    }
    int64_t n = 1;
    for (auto v : sz)
        n *= v;
    TORCH_CHECK(n == t.numel(), "torch_vc4: view: cannot view ", t.sizes(),
                " as ", sizes, " (", n, " != ", t.numel(), " elements)");
    std::vector<int64_t> strides(sz.size());
    int64_t acc = 1;
    for (int64_t i = (int64_t)sz.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= sz[i];
    }
    auto out = make_arena_tensor({0});
    /* TensorImpl-level, not Tensor::set_(): the latter is an aten op and
     * would dispatch back into this backend. */
    auto *impl = out.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(t.storage());
    impl->set_storage_offset(t.storage_offset());
    impl->set_sizes_and_strides(sz, strides);
    return out;
}

static at::Tensor vc4_as_strided(const at::Tensor &t, c10::IntArrayRef sizes,
                                 c10::IntArrayRef strides,
                                 std::optional<int64_t> storage_offset)
{
    ensure_init();
    check_vc4_f32(t, "as_strided");
    TORCH_CHECK(sizes.size() == strides.size(),
                "torch_vc4: as_strided: size/stride rank mismatch");
    int64_t off = storage_offset ? *storage_offset : t.storage_offset();
    int64_t max_off = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        TORCH_CHECK(sizes[i] >= 0, "torch_vc4: as_strided: negative size");
        /* negative strides would need a base pointer past the end */
        TORCH_CHECK(strides[i] >= 0, "torch_vc4: as_strided: negative strides "
                                     "are not supported");
        if (sizes[i] > 1)
            max_off += (sizes[i] - 1) * strides[i];
    }
    int64_t elems = (int64_t)(t.storage().nbytes() / sizeof(float));
    TORCH_CHECK(off >= 0 && off + max_off <= elems,
                "torch_vc4: as_strided: view [", off, ", ", off + max_off,
                "] is outside the ", elems, "-element storage");
    auto out = make_arena_tensor({0});
    auto *impl = out.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(t.storage());
    impl->set_storage_offset(off);
    impl->set_sizes_and_strides(sizes, strides);
    return out;
}

/* -------------------------------------------------------- fill / scalar */

static at::Tensor &vc4_fill_(at::Tensor &self, const at::Scalar &value)
{
    ensure_init();
    TORCH_CHECK(self.device().type() == kDev && self.is_contiguous(),
                "torch_vc4: fill_ needs a contiguous vc4 tensor");
    size_t n = (size_t)self.numel();
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf b = buf_of(self);
    vc4_status st = VC4_OK;
    if (self.scalar_type() == at::kLong) {
        std::vector<int64_t> buf(n, value.toLong());
        st = vc4_upload(s.ctx, &b, buf.data(), n * sizeof(int64_t));
    } else {
        check_vc4_f32(self, "fill_");
        std::vector<float> buf(n, (float)value.toDouble());
        st = vc4_upload(s.ctx, &b, buf.data(), n * sizeof(float));
    }
    if (st != VC4_OK)
        throw_st(st, "fill_");
    return self;
}

static at::Tensor &vc4_zero_(at::Tensor &self)
{
    return vc4_fill_(self, 0.0);
}

/* Materialise a strided view into a contiguous tensor.  Pure byte gather on
 * the ARM (no arithmetic), used by contiguous()/clone() and by autograd for
 * gradients that arrive as expanded/0-stride views. */
static at::Tensor vc4_materialize(const at::Tensor &t)
{
    ensure_init();
    check_vc4_f32(t, "contiguous");
    std::vector<int64_t> sz(t.sizes().begin(), t.sizes().end());
    int64_t n = 1;
    for (auto v : sz)
        n *= v;
    std::vector<float> dst((size_t)n);
    if (n) {
        /* read the whole storage once (this also invalidates the CPU's view
         * of it), then gather in host memory */
        int64_t elems = (int64_t)(t.storage().nbytes() / sizeof(float));
        auto flat = make_arena_tensor({0});
        auto *impl = flat.unsafeGetTensorImpl();
        impl->set_storage_keep_dtype(t.storage());
        impl->set_storage_offset(0);
        std::vector<int64_t> fsz{(int64_t)elems}, fst{1};
        impl->set_sizes_and_strides(fsz, fst);
        std::vector<float> src = host_get(flat);
        int64_t off = t.storage_offset();
        std::vector<int64_t> idx(sz.size(), 0);
        for (int64_t lin = 0; lin < n; ++lin) {
            int64_t rem = lin, so = off;
            for (int64_t d = (int64_t)sz.size() - 1; d >= 0; --d) {
                int64_t i = sz[d] ? rem % sz[d] : 0;
                rem = sz[d] ? rem / sz[d] : 0;
                so += i * t.stride(d);
            }
            TORCH_CHECK(so >= 0 && so < elems, "torch_vc4: contiguous: gather "
                        "index ", so, " outside the ", elems, "-element "
                        "storage");
            dst[(size_t)lin] = src[so];
        }
    }
    return host_put(sz, dst.data(), dst.size());
}

static at::Tensor vc4_contiguous(const at::Tensor &t, c10::MemoryFormat fmt)
{
    TORCH_CHECK(fmt == at::MemoryFormat::Contiguous ||
                    fmt == at::MemoryFormat::Preserve,
                "torch_vc4: only the contiguous memory format is supported");
    if (t.is_contiguous())
        return t;
    return vc4_materialize(t);
}

static at::Tensor vc4_clone(const at::Tensor &t,
                            std::optional<c10::MemoryFormat> fmt)
{
    (void)fmt;
    return vc4_materialize(t);
}

static at::Scalar vc4_local_scalar_dense(const at::Tensor &t)
{
    TORCH_CHECK(t.numel() == 1, "torch_vc4: item() needs a one-element tensor, "
                               "got ", t.sizes());
    ensure_init();
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf b = buf_of(t);
    if (t.scalar_type() == at::kLong) {
        int64_t v = 0;
        vc4_status st = vc4_download(s.ctx, &b, &v, sizeof(v));
        if (st != VC4_OK)
            throw_st(st, "item");
        return at::Scalar(v);
    }
    float v = 0;
    vc4_status st = vc4_download(s.ctx, &b, &v, sizeof(v));
    if (st != VC4_OK)
        throw_st(st, "item");
    return at::Scalar((double)v);
}

/* --------------------------------------------------- elementwise bindings */

static at::Tensor vc4_sigmoid_op(const at::Tensor &a)
{
    ensure_init();
    check_vc4_f32(a, "sigmoid");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: sigmoid needs a contiguous "
                                  "tensor");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_sigmoid(s.ctx, &ba, &bo, (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "sigmoid");
    return out;
}

static at::Tensor vc4_silu_op(const at::Tensor &a)
{
    ensure_init();
    check_vc4_f32(a, "silu");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: silu needs a contiguous tensor");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_silu(s.ctx, &ba, &bo, (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "silu");
    return out;
}

static at::Tensor vc4_rsqrt_op(const at::Tensor &a)
{
    ensure_init();
    check_vc4_f32(a, "rsqrt");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: rsqrt needs a contiguous tensor");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_rsqrt(s.ctx, &ba, &bo, (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "rsqrt");
    return out;
}

static at::Tensor vc4_relu_bwd(const at::Tensor &a, const at::Tensor &dy)
{
    ensure_init();
    check_vc4_f32(a, "relu_backward(a)");
    check_vc4_f32(dy, "relu_backward(dy)");
    TORCH_CHECK(a.sizes() == dy.sizes(), "torch_vc4: relu_backward shape "
                "mismatch ", a.sizes(), " vs ", dy.sizes());
    TORCH_CHECK(a.is_contiguous() && dy.is_contiguous(),
                "torch_vc4: relu_backward needs contiguous tensors");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bdy = buf_of(dy), bo = buf_of(out);
    vc4_status st = vc4_relu_backward(s.ctx, &ba, &bdy, &bo,
                                      (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "relu_backward");
    return out;
}

/* t * s, s a host scalar.  The constant tensor is materialised on the host
 * (bytes, like normal_) and the multiply itself is the QPU kernel. */
static at::Tensor vc4_mul_scalar(const at::Tensor &a, double s)
{
    ensure_init();
    check_vc4_f32(a, "mul_scalar");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: mul_scalar needs a contiguous "
                                  "tensor");
    size_t n = (size_t)a.numel();
    std::vector<float> cbuf(n, (float)s);
    auto k = host_put(a.sizes(), cbuf.data(), n);
    auto out = make_arena_tensor(a.sizes());
    State &s_ = state();
    std::lock_guard<std::mutex> g(s_.gpu_mu);
    vc4_buf ba = buf_of(a), bk = buf_of(k), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s_.ctx, VC4_OP_MUL, &ba, &bk, &bo, n);
    if (st != VC4_OK)
        throw_st(st, "mul_scalar");
    return out;
}

static at::Tensor vc4_add_scalar(const at::Tensor &a, double s)
{
    ensure_init();
    check_vc4_f32(a, "add_scalar");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: add_scalar needs a contiguous "
                                  "tensor");
    size_t n = (size_t)a.numel();
    std::vector<float> cbuf(n, (float)s);
    auto k = host_put(a.sizes(), cbuf.data(), n);
    auto out = make_arena_tensor(a.sizes());
    State &s_ = state();
    std::lock_guard<std::mutex> g(s_.gpu_mu);
    vc4_buf ba = buf_of(a), bk = buf_of(k), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s_.ctx, VC4_OP_ADD, &ba, &bk, &bo, n);
    if (st != VC4_OK)
        throw_st(st, "add_scalar");
    return out;
}

/* ------------------------------------------------------ layout primitives */

/* [C][...] -> [cpad][...] with the extra channels zero (1-D and 3-D). */
static at::Tensor vc4_pad_ch(const at::Tensor &t, int64_t cpad)
{
    ensure_init();
    check_vc4_f32(t, "pad_ch");
    TORCH_CHECK(t.is_contiguous(), "torch_vc4: pad_ch needs a contiguous tensor");
    TORCH_CHECK(t.dim() == 1 || t.dim() == 3,
                "torch_vc4: pad_ch expects [C] or [C][H][W], got ", t.sizes());
    int64_t C = t.size(0);
    TORCH_CHECK(cpad >= C, "torch_vc4: pad_ch: cpad ", cpad, " < C ", C);
    std::vector<int64_t> osz(t.sizes().begin(), t.sizes().end());
    osz[0] = cpad;
    std::vector<float> src = host_get(t);
    size_t per = (size_t)(t.numel() / C);
    std::vector<float> dst((size_t)cpad * per, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        memcpy(dst.data() + (size_t)c * per, src.data() + (size_t)c * per,
               per * sizeof(float));
    return host_put(osz, dst.data(), dst.size());
}

/* [C][...] -> [c1-c0][...] (channel slice, 1-D and 3-D). */
static at::Tensor vc4_take_ch(const at::Tensor &t, int64_t c0, int64_t c1)
{
    ensure_init();
    check_vc4_f32(t, "take_ch");
    TORCH_CHECK(t.is_contiguous(), "torch_vc4: take_ch needs a contiguous "
                                  "tensor");
    TORCH_CHECK(t.dim() == 1 || t.dim() == 3,
                "torch_vc4: take_ch expects [C] or [C][H][W], got ", t.sizes());
    int64_t C = t.size(0);
    TORCH_CHECK(c0 >= 0 && c0 <= c1 && c1 <= C,
                "torch_vc4: take_ch range [", c0, ", ", c1, ") outside ", C);
    std::vector<int64_t> osz(t.sizes().begin(), t.sizes().end());
    osz[0] = c1 - c0;
    std::vector<float> src = host_get(t);
    size_t per = (size_t)(t.numel() / C);
    std::vector<float> dst((size_t)(c1 - c0) * per);
    memcpy(dst.data(), src.data() + (size_t)c0 * per, dst.size() * sizeof(float));
    return host_put(osz, dst.data(), dst.size());
}

/* Channel concat of [C][H][W] tensors (same H, W). */
static at::Tensor vc4_cat_ch(at::TensorList ts)
{
    ensure_init();
    TORCH_CHECK(!ts.empty(), "torch_vc4: cat_ch of an empty list");
    for (const auto &t : ts) {
        check_vc4_f32(t, "cat_ch");
        TORCH_CHECK(t.dim() == 3, "torch_vc4: cat_ch expects [C][H][W] tensors");
        TORCH_CHECK(t.is_contiguous(), "torch_vc4: cat_ch needs contiguous "
                                      "tensors");
        TORCH_CHECK(t.sizes() == ts[0].sizes() || (t.size(1) == ts[0].size(1) &&
                    t.size(2) == ts[0].size(2)),
                    "torch_vc4: cat_ch needs equal spatial sizes");
    }
    int64_t C = 0, H = ts[0].size(1), W = ts[0].size(2);
    for (const auto &t : ts)
        C += t.size(0);
    std::vector<float> dst((size_t)C * H * W);
    size_t at = 0;
    for (const auto &t : ts) {
        std::vector<float> src = host_get(t);
        memcpy(dst.data() + at, src.data(), src.size() * sizeof(float));
        at += src.size();
    }
    return host_put({C, H, W}, dst.data(), dst.size());
}

/* [N] -> [M][N], row replication (bias broadcast for linear). */
static at::Tensor vc4_bcast_rows(const at::Tensor &t, int64_t M)
{
    ensure_init();
    check_vc4_f32(t, "bcast_rows");
    TORCH_CHECK(t.dim() == 1 && t.is_contiguous(),
                "torch_vc4: bcast_rows expects a contiguous 1-D tensor");
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)M * src.size());
    for (int64_t r = 0; r < M; ++r)
        memcpy(dst.data() + (size_t)r * src.size(), src.data(),
               src.size() * sizeof(float));
    return host_put({M, (int64_t)src.size()}, dst.data(), dst.size());
}

/* [C] -> [C][H][W], per-channel replication (batch_norm broadcast). */
static at::Tensor vc4_bcast_ch(const at::Tensor &t, int64_t H, int64_t W)
{
    ensure_init();
    check_vc4_f32(t, "bcast_ch");
    TORCH_CHECK(t.dim() == 1 && t.is_contiguous(),
                "torch_vc4: bcast_ch expects a contiguous 1-D tensor");
    int64_t C = t.size(0);
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)C * H * W);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < H * W; ++i)
            dst[(size_t)c * H * W + i] = src[c];
    return host_put({C, H, W}, dst.data(), dst.size());
}

/* Nearest-neighbour resize with PyTorch's nearest (not -exact) index rule:
 * src = min(floor(dst * scale), in-1), scale = 1/scales when scales is given
 * (F.interpolate does that), else in/out. */
static at::Tensor vc4_up_nearest(const at::Tensor &t, int64_t ho, int64_t wo,
                                 double scale_h, double scale_w)
{
    ensure_init();
    check_vc4_f32(t, "upsample_nearest2d");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: upsample_nearest2d expects contiguous [C][H][W]");
    int64_t C = t.size(0), H = t.size(1), W = t.size(2);
    double sh = scale_h > 0 ? 1.0 / scale_h : (double)H / (double)ho;
    double sw = scale_w > 0 ? 1.0 / scale_w : (double)W / (double)wo;
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)C * ho * wo);
    std::vector<int64_t> hy((size_t)ho), wx((size_t)wo);
    for (int64_t y = 0; y < ho; ++y) {
        int64_t v = (int64_t)std::floor((double)y * sh);
        hy[y] = v < H - 1 ? v : H - 1;
    }
    for (int64_t x = 0; x < wo; ++x) {
        int64_t v = (int64_t)std::floor((double)x * sw);
        wx[x] = v < W - 1 ? v : W - 1;
    }
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            for (int64_t x = 0; x < wo; ++x)
                dst[((size_t)c * ho + y) * wo + x] =
                    src[((size_t)c * H + hy[y]) * W + wx[x]];
    return host_put({C, ho, wo}, dst.data(), dst.size());
}

/* ------------------------------------------------------------ weight maps */
/* All of these are byte copies / permutations of the weight tensor; no
 * arithmetic.  They exist because the engine's conv wants [Cout][Cin][k][k]
 * with Cin, Cout multiples of 16, and because the stride-2 and depthwise
 * cases are expressed as dense weight layouts (see docs/ONNX.md §6). */

/* [Cout][Cin][kh][kw] -> [cout_pad][cin_pad][kh][kw], zero padded. */
static at::Tensor vc4_pad_w(const at::Tensor &w, int64_t cin_pad,
                            int64_t cout_pad)
{
    ensure_init();
    check_vc4_f32(w, "pad_w");
    TORCH_CHECK(w.dim() == 4 && w.is_contiguous(),
                "torch_vc4: pad_w expects a contiguous 4-D weight");
    int64_t Cout = w.size(0), Cin = w.size(1), kh = w.size(2), kw = w.size(3);
    TORCH_CHECK(cin_pad >= Cin && cout_pad >= Cout, "torch_vc4: pad_w shrinks");
    std::vector<float> src = host_get(w);
    std::vector<float> dst((size_t)cout_pad * cin_pad * kh * kw, 0.0f);
    for (int64_t co = 0; co < Cout; ++co)
        for (int64_t ci = 0; ci < Cin; ++ci)
            memcpy(dst.data() + ((size_t)co * cin_pad + ci) * kh * kw,
                   src.data() + ((size_t)co * Cin + ci) * kh * kw,
                   (size_t)(kh * kw) * sizeof(float));
    return host_put({cout_pad, cin_pad, kh, kw}, dst.data(), dst.size());
}

/* depthwise [C][1][kh][kw] -> dense block-diagonal [cout_pad][cin_pad][kh][kw]. */
static at::Tensor vc4_dense_dw_w(const at::Tensor &w, int64_t cin_pad,
                                 int64_t cout_pad)
{
    ensure_init();
    check_vc4_f32(w, "dense_dw_w");
    TORCH_CHECK(w.dim() == 4 && w.size(1) == 1 && w.is_contiguous(),
                "torch_vc4: dense_dw_w expects a [C][1][kh][kw] weight");
    int64_t C = w.size(0), kh = w.size(2), kw = w.size(3);
    TORCH_CHECK(cout_pad >= C && cin_pad >= C, "torch_vc4: dense_dw_w shrinks");
    std::vector<float> src = host_get(w);
    std::vector<float> dst((size_t)cout_pad * cin_pad * kh * kw, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        memcpy(dst.data() + ((size_t)c * cin_pad + c) * kh * kw,
               src.data() + (size_t)c * kh * kw,
               (size_t)(kh * kw) * sizeof(float));
    return host_put({cout_pad, cin_pad, kh, kw}, dst.data(), dst.size());
}

/* stride-1 weight extraction back to the torch layout. */
static at::Tensor vc4_w_block(const at::Tensor &w, int64_t cout, int64_t cin,
                              bool depthwise)
{
    ensure_init();
    check_vc4_f32(w, "w_block");
    TORCH_CHECK(w.dim() == 4 && w.is_contiguous(),
                "torch_vc4: w_block expects a contiguous 4-D weight");
    int64_t kh = w.size(2), kw = w.size(3), Cin = w.size(1);
    std::vector<float> src = host_get(w);
    int64_t ncin = depthwise ? 1 : cin;
    std::vector<float> dst((size_t)cout * ncin * kh * kw, 0.0f);
    for (int64_t co = 0; co < cout; ++co) {
        if (depthwise) {
            memcpy(dst.data() + (size_t)co * kh * kw,
                   src.data() + ((size_t)co * Cin + co) * kh * kw,
                   (size_t)(kh * kw) * sizeof(float));
            continue;
        }
        for (int64_t ci = 0; ci < cin; ++ci)
            memcpy(dst.data() + ((size_t)co * ncin + ci) * kh * kw,
                   src.data() + ((size_t)co * Cin + ci) * kh * kw,
                   (size_t)(kh * kw) * sizeof(float));
    }
    return host_put({cout, ncin, kh, kw}, dst.data(), dst.size());
}

/* Phase-map (a, d) for one kernel tap: u = k - pad, a = u mod 2,
 * d = (u - a)/2, phase kernel row/col = d + q. */
struct PhaseTap {
    int64_t a, d;
    PhaseTap(int64_t k, int64_t pad)
    {
        int64_t u = k - pad;
        a = ((u % 2) + 2) % 2;
        d = (u - a) / 2;
    }
};

/* Dense [Cout][Cin][3][3] -> [Cout][4*Cin][3][3] phase weight. */
static at::Tensor vc4_phase_w(const at::Tensor &w, int64_t cin_pad,
                              int64_t cout_pad, int64_t pad, int64_t q)
{
    ensure_init();
    check_vc4_f32(w, "phase_w");
    TORCH_CHECK(w.dim() == 4 && w.size(2) == 3 && w.size(3) == 3 &&
                w.is_contiguous(),
                "torch_vc4: phase_w expects a contiguous 3x3 weight");
    int64_t Cout = w.size(0), Cin = w.size(1);
    TORCH_CHECK(Cin == cin_pad && Cout == cout_pad,
                "torch_vc4: phase_w expects an already padded weight");
    std::vector<float> src = host_get(w);
    int64_t CW = 4 * cin_pad;
    std::vector<float> dst((size_t)cout_pad * CW * 9, 0.0f);
    for (int64_t co = 0; co < cout_pad; ++co)
        for (int64_t ci = 0; ci < cin_pad; ++ci)
            for (int64_t kh = 0; kh < 3; ++kh) {
                PhaseTap th(kh, pad);
                for (int64_t kw = 0; kw < 3; ++kw) {
                    PhaseTap tw(kw, pad);
                    float v = src[((size_t)co * cin_pad + ci) * 9 + kh * 3 + kw];
                    int64_t blk = (th.a * 2 + tw.a) * cin_pad + ci;
                    dst[((size_t)co * CW + blk) * 9 + (th.d + q) * 3 +
                        (tw.d + q)] = v;
                }
            }
    return host_put({cout_pad, CW, 3, 3}, dst.data(), dst.size());
}

/* Phase weight extraction: [Cout_pad][4*Cin_pad][3][3] -> torch layout. */
static at::Tensor vc4_w_phase_block(const at::Tensor &w, int64_t cout,
                                    int64_t cin, int64_t pad, int64_t q,
                                    bool depthwise)
{
    ensure_init();
    check_vc4_f32(w, "w_phase_block");
    TORCH_CHECK(w.dim() == 4 && w.size(2) == 3 && w.size(3) == 3 &&
                w.is_contiguous(),
                "torch_vc4: w_phase_block expects a contiguous 3x3 weight");
    int64_t CW = w.size(1), cpad = CW / 4;
    std::vector<float> src = host_get(w);
    int64_t ncin = depthwise ? 1 : cin;
    std::vector<float> dst((size_t)cout * ncin * 9, 0.0f);
    for (int64_t co = 0; co < cout; ++co)
        for (int64_t ci = 0; ci < ncin; ++ci) {
            int64_t sci = depthwise ? co : ci;
            for (int64_t kh = 0; kh < 3; ++kh) {
                PhaseTap th(kh, pad);
                for (int64_t kw = 0; kw < 3; ++kw) {
                    PhaseTap tw(kw, pad);
                    int64_t blk = (th.a * 2 + tw.a) * cpad + sci;
                    dst[((size_t)co * ncin + ci) * 9 + kh * 3 + kw] =
                        src[((size_t)co * CW + blk) * 9 + (th.d + q) * 3 +
                            (tw.d + q)];
                }
            }
        }
    return host_put({cout, ncin, 3, 3}, dst.data(), dst.size());
}

/* ------------------------------------------------ phase / subsample images */

/* X[C][H][W] -> P[4C][hp][wp], P[(a*2+b)*C+c][i][j] = X[c][2i+a][2j+b]
 * (zero outside the image). */
static at::Tensor vc4_phase_split(const at::Tensor &t, int64_t hp, int64_t wp)
{
    ensure_init();
    check_vc4_f32(t, "phase_split");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: phase_split expects contiguous [C][H][W]");
    int64_t C = t.size(0), H = t.size(1), W = t.size(2);
    TORCH_CHECK(hp > 0 && wp > 0, "torch_vc4: phase_split: empty phase image");
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)4 * C * hp * wp, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < hp; ++i)
            for (int64_t j = 0; j < wp; ++j)
                for (int64_t a = 0; a < 2; ++a)
                    for (int64_t b = 0; b < 2; ++b) {
                        int64_t y = 2 * i + a, x = 2 * j + b;
                        if (y >= H || x >= W)
                            continue;
                        dst[(((size_t)(a * 2 + b) * C + c) * hp + i) * wp + j] =
                            src[((size_t)c * H + y) * W + x];
                    }
    return host_put({4 * C, hp, wp}, dst.data(), dst.size());
}

/* Inverse of phase_split: P[4C][hp][wp] -> X[C][H][W]. */
static at::Tensor vc4_phase_merge(const at::Tensor &t, int64_t C, int64_t H,
                                  int64_t W)
{
    ensure_init();
    check_vc4_f32(t, "phase_merge");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous() && t.size(0) == 4 * C,
                "torch_vc4: phase_merge expects contiguous [4C][hp][wp]");
    int64_t hp = t.size(1), wp = t.size(2);
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)C * H * W, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < hp; ++i)
            for (int64_t j = 0; j < wp; ++j)
                for (int64_t a = 0; a < 2; ++a)
                    for (int64_t b = 0; b < 2; ++b) {
                        int64_t y = 2 * i + a, x = 2 * j + b;
                        if (y >= H || x >= W)
                            continue;
                        dst[((size_t)c * H + y) * W + x] =
                            src[(((size_t)(a * 2 + b) * C + c) * hp + i) * wp + j];
                    }
    return host_put({C, H, W}, dst.data(), dst.size());
}

/* X[C][H][W] -> X[:, ::2, ::2] */
static at::Tensor vc4_subsample2(const at::Tensor &t)
{
    ensure_init();
    check_vc4_f32(t, "subsample2");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: subsample2 expects contiguous [C][H][W]");
    int64_t C = t.size(0), H = t.size(1), W = t.size(2);
    int64_t ho = (H - 1) / 2 + 1, wo = (W - 1) / 2 + 1;
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)C * ho * wo);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            for (int64_t x = 0; x < wo; ++x)
                dst[((size_t)c * ho + y) * wo + x] =
                    src[((size_t)c * H + 2 * y) * W + 2 * x];
    return host_put({C, ho, wo}, dst.data(), dst.size());
}

/* Inverse of subsample2: scatter [C][ho][wo] into even positions of
 * [C][H][W], zeros elsewhere. */
static at::Tensor vc4_scatter2(const at::Tensor &t, int64_t H, int64_t W)
{
    ensure_init();
    check_vc4_f32(t, "scatter2");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: scatter2 expects contiguous [C][ho][wo]");
    int64_t C = t.size(0), ho = t.size(1), wo = t.size(2);
    TORCH_CHECK(2 * ho - 1 <= H && 2 * wo - 1 <= W,
                "torch_vc4: scatter2 shape mismatch");
    std::vector<float> src = host_get(t);
    std::vector<float> dst((size_t)C * H * W, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            for (int64_t x = 0; x < wo; ++x)
                dst[((size_t)c * H + 2 * y) * W + 2 * x] =
                    src[((size_t)c * ho + y) * wo + x];
    return host_put({C, H, W}, dst.data(), dst.size());
}

/* --------------------------------------------------------- engine: conv */

/* The arena allocator keeps 64 B of slack after every block; some engine
 * kernels require it (maxpool's 2x2 gather over-reads a row). */
static vc4_buf buf_of_slack(const at::Tensor &t)
{
    State &s = state();
    vc4_buf b = buf_of(t);
    ptrdiff_t local = (ptrdiff_t)((const char *)t.const_data_ptr() -
                                  (const char *)s.base);
    size_t room = s.cap - (size_t)local;
    if (b.bytes + 64 <= room)
        b.bytes += 64;
    return b;
}

/* conv2d, stride 1, NCHW, Cin/Cout multiples of 16 (the engine's im2col
 * path).  Everything above this line that pads or slices channels is a byte
 * copy; the MACs are in vc4_conv2d. */
static at::Tensor vc4_conv_raw(const at::Tensor &in, const at::Tensor &w,
                               const std::optional<at::Tensor> &bias,
                               int64_t pad)
{
    ensure_init();
    check_vc4_f32(in, "conv2d(input)");
    check_vc4_f32(w, "conv2d(weight)");
    /* [C][H][W] (what the Python layer passes) or [1][C][H][W] */
    TORCH_CHECK(w.dim() == 4, "torch_vc4: conv2d needs a 4-D weight");
    TORCH_CHECK(in.dim() == 3 || (in.dim() == 4 && in.size(0) == 1),
                "torch_vc4: conv2d needs [C][H][W] or [1][C][H][W] input, "
                "got ", in.sizes());
    TORCH_CHECK(in.is_contiguous() && w.is_contiguous(),
                "torch_vc4: conv2d needs contiguous tensors");
    const bool batched = in.dim() == 4;
    const int64_t C = in.size(batched ? 1 : 0),
                  H = in.size(batched ? 2 : 1),
                  W = in.size(batched ? 3 : 2);
    const int64_t Cout = w.size(0), KH = w.size(2), KW = w.size(3);
    TORCH_CHECK(w.size(1) == C, "torch_vc4: conv2d: weight has ", w.size(1),
                " input channels, input has ", C);
    TORCH_CHECK(KH == KW && (KH == 3 || KH == 1),
                "torch_vc4: conv2d supports 3x3 and 1x1 kernels, got ", KH,
                "x", KW);
    TORCH_CHECK(KH == 3 || pad == 0,
                "torch_vc4: conv2d 1x1 does not take padding");
    TORCH_CHECK(C % 16 == 0 && Cout % 16 == 0,
                "torch_vc4: conv2d engine path needs Cin%16==0 and Cout%16==0 "
                "(pad the channels first), got ", C, " and ", Cout);
    int64_t ho = H + 2 * pad - KH + 1, wo = W + 2 * pad - KW + 1;
    TORCH_CHECK(ho > 0 && wo > 0, "torch_vc4: conv2d: empty output");
    auto out = make_arena_tensor({Cout, ho, wo});
    (void)batched;
    invalidate_for_cpu(in);
    invalidate_for_cpu(w);
    if (bias) {
        check_vc4_f32(*bias, "conv2d(bias)");
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == Cout,
                    "torch_vc4: conv2d bias must have ", Cout, " elements");
        invalidate_for_cpu(*bias);
    }
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bi = buf_of(in), bw = buf_of(w), bo = buf_of(out), bb{};
    const vc4_buf *pbb = nullptr;
    if (bias) {
        bb = buf_of(*bias);
        pbb = &bb;
    }
    int gho = (int)ho, gwo = (int)wo;
    vc4_status st = vc4_conv2d(s.ctx, &bi, (int)C, (int)H, (int)W, &bw,
                               (int)Cout, (int)KH, (int)KW, pbb, 1, (int)pad,
                               &bo, &gho, &gwo);
    if (st != VC4_OK)
        throw_st(st, "conv2d");
    TORCH_CHECK(gho == ho && gwo == wo, "torch_vc4: conv2d output size drift");
    return out;
}

/* conv2d backward (stride 1, NCHW, Cin/Cout multiples of 16): dX, dW. */
static std::tuple<at::Tensor, at::Tensor> vc4_conv_bwd_raw(
    const at::Tensor &in, const at::Tensor &w, const at::Tensor &dy,
    int64_t pad)
{
    ensure_init();
    check_vc4_f32(in, "conv2d_backward(input)");
    check_vc4_f32(w, "conv2d_backward(weight)");
    check_vc4_f32(dy, "conv2d_backward(dy)");
    TORCH_CHECK(w.dim() == 4, "torch_vc4: conv2d_backward needs a 4-D weight");
    TORCH_CHECK(in.dim() == 3 || (in.dim() == 4 && in.size(0) == 1),
                "torch_vc4: conv2d_backward needs [C][H][W] or [1][C][H][W] "
                "input");
    TORCH_CHECK(dy.dim() == 3 || (dy.dim() == 4 && dy.size(0) == 1),
                "torch_vc4: conv2d_backward needs [Cout][ho][wo] or "
                "[1][Cout][ho][wo] dy");
    TORCH_CHECK(in.is_contiguous() && w.is_contiguous() && dy.is_contiguous(),
                "torch_vc4: conv2d_backward needs contiguous tensors");
    const bool bi4 = in.dim() == 4, bdy4 = dy.dim() == 4;
    const int64_t C = in.size(bi4 ? 1 : 0), H = in.size(bi4 ? 2 : 1),
                  W = in.size(bi4 ? 3 : 2);
    const int64_t Cout = w.size(0), KH = w.size(2), KW = w.size(3);
    TORCH_CHECK(KH == KW && (KH == 3 || KH == 1), "torch_vc4: conv2d_backward "
                "supports 3x3 and 1x1 kernels");
    TORCH_CHECK(C % 16 == 0 && Cout % 16 == 0, "torch_vc4: conv2d_backward "
                "needs Cin%16==0 and Cout%16==0");
    int64_t ho = H + 2 * pad - KH + 1, wo = W + 2 * pad - KW + 1;
    TORCH_CHECK(dy.size(bdy4 ? 1 : 0) == Cout &&
                    dy.size(bdy4 ? 2 : 1) == ho &&
                    dy.size(bdy4 ? 3 : 2) == wo,
                "torch_vc4: conv2d_backward: dy ", dy.sizes(), " does not "
                "match output ", Cout, "x", ho, "x", wo);
    auto dW = make_arena_tensor({Cout, C, KH, KW});
    auto dX = make_arena_tensor({C, H, W});
    invalidate_for_cpu(in);
    invalidate_for_cpu(w);
    invalidate_for_cpu(dy);
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bi = buf_of(in), bw = buf_of(w), bdy = buf_of(dy);
    vc4_buf bdw = buf_of(dW), bdx = buf_of(dX);
    vc4_status st = vc4_conv2d_backward(s.ctx, &bi, (int)C, (int)H, (int)W,
                                        &bw, (int)Cout, (int)KH, (int)KW,
                                        &bdy, (int)pad, &bdw, &bdx);
    if (st != VC4_OK)
        throw_st(st, "conv2d_backward");
    return {dX, dW};
}

/* ---------------------------------------------------------- engine: GEMM */

/* C = A * B^T, A [M][K], B [N][K] (padding to multiples of 16 is done here
 * with zero bytes; the multiply is the QPU's). */
static at::Tensor vc4_mm_bt(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    check_vc4_f32(a, "mm_bt(A)");
    check_vc4_f32(b, "mm_bt(B)");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: mm_bt needs 2-D tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: mm_bt needs contiguous tensors");
    int64_t M = a.size(0), K = a.size(1), N = b.size(0);
    TORCH_CHECK(b.size(1) == K, "torch_vc4: mm_bt shape mismatch ", a.sizes(),
                " and ", b.sizes());
    int64_t Mp = al16(M), Np = al16(N), Kp = al16(K);
    if (Mp == M && Kp == K && Np == N) {
        auto out = make_arena_tensor({M, N});
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf ba = buf_of(a), bb = buf_of(b), bo = buf_of(out);
        vc4_status st = vc4_matmul_bt(s.ctx, &ba, &bb, &bo, (uint32_t)M,
                                      (uint32_t)N, (uint32_t)K);
        if (st != VC4_OK)
            throw_st(st, "mm_bt");
        return out;
    }
    std::vector<float> av = host_get(a), bv = host_get(b);
    std::vector<float> ap((size_t)Mp * Kp, 0.0f), bp((size_t)Np * Kp, 0.0f);
    for (int64_t r = 0; r < M; ++r)
        memcpy(ap.data() + (size_t)r * Kp, av.data() + (size_t)r * K,
               (size_t)K * sizeof(float));
    for (int64_t r = 0; r < N; ++r)
        memcpy(bp.data() + (size_t)r * Kp, bv.data() + (size_t)r * K,
               (size_t)K * sizeof(float));
    auto A = host_put({Mp, Kp}, ap.data(), ap.size());
    auto B = host_put({Np, Kp}, bp.data(), bp.size());
    auto C = make_arena_tensor({Mp, Np});
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf ba = buf_of(A), bb = buf_of(B), bc = buf_of(C);
        vc4_status st = vc4_matmul_bt(s.ctx, &ba, &bb, &bc, (uint32_t)Mp,
                                      (uint32_t)Np, (uint32_t)Kp);
        if (st != VC4_OK)
            throw_st(st, "mm_bt");
    }
    std::vector<float> cv = host_get(C);
    std::vector<float> out((size_t)M * N);
    for (int64_t r = 0; r < M; ++r)
        memcpy(out.data() + (size_t)r * N, cv.data() + (size_t)r * Np,
               (size_t)N * sizeof(float));
    return host_put({M, N}, out.data(), out.size());
}

/* C = A^T * B, A [M][K], B [M][N].  Uses the engine's 16x16 transpose and
 * the verified GEMM (both on the QPU). */
static at::Tensor vc4_mm_at(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    check_vc4_f32(a, "mm_at(A)");
    check_vc4_f32(b, "mm_at(B)");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: mm_at needs 2-D tensors");
    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(b.size(0) == M, "torch_vc4: mm_at shape mismatch ", a.sizes(),
                " and ", b.sizes());
    int64_t Mp = al16(M), Kp = al16(K), Np = al16(N);
    std::vector<float> av = host_get(a), bv = host_get(b);
    std::vector<float> ap((size_t)Mp * Kp, 0.0f), bp((size_t)Mp * Np, 0.0f);
    for (int64_t r = 0; r < M; ++r) {
        memcpy(ap.data() + (size_t)r * Kp, av.data() + (size_t)r * K,
               (size_t)K * sizeof(float));
        memcpy(bp.data() + (size_t)r * Np, bv.data() + (size_t)r * N,
               (size_t)N * sizeof(float));
    }
    auto A = host_put({Mp, Kp}, ap.data(), ap.size());
    auto B = host_put({Mp, Np}, bp.data(), bp.size());
    auto AT = make_arena_tensor({Kp, Mp});
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf ba = buf_of(A), bt = buf_of(AT);
        vc4_status st = vc4_transpose(s.ctx, &ba, &bt, (uint32_t)Mp,
                                      (uint32_t)Kp);
        if (st != VC4_OK)
            throw_st(st, "mm_at transpose");
    }
    auto C = make_arena_tensor({Kp, Np});
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf bt = buf_of(AT), bb = buf_of(B), bc = buf_of(C);
        vc4_status st = vc4_matmul(s.ctx, &bt, &bb, &bc, (uint32_t)Kp,
                                   (uint32_t)Np, (uint32_t)Mp,
                                   VC4_LAYOUT_ROW_MAJOR);
        if (st != VC4_OK)
            throw_st(st, "mm_at matmul");
    }
    std::vector<float> cv = host_get(C);
    std::vector<float> out((size_t)K * N);
    for (int64_t r = 0; r < K; ++r)
        memcpy(out.data() + (size_t)r * N, cv.data() + (size_t)r * Np,
               (size_t)N * sizeof(float));
    return host_put({K, N}, out.data(), out.size());
}

/* Full reduction over all elements: out = ones[1][n] @ x[n][1], one GEMM.
 * The ones vector is host-materialised data, not arithmetic. */
static at::Tensor vc4_sum_all(const at::Tensor &t)
{
    ensure_init();
    check_vc4_f32(t, "sum");
    TORCH_CHECK(t.is_contiguous(), "torch_vc4: sum needs a contiguous tensor");
    int64_t n = t.numel();
    TORCH_CHECK(n > 0, "torch_vc4: sum of an empty tensor is not supported");
    std::vector<float> ones((size_t)n, 1.0f);
    auto A = host_put({1, n}, ones.data(), ones.size());
    std::vector<float> tv = host_get(t);
    auto B = host_put({n, 1}, tv.data(), tv.size());
    auto C = make_arena_tensor({1, 1});
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf ba = buf_of(A), bb = buf_of(B), bc = buf_of(C);
        vc4_status st = vc4_matmul(s.ctx, &ba, &bb, &bc, 1, 1, (uint32_t)n,
                                   VC4_LAYOUT_ROW_MAJOR);
        if (st != VC4_OK)
            throw_st(st, "sum");
    }
    return C;
}

/* ------------------------------------------------------- engine: pooling */

/* 2x2 stride-2 max pool, [C][H][W] with H%8==0, W%16==0 (the engine's
 * scheduling constraint). */
static at::Tensor vc4_maxpool2x2_op(const at::Tensor &t)
{
    ensure_init();
    check_vc4_f32(t, "max_pool2d");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: max_pool2d expects contiguous [C][H][W]");
    int64_t C = t.size(0), H = t.size(1), W = t.size(2);
    TORCH_CHECK(H % 8 == 0 && W % 16 == 0,
                "torch_vc4: max_pool2d 2x2 engine path needs H%8==0 and "
                "W%16==0, got ", H, "x", W);
    int64_t ho = H / 2, wo = W / 2, pitch = al16(wo);
    auto tmp = make_arena_tensor({C, ho, pitch});
    invalidate_for_cpu(t);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf bi = buf_of_slack(t), bo = buf_of(tmp);
        int gho = 0, gwo = 0;
        vc4_status st = vc4_maxpool2x2(s.ctx, &bi, (int)C, (int)H, (int)W,
                                       (int)pitch, &bo, &gho, &gwo);
        if (st != VC4_OK)
            throw_st(st, "max_pool2d");
        TORCH_CHECK(gho == ho && gwo == wo, "torch_vc4: max_pool2d size drift");
    }
    if (pitch == wo)
        return tmp;   /* already compact */
    std::vector<float> tv = host_get(tmp);
    std::vector<float> out((size_t)C * ho * wo);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            memcpy(out.data() + ((size_t)c * ho + y) * wo,
                   tv.data() + ((size_t)c * ho + y) * pitch,
                   (size_t)wo * sizeof(float));
    return host_put({C, ho, wo}, out.data(), out.size());
}

/* 2x2 stride-2 max pool backward, [C][H][W] with H%4==0, W%16==0. */
static at::Tensor vc4_maxpool2x2_bwd_op(const at::Tensor &x,
                                        const at::Tensor &dy)
{
    ensure_init();
    check_vc4_f32(x, "max_pool2d_backward(input)");
    check_vc4_f32(dy, "max_pool2d_backward(dy)");
    TORCH_CHECK(x.dim() == 3 && dy.dim() == 3,
                "torch_vc4: max_pool2d_backward expects [C][H][W]");
    int64_t C = x.size(0), H = x.size(1), W = x.size(2);
    TORCH_CHECK(H % 4 == 0 && W % 16 == 0, "torch_vc4: max_pool2d_backward "
                "engine path needs H%4==0 and W%16==0");
    int64_t ho = H / 2, wo = W / 2, pitch = al16(wo);
    TORCH_CHECK(dy.size(0) == C && dy.size(1) == ho && dy.size(2) == wo,
                "torch_vc4: max_pool2d_backward: dy ", dy.sizes(), " does not "
                "match ", C, "x", ho, "x", wo);
    std::vector<float> d = host_get(dy);
    std::vector<float> dpp((size_t)C * ho * pitch, 0.0f);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            memcpy(dpp.data() + ((size_t)c * ho + y) * pitch,
                   d.data() + ((size_t)c * ho + y) * wo,
                   (size_t)wo * sizeof(float));
    auto dy_p = host_put({C, ho, pitch}, dpp.data(), dpp.size());
    auto dx = make_arena_tensor({C, H, W});
    invalidate_for_cpu(x);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf bi = buf_of_slack(x), bdy = buf_of(dy_p), bdx = buf_of(dx);
        vc4_status st = vc4_maxpool2x2_backward(s.ctx, &bi, &bdy, (int)C,
                                                (int)H, (int)W, (int)pitch,
                                                &bdx);
        if (st != VC4_OK)
            throw_st(st, "max_pool2d_backward");
    }
    return dx;
}

/* Generic NxN / stride-S / pad-P max pool over [C][H][W]: the engine builds
 * the -inf border itself and every comparison is on the QPU. */
static at::Tensor vc4_maxpooln_op(const at::Tensor &t, int64_t k, int64_t stride,
                                  int64_t pad)
{
    ensure_init();
    check_vc4_f32(t, "max_pool2d");
    TORCH_CHECK(t.dim() == 3 && t.is_contiguous(),
                "torch_vc4: max_pool2d expects contiguous [C][H][W]");
    TORCH_CHECK(k > 0 && stride > 0 && pad >= 0, "torch_vc4: max_pool2d: bad "
                "kernel/stride/padding");
    int64_t C = t.size(0), H = t.size(1), W = t.size(2);
    int64_t ho = (H + 2 * pad - k) / stride + 1;
    int64_t wo = (W + 2 * pad - k) / stride + 1;
    TORCH_CHECK(ho > 0 && wo > 0, "torch_vc4: max_pool2d: empty output");
    int64_t hop = al4(ho), op = al16(wo);
    auto tmp = make_arena_tensor({C, hop, op});
    invalidate_for_cpu(t);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf bi = buf_of(t), bo = buf_of(tmp);
        int gho = 0, gwo = 0;
        vc4_status st = ::vc4_maxpooln(s.ctx, &bi, (int)C, (int)H, (int)W, (int)k,
                                     (int)stride, (int)pad, &bo, &gho, &gwo);
        if (st != VC4_OK)
            throw_st(st, "max_pool2d");
        TORCH_CHECK(gho == ho && gwo == wo, "torch_vc4: max_pool2d size drift");
    }
    std::vector<float> tv = host_get(tmp);
    std::vector<float> out((size_t)C * ho * wo);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t y = 0; y < ho; ++y)
            memcpy(out.data() + ((size_t)c * ho + y) * wo,
                   tv.data() + ((size_t)c * hop + y) * op,
                   (size_t)wo * sizeof(float));
    return host_put({C, ho, wo}, out.data(), out.size());
}

/* ------------------------------------------------------- engine: softmax */

/* Row-wise softmax over the last dim (<= 16 classes); rows padded to 8 with
 * zero bytes (the engine's lane width). */
static at::Tensor vc4_softmax_op(const at::Tensor &t)
{
    ensure_init();
    check_vc4_f32(t, "softmax");
    TORCH_CHECK(t.dim() >= 1 && t.is_contiguous(),
                "torch_vc4: softmax needs a contiguous tensor");
    int64_t cols = t.size(-1), rows = t.numel() / cols;
    TORCH_CHECK(cols <= 16, "torch_vc4: softmax supports at most 16 classes "
                "on the last dim (engine lane width), got ", cols);
    TORCH_CHECK(rows > 0, "torch_vc4: softmax of an empty tensor");
    int64_t rp = al8(rows);
    std::vector<float> src = host_get(t);
    std::vector<float> in((size_t)rp * cols, 0.0f);
    memcpy(in.data(), src.data(), src.size() * sizeof(float));
    auto bin = host_put({rp, cols}, in.data(), in.size());
    auto bout = make_arena_tensor({rp, cols});
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf bi = buf_of(bin), bo = buf_of(bout);
        vc4_status st = ::vc4_softmax_fwd(s.ctx, &bi, (uint32_t)rp,
                                        (uint32_t)cols, &bo);
        if (st != VC4_OK)
            throw_st(st, "softmax");
    }
    std::vector<float> ov = host_get(bout);
    std::vector<float> out((size_t)rows * cols);
    memcpy(out.data(), ov.data(), out.size() * sizeof(float));
    std::vector<int64_t> osz(t.sizes().begin(), t.sizes().end());
    return host_put(osz, out.data(), out.size());
}

/* ------------------------------------------------------------------ misc */

/* Drop the CPU's cache lines for a tensor without keeping the copy (used by
 * the Python layer after in-place work it did not route through upload). */
static void vc4_touch(const at::Tensor &t)
{
    invalidate_for_cpu(t);
}

/* ------------------------------------------------------------ uniform_ (RNG) */
/* Module initialisation (`nn.Linear`, `nn.Conv2d` constructed on the device)
 * calls uniform_; like normal_ the numbers come from the host std::mt19937 and
 * are uploaded.  Data generation, not compute. */
static at::Tensor &vc4_uniform_(at::Tensor &self, double from, double to,
                                std::optional<at::Generator> generator)
{
    (void)generator;
    ensure_init();
    check_vc4_f32(self, "uniform_");
    TORCH_CHECK(self.is_contiguous(),
                "torch_vc4: uniform_ needs a contiguous tensor");
    TORCH_CHECK(from <= to, "torch_vc4: uniform_: from > to");
    size_t n = (size_t)self.numel();
    std::vector<float> host(n);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.alloc_mu);
        std::uniform_real_distribution<float> dist((float)from, (float)to);
        for (size_t i = 0; i < n; ++i)
            host[i] = dist(s.rng);
    }
    std::lock_guard<std::mutex> g(state().gpu_mu);
    vc4_buf b = buf_of(self);
    vc4_status st = vc4_upload(state().ctx, &b, host.data(),
                               n * sizeof(float));
    if (st != VC4_OK)
        throw_st(st, "uniform_");
    return self;
}


/* ======================================================================
 * Part 3: batches, arbitrary-dim reductions, softmax for any class count and
 * the verified unary subset.
 *
 * IMPORTANT, measured on this board: the engine's generic kernels
 * `vc4_reduce` and `vc4_gather` are linked in libvc4infer.a but HANG the GPU
 * in this build -- every call returns VC4_ERR_TIMEOUT (-4) and kills the
 * context.  Nothing below calls them.  Reductions are GEMMs (sum/mean) or a
 * relu/sign max tree (amax/amin); gather/index_select are a host word gather
 * (a pure copy, exactly like contiguous()); argmax is a host integer scan.
 * COMPLETENESS.md records this with the measured status codes.
 * ====================================================================== */

/* int64 tensors are index bookkeeping, never compute: the allocator and the
 * upload path handle them, the kernels never see them. */
static std::vector<int64_t> host_get_i64(const at::Tensor &t)
{
    TORCH_CHECK(t.scalar_type() == at::kLong,
                "torch_vc4: expected an int64 index tensor, got ",
                t.scalar_type());
    int64_t n = t.numel();
    std::vector<int64_t> v((size_t)n);
    if (!n)
        return v;
    if (t.device().type() == at::kCPU) {
        TORCH_CHECK(t.is_contiguous(), "torch_vc4: index tensor must be "
                    "contiguous");
        memcpy(v.data(), t.const_data_ptr(), (size_t)n * 8);
        return v;
    }
    TORCH_CHECK(t.device().type() == kDev, "torch_vc4: index tensor must be "
                "on vc4 or the CPU");
    TORCH_CHECK(t.is_contiguous(), "torch_vc4: index tensor must be contiguous");
    ensure_init();
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf b = buf_of(t);
    vc4_status st = vc4_download(s.ctx, &b, v.data(), (size_t)n * 8);
    if (st != VC4_OK)
        throw_st(st, "index download");
    return v;
}

static at::Tensor host_put_i64(c10::IntArrayRef sizes, const int64_t *src,
                               size_t n)
{
    auto t = make_arena_tensor_dtype(sizes, at::kLong);
    if (n) {
        TORCH_CHECK(n == (size_t)t.numel(), "torch_vc4: internal size mismatch");
        ensure_init();
        State &s = state();
        std::lock_guard<std::mutex> g(s.gpu_mu);
        vc4_buf b = buf_of(t);
        vc4_status st = vc4_upload(s.ctx, &b, src, n * 8);
        if (st != VC4_OK)
            throw_st(st, "index upload");
    }
    return t;
}

/* out = f(a), the engine's vc4_unary family.  Only the ops the Python layer
 * has measured correct on the board are routed here (see COMPLETENESS.md):
 * abs/neg/sign/clip/leakyrelu/hardsigmoid/hardswish.  The other bodies in
 * this engine build return wrong numbers, so the Python layer refuses them. */
static at::Tensor vc4_unary_op(const at::Tensor &a, int64_t op, double p0,
                               double p1, double p2)
{
    ensure_init();
    check_vc4_f32(a, "unary");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: unary needs a contiguous tensor");
    auto out = make_arena_tensor(a.sizes());
    size_t n = (size_t)a.numel();
    if (!n)
        return out;
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_unary(s.ctx, (vc4_uop)op, &ba, &bo, n, (float)p0,
                              (float)p1, (float)p2);
    if (st != VC4_OK)
        throw_st(st, "unary");
    return out;
}

/* ------------------------------------------------------------ NHWC conv */

/* NCHW [N][C][H][W] -> NHWC [N][H][W][cpad]; channels C..cpad-1 are zero.
 * A byte permutation on the ARM (no arithmetic), the layout the engine's
 * batched/stride-2/depthwise NHWC kernels read. */
static at::Tensor vc4_nhwc_pack(const at::Tensor &x, int64_t cpad)
{
    ensure_init();
    check_vc4_f32(x, "conv2d(nhwc pack)");
    TORCH_CHECK(x.dim() == 4 && x.is_contiguous(),
                "torch_vc4: nhwc_pack expects contiguous NCHW");
    int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(cpad >= C, "torch_vc4: nhwc_pack: cpad ", cpad, " < C ", C);
    std::vector<float> src = host_get(x);
    std::vector<float> dst((size_t)N * H * W * cpad, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t y = 0; y < H; ++y)
            for (int64_t xx = 0; xx < W; ++xx) {
                const float *sp = src.data() +
                    ((size_t)n * C * H * W + (size_t)y * W + xx);
                float *dp = dst.data() +
                    (((size_t)n * H + y) * W + xx) * cpad;
                for (int64_t c = 0; c < C; ++c)
                    dp[c] = sp[(size_t)c * H * W];
            }
    return host_put({N, H, W, cpad}, dst.data(), dst.size());
}

/* NHWC [N][H][pitch][Cp] -> NCHW [N][C][H][W] (drops channels C..Cp-1 and
 * the pitch padding columns).  Byte permutation only. */
static at::Tensor vc4_nhwc_unpack(const at::Tensor &y, int64_t N, int64_t C,
                                  int64_t H, int64_t W)
{
    ensure_init();
    check_vc4_f32(y, "conv2d(nhwc unpack)");
    TORCH_CHECK(y.dim() == 4 && y.is_contiguous(),
                "torch_vc4: nhwc_unpack expects contiguous NHWC");
    int64_t pitch = y.size(2), Cp = y.size(3);
    TORCH_CHECK(y.size(0) == N && y.size(1) == H && Cp >= C && pitch >= W,
                "torch_vc4: nhwc_unpack shape ", y.sizes(), " vs N=", N,
                " C=", C, " H=", H, " W=", W);
    std::vector<float> src = host_get(y);
    std::vector<float> dst((size_t)N * C * H * W);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t yy = 0; yy < H; ++yy)
                for (int64_t xx = 0; xx < W; ++xx)
                    dst[(((size_t)n * C + c) * H + yy) * W + xx] =
                        src[(((size_t)n * H + yy) * pitch + xx) * Cp + c];
    return host_put({N, C, H, W}, dst.data(), dst.size());
}

/* Pack [Cout][Cin][KH][KW] (+ optional [Cout] bias) into the engine's GEMM
 * weight panel.  The engine does the ci-major -> tap-major permutation. */
static at::Tensor vc4_conv_nhwc_pack_w_op(const at::Tensor &wt,
                                          const std::optional<at::Tensor> &bias,
                                          int64_t cin, int64_t cout)
{
    ensure_init();
    check_vc4_f32(wt, "conv2d(weight pack)");
    TORCH_CHECK(wt.dim() == 4 && wt.is_contiguous(),
                "torch_vc4: conv weight must be a contiguous 4-D tensor");
    int64_t KH = wt.size(2), KW = wt.size(3);
    TORCH_CHECK(wt.size(0) == cout && wt.size(1) == cin,
                "torch_vc4: weight is ", wt.sizes(), ", expected [", cout,
                "][", cin, "][", KH, "][", KW, "]");
    size_t bytes = vc4_conv_nhwc_wpanel_bytes((int)cin, (int)cout, (int)KH,
                                              (int)KW, bias ? 1 : 0);
    TORCH_CHECK(bytes > 0, "torch_vc4: conv weight panel needs Cin%16==0 and "
                "Cout%16==0, got ", cin, " and ", cout);
    auto panel = make_arena_tensor({(int64_t)(bytes / 4)});
    vc4_buf bb{};
    const vc4_buf *pbb = nullptr;
    if (bias) {
        check_vc4_f32(*bias, "conv2d(bias pack)");
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == cout &&
                    bias->is_contiguous(),
                    "torch_vc4: conv bias must be [", cout, "]");
    }
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bw = buf_of(wt), bp = buf_of(panel);
    if (bias) {
        bb = buf_of(*bias);
        pbb = &bb;
    }
    vc4_status st = vc4_conv_nhwc_pack_w(s.ctx, &bw, (int)cin, (int)cout,
                                         (int)KH, (int)KW, pbb, &bp);
    if (st != VC4_OK)
        throw_st(st, "conv2d weight pack");
    return panel;
}

/* out = conv(in) over the whole batch, in NHWC.  Returns
 * [nimg][ho][align16(wo)][Cout] (the engine's padded output pitch). */
static at::Tensor vc4_conv_nhwc_op(const at::Tensor &in,
                                   const at::Tensor &panel, int64_t nimg,
                                   int64_t cin, int64_t H, int64_t W,
                                   int64_t cout, int64_t KH, int64_t KW,
                                   int64_t stride, int64_t pad, int64_t dil)
{
    ensure_init();
    check_vc4_f32(in, "conv2d(nhwc)");
    check_vc4_f32(panel, "conv2d(wpanel)");
    TORCH_CHECK(in.dim() == 4 && in.is_contiguous(),
                "torch_vc4: conv2d(nhwc) expects contiguous NHWC");
    TORCH_CHECK(in.size(0) == nimg && in.size(1) == H && in.size(2) == W &&
                in.size(3) == cin,
                "torch_vc4: conv2d(nhwc) input ", in.sizes(), " != [", nimg,
                "][", H, "][", W, "][", cin, "]");
    int64_t effh = (KH - 1) * dil + 1, effw = (KW - 1) * dil + 1;
    int64_t ho = (H + 2 * pad - effh) / stride + 1;
    int64_t wo = (W + 2 * pad - effw) / stride + 1;
    TORCH_CHECK(ho > 0 && wo > 0, "torch_vc4: conv2d: empty output");
    int64_t Wg = (wo + 15) & ~int64_t(15);
    auto out = make_arena_tensor({nimg, ho, Wg, cout});
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bi = buf_of(in), bp = buf_of(panel), bo = buf_of(out);
    int gho = 0, gwo = 0, op = 0;
    vc4_status st = vc4_conv_nhwc_dil(s.ctx, &bi, (int)nimg, (int)cin, (int)H,
                                      (int)W, (int)W, &bp, (int)cout, (int)KH,
                                      (int)KW, (int)stride, (int)pad,
                                      (int)dil, &bo, &gho, &gwo, &op);
    if (st != VC4_OK)
        throw_st(st, "conv2d(nhwc)");
    TORCH_CHECK(gho == ho && gwo == wo, "torch_vc4: conv2d nhwc size drift");
    return out;
}

/* 3x3 depthwise panel: [C][1][3][3] -> engine block-diagonal panel, with the
 * channel count padded to a multiple of 16 (zero weights). */
static at::Tensor vc4_depthwise_panel(const at::Tensor &w, int64_t cpad)
{
    ensure_init();
    check_vc4_f32(w, "conv2d(depthwise weight)");
    TORCH_CHECK(w.dim() == 4 && w.size(1) == 1 && w.size(2) == 3 &&
                w.size(3) == 3 && w.is_contiguous(),
                "torch_vc4: depthwise weight must be [C][1][3][3]");
    int64_t C = w.size(0);
    TORCH_CHECK(cpad >= C && cpad % 16 == 0,
                "torch_vc4: depthwise channel pad must be a multiple of 16");
    std::vector<float> src = host_get(w);
    std::vector<float> dst((size_t)cpad * 9, 0.0f);
    memcpy(dst.data(), src.data(), (size_t)C * 9 * sizeof(float));
    auto wt = host_put({cpad, 1, 3, 3}, dst.data(), dst.size());
    size_t bytes = vc4_depthwise_wpanel_bytes((int)cpad);
    TORCH_CHECK(bytes > 0, "torch_vc4: depthwise panel size");
    auto panel = make_arena_tensor({(int64_t)(bytes / 4)});
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bw = buf_of(wt), bp = buf_of(panel);
    vc4_status st = vc4_depthwise_pack_w(s.ctx, &bw, (int)cpad, &bp);
    if (st != VC4_OK)
        throw_st(st, "conv2d depthwise pack");
    return panel;
}

/* 3x3 stride-1 pad-1 depthwise, NHWC, whole batch. */
static at::Tensor vc4_depthwise_nhwc_op(const at::Tensor &in,
                                        const at::Tensor &panel, int64_t nimg,
                                        int64_t C, int64_t H, int64_t W)
{
    ensure_init();
    check_vc4_f32(in, "conv2d(depthwise)");
    TORCH_CHECK(in.dim() == 4 && in.is_contiguous() && in.size(3) == C &&
                in.size(0) == nimg && in.size(1) == H && in.size(2) == W,
                "torch_vc4: depthwise expects NHWC [N][H][W][C]");
    int64_t Wg = (W + 15) & ~int64_t(15);
    auto out = make_arena_tensor({nimg, H, Wg, C});
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf bi = buf_of(in), bp = buf_of(panel), bo = buf_of(out);
    int gho = 0, gwo = 0, op = 0;
    vc4_status st = vc4_depthwise_nhwc(s.ctx, &bi, (int)nimg, (int)C, (int)H,
                                       (int)W, (int)W, &bp, &bo, &gho, &gwo,
                                       &op);
    if (st != VC4_OK)
        throw_st(st, "conv2d(depthwise)");
    TORCH_CHECK(gho == H && gwo == W, "torch_vc4: depthwise size drift");
    return out;
}

/* ------------------------------------------------------- index selection */

/* out[b][r] = src[row][r], where row = idx[b] (idx 1-D) or idx[b][r]
 * (idx 2-D with idx.size(1) == rest).  A host word gather: index arithmetic
 * plus memcpy, no floating-point arithmetic.  This is the layout primitive
 * gather/index_select are built from (vc4_gather hangs, see above). */
static at::Tensor vc4_take_host(const at::Tensor &src, const at::Tensor &idx,
                                int64_t rest)
{
    ensure_init();
    check_vc4_f32(src, "gather(source)");
    TORCH_CHECK(src.is_contiguous(), "torch_vc4: gather needs a contiguous "
                "source");
    TORCH_CHECK(rest > 0 && src.numel() % rest == 0,
                "torch_vc4: gather: rest ", rest, " does not divide ",
                src.numel());
    int64_t A = src.numel() / rest;
    TORCH_CHECK(idx.dim() == 1 || (idx.dim() == 2 && idx.size(1) == rest),
                "torch_vc4: gather index must be 1-D or [B][", rest, "], got ",
                idx.sizes());
    std::vector<int64_t> ii = host_get_i64(idx);
    int64_t B = idx.dim() == 1 ? idx.numel() : idx.size(0);
    std::vector<float> s = host_get(src);
    std::vector<float> out((size_t)B * rest);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t r = 0; r < rest; ++r) {
            int64_t row = idx.dim() == 1 ? ii[(size_t)b]
                                         : ii[(size_t)b * rest + r];
            if (row < 0)
                row += A;
            TORCH_CHECK(row >= 0 && row < A, "torch_vc4: gather index ", row,
                        " is outside [0, ", A, ")");
            out[(size_t)b * rest + r] = s[(size_t)row * rest + r];
        }
    if (idx.dim() == 1)
        return host_put({B, rest}, out.data(), out.size());
    return host_put({B, rest}, out.data(), out.size());
}

/* argmax/argmin over [outer][red]: an integer scan on the ARM.  There is no
 * working device arg-reduction in this engine build (vc4_reduce hangs); the
 * scan only compares already-computed values and returns indices. */
static at::Tensor vc4_arg_host(const at::Tensor &a, int64_t outer,
                               int64_t red, bool is_max)
{
    ensure_init();
    check_vc4_f32(a, "argmax/argmin");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: argmax needs a contiguous "
                "tensor");
    TORCH_CHECK(outer > 0 && red > 0 && a.numel() == outer * red,
                "torch_vc4: argmax: shape ", a.sizes(), " != ", outer, "x",
                red);
    std::vector<float> v = host_get(a);
    std::vector<int64_t> idx((size_t)outer, 0);
    for (int64_t o = 0; o < outer; ++o) {
        int64_t best = 0;
        for (int64_t r = 1; r < red; ++r) {
            float x = v[(size_t)o * red + r];
            float bst = v[(size_t)o * red + best];
            bool take;
            if (std::isnan(x))
                take = !std::isnan(bst);
            else if (std::isnan(bst))
                take = false;
            else
                take = is_max ? (x > bst) : (x < bst);
            if (take)
                best = r;
        }
        idx[(size_t)o] = best;
    }
    return host_put_i64({outer}, idx.data(), idx.size());
}

/* ---------------------------------------------------------------- fallback */

static void vc4_fallback(const c10::OperatorHandle &op, c10::DispatchKeySet,
                         torch::jit::Stack *)
{
    TORCH_CHECK(false, "torch_vc4: '", op.schema().operator_name().name,
                "' is not implemented for device vc4.  This backend has no "
                "CPU fallback by design; see README.md for the op table "
                "(matmul/linear, conv2d, pools, softmax, elementwise, "
                "batch_norm inference, views).");
}

/* ------------------------------------------------------------------- misc */

/* ---------------------------------------------------------------- hooks */
/* torch requires a PrivateUse1HooksInterface before a privateuse1 device can
 * be used at all ("Please register PrivateUse1HooksInterface ... first"). */
struct Vc4Hooks final : public at::PrivateUse1HooksInterface {
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return true; }
    bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
    c10::DeviceIndex deviceCount() const override { return 1; }
    c10::DeviceIndex getCurrentDevice() const override { return 0; }
    void setCurrentDevice(c10::DeviceIndex) const override {}

    const at::Generator &getDefaultGenerator(c10::DeviceIndex) const override
    {
        /* torch only ever asks for a generator to pass to normal_-style
         * kernels; ours ignores it and uses the host std::mt19937. */
        static at::Generator gen = at::detail::createCPUGenerator(42u);
        return gen;
    }

    at::Device getDeviceFromPtr(void *) const override
    {
        /* There is exactly one vc4 device, so every pointer torch asks about
         * here belongs to it (this is also called for pointers that are not
         * arena allocations, e.g. while a factory op is being set up). */
        return at::Device(kDev, kDevIndex);
    }

    void resizePrivateUse1Bytes(const c10::Storage &, size_t) const override
    {
        TORCH_CHECK(false, "torch_vc4: resizing vc4 tensor storage is not "
                           "supported (the arena allocator has no realloc)");
    }
};

/* Kick the hooks/allocator in at library load time. */
static void register_backend()
{
    static std::once_flag once;
    std::call_once(once, [] {
        c10::SetAllocator(kDev, new Vc4Allocator());
        at::RegisterPrivateUse1HooksInterface(new Vc4Hooks());
    });
}

}  // namespace torch_vc4

/* ------------------------------------------------------------ registration */

static struct AllocatorRegistrar {
    AllocatorRegistrar() { torch_vc4::register_backend(); }
} allocator_registrar;

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(torch_vc4::vc4_empty));
    m.impl("empty_strided", TORCH_FN(torch_vc4::vc4_empty_strided));
    m.impl("copy_", TORCH_FN(torch_vc4::vc4_copy_));
    m.impl("mm", TORCH_FN(torch_vc4::vc4_mm));
    m.impl("matmul", TORCH_FN(torch_vc4::vc4_matmul));
    m.impl("add.Tensor", TORCH_FN(torch_vc4::vc4_add));
    m.impl("mul.Tensor", TORCH_FN(torch_vc4::vc4_mul));
    m.impl("relu", TORCH_FN(torch_vc4::vc4_relu));
    m.impl("normal_", TORCH_FN(torch_vc4::vc4_normal_));
    m.impl("uniform_", TORCH_FN(torch_vc4::vc4_uniform_));
    m.impl("fill_.Scalar", TORCH_FN(torch_vc4::vc4_fill_));
    m.impl("zero_", TORCH_FN(torch_vc4::vc4_zero_));
    m.impl("view", TORCH_FN(torch_vc4::vc4_view));
    m.impl("as_strided", TORCH_FN(torch_vc4::vc4_as_strided));
    m.impl("_local_scalar_dense",
           TORCH_FN(torch_vc4::vc4_local_scalar_dense));
    m.impl("contiguous", TORCH_FN(torch_vc4::vc4_contiguous));
    m.impl("clone", TORCH_FN(torch_vc4::vc4_clone));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<
               &torch_vc4::vc4_fallback>());
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_vc4::Vc4GuardImpl);

/* Diagnostics, reachable as torch.ops.torch_vc4.stats() etc.  (No pybind11:
 * it is not worth the compile memory on this board.  The returns are plain
 * double lists; torch_vc4/__init__.py turns them into named dicts.) */
TORCH_LIBRARY(torch_vc4, m)
{
    m.def("stats", []() -> std::vector<double> {
        torch_vc4::ensure_init();
        vc4_stats st{};
        vc4_get_stats(torch_vc4::state().ctx, &st);
        return {(double)st.launches,
                (double)st.pack_ns / 1e6,
                (double)st.kernel_ns / 1e6,
                (double)st.gpu_cycles,
                (double)st.flops,
                st.kernel_ns ? (double)st.flops / (double)st.kernel_ns : 0.0};
    });
    m.def("reset_stats", []() {
        torch_vc4::ensure_init();
        vc4_reset_stats(torch_vc4::state().ctx);
    });
    m.def("seed", [](int64_t s) {
        torch_vc4::ensure_init();
        std::lock_guard<std::mutex> g(torch_vc4::state().alloc_mu);
        torch_vc4::state().rng.seed((uint32_t)s);
    });
    m.def("alloc_info", []() -> std::vector<double> {
        torch_vc4::State &s = torch_vc4::state();
        std::lock_guard<std::mutex> g(s.alloc_mu);
        size_t used = 0, freeb = 0, largest = 0;
        for (const auto &b : s.blocks) {
            if (b.used)
                used += b.size;
            else {
                freeb += b.size;
                if (b.size > largest)
                    largest = b.size;
            }
        }
        return {(double)s.blocks.size(), (double)used, (double)freeb,
                (double)largest, (double)s.cap};
    });

    m.def("arena", []() -> std::vector<double> {
        torch_vc4::ensure_init();
        torch_vc4::State &s = torch_vc4::state();
        return {(double)s.region_off,
                (double)s.cap,
                (double)vc4_arena_bytes(s.ctx),
                (double)vc4_arena_used(s.ctx)};
    });

    /* ---- byte-level layout primitives (no arithmetic) ---- */
    m.def("pad_ch(Tensor t, int cpad) -> Tensor", &torch_vc4::vc4_pad_ch);
    m.def("take_ch(Tensor t, int c0, int c1) -> Tensor",
          &torch_vc4::vc4_take_ch);
    m.def("cat_ch(Tensor[] ts) -> Tensor", &torch_vc4::vc4_cat_ch);
    m.def("bcast_rows(Tensor t, int m) -> Tensor", &torch_vc4::vc4_bcast_rows);
    m.def("bcast_ch(Tensor t, int h, int w) -> Tensor",
          &torch_vc4::vc4_bcast_ch);
    m.def("up_nearest(Tensor t, int ho, int wo, float sh, float sw) -> Tensor",
          &torch_vc4::vc4_up_nearest);
    m.def("pad_w(Tensor w, int cin_pad, int cout_pad) -> Tensor",
          &torch_vc4::vc4_pad_w);
    m.def("dense_dw_w(Tensor w, int cin_pad, int cout_pad) -> Tensor",
          &torch_vc4::vc4_dense_dw_w);
    m.def("w_block(Tensor w, int cout, int cin, bool depthwise) -> Tensor",
          &torch_vc4::vc4_w_block);
    m.def("phase_w(Tensor w, int cin_pad, int cout_pad, int pad, int q) "
          "-> Tensor", &torch_vc4::vc4_phase_w);
    m.def("w_phase_block(Tensor w, int cout, int cin, int pad, int q, "
          "bool depthwise) -> Tensor", &torch_vc4::vc4_w_phase_block);
    m.def("phase_split(Tensor t, int hp, int wp) -> Tensor",
          &torch_vc4::vc4_phase_split);
    m.def("phase_merge(Tensor t, int c, int h, int w) -> Tensor",
          &torch_vc4::vc4_phase_merge);
    m.def("subsample2(Tensor t) -> Tensor", &torch_vc4::vc4_subsample2);
    m.def("scatter2(Tensor t, int h, int w) -> Tensor",
          &torch_vc4::vc4_scatter2);
    m.def("touch(Tensor t) -> ()", &torch_vc4::vc4_touch);
    /* aliases of the aten view kernels, for the Python layer (which overrides
     * aten::view/aten::as_strided and must not call back into itself) */
    m.def("view_alias(Tensor t, int[] size) -> Tensor",
          &torch_vc4::vc4_view);
    m.def("as_strided_alias(Tensor t, int[] size, int[] stride, int? off) "
          "-> Tensor", &torch_vc4::vc4_as_strided);

    /* ---- engine ops (everything that computes is on the QPU) ---- */
    m.def("conv_raw(Tensor input, Tensor weight, Tensor? bias, int pad) "
          "-> Tensor", &torch_vc4::vc4_conv_raw);
    m.def("conv_bwd_raw(Tensor input, Tensor weight, Tensor dy, int pad) "
          "-> (Tensor, Tensor)", &torch_vc4::vc4_conv_bwd_raw);
    m.def("mm_bt(Tensor a, Tensor b) -> Tensor", &torch_vc4::vc4_mm_bt);
    m.def("mm_at(Tensor a, Tensor b) -> Tensor", &torch_vc4::vc4_mm_at);
    m.def("sum_all(Tensor t) -> Tensor", &torch_vc4::vc4_sum_all);
    m.def("maxpool2x2(Tensor t) -> Tensor", &torch_vc4::vc4_maxpool2x2_op);
    m.def("maxpool2x2_bwd(Tensor x, Tensor dy) -> Tensor",
          &torch_vc4::vc4_maxpool2x2_bwd_op);
    m.def("maxpooln(Tensor t, int k, int stride, int pad) -> Tensor",
          &torch_vc4::vc4_maxpooln_op);
    m.def("softmax(Tensor t) -> Tensor", &torch_vc4::vc4_softmax_op);
    m.def("sigmoid(Tensor t) -> Tensor", &torch_vc4::vc4_sigmoid_op);
    m.def("silu(Tensor t) -> Tensor", &torch_vc4::vc4_silu_op);
    m.def("relu_bwd(Tensor a, Tensor dy) -> Tensor",
          &torch_vc4::vc4_relu_bwd);
    m.def("rsqrt(Tensor t) -> Tensor", &torch_vc4::vc4_rsqrt_op);
    m.def("add(Tensor a, Tensor b, Scalar alpha=1) -> Tensor",
          [](const at::Tensor &a, const at::Tensor &b, const at::Scalar &al) {
              return torch_vc4::vc4_add(a, b, al);
          });
    m.def("mul(Tensor a, Tensor b) -> Tensor", &torch_vc4::vc4_mul);
    m.def("mul_scalar(Tensor t, float s) -> Tensor",
          &torch_vc4::vc4_mul_scalar);
    m.def("add_scalar(Tensor t, float s) -> Tensor",
          &torch_vc4::vc4_add_scalar);
    m.def("unary(Tensor t, int op, float p0, float p1, float p2) -> Tensor",
          &torch_vc4::vc4_unary_op);
    m.def("materialize(Tensor t) -> Tensor", &torch_vc4::vc4_materialize);
    m.def("nhwc_pack(Tensor t, int cpad) -> Tensor",
          &torch_vc4::vc4_nhwc_pack);
    m.def("nhwc_unpack(Tensor t, int n, int c, int h, int w) -> Tensor",
          &torch_vc4::vc4_nhwc_unpack);
    m.def("conv_nhwc_pack_w(Tensor w, Tensor? bias, int cin, int cout) "
          "-> Tensor", &torch_vc4::vc4_conv_nhwc_pack_w_op);
    m.def("conv_nhwc(Tensor src, Tensor panel, int nimg, int cin, int h, "
          "int w, int cout, int kh, int kw, int stride, int pad, int dil) "
          "-> Tensor", &torch_vc4::vc4_conv_nhwc_op);
    m.def("depthwise_panel(Tensor w, int cpad) -> Tensor",
          &torch_vc4::vc4_depthwise_panel);
    m.def("depthwise_nhwc(Tensor src, Tensor panel, int nimg, int c, int h, "
          "int w) -> Tensor", &torch_vc4::vc4_depthwise_nhwc_op);
    m.def("take(Tensor src, Tensor idx, int rest) -> Tensor",
          &torch_vc4::vc4_take_host);
    m.def("argmax_host(Tensor t, int outer, int red, bool is_max) -> Tensor",
          &torch_vc4::vc4_arg_host);
}
