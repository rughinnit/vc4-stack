"""Autograd for the vc4 subset backend.

The graph is built from ``torch.autograd.Function``s whose backward calls the
engine's own backward kernels:

  mm/matmul    vc4_matmul_bt (dA = g B^T) and vc4_transpose + vc4_matmul
               (dB = A^T g)
  add/mul/sub/div  the engine's elementwise kernels; broadcast gradients are
               summed back to the input shape with the GEMM reduction
  relu         vc4_relu_backward (masked copy)
  silu/sigmoid/tanh/sqrt/gelu  composites of vc4_sigmoid / MUL / ADD
               (the exact GELU uses the erf and exp bodies)
  conv2d       vc4_conv2d_backward (im2col -> gemm3 -> col2im, dW and dX)
  max_pool2d   vc4_maxpool2x2_backward (argmax mask)
  softmax      y * (g - sum(g*y)) with the reduction as a GEMM
  linear       mm_bt / mm_at
  batch_norm   per-channel GEMM reductions + the engine's elementwise ops
  sum/mean     GEMM; the backward is a 0-stride broadcast of the scalar grad
  view         metadata (a view is materialised only when grad is needed)

The ``Autograd`` dispatch key is served by these Functions; inside a Function
we run below autograd (``_AutoDispatchBelowAutograd``) so the raw PrivateUse1
kernels are reached.  There is no CPU fallback: an op without an Autograd
kernel still raises instead of silently detaching, and ops whose backward
would need a scatter/sort kernel raise a precise message in backward.
"""

import torch

from . import ops

T = torch.ops.torch_vc4
_AdBelow = torch._C._AutoDispatchBelowAutograd

_LIBS = []


def _keep(lib):
    _LIBS.append(lib)
    return lib


__all__ = ["register", "mse_loss"]


def _g(g):
    """Gradients can arrive as expanded/0-stride views; the engine needs
    contiguous buffers, so materialise them (a byte gather, no arithmetic)."""
    return ops.contiguous(g)


def _below(fn):
    with _AdBelow():
        return fn()


def _ones(shape, like):
    return torch.ones(shape, dtype=torch.float32, device=like.device)


def _sub(a, b):
    return T.add(a, T.mul_scalar(b, -1.0))


def _col_sums(t2):
    """Sum a [M][N] tensor over rows -> [N] (one GEMM with a ones row)."""
    m, n = t2.shape
    return torch.mm(_ones((1, m), t2), t2).view(n)


def _recip(y):
    """1/y for y > 0, from the verified engine rsqrt."""
    r = T.rsqrt(y)
    return T.mul(r, r)


# ------------------------------------------------------------------- mm

class MmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _below(lambda: torch.mm(a, b))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        a, b = ctx.saved_tensors
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = _below(lambda: T.mm_bt(g, b))       # g @ b^T
        if ctx.needs_input_grad[1]:
            gb = _below(lambda: T.mm_at(a, g))       # a^T @ g
        return ga, gb


# --------------------------------------------------------------- binary

class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, alpha):
        ctx.alpha = float(alpha)
        ctx.sa, ctx.sb = tuple(a.shape), tuple(b.shape)
        return _below(lambda: ops._binary(a, b, "add", ctx.alpha))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        ga = ops._sum_to(g, ctx.sa) if ctx.needs_input_grad[0] else None
        gb = None
        if ctx.needs_input_grad[1]:
            ga2 = T.mul_scalar(g, ctx.alpha)
            gb = ops._sum_to(ga2, ctx.sb)
        return ga, gb, None


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.sa, ctx.sb = tuple(a.shape), tuple(b.shape)
        return _below(lambda: ops._binary(a, b, "mul"))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        a, b = ctx.saved_tensors
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = ops._sum_to(ops._mul_tensor(g, b), ctx.sa)
        if ctx.needs_input_grad[1]:
            gb = ops._sum_to(ops._mul_tensor(g, a), ctx.sb)
        return ga, gb


class SubFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, alpha):
        ctx.alpha = float(alpha)
        ctx.sa, ctx.sb = tuple(a.shape), tuple(b.shape)
        return _below(lambda: ops._binary(a, b, "sub", ctx.alpha))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        ga = ops._sum_to(g, ctx.sa) if ctx.needs_input_grad[0] else None
        gb = None
        if ctx.needs_input_grad[1]:
            gb = ops._sum_to(T.mul_scalar(g, -ctx.alpha), ctx.sb)
        return ga, gb, None


class DivFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.sa, ctx.sb = tuple(a.shape), tuple(b.shape)
        return _below(lambda: ops._binary(a, b, "div"))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        a, b = ctx.saved_tensors
        ga = gb = None
        if ctx.needs_input_grad[0]:
            ga = ops._sum_to(ops._mul_tensor(g, _recip(b)), ctx.sa)
        if ctx.needs_input_grad[1]:
            # d(a/b)/db = -a/b^2 = -(a/b)/b = -y/b
            y = ops._mul_tensor(a, _recip(b))
            gb = ops._sum_to(ops._mul_tensor(
                ops._mul_tensor(g, T.mul_scalar(y, -1.0)), _recip(b)), ctx.sb)
        return ga, gb


# --------------------------------------------------------------- scalars

class ScalarMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = float(s)
        return _below(lambda: T.mul_scalar(x, ctx.s))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (_below(lambda: T.mul_scalar(g, ctx.s))
                if ctx.needs_input_grad[0] else None), None


class ScalarAddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        return _below(lambda: T.add_scalar(x, float(s)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (g if ctx.needs_input_grad[0] else None), None


class NegFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _below(lambda: T.mul_scalar(x, -1.0))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return (_below(lambda: T.mul_scalar(g, -1.0))
                if ctx.needs_input_grad[0] else None)


class Pow2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.save_for_backward(x)
        return _below(lambda: T.mul(x, x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return (_below(lambda: T.mul_scalar(T.mul(g, x), 2.0))
                if ctx.needs_input_grad[0] else None), None


# --------------------------------------------------------------- unary

class ReluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: torch.relu(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return _below(lambda: T.relu_bwd(x, g))


class SigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _below(lambda: T.sigmoid(x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (s,) = ctx.saved_tensors
        def f():
            one_minus = T.add_scalar(T.mul_scalar(s, -1.0), 1.0)
            return T.mul(T.mul(g, s), one_minus)
        return _below(f)


class SiluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: T.silu(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            s = T.sigmoid(x)
            one_minus = T.add_scalar(T.mul_scalar(s, -1.0), 1.0)
            # d/dx [x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
            factor = T.add(s, T.mul(T.mul(x, s), one_minus))
            return T.mul(g, factor)
        return _below(f)


class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        y = _below(lambda: ops.softmax(x, dim))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (y,) = ctx.saved_tensors
        cols = y.shape[-1]
        rows = y.numel() // cols
        def f():
            gy = T.mul(g, y)
            rs = torch.mm(gy.view(rows, cols), _ones((cols, 1), y))  # [rows][1]
            b = torch.mm(rs, _ones((1, cols), y))                   # [rows][cols]
            b = b.view(g.shape)                                     # 3-D inputs
            return T.mul(y, T.add(g, T.mul_scalar(b, -1.0)))
        return _below(f), None


class AbsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: ops.unary_abs(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return _below(lambda: T.mul(g, ops.unary_sign(x)))


class SignFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _below(lambda: ops.unary_sign(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.mul_scalar(g, 0.0))


class ClipFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return _below(lambda: ops.unary_clip(x, lo, hi))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            zero = T.mul_scalar(x, 0.0)
            # 1 where x < lo, 1 where x > hi (relu(lo - x) / relu(x - hi))
            below = (ops.unary_sign(torch.relu(
                T.add_scalar(T.mul_scalar(x, -1.0), float(ctx.lo))))
                if ctx.lo is not None else zero)
            above = (ops.unary_sign(torch.relu(
                T.add_scalar(x, -float(ctx.hi))))
                if ctx.hi is not None else zero)
            mask = T.add_scalar(T.mul_scalar(T.add(below, above), -1.0), 1.0)
            return T.mul(g, mask)
        return _below(f), None, None


class LeakyReluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return _below(lambda: ops.unary_leaky_relu(x, slope))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            pos = ops.unary_sign(torch.relu(x))
            neg = T.add_scalar(T.mul_scalar(pos, -1.0), 1.0)
            factor = T.add(pos, T.mul_scalar(neg, ctx.slope))
            return T.mul(g, factor)
        return _below(f), None


class HardSigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: ops.unary_hardsigmoid(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            lo = ops.unary_sign(torch.relu(T.add_scalar(x, 3.0)))
            hi = ops.unary_sign(torch.relu(T.add_scalar(T.mul_scalar(x, -1.0),
                                                   3.0)))
            return T.mul_scalar(T.mul(g, T.mul(lo, hi)), 1.0 / 6.0)
        return _below(f)


class HardSwishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: ops.unary_hardswish(x))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        def f():
            lo = ops.unary_sign(torch.relu(T.add_scalar(x, 3.0)))     # x > -3
            lt3 = ops.unary_sign(torch.relu(T.add_scalar(T.mul_scalar(x, -1.0),
                                                    3.0)))        # x < 3
            mid = T.mul(lo, lt3)
            hi = T.add_scalar(T.mul_scalar(lt3, -1.0), 1.0)       # x >= 3
            slope = T.mul_scalar(T.add_scalar(T.mul_scalar(x, 2.0), 3.0),
                                 1.0 / 6.0)
            factor = T.add(T.mul(mid, slope), hi)
            return T.mul(g, factor)
        return _below(f)


class TanhFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _below(lambda: ops.tanh(x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (y,) = ctx.saved_tensors
        def f():
            one_minus = T.add_scalar(T.mul_scalar(T.mul(y, y), -1.0), 1.0)
            return T.mul(g, one_minus)
        return _below(f)


class SqrtFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _below(lambda: ops.sqrt(x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (y,) = ctx.saved_tensors
        return _below(lambda: T.mul_scalar(T.mul(g, _recip(y)), 0.5))


class ReciprocalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _below(lambda: ops.reciprocal(x))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (y,) = ctx.saved_tensors
        return _below(lambda: T.mul_scalar(T.mul(g, T.mul(y, y)), -1.0))


class GeluTanhFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: ops.gelu(x, "tanh"))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        import math
        c2 = 2.0 * math.sqrt(2.0 / math.pi)

        def f():
            x2 = T.mul(x, x)
            u = T.mul_scalar(T.add(x, T.mul_scalar(T.mul(x, x2), 0.044715)),
                             c2)
            s = T.sigmoid(u)
            du = T.mul_scalar(T.add_scalar(T.mul_scalar(x2, 3.0 * 0.044715),
                                           1.0), c2)
            ds = T.mul(T.mul(s, T.add_scalar(T.mul_scalar(s, -1.0), 1.0)), du)
            factor = T.add(s, T.mul(x, ds))
            return T.mul(g, factor)
        return _below(f)


class GeluErfFn(torch.autograd.Function):
    """Exact GELU.  The derivative is composed from the verified erf and exp
    bodies: 0.5(1+erf(x/sqrt2)) + x*exp(-x^2/2)/sqrt(2*pi)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _below(lambda: ops.gelu(x, "none"))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        import math

        def f():
            z = T.mul_scalar(x, 1.0 / math.sqrt(2.0))
            cdf = T.mul_scalar(T.add_scalar(ops.unary_erf(z), 1.0), 0.5)
            e = ops.unary_exp(T.mul_scalar(T.mul(x, x), -0.5))
            pdf = T.mul_scalar(e, 1.0 / math.sqrt(2.0 * math.pi))
            return T.mul(g, T.add(cdf, T.mul(x, pdf)))

        return _below(f)


# ---------------------------------------------------------------- conv

class Conv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, stride, pad, groups):
        ctx.save_for_backward(x, w)
        ctx.stride, ctx.pad, ctx.groups = stride, pad, groups
        ctx.has_bias = b is not None
        return _below(lambda: ops.conv2d(x, w, b, stride, pad, groups))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        x, w = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0:3]
        if not (need_x or need_w or need_b):
            return None, None, None, None, None, None
        dx, dw = _below(lambda: ops.conv2d_backward(
            x, w, g, ctx.stride, ctx.pad, ctx.groups))
        db = None
        if need_b:
            db = _below(lambda: ops._dims_reduce(g, [0, 2, 3], False, "sum"))
        return (dx if need_x else None, dw if need_w else None, db,
                None, None, None)


# ---------------------------------------------------------------- pooling

class MaxPool2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, s, p):
        ctx.save_for_backward(x)
        ctx.k, ctx.s, ctx.p = k, s, p
        return _below(lambda: ops.max_pool2d(x, k, s, p))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        return _below(lambda: ops.max_pool2d_backward(
            x, g, ctx.k, ctx.s, ctx.p)), None, None, None


class AdaptiveAvgPool2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_size):
        ctx.save_for_backward(x)
        ctx.out = tuple(int(v) for v in out_size)
        return _below(lambda: ops.adaptive_avg_pool2d(x, ctx.out))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        (x,) = ctx.saved_tensors
        C, H, W = x.shape[-3:]
        oh, ow = ctx.out
        def f():
            a = ops.adaptive_matrix(H, W, oh, ow).to(x.device)
            rows = g.numel() // (oh * ow)
            dx = torch.mm(g.reshape(rows, oh * ow), a)
            return dx.view(x.shape)
        return _below(f), None


# ---------------------------------------------------------------- views

class ViewFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        ctx.shape = tuple(x.shape)
        return _below(lambda: T.view_alias(x.detach(), list(size)))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.view_alias(g, list(ctx.shape))), None


class AsStridedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, stride, off):
        ctx.shape = tuple(x.shape)
        y = _below(lambda: T.as_strided_alias(x.detach(), list(size),
                                              list(stride), off))
        if not y.is_contiguous():
            raise RuntimeError(
                "torch_vc4: as_strided/permute views are supported as "
                "metadata only; autograd through a non-contiguous vc4 view "
                "is not implemented (the engine has no strided kernels)")
        return y

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.view_alias(g, list(ctx.shape))), None, None, None


# ---------------------------------------------------------------- linear

class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        return _below(lambda: ops.linear(x, w, b))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        x, w = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0:3]
        g2 = g.view(-1, g.shape[-1])
        # dx = g @ W   (W is [out][in]); dw = g^T @ x
        dx = _below(lambda: torch.mm(g2, w)) if need_x else None
        if dx is not None:
            dx = dx.view(x.shape)
        dw = _below(lambda: T.mm_at(g2, x.view(-1, x.shape[-1]))) \
            if need_w else None
        db = _below(lambda: _col_sums(g2)) if need_b else None
        return dx, dw, db


# ------------------------------------------------------------- reductions

class SumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = tuple(x.shape)
        return _below(lambda: ops.sum_all(x))

    @staticmethod
    def backward(ctx, g):
        gv = float(g)
        def f():
            ones = torch.ones(ctx.shape, dtype=torch.float32,
                              device=g.device)
            return T.mul_scalar(ones, gv)
        return _below(f)


class MeanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = tuple(x.shape)
        return _below(lambda: ops.mean_all(x))

    @staticmethod
    def backward(ctx, g):
        n = 1
        for d in ctx.shape:
            n *= d
        gv = float(g) / n
        def f():
            ones = torch.ones(ctx.shape, dtype=torch.float32,
                              device=g.device)
            return T.mul_scalar(ones, gv)
        return _below(f)


class SumDimFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, keepdim):
        ctx.shape = tuple(x.shape)
        ctx.dims = [int(d) for d in dims]
        ctx.keepdim = bool(keepdim)
        return _below(lambda: ops.sum_dims(x, ctx.dims, ctx.keepdim))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: ops._expand_reduced(g, ctx.shape, ctx.dims)), \
            None, None


class MeanDimFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, keepdim):
        ctx.shape = tuple(x.shape)
        ctx.dims = [int(d) for d in dims]
        ctx.keepdim = bool(keepdim)
        n = 1
        for d in ctx.dims:
            n *= ctx.shape[d]
        ctx.n = n
        return _below(lambda: ops.mean_dims(x, ctx.dims, ctx.keepdim))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        return _below(lambda: T.mul_scalar(
            ops._expand_reduced(g, ctx.shape, ctx.dims), 1.0 / ctx.n)), \
            None, None


class AmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, keepdim, is_max):
        ctx.save_for_backward(x)
        ctx.dims = dims
        ctx.keepdim = keepdim
        return _below(lambda: ops.amax_dims(x, dims, keepdim) if is_max
                      else ops.amin_dims(x, dims, keepdim))

    @staticmethod
    def backward(ctx, g):
        raise RuntimeError(
            "torch_vc4: amax/amin backward is unavailable on this backend: "
            "it needs an equality mask plus a scatter of the gradient among "
            "ties, and the engine has no scatter kernel (the table gather "
            "only reads, and it hangs on this board).  Use sum/mean, or "
            "rewrite the model with a differentiable reduction.")


# ------------------------------------------------------------- batch norm

class BatchNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training,
                momentum, eps):
        y, bmean, invstd = _below(lambda: ops._bn_forward(
            x, weight, bias, running_mean, running_var, bool(training),
            float(momentum), float(eps)))
        C = x.shape[1]
        shape = tuple(x.shape)
        xhat = _below(lambda: T.mul(
            T.add(x, T.mul_scalar(ops._expand(bmean.view(1, C, 1, 1), shape),
                                  -1.0)),
            ops._expand(invstd.view(1, C, 1, 1), shape)))
        ctx.save_for_backward(x, weight, bias, xhat, invstd)
        ctx.bn_shape = shape
        ctx.mark_non_differentiable(bmean, invstd)
        return y, bmean, invstd

    @staticmethod
    def backward(ctx, g, gb, gi):
        g = _g(g)
        x, weight, bias, xhat, invstd = ctx.saved_tensors
        N, C, H, W = ctx.bn_shape
        n = N * H * W
        def f():
            sum_dy = ops._dims_reduce(g, [0, 2, 3], True, "sum")
            sum_dy_x = ops._dims_reduce(T.mul(g, xhat), [0, 2, 3], True, "sum")
            mean_dy = T.mul_scalar(sum_dy, 1.0 / n)
            mean_dyx = T.mul_scalar(sum_dy_x, 1.0 / n)
            t = T.add(g, T.mul_scalar(ops._expand(mean_dy, g.shape), -1.0))
            t = T.add(t, T.mul_scalar(
                T.mul(xhat, ops._expand(mean_dyx, g.shape)), -1.0))
            dx = T.mul(t, ops._expand(invstd.view(1, C, 1, 1), g.shape))
            if weight is not None:
                dx = T.mul(dx, ops._expand(weight.view(1, C, 1, 1), g.shape))
            return dx, sum_dy_x.view(C), sum_dy.view(C)
        dx, dgamma, dbeta = _below(f)
        if weight is None or not ctx.needs_input_grad[1]:
            dgamma = None
        if bias is None or not ctx.needs_input_grad[2]:
            dbeta = None
        return (dx if ctx.needs_input_grad[0] else None, dgamma, dbeta,
                None, None, None, None, None)


# ------------------------------------------------------------- cat / resize

class CatFn(torch.autograd.Function):
    # The tensors are passed as *args, not as one list: a TensorList argument
    # is not scanned for requires_grad by Function.apply, so its graph node
    # was silently dropped.
    @staticmethod
    def forward(ctx, dim, *tensors):
        ctx.dim = int(dim)
        ctx.shapes = [tuple(t.shape) for t in tensors]
        return _below(lambda: ops.cat(list(tensors), ctx.dim))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        dim = ctx.dim
        nd = g.dim()
        if nd == 3 and dim in (0, -3):
            off, outs = 0, []
            for sh in ctx.shapes:
                c = sh[0]
                outs.append(_below(lambda o=off, c=c: T.take_ch(g, o, o + c)))
                off += c
            return (None,) + tuple(outs)
        if nd == 4 and dim in (1, -3):
            N = g.shape[0]
            off, outs = 0, []
            for sh in ctx.shapes:
                c = sh[1]
                parts = []
                for n in range(N):
                    parts.append(_below(
                        lambda n=n, o=off, c=c: T.take_ch(g[n], o, o + c)))
                outs.append(_below(lambda p=parts: T.cat_ch(p)).view(sh))
                off += c
            return (None,) + tuple(outs)
        raise RuntimeError("torch_vc4: cat backward for rank %d dim %d is not "
                           "implemented" % (nd, dim))


def _up_matrix(H, W, ho, wo, sh, sw):
    """0/1 matrix A[o][p] for nearest resize (PyTorch's floor index rule)."""
    import math
    a = torch.zeros((ho * wo, H * W), dtype=torch.float32)
    sy = 1.0 / sh if sh else H / ho
    sx = 1.0 / sw if sw else W / wo
    for y in range(ho):
        hy = min(int(math.floor(y * sy)), H - 1)
        for x in range(wo):
            hx = min(int(math.floor(x * sx)), W - 1)
            a[y * wo + x, hy * W + hx] = 1.0
    return a


class UpsampleNearestFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ho, wo, sh, sw):
        ctx.shape = tuple(x.shape)
        ctx.out = (ho, wo)
        ctx.sc = (sh, sw)
        return _below(lambda: ops.upsample_nearest2d(x, (ho, wo), sh, sw))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        N, C, H, W = ctx.shape
        oh, ow = ctx.out
        sh, sw = ctx.sc
        def f():
            a = _up_matrix(H, W, oh, ow, sh, sw).to(g.device)
            return torch.mm(g.view(N * C, oh * ow), a).view(ctx.shape)
        return _below(f), None, None, None, None


def _bilin_matrix(H, W, ho, wo, align_corners):
    """A[o][p]: the four bilinear weights of output o over source p.

    Geometry only -- the same taps and weights the forward uses, built once
    per backward from the output grid; no activation value is read.  The
    adjoint is then ``g @ A``, a device GEMM (the pattern
    ``UpsampleNearestFn`` already uses for the nearest mapping).
    """
    import math
    a = torch.zeros((ho * wo, H * W), dtype=torch.float32)

    def axis(n_out, n_in):
        taps = []
        for o in range(n_out):
            if align_corners:
                s = (o * (n_in - 1) / (n_out - 1)) if n_out > 1 else 0.0
            else:
                s = (o + 0.5) * n_in / n_out - 0.5
                if s < 0.0:
                    s = 0.0
            i0 = int(math.floor(s))
            f = s - i0
            if i0 < 0:
                i0, f = 0, 0.0
            if i0 >= n_in - 1:
                i0, f = n_in - 1, 0.0
            i1 = i0 + 1 if i0 + 1 < n_in else n_in - 1
            taps.append((i0, i1, f))
        return taps

    ty, tx = axis(ho, H), axis(wo, W)
    for oy, (y0, y1, fy) in enumerate(ty):
        for ox, (x0, x1, fx) in enumerate(tx):
            o = oy * wo + ox
            a[o, y0 * W + x0] += (1.0 - fy) * (1.0 - fx)
            a[o, y0 * W + x1] += (1.0 - fy) * fx
            a[o, y1 * W + x0] += fy * (1.0 - fx)
            a[o, y1 * W + x1] += fy * fx
    return a


class UpsampleBilinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ho, wo, align_corners):
        ctx.shape = tuple(x.shape)
        ctx.out = (ho, wo)
        ctx.ac = bool(align_corners)
        return _below(lambda: ops.upsample_bilinear2d(x, (ho, wo), ctx.ac))

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        N, C, H, W = ctx.shape
        oh, ow = ctx.out
        n_in, n_out = H * W, oh * ow
        if n_out * n_in * 4 > (32 << 20):
            raise RuntimeError(
                "torch_vc4: upsample_bilinear2d backward builds a %dx%d "
                "geometry matrix (%.1f MiB); over the 32 MiB cap.  Decode at a "
                "smaller stage or run the backward on the CPU."
                % (n_out, n_in, n_out * n_in * 4 / float(1 << 20)))

        def f():
            a = _bilin_matrix(H, W, oh, ow, ctx.ac).to(g.device)
            return torch.mm(g.view(N * C, n_out), a).view(ctx.shape)
        return _below(f), None, None, None


# ------------------------------------------------------------- indexing

class GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, index, is_index_select):
        ctx.dim = dim
        return _below(lambda: (ops.index_select(x, dim, index)
                               if is_index_select
                               else ops.gather(x, dim, index)))

    @staticmethod
    def backward(ctx, g):
        raise RuntimeError(
            "torch_vc4: gather/index_select backward is unavailable: it needs "
            "scatter_add, and the engine has no scatter kernel (the table "
            "gather only reads, and it hangs on this board).  Keep gathered "
            "features read-only, or run the model on CPU.")


# ------------------------------------------------------------------ loss

class MSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        ctx.save_for_backward(pred, target)
        def f():
            d = _sub(pred, target)
            s = T.sum_all(T.mul(d, d).view(-1))
            return T.mul_scalar(s, 1.0 / pred.numel()).view([])
        return _below(f)

    @staticmethod
    def backward(ctx, g):
        g = _g(g)
        pred, target = ctx.saved_tensors
        n = pred.numel()
        scale = float(g) * 2.0 / n
        def f():
            d = _sub(pred, target)
            return (T.mul_scalar(d, scale), T.mul_scalar(d, -scale))
        ga, gb = _below(f)
        return (ga if ctx.needs_input_grad[0] else None,
                gb if ctx.needs_input_grad[1] else None)


def mse_loss(pred, target):
    """Mean squared error through the backend (forward and backward on vc4).

    ``F.mse_loss`` decomposes into ops this backend does not implement
    (``resize_`` and friends); this is the same reduction written with the
    engine's GEMM-based sum, so the whole loss stays on the device.
    """
    ops.f32(pred, "mse_loss(pred)")
    ops.f32(target, "mse_loss(target)")
    if pred.shape != target.shape:
        raise RuntimeError("torch_vc4: mse_loss needs equal shapes, got %s "
                           "and %s" % (tuple(pred.shape), tuple(target.shape)))
    return MSELossFn.apply(pred, target)


def _no_div(a, b):
    raise RuntimeError("torch_vc4: division of two device tensors is not "
                       "implemented on this backend")


def _k(v):
    p = ops._pair(v)
    if p[0] != p[1]:
        raise RuntimeError("torch_vc4: square kernel/stride required, got %r"
                           % (v,))
    return p[0]


# ------------------------------------------------------------ registration

def register():
    # torch deregisters a Library when it is collected; keep it alive
    lib = _keep(torch.library.Library("aten", "IMPL",
                                      "AutogradPrivateUse1"))

    def conv(x, w, b, stride, pad, dil, transposed, out_pad, groups):
        if transposed or any(int(v) for v in out_pad):
            raise RuntimeError("torch_vc4: transposed conv has no autograd")
        if ops._pair(dil) != (1, 1):
            raise RuntimeError("torch_vc4: conv dilation != 1 is not supported")
        s, p = ops._pair(stride), ops._pair(pad)
        if s[0] != s[1] or p[0] != p[1]:
            raise RuntimeError("torch_vc4: conv needs square stride/padding")
        return Conv2dFn.apply(x, w, b, s[0], p[0], int(groups))

    lib.impl("convolution_overrideable", conv)
    lib.impl("mm", lambda a, b: MmFn.apply(a, b))
    lib.impl("matmul", lambda a, b: MmFn.apply(a, b))

    def add(a, b, alpha=1):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarAddFn.apply(a, s * float(alpha))
        s = ops._scalar_of(a)
        if s is not None:
            return ScalarAddFn.apply(b, s * float(alpha))
        return AddFn.apply(a, b, alpha)

    def mul(a, b):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarMulFn.apply(a, s)
        s = ops._scalar_of(a)
        if s is not None:
            return ScalarMulFn.apply(b, s)
        return MulFn.apply(a, b)

    def sub(a, b, alpha=1):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarAddFn.apply(a, -s * float(alpha))
        if ops._scalar_of(a) is not None:
            raise RuntimeError("torch_vc4: (scalar - tensor) has no autograd "
                               "kernel; rewrite it as a - x on the device")
        return SubFn.apply(a, b, alpha)

    def div(a, b):
        s = ops._scalar_of(b)
        if s is not None:
            return ScalarMulFn.apply(a, 1.0 / s)
        return DivFn.apply(a, b)

    lib.impl("add.Tensor", add)
    lib.impl("mul.Tensor", mul)
    lib.impl("sub.Tensor", sub)
    lib.impl("div.Tensor", div)
    lib.impl("relu", lambda x: ReluFn.apply(x))
    lib.impl("silu", lambda x: SiluFn.apply(x))
    lib.impl("sigmoid", lambda x: SigmoidFn.apply(x))
    lib.impl("abs", lambda x: AbsFn.apply(x))
    lib.impl("sign", lambda x: SignFn.apply(x))
    lib.impl("reciprocal", lambda x: ReciprocalFn.apply(x))
    lib.impl("sqrt", lambda x: SqrtFn.apply(x))
    lib.impl("tanh", lambda x: TanhFn.apply(x))
    lib.impl("clip", lambda x, lo=None, hi=None: ClipFn.apply(x, lo, hi))
    lib.impl("leaky_relu", lambda x, slope=0.01:
             LeakyReluFn.apply(x, float(slope)))
    lib.impl("hardsigmoid", lambda x: HardSigmoidFn.apply(x))
    lib.impl("hardswish", lambda x: HardSwishFn.apply(x))

    def gelu(x, approximate="none"):
        if approximate == "none":
            return GeluErfFn.apply(x)
        if approximate != "tanh":
            return ops.gelu(x, approximate)  # raises with the precise reason
        return GeluTanhFn.apply(x)

    lib.impl("gelu", gelu)
    lib.impl("_softmax", lambda x, dim, half=False: SoftmaxFn.apply(x, dim))
    def max_pool2d(x, k, s=None, p=0, d=1, ceil=False):
        if ceil:
            raise RuntimeError("torch_vc4: max_pool2d ceil_mode is not "
                               "implemented (the engine pools over a "
                               "floor-shaped window)")
        if ops._pair(d) != (1, 1):
            raise RuntimeError("torch_vc4: max_pool2d dilation != 1 is not "
                               "implemented")
        return MaxPool2dFn.apply(x, _k(k), _k(s if s not in (None, []) else k),
                                 _k(p))

    lib.impl("max_pool2d", max_pool2d)
    lib.impl("_adaptive_avg_pool2d",
             lambda x, out: AdaptiveAvgPool2dFn.apply(x, out))
    lib.impl("adaptive_avg_pool2d",
             lambda x, out: AdaptiveAvgPool2dFn.apply(x, out))
    lib.impl("view", lambda x, size: ViewFn.apply(x, list(size)))
    lib.impl("as_strided",
             lambda x, size, stride, off=None:
             AsStridedFn.apply(x, list(size), list(stride), off))
    lib.impl("linear", lambda x, w, b=None: LinearFn.apply(x, w, b))
    lib.impl("mul.Scalar", lambda x, s: ScalarMulFn.apply(x, float(s)))
    lib.impl("div.Scalar", lambda x, s: ScalarMulFn.apply(x, 1.0 / float(s)))
    lib.impl("add.Scalar", lambda x, s, alpha=1: ScalarAddFn.apply(
        x, float(s) * float(alpha)))
    lib.impl("sub.Scalar", lambda x, s, alpha=1: ScalarAddFn.apply(
        x, -float(s) * float(alpha)))
    lib.impl("neg", lambda x: NegFn.apply(x))
    lib.impl("pow.Tensor_Scalar", lambda x, p: Pow2Fn.apply(x, float(p)))
    lib.impl("sum", lambda x, dtype=None: SumFn.apply(x))

    def sum_dim(x, dim, keepdim=False, dtype=None):
        return SumDimFn.apply(x, list(dim), bool(keepdim))

    def mean_dim(x, dim, keepdim=False, dtype=None):
        return MeanDimFn.apply(x, list(dim), bool(keepdim))

    lib.impl("sum.dim_IntList", sum_dim)
    lib.impl("mean.dim", mean_dim)
    lib.impl("mean", lambda x, dtype=None: MeanFn.apply(x))

    def amax(x, dim=None, keepdim=False):
        d = [] if dim is None else list(dim)
        return AmaxFn.apply(x, d, bool(keepdim), True)

    def amin(x, dim=None, keepdim=False):
        d = [] if dim is None else list(dim)
        return AmaxFn.apply(x, d, bool(keepdim), False)

    lib.impl("amax", amax)
    lib.impl("amin", amin)

    def max_full(x):
        return AmaxFn.apply(x, [], False, True)

    def min_full(x):
        return AmaxFn.apply(x, [], False, False)

    def max_dim(x, dim, keepdim=False):
        return (AmaxFn.apply(x, [int(dim)], bool(keepdim), True),
                ops.argmax(x, int(dim), bool(keepdim)))

    def min_dim(x, dim, keepdim=False):
        return (AmaxFn.apply(x, [int(dim)], bool(keepdim), False),
                ops.argmin(x, int(dim), bool(keepdim)))

    lib.impl("max", max_full)
    lib.impl("min", min_full)
    lib.impl("max.dim", max_dim)
    lib.impl("min.dim", min_dim)
    lib.impl("cat", lambda ts, dim=0: CatFn.apply(int(dim), *list(ts)))
    lib.impl("upsample_nearest2d",
             lambda x, size, sh=None, sw=None:
             UpsampleNearestFn.apply(x, int(size[0]), int(size[1]), sh, sw))

    def up_vec(x, size, scales):
        if size is not None:
            return UpsampleNearestFn.apply(x, int(size[0]), int(size[1]),
                                           None, None)
        H, W = x.shape[-2], x.shape[-1]
        sh, sw = float(scales[0]), float(scales[1])
        import math
        return UpsampleNearestFn.apply(x, int(math.floor(H * sh)),
                                       int(math.floor(W * sw)), sh, sw)

    lib.impl("upsample_nearest2d.vec", up_vec)
    lib.impl("upsample_bilinear2d",
             lambda x, size, ac, sh=None, sw=None:
             UpsampleBilinearFn.apply(x, int(size[0]), int(size[1]), ac))

    def up_bilin_vec(x, size, ac, scales):
        if size is not None:
            return UpsampleBilinearFn.apply(x, int(size[0]), int(size[1]), ac)
        import math
        H, W = x.shape[-2], x.shape[-1]
        sh, sw = float(scales[0]), float(scales[1])
        return UpsampleBilinearFn.apply(x, int(math.floor(H * sh)),
                                        int(math.floor(W * sw)), ac)

    lib.impl("upsample_bilinear2d.vec", up_bilin_vec)
    lib.impl("gather", lambda x, dim, index, sparse_grad=False:
             GatherFn.apply(x, int(dim), index, False))
    lib.impl("index_select", lambda x, dim, index:
             GatherFn.apply(x, int(dim), index, True))

    def native_bn(x, w, b, rm, rv, training, momentum, eps):
        return BatchNormFn.apply(x, w, b, rm, rv, bool(training),
                                 float(momentum), float(eps))

    lib.impl("native_batch_norm", native_bn)

    def batch_bn(x, w, b, rm, rv, training, momentum, eps,
                 cudnn_enabled=False):
        return BatchNormFn.apply(x, w, b, rm, rv, bool(training),
                                 float(momentum), float(eps))[0]

    lib.impl("batch_norm", batch_bn)
    return lib
