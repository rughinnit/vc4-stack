"""Runner handlers for the recurrent operators: ``RNN``, ``GRU``, ``LSTM``.

``docs/OPS-RNN.md`` is the measurement record for this file.  The method is a
host unroll of the *static* sequence: every timestep is one matrix multiply
against a constant weight panel plus elementwise gate arithmetic, all of it on
the QPUs.  Nothing on the ARM touches an activation value.

Per timestep and direction the shape is

    pre = X_t @ W^T + H_{t-1} @ R^T + (Wb + Rb)

with ``X_t`` read straight out of the input tensor (one row block of the
time-major layout) and ``H_{t-1}`` the compact ``[batch][hidden]`` buffer the
previous step produced.  The products are QPU GEMMs, the biases are constant
panels, and each gate is a lane copy of one column block of ``pre``.

Three facts about this engine shape the code and are not stylistic:

* ``vc4_matmul`` pads an unaligned A operand by copying it **with the ARM**
  from the cacheable arena mapping.  A buffer the QPU wrote can therefore be
  read stale (docs/OPS-NORM.md, ``sum_last_aligned``).  Any GEMM whose A is a
  QPU-produced buffer gets an explicitly zero-padded A copied by
  ``copy2d_x`` on the QPUs, so the library's own pad path never runs.
* the GEMM writes full 8x64 output tiles, so a gate buffer is
  ``[align8(batch)][align64(gates*hidden)]`` and the column blocks are sliced
  at that row pitch.
* every buffer a timestep allocates is released at the end of that timestep
  (a long unroll would otherwise hold thousands of dead buffers).

Forward inference only: there is no backward pass in this file, and nothing
here claims training support.
"""
import numpy as np
import onnx

from . import _binding as vc4infer
from ._dispatch_extra import _Base, _Tensor, _attr, _prod
from ._errors import UnsupportedNode

# gates per direction, in the order the ONNX weight matrices are laid out
NGATE = {"RNN": 1, "GRU": 3, "LSTM": 4}
# ONNX default activations for one direction: RNN (f), GRU (f, g),
# LSTM (f, g, h) -- schema attribute ``activations``
DEF_ACT = {"RNN": ("Tanh",), "GRU": ("Sigmoid", "Tanh"),
           "LSTM": ("Sigmoid", "Tanh", "Tanh")}


def _al(n, a):
    return (int(n) + a - 1) // a * a


def _strs(node, name):
    """A repeated-string attribute (``activations``).

    ``_dispatch_extra._attr`` decodes a single STRING; a STRINGS list needs
    its own reader, and returning None for it silently fell back to the
    default activations (the mismatch this fixed).
    """
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.STRINGS:
            return [v.decode() for v in a.strings]
    return None


class Recurrent(_Base):
    """One handler for all three recurrent operators."""

    def run(self):
        node = self.node
        self.op = node.op_type
        self.G = NGATE[self.op]
        self.step = []

        X = self.ten(node.input[0])
        xs = tuple(int(d) for d in X.shape)
        if len(xs) != 3:
            self.fail("needs a rank-3 input ([seq, batch, input] with "
                      "layout 0, [batch, seq, input] with layout 1); got %s"
                      % (xs,))
        self.layout = int(_attr(node, "layout", 0))
        if self.layout not in (0, 1):
            self.fail("layout=%d (ONNX defines 0 and 1)" % self.layout)
        self.direction = str(_attr(node, "direction", "forward"))
        if self.direction == "bidirectional":
            self.nd = 2
        elif self.direction in ("forward", "reverse"):
            self.nd = 1
        else:
            self.fail("direction=%r is not one of forward/reverse/"
                      "bidirectional" % self.direction)

        if self.layout == 0:
            T, B, I = xs
        else:
            B, T, I = xs
        if min(T, B, I) < 0:
            self.fail("negative extent %s" % (xs,))
        self.T, self.B, self.I = T, B, I

        if len(node.input) > 4 and node.input[4]:
            self.fail("sequence_lens (%s) is not implemented: the kernel "
                      "unrolls one static sequence and every batch row runs "
                      "the full length" % node.input[4])

        W = self.param(node.input[1], "W")
        R = self.param(node.input[2], "R")
        if W.ndim != 3 or R.ndim != 3:
            self.fail("W/R must be rank 3 [num_directions][gates*hidden]"
                      "[input|hidden] (got %s / %s)" % (W.shape, R.shape))
        if W.shape[0] != self.nd or R.shape[0] != self.nd:
            self.fail("W has %d directions and R %d but direction=%r"
                      % (W.shape[0], R.shape[0], self.direction))
        H = int(R.shape[2])
        if int(W.shape[1]) != self.G * H or int(R.shape[1]) != self.G * H:
            self.fail("W/R have %d/%d gate rows for hidden_size=%d (want %d)"
                      % (W.shape[1], R.shape[1], H, self.G * H))
        if int(W.shape[2]) != I:
            self.fail("W input extent %d != input_size %d" % (W.shape[2], I))
        if int(R.shape[2]) != H:
            self.fail("R hidden extent %d != hidden_size %d" % (R.shape[2], H))
        hs = _attr(node, "hidden_size", None)
        if hs is not None and int(hs) != H:
            self.fail("hidden_size=%d but R says %d" % (int(hs), H))
        if H < 1:
            self.fail("hidden_size must be >= 1")
        self.H, self.nd = H, self.nd

        Bt = None
        if len(node.input) > 3 and node.input[3]:
            Bt = self.param(node.input[3], "B")
            if Bt.ndim != 2 or int(Bt.shape[0]) != self.nd or \
                    int(Bt.shape[1]) != 2 * self.G * H:
                self.fail("B must be [%d][%d] (got %s)"
                          % (self.nd, 2 * self.G * H, Bt.shape))

        h0 = c0 = P = None
        if len(node.input) > 5 and node.input[5]:
            h0 = self.state_in(node.input[5], "initial_h")
        if self.op == "LSTM":
            if len(node.input) > 6 and node.input[6]:
                c0 = self.state_in(node.input[6], "initial_c")
            if len(node.input) > 7 and node.input[7]:
                P = self.param(node.input[7], "P")
                if P.ndim != 2 or int(P.shape[0]) != self.nd or \
                        int(P.shape[1]) != 3 * H:
                    self.fail("P must be [%d][%d] (got %s)"
                              % (self.nd, 3 * H, P.shape))

        self.acts, self.alphas, self.betas = self.activation_table(self.nd)
        clip = _attr(node, "clip", None)
        self.clip = None if clip is None else abs(float(clip))
        self.lbr = int(_attr(node, "linear_before_reset", 0))
        if self.op != "GRU" and self.lbr:
            self.fail("linear_before_reset is a GRU attribute")

        self.Mp = _al(B, 8)
        self.Np = _al(self.G * H, 64)
        self.NpH = _al(H, 64)

        self.panels = []
        for d in range(self.nd):
            if Bt is None:
                bias = np.zeros((2 * self.G * H,), np.float32)
            else:
                bias = np.ascontiguousarray(Bt[d], np.float32)
            wb, rb = bias[:self.G * H], bias[self.G * H:]
            self.panels.append(self.weight_panel(
                d, W[d], R[d], wb, rb, None if P is None else P[d]))

        if self.layout == 0:
            yshape = (T, self.nd, B, H)
        else:
            yshape = (B, T, self.nd, H)
        Y = self.alloc(yshape)
        Y_h = self.alloc((self.nd, B, H) if self.layout == 0
                         else (B, self.nd, H))
        Y_c = None
        if self.op == "LSTM":
            Y_c = self.alloc((self.nd, B, H) if self.layout == 0
                             else (B, self.nd, H))

        if T == 0:
            # an empty sequence: Y has no rows, and ONNX defines the final
            # state as the initial one
            for d in range(self.nd):
                if h0 is not None:
                    self.store_state(h0, d, Y_h)
                else:
                    self.fill_state(Y_h, d, 0.0)
                if Y_c is not None:
                    if c0 is not None:
                        self.store_state(c0, d, Y_c)
                    else:
                        self.fill_state(Y_c, d, 0.0)
            return self.outputs(Y, Y_h, Y_c)

        for d in range(self.nd):
            h = self.load_state(h0, d, 0.0)
            c = self.load_state(c0, d, 0.0) if self.op == "LSTM" else None
            reverse = (d == 1) or (self.direction == "reverse")
            order = range(T - 1, -1, -1) if reverse else range(T)
            for t in order:
                xt = self.step_input(X, t)
                h, c = self.step_once(d, xt, h, c)
                self.store_h(Y, t, d, h)
                self.sweep(keep=(h, c))
            self.store_state_dev(h, d, Y_h)
            if c is not None:
                self.store_state_dev(c, d, Y_c)
        return self.outputs(Y, Y_h, Y_c)

    # ------------------------------------------------------- allocation
    def salloc(self, shape, fill=None):
        """A per-step buffer, released by :meth:`sweep`."""
        shape = tuple(int(d) for d in shape)
        n = _prod(shape)
        if fill is None:
            t = self.r._alloc_t(shape)
        else:
            words = ((n + 63) // 64) * 64
            t = self.r._alloc_t(shape, extra=words - n)
            self.fill(t.buf, n, fill)
        self.step.append(t)
        return t

    def track(self, t):
        self.step.append(t)
        return t

    def sweep(self, keep=()):
        live = set(id(t) for t in keep if t is not None)
        if not self.step:
            return
        dead = [t for t in self.step if id(t) not in live]
        ids = set(id(t) for t in dead)
        for t in dead:
            self.r.pool.release(t.buf)
        self.r._garbage = [g for g in self.r._garbage if id(g) not in ids]
        self.step = [t for t in self.step if id(t) in live]

    # ------------------------------------------------------------ inputs
    def param(self, name, what):
        """A weight/parameter tensor as a host float32 numpy array.

        A graph initialiser is already on the host; a runtime tensor is
        downloaded once (``vc4_invalidate_buf`` makes that read correct) and
        then used as a constant panel.  Weights are parameters, not
        activations.
        """
        a = self.r.host.get(name)
        if a is None:
            from ._onnxrun import Tensor
            v = self.r.values.get(name)
            if not isinstance(v, Tensor):
                self.fail("%s %s is neither a constant nor an fp32 tensor"
                          % (what, name))
            a = self.r.download(v)
        return np.ascontiguousarray(a, np.float32)

    def state_in(self, name, what):
        """initial_h / initial_c in the layout the graph declared."""
        a = self.param(name, what)
        if a.ndim != 3:
            self.fail("%s must be rank 3 (got %s)" % (what, a.shape))
        want = (self.nd, self.B, self.H) if self.layout == 0 \
            else (self.B, self.nd, self.H)
        if tuple(a.shape) != want:
            self.fail("%s is %s, want %s" % (what, a.shape, want))
        return a

    def load_state(self, arr, d, fill):
        """Direction d of a state tensor as a compact [B][H] (or zeros)."""
        t = self.salloc((self.B, self.H), fill)
        if arr is not None:
            host = self.salloc((self.B, self.H))
            src = arr[d] if self.layout == 0 else \
                np.transpose(arr, (1, 0, 2))[d]
            self.r.eng.upload(np.ascontiguousarray(src, np.float32), host.buf)
            self.lane_runs(host.buf, t.buf, self.B, self.H, self.H * 4,
                           self.H * 4, 4)
        return t

    def step_input(self, X, t):
        """X[t] as a compact [B][I] (one lane copy, layout 1 included)."""
        out = self.salloc((self.B, self.I))
        if self.layout == 0:
            self.lane_runs(self.r._buf_at(X, t * self.B * self.I * 4),
                           out.buf, self.B, self.I, self.I * 4, self.I * 4, 4)
        else:
            self.lane_runs(self.r._buf_at(X, t * self.I * 4), out.buf,
                           self.B, self.I, self.T * self.I * 4,
                           self.I * 4, 4)
        return out

    # ------------------------------------------------------- weight panels
    def weight_panel(self, d, W, R, wb, rb, P):
        """The transposed, zero-padded constants for one direction.

        A panel is never re-transposed at run time: the whole node runs off
        these device buffers, and they are cached across runs like a packed
        conv weight.
        """
        H, I = self.H, self.I
        key = ("rnn", self.node.name, d)

        def pad(a, rows, cols):
            v = np.zeros((rows, cols), np.float32)
            v[:a.shape[0], :a.shape[1]] = a
            return v

        def vec(a, n):
            v = np.zeros((n,), np.float32)
            v[:a.size] = a
            return v

        pan = {}
        W = np.ascontiguousarray(W, np.float32).T   # [I][G*H]
        R = np.ascontiguousarray(R, np.float32).T   # [H][G*H]
        if self.op == "GRU":
            N2 = _al(2 * H, 64)
            pan["Wzr"] = self.const(key + ("Wzr",),
                                    pad(W[:, :2 * H], _al(I, 16), N2))
            pan["Rzr"] = self.const(key + ("Rzr",),
                                    pad(R[:, :2 * H], _al(H, 16), N2))
            pan["Wh"] = self.const(key + ("Wh",),
                                   pad(W[:, 2 * H:], _al(I, 16), self.NpH))
            pan["Rh"] = self.const(key + ("Rh",),
                                   pad(R[:, 2 * H:], _al(H, 16), self.NpH))
            pan["bzr"] = self.bcast(key + ("bzr",),
                                    vec(wb[:2 * H] + rb[:2 * H], N2))
            pan["bwh"] = self.bcast_c(key + ("bwh",),
                                      vec(wb[2 * H:], H))
            pan["brh"] = self.bcast_c(key + ("brh",),
                                      vec(rb[2 * H:], H))
            return pan
        N = self.G * H
        pan["W"] = self.const(key + ("W",), pad(W, _al(I, 16), self.Np))
        pan["R"] = self.const(key + ("R",), pad(R, _al(H, 16), self.Np))
        pan["b"] = self.bcast(key + ("b",), vec(wb + rb, self.Np))
        if P is not None:
            P = np.asarray(P, np.float32)
            for i, nm in enumerate(("Pi", "Po", "Pf")):
                pan[nm] = self.bcast_c(key + (nm,),
                                       vec(P[i * H:(i + 1) * H], H))
            pan["peep"] = True
        return pan

    def bcast(self, key, row):
        """A [N] constant as a [Mp][N] panel (a plan-time broadcast)."""
        return self.const(key, np.tile(row.reshape(1, -1), (self.Mp, 1)))

    def bcast_c(self, key, row):
        """A [N] constant as a compact [B][N] tile (gate-sized)."""
        return self.const(key, np.tile(row.reshape(1, -1), (self.B, 1)))

    # -------------------------------------------------------- activations
    def activation_table(self, nd):
        node = self.node
        K = len(DEF_ACT[self.op])
        acts = _strs(node, "activations")
        if not acts:
            acts = list(DEF_ACT[self.op]) * nd
        else:
            acts = [str(a) for a in acts]
            if len(acts) != K * nd:
                self.fail("activations has %d names for num_directions=%d "
                          "(%d per direction)" % (len(acts), nd, K))
        alpha = list(_attr(node, "activation_alpha", None) or [])
        beta = list(_attr(node, "activation_beta", None) or [])
        for nm, v in (("activation_alpha", alpha), ("activation_beta", beta)):
            if len(v) not in (0, K * nd):
                self.fail("%s has %d values (want 0 or %d)"
                          % (nm, len(v), K * nd))
        al = alpha + [None] * (K * nd - len(alpha))
        be = beta + [None] * (K * nd - len(beta))
        return acts, al, be

    def act(self, t, d, k):
        """Apply activation k of direction d to a compact buffer."""
        K = len(DEF_ACT[self.op])
        name = self.acts[d * K + k]
        a = self.alphas[d * K + k]
        b = self.betas[d * K + k]
        n = name.lower()
        if n == "sigmoid":
            return self.track(self.ew1(t, vc4infer.OP_SIGMOID))
        if n == "tanh":
            return self.track(self.unary(t, vc4infer.UOP_TANH))
        if n == "relu":
            return self.track(self.ew1(t, vc4infer.OP_RELU))
        if n == "affine":
            if a is None or b is None:
                self.fail("the Affine activation needs activation_alpha and "
                          "activation_beta")
            return self.track(self.affine(t, float(a), float(b)))
        if n == "hardsigmoid":
            return self.track(self.unary(
                t, vc4infer.UOP_HARDSIGMOID,
                0.2 if a is None else float(a),
                 0.5 if b is None else float(b)))
        if n == "leakyrelu":
            return self.track(self.unary(
                t, vc4infer.UOP_LEAKYRELU, 0.01 if a is None else float(a)))
        if n == "elu":
            return self.track(self.unary(
                t, vc4infer.UOP_ELU, 1.0 if a is None else float(a)))
        if n == "softsign":
            return self.track(self.unary(t, vc4infer.UOP_SOFTSIGN))
        if n == "softplus":
            return self.track(self.unary(t, vc4infer.UOP_SOFTPLUS))
        self.fail("activation %r has no QPU body in this dispatch (supported: "
                  "Sigmoid, Tanh, Relu, Affine, HardSigmoid, LeakyRelu, Elu, "
                  "Softsign, Softplus)" % name)

    # --------------------------------------------------------------- GEMM
    def gemm(self, a, K, wt, N):
        """out = a[B][K] @ wt[K][N] in padded space; returns [Mp][Np]."""
        Mp, Np, Kp = self.Mp, _al(N, 64), _al(K, 16)
        if Mp != self.B or Kp != K:
            Ap = self.salloc((Mp * Kp,), 0.0)
            self.lane_runs(a.buf, Ap.buf, self.B, K, K * 4, Kp * 4, 4)
            src = Ap
        else:
            src = a
        out = self.salloc((Mp, Np))
        self.r.eng.matmul(src.buf, wt.buf, out.buf, Mp, Np, Kp)
        return out

    def gate(self, pre, g, n=None, stride=None):
        """Column block g (n wide) of a padded gate buffer, compact [B][n]."""
        n = self.H if n is None else int(n)
        stride = pre.shape[1] if stride is None else int(stride)
        out = self.salloc((self.B, n))
        self.lane_runs(self.r._buf_at(pre, g * n * 4), out.buf, self.B, n,
                       stride * 4, n * 4, 4)
        return out

    def clip_pre(self, pre):
        if self.clip is None:
            return pre
        return self.track(self.unary(pre, vc4infer.UOP_CLIP,
                                     -self.clip, self.clip))

    # ---------------------------------------------------------------- step
    def step_once(self, d, xt, h, c):
        pan = self.panels[d]
        if self.op == "RNN":
            pre = self.track(self.add(self.gemm(xt, self.I, pan["W"], self.H),
                                      self.gemm(h, self.H, pan["R"], self.H)))
            pre = self.track(self.add(pre, pan["b"]))
            pre = self.clip_pre(pre)
            return self.act(self.gate(pre, 0), d, 0), c

        if self.op == "GRU":
            pre = self.track(self.add(
                self.gemm(xt, self.I, pan["Wzr"], 2 * self.H),
                self.gemm(h, self.H, pan["Rzr"], 2 * self.H)))
            pre = self.track(self.add(pre, pan["bzr"]))
            pre = self.clip_pre(pre)
            z = self.act(self.gate(pre, 0), d, 0)
            r = self.act(self.gate(pre, 1), d, 0)
            # the h gate mixes three different GEMM operands, so its
            # arithmetic stays in compact gate space (a padded panel and a
            # compact gate must never share one elementwise call: the kernel
            # runs over the *first* operand's element count)
            gx = self.gate(self.gemm(xt, self.I, pan["Wh"], self.H), 0)
            gx = self.track(self.add(gx, pan["bwh"]))
            if self.lbr:
                gh = self.gate(self.gemm(h, self.H, pan["Rh"], self.H), 0)
                gh = self.track(self.add(gh, pan["brh"]))
                hpre = self.track(self.add(gx, self.track(self.mul(r, gh))))
            else:
                rh = self.track(self.mul(r, h))
                gh = self.gate(self.gemm(rh, self.H, pan["Rh"], self.H), 0)
                gh = self.track(self.add(gh, pan["brh"]))
                hpre = self.track(self.add(gx, gh))
            hpre = self.clip_pre(hpre)
            hn = self.act(hpre, d, 1)
            zm = self.track(self.affine(z, -1.0, 1.0))
            out = self.track(self.add(self.track(self.mul(zm, hn)),
                                      self.track(self.mul(z, h))))
            return out, c

        # LSTM: gate order i, o, f, c (ONNX weight layout)
        N = self.G * self.H
        pre = self.track(self.add(self.gemm(xt, self.I, pan["W"], N),
                                  self.gemm(h, self.H, pan["R"], N)))
        pre = self.track(self.add(pre, pan["b"]))
        pre = self.clip_pre(pre)
        gi = self.gate(pre, 0)
        gp = self.gate(pre, 1)
        gf = self.gate(pre, 2)
        gc = self.gate(pre, 3)
        peep = pan.get("peep", False)
        if peep:
            gi = self.track(self.add(gi, self.track(self.mul(pan["Pi"], c))))
            gf = self.track(self.add(gf, self.track(self.mul(pan["Pf"], c))))
        gi = self.act(gi, d, 0)
        gf = self.act(gf, d, 0)
        gc = self.act(gc, d, 1)
        cn = self.track(self.add(self.track(self.mul(gf, c)),
                                 self.track(self.mul(gi, gc))))
        if peep:
            gp = self.track(self.add(gp, self.track(self.mul(pan["Po"], cn))))
        go = self.act(gp, d, 0)
        tcn = self.track(self.unary(cn, vc4infer.UOP_TANH))
        return self.track(self.mul(go, tcn)), cn

    # -------------------------------------------------------------- stores
    def store_h(self, Y, t, d, h):
        H, B = self.H, self.B
        if self.layout == 0:
            off, drow = (t * self.nd + d) * B * H * 4, H * 4
        else:
            off = (t * self.nd + d) * H * 4
            drow = self.T * self.nd * H * 4
        self.lane_runs(h.buf, self.r._buf_at(Y, off), B, H, H * 4, drow, 4)

    def store_state_dev(self, h, d, Y):
        """Y[d] = h for a compact device [B][H] tensor (one lane copy)."""
        B, H = self.B, self.H
        if self.layout == 0:
            self.lane_runs(h.buf, self.r._buf_at(Y, d * B * H * 4),
                           B, H, H * 4, H * 4, 4)
        else:
            self.lane_runs(h.buf, self.r._buf_at(Y, d * H * 4),
                           B, H, H * 4, self.nd * H * 4, 4)

    def store_state(self, arr, d, Y):
        """Y[d] = arr[d] for a host [nd][B][H] / [B][nd][H] array."""
        B, H = self.B, self.H
        tmp = self.r._tmp(_Tensor()(self.r.pool.alloc(B * H * 4 + 64), (B, H)))
        src = np.ascontiguousarray(arr[d]) if self.layout == 0 else \
            np.ascontiguousarray(np.transpose(arr, (1, 0, 2))[d])
        self.r.eng.upload(np.ascontiguousarray(src, np.float32), tmp.buf)
        self.store_state_dev(tmp, d, Y)

    def fill_state(self, Y, d, value):
        B, H, nd = self.B, self.H, self.nd
        n = B * H
        words = ((n + 63) // 64) * 64
        zero = self.salloc((words,), value)
        if self.layout == 0:
            self.lane_runs(zero.buf, self.r._buf_at(Y, d * B * H * 4),
                           B, H, H * 4, H * 4, 4)
        else:
            self.lane_runs(zero.buf, self.r._buf_at(Y, d * H * 4),
                           B, H, H * 4, nd * H * 4, 4)

    def outputs(self, Y, Y_h, Y_c):
        node = self.node
        outs = []
        for i, name in enumerate(node.output):
            if not name:
                continue
            if i == 0:
                outs.append(Y)
            elif i == 1:
                outs.append(Y_h)
            elif i == 2 and self.op == "LSTM":
                outs.append(Y_c)
        return outs


def _mk(cls):
    def run(r, node):
        return cls(r, node).run()
    return run


# operators the main table also knows and this family owns: none
OVERRIDES = ()

HANDLERS = {
    "RNN": _mk(Recurrent),
    "GRU": _mk(Recurrent),
    "LSTM": _mk(Recurrent),
}
