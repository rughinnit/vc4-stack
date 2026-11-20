#!/usr/bin/env python3
"""test_onnx_geom.py -- the geometry / detection block.

The generated matrix (``infer/tools/op_matrix.py``) runs one case per operator.
This file is the *positive* evidence for the forms it cannot reach, one ONNX
graph per case, run through the shipped runner (``mode="hybrid"``,
``host_policy="device"``) and compared with ``onnx.reference``:

  ConvTranspose   stride 1/2, 2x2/3x3/4x4, pads, output_padding, groups,
                  depthwise, dilation, auto_pad SAME_UPPER, output_shape,
                  no bias, 1x1 kernel
  Col2Im          block 2x2/3x3, pads, stride, dilation, multi-channel
  AffineGrid      align_corners 0/1, identity theta, batch 2, non-square
  GridSample      linear and nearest, zeros/border/reflection, align_corners
                  0/1, all four nearest_mode forms, out-of-range coordinates,
                  identity grid
  RoiAlign        avg/max, sampling_ratio 1/2/4, spatial_scale,
                  half_pixel / output_half_pixel, 3x3 output
  MaxRoiPool      pooled 2x2/3x3, spatial_scale, several ROIs
  NonMaxSuppression  center_point_box 0/1, ties, score_threshold, the
                  max_output cap, an empty result, several batches/classes
  Det             2x2/3x3/4x4, batched, singular
  Einsum          ij,jk->ik; bij,bjk->bik; ijk,ikl->jl; transposes; partial and
                  full reductions; outer product; the refusals
  DeformConv      3x3 and 1x1, offset_group 1/2, mask absent, group 2

Three operators have no usable oracle in ``onnx.reference`` 1.17 and their
cases carry a numpy expectation instead (``Case.ref_override``):
``Col2Im`` (its reference passes the 4-D image_shape where a 2-D pads array is
expected), ``MaxRoiPool`` (no implementation) and ``RoiAlign`` ``mode="max"``
(the reference maxes the four *weighted tap products* instead of the
interpolated values, which is not the operator).  Each override is the ONNX
formula, and the test says so beside the case.

    TMPDIR=$HOME/scratch/tmp VC4INFER_DAEMON=1 python3 tests/test_onnx_geom.py
    python3 tests/test_onnx_geom.py --only GridSample -v

Exit status is 0 only when every case behaved as declared.
"""
import argparse
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from test_onnx_ops import Case, build, compare, f32, i64, rnd  # noqa: E402
from test_onnx_gaps import no, ref                              # noqa: E402

from vc4infer import _binding as vc4infer                       # noqa: E402
from vc4infer._errors import HostNotAllowed, UnsupportedNode    # noqa: E402
from vc4infer._onnxrun import Runner                            # noqa: E402

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64


# =====================================================================
# numpy oracles for the three forms onnx.reference cannot answer
# =====================================================================
def col2im_ref(data, image_shape, block_shape, dil, pads, strides):
    """ONNX Col2Im, the 2-D accumulation written out (spec formula)."""
    N, C, H, W = image_shape
    kh, kw = block_shape
    dh, dw = dil
    ph, pw, ph2, pw2 = pads
    sh, sw = strides
    nb_h = (H + ph + ph2 - dh * (kh - 1) - 1) // sh + 1
    nb_w = (W + pw + pw2 - dw * (kw - 1) - 1) // sw + 1
    out = np.zeros((N, C, H, W), np.float64)
    for n in range(N):
        for c in range(C):
            for t in range(kh * kw):
                ky, kx = divmod(t, kw)
                for iy in range(nb_h):
                    y = iy * sh - ph + ky * dh
                    if not 0 <= y < H:
                        continue
                    for ix in range(nb_w):
                        x = ix * sw - pw + kx * dw
                        if 0 <= x < W:
                            out[n, c, y, x] += data[n, c * kh * kw + t,
                                                    iy * nb_w + ix]
    return out.astype(np.float32)


def convtranspose_ref(X, W, B, strides, pads, output_padding, dil, group):
    """ONNX ConvTranspose, the scatter written out (spec form).

    onnx.reference 1.17's grouped branch writes the whole group's output into a
    single channel (`final[image, g:g+1] = group_output[image]`), so a group
    with more than one output channel cannot be checked against it.
    """
    N, Cin, Hin, Win = X.shape
    Coutg = W.shape[1]
    kh, kw = W.shape[2], W.shape[3]
    Cout = Coutg * group
    Hout = strides[0] * (Hin - 1) + output_padding[0] + \
        dil[0] * (kh - 1) + 1 - pads[0] - pads[2]
    Wout = strides[1] * (Win - 1) + output_padding[1] + \
        dil[1] * (kw - 1) + 1 - pads[1] - pads[3]
    out = np.zeros((N, Cout, Hout, Wout), np.float64)
    if B is not None:
        out += np.asarray(B, np.float64).reshape(1, Cout, 1, 1)
    Cin_g = Cin // group
    for n in range(N):
        for g in range(group):
            for ci in range(Cin_g):
                for c in range(Coutg):
                    oc = g * Coutg + c
                    for ky in range(kh):
                        for kx in range(kw):
                            w = W[g * Cin_g + ci, c, ky, kx]
                            if w == 0.0:
                                continue
                            for iy in range(Hin):
                                y = iy * strides[0] - pads[0] + ky * dil[0]
                                if not 0 <= y < Hout:
                                    continue
                                for ix in range(Win):
                                    x = ix * strides[1] - pads[1] + \
                                        kx * dil[1]
                                    if 0 <= x < Wout:
                                        out[n, oc, y, x] += w * \
                                            X[n, g * Cin_g + ci, iy, ix]
    return out.astype(np.float32)


def maxroipool_ref(X, rois, pooled_shape, spatial_scale):
    """MaxRoiPool (torchvision geometry), the ONNX operator's definition."""
    N, C, H, W = X.shape
    ph, pw = pooled_shape
    out = np.zeros((rois.shape[0], C, ph, pw), np.float64)
    for r in range(rois.shape[0]):
        b = int(rois[r, 0])
        x1, y1, x2, y2 = (float(v) * spatial_scale for v in rois[r, 1:])
        # round half away from zero, as torchvision's MaxRoiPool does
        rx1, ry1, rx2, ry2 = (int(np.sign(v) * np.floor(abs(v) + 0.5))
                              for v in (x1, y1, x2, y2))
        rh = max(ry2 - ry1 + 1, 1)
        rw = max(rx2 - rx1 + 1, 1)
        bin_h = rh / ph
        bin_w = rw / pw
        for i in range(ph):
            hs = int(np.floor(i * bin_h + ry1))
            he = int(np.ceil((i + 1) * bin_h + ry1))
            hs, he = max(0, min(H, hs)), max(0, min(H, he))
            hh = max(he - hs, 1)
            for j in range(pw):
                ws = int(np.floor(j * bin_w + rx1))
                we = int(np.ceil((j + 1) * bin_w + rx1))
                ws, we = max(0, min(W, ws)), max(0, min(W, we))
                ww = max(we - ws, 1)
                out[r, :, i, j] = X[b, :, hs:hs + hh, ws:ws + ww] \
                    .reshape(C, -1).max(axis=1)
    return out.astype(np.float32)


def roialign_ref(X, rois, batch, oh, ow, sr, scale, mode, half_pixel):
    """ONNX RoiAlign, avg and max, with torch's max-of-interpolated rule."""
    N, C, H, W = X.shape
    off = 0.5 if half_pixel else 0.0
    out = np.zeros((rois.shape[0], C, oh, ow), np.float64)
    for r in range(rois.shape[0]):
        b = int(batch[r])
        x1, y1, x2, y2 = (float(v) * scale - off for v in rois[r])
        rw, rh = x2 - x1, y2 - y1
        if not half_pixel:
            rw, rh = max(rw, 1.0), max(rh, 1.0)
        bw, bh = rw / ow, rh / oh
        for i in range(oh):
            for j in range(ow):
                vals = np.zeros((C, sr * sr), np.float64)
                k = 0
                for sy in range(sr):
                    for sx in range(sr):
                        y = y1 + i * bh + (sy + 0.5) * bh / sr
                        x = x1 + j * bw + (sx + 0.5) * bw / sr
                        if y < -1.0 or y > H or x < -1.0 or x > W:
                            k += 1
                            continue
                        y = min(max(y, 0.0), H - 1.0)
                        x = min(max(x, 0.0), W - 1.0)
                        y0, x0 = int(np.floor(y)), int(np.floor(x))
                        y1i, x1i = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
                        ly, lx = y - y0, x - x0
                        vals[:, k] = ((1 - lx) * (1 - ly) * X[b, :, y0, x0] +
                                      lx * (1 - ly) * X[b, :, y0, x1i] +
                                      (1 - lx) * ly * X[b, :, y1i, x0] +
                                      lx * ly * X[b, :, y1i, x1i])
                        k += 1
                if mode == "avg":
                    out[r, :, i, j] = vals.sum(axis=1) / (sr * sr)
                else:
                    out[r, :, i, j] = vals.max(axis=1)
    return out.astype(np.float32)


# =====================================================================
# cases
# =====================================================================
def convtranspose_cases(rng):
    def sp(name, xs, ws, attrs, bias=True, note=""):
        feeds = {"X": rnd(rng, xs)}
        inits = {"W": rnd(rng, ws, -0.4, 0.4)}
        ins = ["X", "W"]
        if bias:
            inits["B"] = rnd(rng, (ws[1] * attrs.get("group", 1),), -0.3, 0.3)
            ins.append("B")
        return ref("ConvTranspose", name, feeds, inits, attrs, inputs=ins,
                   note=note)

    return [
        sp("s1-3x3-p1", (1, 4, 6, 6), (4, 2, 3, 3),
           {"strides": [1, 1], "pads": [1, 1, 1, 1]},
           note="stride 1: every tap writes the whole plane"),
        sp("s2-3x3-p1", (1, 4, 6, 6), (4, 2, 3, 3),
           {"strides": [2, 2], "pads": [1, 1, 1, 1]}),
        sp("s1-2x2-p0", (1, 3, 5, 5), (3, 2, 2, 2), {"strides": [1, 1]}),
        sp("s2-4x4-p1-opad1", (1, 2, 4, 4), (2, 3, 4, 4),
           {"strides": [2, 2], "pads": [1, 1, 1, 1],
            "output_padding": [1, 1]},
           note="output_padding adds an all-zero last row/column"),
        sp("s1-3x3-p0-nobias", (1, 2, 4, 4), (2, 3, 3, 3),
           {"strides": [1, 1]}, bias=False),
        _grouped(rng, "depthwise-s2", (1, 4, 5, 5), (4, 1, 3, 3),
                 {"strides": [2, 2], "pads": [1, 1, 1, 1], "group": 4}),
        _grouped(rng, "group2-s1", (1, 4, 5, 5), (4, 2, 3, 3),
                 {"strides": [1, 1], "pads": [1, 1, 1, 1], "group": 2}),
        sp("same_upper-s2", (1, 3, 5, 5), (3, 2, 3, 3),
           {"strides": [2, 2], "auto_pad": "SAME_UPPER"}),
        sp("output_shape-attr", (1, 2, 3, 3), (2, 2, 3, 3),
           {"strides": [2, 2], "output_shape": [7, 7]}),
        sp("s2-1x1", (1, 4, 4, 4), (4, 2, 1, 1), {"strides": [2, 2]}),
        sp("dil2-3x3", (1, 2, 4, 4), (2, 2, 3, 3),
           {"strides": [1, 1], "dilations": [2, 2], "pads": [2, 2, 2, 2]}),
        ref("ConvTranspose", "s1-nonsquare-3x2", {"X": rnd(rng, (1, 3, 4, 4))},
            {"W": rnd(rng, (3, 2, 3, 2), -0.4, 0.4)},
            {"strides": [1, 1], "pads": [1, 1, 1, 1]},
            inputs=["X", "W"], note="kernel 3x2"),
    ]


def _col2im_L(image, block, attrs):
    H, W = image[2], image[3]
    kh, kw = block
    dh, dw = attrs.get("dilations", [1, 1])
    ph, pw, ph2, pw2 = attrs.get("pads", [0, 0, 0, 0])
    sh, sw = attrs.get("strides", [1, 1])
    nb_h = (H + ph + ph2 - dh * (kh - 1) - 1) // sh + 1
    nb_w = (W + pw + pw2 - dw * (kw - 1) - 1) // sw + 1
    return nb_h * nb_w


def _grouped(rng, name, xs, ws, attrs):
    """A grouped ConvTranspose: the reference's grouped branch is broken, so
    the expectation is the spec's scatter (convtranspose_ref)."""
    feeds = {"X": rnd(rng, xs)}
    inits = {"W": rnd(rng, ws, -0.4, 0.4),
             "B": rnd(rng, (ws[1] * attrs.get("group", 1),), -0.3, 0.3)}
    c = ref("ConvTranspose", name, feeds, inits, attrs,
            inputs=["X", "W", "B"],
            note="grouped: onnx.reference writes the group into one channel")
    X = feeds["X"]
    c.ref_override = lambda x, *rest, _w=inits["W"], _b=inits["B"], _a=attrs: \
        convtranspose_ref(x, _w, _b, _a["strides"],
                          _a.get("pads", [0, 0, 0, 0]),
                          _a.get("output_padding", [0, 0]),
                          _a.get("dilations", [1, 1]), _a["group"])
    return c


def col2im_cases(rng):
    def sp(name, image, block, attrs, note=""):
        N, C = image[0], image[1]
        chan = C * block[0] * block[1]
        data_shape = (N, chan, _col2im_L(image, block, attrs))
        feeds = {"input": rnd(rng, data_shape)}
        inits = {"image_shape": np.asarray(image, np.int64),
                 "block_shape": np.asarray(block, np.int64)}
        c = ref("Col2Im", name, feeds, inits, attrs,
                inputs=["input", "image_shape", "block_shape"], note=note)
        img = tuple(int(v) for v in image)
        blk = tuple(int(v) for v in block)
        dil = attrs.get("dilations", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        st = attrs.get("strides", [1, 1])
        c.ref_override = lambda d, **_kw: col2im_ref(
            d.reshape(data_shape), img, blk, dil, pads, st)
        return c

    return [
        sp("block2-s1-p0", [1, 2, 4, 4], [2, 2], {}),
        sp("block2-pad1", [1, 1, 5, 5], [2, 2], {"pads": [1, 1, 1, 1]},
           note="the padded border stays zero"),
        sp("block2-s2-p1", [1, 3, 7, 7], [2, 2],
           {"strides": [2, 2], "pads": [1, 1, 1, 1]},
           note="stride 2: k^2 copies into disjoint phases"),
        sp("block3-dil2", [1, 1, 8, 8], [3, 3],
           {"dilations": [2, 2], "pads": [2, 2, 2, 2]}),
        sp("block2-nonsquare", [1, 3, 4, 5], [2, 3], {}),
    ]


def affinegrid_cases(rng):
    return [
        ref("AffineGrid", "align0-3x4", {"theta": rnd(rng, (1, 2, 3), -0.5, 0.5)},
            {"size": i64(1, 3, 3, 4)}, {"align_corners": 0}),
        ref("AffineGrid", "align1-3x4", {"theta": rnd(rng, (1, 2, 3), -0.5, 0.5)},
            {"size": i64(1, 3, 3, 4)}, {"align_corners": 1}),
        ref("AffineGrid", "identity", {"theta": f32(1, 0, 0, 0, 1, 0).reshape(1, 2, 3)},
            {"size": i64(1, 2, 5, 5)}, {"align_corners": 1},
            note="the identity theta reproduces the normalised grid"),
        ref("AffineGrid", "batch2", {"theta": rnd(rng, (2, 2, 3), -0.5, 0.5)},
            {"size": i64(2, 1, 2, 7)}, {"align_corners": 0},
            note="two batches, 1 channel, a 2x7 grid"),
    ]


def gridsample_cases(rng):
    def sp(name, xs, gs, attrs, note=""):
        return ref("GridSample", name,
                   {"X": rnd(rng, xs), "grid": rnd(rng, gs, -0.9, 0.9)},
                   attrs=attrs, inputs=["X", "grid"], note=note)

    out = [
        sp("linear-zeros-a0", (1, 3, 6, 6), (1, 5, 5, 2),
           {"mode": "linear", "padding_mode": "zeros", "align_corners": 0}),
        sp("linear-border-a0", (1, 2, 5, 5), (1, 4, 4, 2),
           {"mode": "linear", "padding_mode": "border", "align_corners": 0}),
        sp("linear-reflection-a0", (1, 1, 5, 5), (1, 4, 4, 2),
           {"mode": "linear", "padding_mode": "reflection",
            "align_corners": 0}),
        sp("linear-reflection-a1", (1, 1, 6, 6), (1, 4, 4, 2),
           {"mode": "linear", "padding_mode": "reflection",
            "align_corners": 1}),
        sp("linear-border-a1", (1, 2, 4, 4), (1, 3, 3, 2),
           {"mode": "linear", "padding_mode": "border", "align_corners": 1}),
        sp("nearest-zeros-a0", (1, 2, 5, 5), (1, 4, 4, 2),
           {"mode": "nearest", "padding_mode": "zeros", "align_corners": 0}),
        sp("nearest-border-a1", (1, 1, 5, 5), (1, 3, 3, 2),
           {"mode": "nearest", "padding_mode": "border",
            "align_corners": 1},
           note="nearest rounds to even (onnx.reference / torch)"),
        sp("nearest-reflection-a0", (1, 1, 5, 5), (1, 3, 3, 2),
           {"mode": "nearest", "padding_mode": "reflection",
            "align_corners": 0}),
        sp("nearest-multichannel", (1, 3, 4, 4), (1, 2, 2, 2),
           {"mode": "nearest", "align_corners": 0}),
    ]
    # an identity grid must reproduce the input exactly
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3)),
                 axis=-1).astype(np.float32)[None]
    out.append(ref("GridSample", "identity-grid",
                   {"X": rnd(rng, (1, 2, 3, 4)), "grid": g},
                   {"mode": "linear", "align_corners": 1},
                   inputs=["X", "grid"],
                   note="align_corners=1 identity sampling is exact"))
    # coordinates far outside the input: zeros padding must mask them
    far = np.asarray([[[-3.0, -3.0], [3.0, 3.0], [0.0, 0.0]]],
                     np.float32).reshape(1, 1, 3, 2)
    out.append(ref("GridSample", "far-outside-zeros",
                   {"X": rnd(rng, (1, 1, 3, 3)), "grid": far},
                   {"mode": "linear", "padding_mode": "zeros"},
                   inputs=["X", "grid"],
                   note="only the in-range pixel contributes"))
    out.append(ref("GridSample", "far-outside-border",
                   {"X": rnd(rng, (1, 1, 3, 3)), "grid": far},
                   {"mode": "linear", "padding_mode": "border"},
                   inputs=["X", "grid"]))
    out.append(no("GridSample", "cubic-refused",
                  {"X": rnd(rng, (1, 1, 3, 3)), "grid": rnd(rng, (1, 2, 2, 2))},
                  attrs={"mode": "cubic"}, inputs=["X", "grid"],
                  note="cubic needs a 4x4 Keys polynomial per axis",
                  why="cubic"))
    return out


def roialign_cases(rng):
    def sp(name, xs, rois, batch, attrs, note=""):
        return ref("RoiAlign", name,
                   {"X": rnd(rng, xs), "rois": np.asarray(rois, np.float32)},
                   {"batch_indices": np.asarray(batch, np.int64)}, attrs,
                   inputs=["X", "rois", "batch_indices"], note=note)

    rois = [[0.0, 0.0, 6.0, 6.0], [1.5, 1.0, 5.0, 5.5]]
    out = [
        sp("avg-sr2", (1, 3, 8, 8), rois, [0, 0],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 2}),
        sp("avg-sr1", (1, 2, 7, 7), rois, [0, 0],
           {"output_height": 3, "output_width": 3, "sampling_ratio": 1}),
        sp("avg-sr4", (1, 1, 6, 6), [[0.0, 0.0, 5.0, 5.0]], [0],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 4}),
        sp("avg-scale05", (1, 2, 8, 8), [[0.0, 0.0, 12.0, 12.0],
                                        [2.0, 2.0, 9.0, 9.0]], [0, 0],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 2,
            "spatial_scale": 0.5},
           note="spatial_scale maps the ROI into the feature map"),
        sp("avg-output_half_pixel", (1, 2, 6, 6), rois, [0, 0],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 2,
            "coordinate_transformation_mode": "output_half_pixel"}),
        sp("avg-batch2", (2, 2, 6, 6),
           [[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 5.0, 5.0]], [0, 1],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 2},
           note="the second ROI samples the second batch"),
        sp("avg-clipped-roi", (1, 2, 5, 5),
           [[-2.0, -2.0, 7.0, 7.0]], [0],
           {"output_height": 2, "output_width": 2, "sampling_ratio": 2},
           note="samples outside the image are clamped"),
    ]
    # mode=max: ONNX/reference maxes the weighted tap products; torch (and the
    # operator) maxes the interpolated values, which is what the engine does.
    xs, rr = (1, 2, 6, 6), np.asarray([[0.0, 0.0, 5.5, 5.5]], np.float32)
    X = rnd(np.random.default_rng(7), xs)
    c = ref("RoiAlign", "max-sr2", {"X": X, "rois": rr},
            {"batch_indices": i64(0)},
            {"output_height": 2, "output_width": 2, "sampling_ratio": 2,
             "mode": "max"},
            inputs=["X", "rois", "batch_indices"],
            note="mode=max is max of the interpolated samples (torch rule)")
    c.ref_override = lambda x, r: roialign_ref(
        x, rr, np.asarray([0]), 2, 2, 2, 1.0, "max", True)
    out.append(c)
    out.append(no("RoiAlign", "adaptive-refused",
                  {"X": rnd(rng, (1, 1, 6, 6)), "rois": rr},
                  {"batch_indices": i64(0)},
                  {"output_height": 2, "output_width": 2, "sampling_ratio": 0},
                  inputs=["X", "rois", "batch_indices"],
                  note="sampling_ratio=0 is adaptive (a per-ROI sample count)",
                  why="adaptive"))
    return out


def maxroipool_cases(rng):
    def sp(name, xs, rois, attrs, note=""):
        rois = np.asarray(rois, np.float32)
        X = rnd(rng, xs)
        c = ref("MaxRoiPool", name, {"X": X, "rois": rois}, attrs=attrs,
                inputs=["X", "rois"], note=note)
        c.ref_override = lambda x, r, _rois=rois, _a=dict(attrs): \
            maxroipool_ref(x, _rois, list(_a["pooled_shape"]),
                           float(_a.get("spatial_scale", 1.0)))
        return c

    return [
        sp("2x2", (1, 3, 8, 8), [[0, 0, 0, 3, 3], [0, 2, 2, 5, 5]],
           {"pooled_shape": [2, 2], "spatial_scale": 1.0}),
        sp("3x3", (1, 2, 9, 9), [[0, 1, 1, 7, 7]],
           {"pooled_shape": [3, 3], "spatial_scale": 1.0},
           note="bin edges are floor/ceil of the scaled ROI"),
        sp("scale05", (1, 2, 8, 8), [[0, 0, 0, 14, 14]],
           {"pooled_shape": [2, 2], "spatial_scale": 0.5}),
        sp("batch2", (2, 2, 6, 6), [[0, 0, 0, 5, 5], [1, 1, 1, 4, 4]],
           {"pooled_shape": [2, 2]}, note="one ROI per batch"),
    ]


def _absent(rng, boxes, scores):
    """max_output_boxes_per_class omitted: the ONNX default is 0 = no output."""
    c = ref("NonMaxSuppression", "max-out-absent-empty",
            {"boxes": np.asarray(boxes, np.float32),
             "scores": np.asarray(scores, np.float32)},
            {"max_output_boxes_per_class": np.zeros(0, np.int64),
             "iou_threshold": f32(0.5), "score_threshold": f32(0.0)},
            inputs=["boxes", "scores", "max_output_boxes_per_class",
                    "iou_threshold", "score_threshold"],
            out_types=[I64],
            note="an empty max_output_boxes_per_class is the ONNX default 0")
    # onnx.reference 1.17 raises on an empty input tensor here ("Issues with
    # types"), so the expectation is the operator's default: no output
    c.ref_override = lambda *a: np.zeros((0, 3), np.int64)
    return c


def nms_cases(rng):
    def sp(name, boxes, scores, attrs=None, extra=None, note=""):
        feeds = {"boxes": np.asarray(boxes, np.float32),
                 "scores": np.asarray(scores, np.float32)}
        inits = {"max_output_boxes_per_class": i64(2),
                 "iou_threshold": f32(0.5), "score_threshold": f32(0.0)}
        for k, v in (extra or {}).items():
            if v is None:
                inits.pop(k, None)
            else:
                inits[k] = v
        return ref("NonMaxSuppression", name, feeds, inits, attrs or {},
                   inputs=["boxes", "scores", "max_output_boxes_per_class",
                           "iou_threshold", "score_threshold"],
                   out_types=[I64], note=note)

    # overlapping boxes, one batch, one class
    boxes = [[[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 1.05, 1.05],
              [0.5, 0.5, 1.5, 1.5], [3.0, 3.0, 4.0, 4.0]]]
    scores = [[[0.9, 0.85, 0.8, 0.3]]]
    tied = [[[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 1.05, 1.05],
             [3.0, 3.0, 4.0, 4.0]]]
    tied_sc = [[[0.6, 0.6, 0.6]]]
    out = [
        sp("overlap-m2", boxes, scores, note="the second box is suppressed"),
        sp("center-point-box", [[[0.5, 0.5, 1.0, 1.0], [0.55, 0.55, 1.0, 1.0],
                                [3.5, 3.5, 1.0, 1.0]]],
           [[[0.9, 0.8, 0.2]]], {"center_point_box": 1},
           note="center_point_box=1: (x, y, w, h)"),
        sp("ties", tied, tied_sc,
           note="equal scores keep the lower index first"),
        sp("score-threshold", boxes, scores, extra={"score_threshold": f32(0.5)},
           note="only the three high scores are candidates"),
        sp("max-out1", boxes, scores,
           extra={"max_output_boxes_per_class": i64(1)}),
        sp("max-out0-empty", boxes, scores,
           extra={"max_output_boxes_per_class": i64(0)}),
        _absent(rng, boxes, scores),
        sp("iou-threshold-1", boxes, scores,
           extra={"iou_threshold": f32(1.0)},
           note="IoU > 1 is never true, so nothing is suppressed"),
        sp("batch2-class2", np.asarray(boxes * 2, np.float32),
           np.asarray(scores * 2, np.float32),
           note="two batches and two classes: four independent selections"),
        sp("no-overlap", [[[0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 2.0, 2.0],
                           [2.0, 0.0, 3.0, 0.5]]],
           [[[0.9, 0.8, 0.7]]], extra={"max_output_boxes_per_class": i64(3)}),
    ]
    return out


def det_cases(rng):
    out = [
        ref("Det", "2x2-batch3", {"X": rnd(rng, (3, 2, 2))}),
        ref("Det", "3x3-batch2", {"X": rnd(rng, (2, 3, 3))}),
        ref("Det", "4x4", {"X": rnd(rng, (1, 4, 4))}),
        ref("Det", "singular-3x3", {"X": np.asarray(
            [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [7.0, 8.0, 10.0]]],
            np.float32)}, note="dependent rows: det == 0"),
        ref("Det", "identity-4x4", {"X": np.eye(4, dtype=np.float32)[None]},
            note="det(I) == 1"),
        no("Det", "5x5-refused", {"X": rnd(rng, (1, 5, 5))},
           why="4", note="M > 4 needs 5! = 120 permutation terms"),
        _nonsq(rng),
    ]
    return out


def _nonsq(rng):
    c = no("Det", "nonsquare-refused", {"X": rnd(rng, (1, 3, 4))}, why="square")
    # onnx.reference raises LinAlgError before the engine is asked, so the
    # expectation is supplied: the engine must refuse the shape by name
    c.ref_override = lambda x: np.zeros((1,), np.float32)
    return c


def einsum_cases(rng):
    out = [
        ref("Einsum", "ij-jk-ik", {"A": rnd(rng, (2, 3)),
                                   "B": rnd(rng, (3, 4))},
            attrs={"equation": "ij,jk->ik"}, inputs=["A", "B"]),
        ref("Einsum", "bij-bjk-bik", {"A": rnd(rng, (2, 3, 4)),
                                      "B": rnd(rng, (2, 4, 5))},
            attrs={"equation": "bij,bjk->bik"}, inputs=["A", "B"],
            note="one batched GEMM per batch element"),
        ref("Einsum", "ijk-ikl-jl", {"A": rnd(rng, (2, 3, 4)),
                                     "B": rnd(rng, (2, 4, 5))},
            attrs={"equation": "ijk,ikl->jl"}, inputs=["A", "B"],
            note="two contracted labels, one free label each"),
        ref("Einsum", "ij-jk-ki", {"A": rnd(rng, (3, 4)),
                                   "B": rnd(rng, (4, 3))},
            attrs={"equation": "ij,jk->ki"}, inputs=["A", "B"],
            note="the output needs a transpose after the GEMM"),
        ref("Einsum", "ij-ji", {"A": rnd(rng, (3, 4))},
            attrs={"equation": "ij->ji"}, inputs=["A"]),
        ref("Einsum", "ij-i", {"A": rnd(rng, (3, 4))},
            attrs={"equation": "ij->i"}, inputs=["A"], note="row sums"),
        ref("Einsum", "ij-scalar", {"A": rnd(rng, (3, 4))},
            attrs={"equation": "ij->"}, inputs=["A"], note="full reduction"),
        ref("Einsum", "ij-ij-dot", {"A": rnd(rng, (3, 4)),
                                    "B": rnd(rng, (3, 4))},
            attrs={"equation": "ij,ij->"}, inputs=["A", "B"],
            note="the Frobenius inner product"),
        ref("Einsum", "i-j-ij", {"A": rnd(rng, (3,)), "B": rnd(rng, (4,))},
            attrs={"equation": "i,j->ij"}, inputs=["A", "B"],
            note="outer product: a K=1 contraction"),
        ref("Einsum", "implicit", {"A": rnd(rng, (2, 3)),
                                   "B": rnd(rng, (3, 4))},
            attrs={"equation": "ij,jk"}, inputs=["A", "B"],
            note="no '->': the labels seen once, sorted"),
        no("Einsum", "diagonal-refused", {"A": rnd(rng, (4, 4))},
           attrs={"equation": "ii->i"}, inputs=["A"],
           note="a repeated label inside one operand is a diagonal",
           why="diagonal"),
        no("Einsum", "ellipsis-refused", {"A": rnd(rng, (2, 3)),
                                          "B": rnd(rng, (3, 4))},
           attrs={"equation": "...ij,jk->...ik"}, inputs=["A", "B"],
           why="ellipsis"),
        no("Einsum", "three-operands-refused",
           {"A": rnd(rng, (2, 3)), "B": rnd(rng, (3, 2)),
            "C": rnd(rng, (2, 2))},
           attrs={"equation": "ij,jk,ki->"}, inputs=["A", "B", "C"],
           why="operands"),
    ]
    return out


def deformconv_cases(rng):
    def sp(name, xs, ws, offs, attrs, mask=True, bias=True, note=""):
        feeds = {"X": rnd(rng, xs), "offset": rnd(rng, offs, -0.3, 0.3)}
        inits = {"W": rnd(rng, ws, -0.3, 0.3)}
        ins = ["X", "W", "offset"]
        if bias:
            inits["B"] = rnd(rng, (ws[0],), -0.3, 0.3)
            ins.append("B")
        if mask:
            feeds["mask"] = rnd(rng, (offs[0], offs[1] // 2,
                                      offs[2], offs[3]), 0.3, 0.7)
            ins.append("mask")
        elif bias:
            ins.append("")          # an omitted optional input
        return ref("DeformConv", name, feeds, inits, attrs, inputs=ins,
                   note=note)

    return [
        sp("3x3-og1", (1, 3, 5, 5), (2, 3, 3, 3), (1, 18, 5, 5),
           {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
            "strides": [1, 1], "dilations": [1, 1], "group": 1,
            "offset_group": 1}),
        sp("3x3-og2", (1, 4, 4, 4), (2, 4, 3, 3), (1, 36, 4, 4),
           {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
            "strides": [1, 1], "group": 1, "offset_group": 2},
           note="two offset groups: channels share their group's offsets"),
        sp("3x3-nomask", (1, 2, 5, 5), (2, 2, 3, 3), (1, 18, 5, 5),
           {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
            "strides": [1, 1]}, mask=False),
        sp("group2", (1, 4, 5, 5), (4, 2, 3, 3), (1, 18, 5, 5),
           {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
            "strides": [1, 1], "group": 2, "offset_group": 1},
           note="grouped contraction: one GEMM per group"),
        sp("1x1-s2", (1, 2, 6, 6), (3, 2, 1, 1), (1, 2, 3, 3),
           {"kernel_shape": [1, 1], "strides": [2, 2]},
           note="1x1 kernel, stride 2: one sample per output pixel"),
    ]


ALL = {}


def _register():
    rng = np.random.default_rng(20261122)
    for fn in (convtranspose_cases, col2im_cases, affinegrid_cases,
               gridsample_cases, roialign_cases, maxroipool_cases, nms_cases,
               det_cases, einsum_cases, deformconv_cases):
        for c in fn(rng):
            ALL.setdefault(c.op, []).append(c)


_register()


# =====================================================================
# runner
# =====================================================================
def run_case(eng, case, workdir, verbose):
    model = build(case)
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        return "invalid-model", "%s: %s" % (type(e).__name__, e)
    path = os.path.join(workdir, "g_%s_%s.onnx"
                        % (case.op, case.name.replace("/", "_")))
    onnx.save(model, path)
    feeds = {k: v for k, v in case.feeds.items()
             if np.asarray(v).dtype == np.float32}
    try:
        if case.ref_override is not None:
            refv = [case.ref_override(*[feeds[k] for k in feeds])]
        else:
            refv = ReferenceEvaluator(model).run(None, dict(feeds))
    except Exception as e:
        return "oracle-error", "%s: %s" % (type(e).__name__, e)
    try:
        got = Runner(path, eng, mode="hybrid", host_policy="device") \
            .run({k: np.ascontiguousarray(v, np.float32)
                  for k, v in feeds.items()})
    except (UnsupportedNode, HostNotAllowed) as e:
        if case.expect == "refused":
            if case.why and case.why not in str(e):
                return "refused", "refused, but not for %r: %s" % (case.why, e)
            return "refused", ""
        return "refused", str(e)
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return "error", "%s: %s" % (type(e).__name__, e)
    if case.expect == "refused":
        return "ran", "the case was expected to be refused (%s) but ran" \
            % case.why
    for i, (w, g) in enumerate(zip(refv, got)):
        msg = compare(w, g)
        if msg:
            return "mismatch", "output %d: %s" % (i, msg)
    return "vc4", ""


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    only = [s for s in args.only.replace(",", " ").split() if s]
    ops = [o for o in sorted(ALL) if not only or o in only]
    if only:
        missing = [o for o in only if o not in ALL]
        if missing:
            print("no cases for: %s" % ", ".join(missing))
            return 2
    workdir = args.work_dir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "vc4geom-models")
    os.makedirs(workdir, exist_ok=True)
    cache = os.environ.get("VC4INFER_CACHE") or None
    eng = vc4infer.Engine(cache_dir=cache)
    # One pool per engine, and libvc4infer has no free(): a later Runner whose
    # model needs more than the current pool makes Runner allocate a second
    # Pool and the first one's segments leak.  Every case here is small, so one
    # pool sized once covers them all.
    from vc4infer._onnxrun import Pool                       # noqa: E402
    eng._vc4_pool = Pool(eng, int(os.environ.get("VC4GEOM_POOL_MB", "12"))
                         << 20)
    bad, total = [], 0
    try:
        for op in ops:
            for case in ALL[op]:
                total += 1
                status, msg = run_case(eng, case, workdir, args.verbose)
                ok = status == case.expect
                print("%-4s %-16s %-28s %-10s%s"
                      % ("ok" if ok else "FAIL", op, case.name, status,
                         "" if ok else "  " + msg[:160]))
                if not ok:
                    bad.append((op, case.name, status, msg))
    finally:
        eng.close()
    print("\n%d cases, %d failed" % (total, len(bad)))
    if bad:
        from collections import Counter
        print("by status:", dict(Counter(b[2] for b in bad)))
        for op, name, status, msg in bad:
            print("  %-16s %-28s %-12s %s" % (op, name, status, msg[:140]))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
