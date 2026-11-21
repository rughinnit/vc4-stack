"""CPU-side checks for the VC4 Relax BYOC backend.

Nothing here dispatches to the QPU: the fused subgraph is executed through the
runtime module's ``host`` and ``record`` modes only, and the engine binary is
inspected with ``readelf``/``nm``.  The silicon pass is
``tvm_byoc/tests/silicon_check.py``.

Run:

    TMPDIR=$HOME/scratch/tmp /path/to/tvm/bin/python tvm_byoc/tests/test_byoc.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import tvm  # noqa: E402
from tvm import relax  # noqa: E402

from tvm_byoc.vc4 import build_for_vc4, codegen, partition_for_vc4, runtime  # noqa: E402

M, K, N = 4, 16, 8

WORK = pathlib.Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / f"vc4-byoc-{os.getpid()}"
WORK.mkdir(parents=True, exist_ok=True)
# The runtime caches its log handle on first use, so the path is fixed for the
# lifetime of the process and truncated between checks.
LOG = WORK / "calls.log"
os.environ["VC4_TVM_LOG"] = str(LOG)
os.environ.pop("VC4_TVM_MODE", None)


def bias_values(n=N):
    return (np.arange(n, dtype="float32") * 0.5).astype("float32")


def func_attr(func, key):
    attrs = func.attrs
    if attrs is None:
        return ""
    try:
        return str(getattr(attrs, key))
    except AttributeError:
        pass
    try:
        return str(attrs[key])
    except Exception:  # noqa: BLE001
        return ""


def codegen_function(mod):
    for _, func in mod.functions.items():
        if func_attr(func, "Codegen") == "vc4":
            return func
    return None


def make_module(bias_as_param=False, bias_shape="row", bias=None):
    """A small Relax module: matmul(x, w) + bias."""
    if bias is None:
        bias = bias_values().reshape({"row": (N,), "scalar": (), "matrix": (M, N)}[bias_shape])
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType((M, K), "float32"))
    w = relax.Var("w", relax.TensorType((K, N), "float32"))
    params = [x, w]
    if bias_as_param:
        b = relax.Var("b", relax.TensorType((N,), "float32"))
        params.append(b)
        bias_expr = b
    else:
        bias_expr = relax.const(bias, "float32")
    with bb.function("main", params):
        with bb.dataflow():
            out = bb.emit(relax.op.add(relax.op.matmul(x, w), bias_expr))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.get()


def truncate_log():
    LOG.write_text("", encoding="utf-8")


def log_events():
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def deterministic_inputs(seed=0):
    x = np.random.RandomState(seed).rand(M, K).astype("float32")
    w = np.random.RandomState(seed + 1).rand(K, N).astype("float32")
    return x, w


def run_layer(mod, args, mode):
    offloaded = build_for_vc4(mod, mode=mode)
    ex = tvm.compile(offloaded, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](*[tvm.runtime.tensor(a) for a in args]).numpy(), offloaded


def test_pattern_table_matches():
    mod = make_module()
    partitioned = partition_for_vc4(mod)
    text = partitioned.script()
    assert 'R.func_attr({"Codegen": "vc4"})' in text, text
    assert 'R.func_attr({"Composite": "vc4.matmul_bias_add"})' in text, text
    # One grouped function, and the entry function calls it instead of the
    # original matmul + add pair.
    assert text.count('"Codegen": "vc4"') == 1, text
    assert "R.matmul(" not in text.split("def main(")[1], text


def test_pattern_rejects_parameter_bias():
    mod = make_module(bias_as_param=True)
    text = partition_for_vc4(mod).script()
    assert "Codegen" not in text, text
    assert "Composite" not in text, text


def test_pattern_rejects_unsupported_bias_shape():
    # (M, 1) cannot broadcast over N and is not an engine-supported form.
    mod = make_module(bias=np.zeros((M, 1), dtype="float32"), bias_shape="matrix")
    text = partition_for_vc4(mod).script()
    assert "Codegen" not in text, text


def test_descriptor_text():
    mod = make_module()
    partitioned = partition_for_vc4(mod)
    func = codegen_function(partitioned)
    assert func is not None

    dump_dir = WORK / "dump"
    os.environ["VC4_TVM_DUMP_DIR"] = str(dump_dir)
    try:
        descriptor = codegen.lower_function(func, {"mode": "record"})
        # The dump is the same descriptor that goes into the module.
        codegen._dump(descriptor)
    finally:
        os.environ.pop("VC4_TVM_DUMP_DIR", None)

    assert descriptor["symbol"] == codegen._function_symbol(func), descriptor["symbol"]
    assert descriptor["symbol"].startswith("fused_"), descriptor["symbol"]
    assert descriptor["mode"] == "record"
    assert [i["shape"] for i in descriptor["inputs"]] == [[M, K], [K, N]]
    assert [i["role"] for i in descriptor["inputs"]] == ["x", "w"]
    assert descriptor["outputs"][0]["shape"] == [M, N]
    node = descriptor["nodes"][0]
    assert node["op"] == "vc4.fused_matmul_add"
    assert (node["attrs"]["M"], node["attrs"]["N"], node["attrs"]["K"]) == (M, N, K)
    assert node["attrs"]["bias_mode"] == 2
    assert node["attrs"]["bias_data"] == [float(v) for v in bias_values()]

    path = dump_dir / f"{descriptor['symbol']}.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == descriptor


def test_host_double_matches_relax_cpu():
    mod = make_module()
    x, w = deterministic_inputs()
    truncate_log()
    got, offloaded = run_layer(mod, (x, w), "host")

    # The offloaded graph calls the module symbol through call_dps_packed.
    symbol = codegen._function_symbol(codegen_function(partition_for_vc4(mod)))
    assert f'R.call_dps_packed("{symbol}"' in offloaded.script(), offloaded.script()

    # CPU reference: the same module, unpartitioned, on llvm.
    ex = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    cpu = vm["main"](tvm.runtime.tensor(x), tvm.runtime.tensor(w)).numpy()

    numpy_ref = x @ w + bias_values()
    assert np.max(np.abs(got - numpy_ref)) < 1e-5, np.max(np.abs(got - numpy_ref))
    assert np.max(np.abs(got - cpu)) < 1e-5, np.max(np.abs(got - cpu))

    calls = [e for e in log_events() if e["event"] == "call"]
    assert len(calls) == 1, calls
    assert calls[0]["mode"] == "host"
    assert (calls[0]["M"], calls[0]["N"], calls[0]["K"]) == (M, N, K)
    assert calls[0]["bias_elems"] == N
    assert calls[0]["in_bytes"] == [M * K * 4, K * N * 4]
    assert calls[0]["out_bytes"] == M * N * 4
    assert not [e for e in log_events() if e["event"] == "engine_init"]


def test_record_mode_is_inert():
    mod = make_module()
    x, w = deterministic_inputs(seed=7)
    truncate_log()
    got, _ = run_layer(mod, (x, w), "record")
    assert np.isnan(got).all(), "record mode must not produce numbers"
    calls = [e for e in log_events() if e["event"] == "call"]
    assert len(calls) == 1 and calls[0]["mode"] == "record", calls
    assert not [e for e in log_events() if e["event"] == "engine_init"]
    assert not [e for e in log_events() if e["event"] == "engine"]


def test_bias_forms():
    """Scalar, row and full-matrix biases all lower and match the CPU."""
    x, w = deterministic_inputs(seed=3)
    cases = {
        "scalar": (np.array(0.25, dtype="float32"), 1),
        "row": (bias_values(), 2),
        "matrix": (np.arange(M * N, dtype="float32").reshape(M, N) / 8.0, 3),
    }
    for name, (bias, expected_mode) in cases.items():
        mod = make_module(bias=bias)
        func = codegen_function(partition_for_vc4(mod))
        assert func is not None, name
        descriptor = codegen.lower_function(func, {"mode": "host"})
        assert descriptor["nodes"][0]["attrs"]["bias_mode"] == expected_mode, name
        truncate_log()
        got, _ = run_layer(mod, (x, w), "host")
        numpy_ref = x @ w + np.asarray(bias, dtype="float32")
        assert np.max(np.abs(got - numpy_ref)) < 1e-5, (name, np.max(np.abs(got - numpy_ref)))


def test_two_fused_layers():
    """Two apart fused subgraphs produce two modules and two recorded calls."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType((M, K), "float32"))
    w1 = relax.Var("w1", relax.TensorType((K, K), "float32"))
    w2 = relax.Var("w2", relax.TensorType((K, N), "float32"))
    b1 = bias_values(K)
    b2 = bias_values(N)
    with bb.function("main", [x, w1, w2]):
        with bb.dataflow():
            h = bb.emit(relax.op.add(relax.op.matmul(x, w1), relax.const(b1, "float32")))
            out = bb.emit(relax.op.add(relax.op.matmul(h, w2), relax.const(b2, "float32")))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    mod = bb.get()
    assert partition_for_vc4(mod).script().count('"Codegen": "vc4"') == 2

    x0 = deterministic_inputs(seed=11)[0]
    m1 = np.random.RandomState(12).rand(K, K).astype("float32")
    m2 = np.random.RandomState(13).rand(K, N).astype("float32")
    truncate_log()
    got, _ = run_layer(mod, (x0, m1, m2), "host")
    calls = [e for e in log_events() if e["event"] == "call"]
    assert len(calls) == 2, calls
    assert {(c["M"], c["N"], c["K"]) for c in calls} == {(M, K, K), (M, N, K)}, calls
    numpy_ref = (x0 @ m1 + b1) @ m2 + b2
    assert np.max(np.abs(got - numpy_ref)) < 1e-5, np.max(np.abs(got - numpy_ref))


def test_engine_path_is_linked():
    lib = pathlib.Path(runtime.load())
    dynamic = subprocess.run(["readelf", "-d", str(lib)], capture_output=True, text=True, check=True).stdout
    symbols = subprocess.run(["nm", "-D", str(lib)], capture_output=True, text=True, check=True).stdout
    if "libvc4infer.so" not in dynamic:
        print("    (built --no-engine; engine path not linked in this artifact)")
        return
    for sym in ("vc4_init", "vc4_matmul", "vc4_elementwise", "vc4_download"):
        assert f" U {sym}\n" in symbols or f" U {sym}@" in symbols, f"{sym} not in {lib}"


TESTS = [
    test_pattern_table_matches,
    test_pattern_rejects_parameter_bias,
    test_pattern_rejects_unsupported_bias_shape,
    test_descriptor_text,
    test_host_double_matches_relax_cpu,
    test_record_mode_is_inert,
    test_bias_forms,
    test_two_fused_layers,
    test_engine_path_is_linked,
]


def main():
    failures = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - the runner reports every failure
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
