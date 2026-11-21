# `vc4infer.ort` — ONNX Runtime execution provider

There are two artifacts here, and they do different things:

* the **graph-partitioning provider** in `infer/python/vc4infer/ort/` (shipped
  in both the wheel and the .deb). It claims a node only when the strict engine
  can execute it exactly, runs the claimed partitions through `libvc4infer`,
  and reports every node that stayed on ORT's CPU EP;
* a **capability-only C plugin** in `c_ep/`. `libvc4_ep.so` is compiled against
  the vendored ORT 1.32.0 plugin headers, registers with the installed runtime,
  advertises one VideoCore IV EP device, and claims no nodes. It has no execute
  path; it exists to prove the plugin ABI is usable at this version, and it says
  so in its metadata and in `vc4_ep_status()`.

## Layout

| path | what it is |
|---|---|
| `export_cnn.py` | `torch.onnx.export` of the small CNN used by the tests |
| `tests/test_ort_claims_cpu.py` | claim table + partition-equivalence test; no daemon, no GPU |
| `tests/test_vc4_ep.py` | device acceptance (needs vc4d and the GPU alone) |
| `tests/test_ep_register.py` | builds, registers and exercises the C plugin; claims nothing |
| `c_ep/vc4_ep.cc`, `c_ep/Makefile` | the capability-only plugin |
| `third_party/onnxruntime-1.32.0/` | the four plugin headers from the matching release, with provenance |
| `bench_yolo160_ort.py` | three-way benchmark (provider / native engine / ORT CPU), one backend per process |

## Running

CPU side (safe while the GPU is busy):

```sh
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_ort_claims_cpu.py \
    --cnn ~/scratch/tmp/vc4ort/smallcnn.onnx
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_ep_register.py
```

Device side (vc4d running, no other client):

```sh
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_vc4_ep.py \
    --data /usr/share/vc4infer/models --work ~/scratch/tmp/vc4ort --reps 3
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/bench_yolo160_ort.py --all --reps 5
```

The provider claims a node only when `vc4infer.coverage` says the strict
engine executes it exactly; everything else is reported as `ort-cpu` and run
by ORT. `fallback='error'` turns that into a refusal, and
`vc4infer.ort.plan(model)` produces the same claim table with no session at
all. See `docs/ONNXRUNTIME.md` for the measured table and for the header and
ABI evidence behind the plugin.
