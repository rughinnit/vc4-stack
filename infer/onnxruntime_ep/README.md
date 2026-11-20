# `vc4infer.ort` — ONNX Runtime execution provider

The implementation lives in `infer/python/vc4infer/ort/` (shipped in both the
wheel and the .deb); this directory holds the acceptance tests and benchmark.

* `export_cnn.py` — `torch.onnx.export` of the small CNN used by the tests.
* `tests/test_vc4_ep.py` — CNN vs ORT CPU, Einsum split (reported, still
  correct), YOLOv8n-160 vs the stored reference, and the ORT-only passthrough.
* `bench_yolo160_ort.py` — three-way benchmark (provider / native engine /
  ORT CPU), each backend in a fresh process because vc4d serves one arena.

Run on the board (install `onnxruntime` first):

```sh
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/tests/test_vc4_ep.py --reps 3
PYTHONPATH=infer/python ~/scratch/ortvenv/bin/python \
    infer/onnxruntime_ep/bench_yolo160_ort.py --all --reps 5
```

The provider claims a node only when `vc4infer.coverage` says the strict
engine executes it exactly; everything else is reported as `ort-cpu` and run
by ORT. `fallback='error'` turns that into a refusal. See
`docs/ONNXRUNTIME.md` for the header evidence behind the design.
