# ONNX Runtime plugin headers, vendored

The onnxruntime 1.32.0 wheel installs no headers (`find site-packages/onnxruntime
-name '*.h' | wc -l` -> 0), so an out-of-tree execution provider cannot be
compiled against it directly. These are the four headers from the matching
upstream release:

| | |
|---|---|
| project | ONNX Runtime |
| release tag | [`v1.32.0`](https://github.com/microsoft/onnxruntime/tree/v1.32.0) |
| source tarball | `onnxruntime-1.32.0.tar.gz` (GitHub tag archive) |
| tarball sha256 | `8bb6dde72d89509b8c014a440bf113505d86af99e8f143da3de1a3feb8132561` |
| obtained from | conda-forge `onnxruntime-cpp` 1.32.0, build `h2a079e4_0_cpu`, linux-aarch64 |
| conda package sha256 | `92417722fc47ba3777026b08bc81c6a756e9e3a888b8e83c66d9d50781046ba7` |
| licence | MIT (see `LICENSE`) |

The GitHub tree is not reachable from this board (`raw.githubusercontent.com`
answers 404 for every onnxruntime path), and PyPI publishes wheels only for
`onnxruntime` - there is no sdist. The conda-forge `-cpp` package is a build of
the same release and is the only route to the header at the exact installed
version. It was unpacked with:

```sh
curl -O https://conda.anaconda.org/conda-forge/linux-aarch64/onnxruntime-cpp-1.32.0-h2a079e4_0_cpu.conda
python3 -c "import zipfile; ..."          # .conda is a zip of .tar.zst members
tar -xf pkg-onnxruntime-cpp-1.32.0-h2a079e4_0_cpu.tar.zst
```

Copied verbatim, unmodified:

* `include/onnxruntime/onnxruntime_c_api.h`
* `include/onnxruntime/onnxruntime_ep_c_api.h`
* `include/onnxruntime/onnxruntime_error_code.h`
* `include/onnxruntime/onnxruntime_ep_device_ep_metadata_keys.h`

`onnxruntime_c_api.h` includes `onnxruntime_error_code.h` and, at its end,
`onnxruntime_ep_c_api.h`; those two include nothing else outside this set. The
unused headers of the source tree (C++ wrapper, training, experimental APIs,
provider option headers) are deliberately not vendored.

`ORT_API_VERSION` in this header is 32, which is the version the installed
1.32.0 runtime exports. A plugin compiled against a different ORT_API_VERSION
fails at `GetApi()` inside `CreateEpFactories` and registers nothing rather
than running with a mismatched struct layout.
