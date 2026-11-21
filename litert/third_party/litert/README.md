# Vendored LiteRT headers

`ai-edge-litert` 2.3.0 ships `libLiteRt.so` and the Python interpreter but no C
or C++ headers, and the delegate has to compile against the exact
`TfLiteTensor` / `TfLiteNode` / `TfLiteRegistration` / `TfLiteDelegate` layouts
the installed shared object was built with.  These files are copied
unmodified from the LiteRT repository at tag **v2.3.0** (the release the wheel
is built from):

```
tflite/builtin_ops.h
tflite/c/common.h                                   (forwarder)
tflite/core/c/common.h
tflite/core/c/c_api_types.h
tflite/core/c/builtin_op_data.h                     (forwarder)
tflite/converter/core/c/tflite_types.h
tflite/converter/core/c/builtin_op_data.h
tflite/delegates/external/external_delegate_interface.h
```

Source: <https://github.com/google-ai-edge/LiteRT/tree/v2.3.0>.  The files
carry their original Apache-2.0 headers.

Only the declarations are used; the delegate links against `libvc4infer` and
libc, not against `libLiteRt.so`, so the runtime can be loaded by the Python
process in any order.  The struct layouts matter: they are why these are
vendored from the matching tag rather than redefined locally.
