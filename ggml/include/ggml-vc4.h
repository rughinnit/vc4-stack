#pragma once

// ggml backend for the VideoCore IV QPUs (Raspberry Pi 3B, V3D).
//
// The backend speaks ggml's backend ABI (ggml_backend_i / ggml_backend_device_i
// / ggml_backend_reg_i recorded in ggml-backend-impl.h of the ggml tree it is
// compiled against) and lowers each graph node to the libvc4infer engine
// (infer/include/vc4infer.h): QPU kernels, one dma-buf arena, strict dispatch.
//
// Two ways to register it:
//   * compiled in:  call ggml_vc4_register() before the scheduler is built
//   * dynamic:      build with GGML_BACKEND_DL (this tree does) and point
//                   GGML_BACKEND_PATH at libggml-vc4.so

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API ggml_backend_reg_t ggml_backend_vc4_reg(void);

// Register with ggml's global backend registry (static registration).
GGML_API void ggml_vc4_register(void);

// GGML_BACKEND_DL score: 0 when the board/daemon is not reachable.
GGML_API int ggml_backend_vc4_score(void);

#ifdef __cplusplus
}
#endif
