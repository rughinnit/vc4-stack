// The only translation unit that exports the GGML_BACKEND_DL entry points.
// Keeping it separate means the statically linked test library never defines
// ggml_backend_init/ggml_backend_score.

#include "ggml-vc4.h"
#include "ggml-backend-impl.h"

GGML_BACKEND_DL_IMPL(ggml_backend_vc4_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_vc4_score)
