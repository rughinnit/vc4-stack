CC      ?= gcc
# The lowering checks import jax (jax.export), so point PYTHON at the venv that
# has jax/jaxlib installed.
PYTHON  ?= python3
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
# The PJRT C API header is vendored: the jaxlib wheel ships only xla/ffi.
PJRT_INC ?= third_party
CPPFLAGS = -I$(PJRT_INC)
# Prefer the engine built from this checkout (infer/build), which carries the
# ungated gather/reduce/select3/conv kernels the lowering uses; fall back to
# the installed libvc4infer.  Override VC4INFER_LIB for a packaged build.
ENGINE_DIR := $(abspath $(CURDIR)/../infer/build)
# Refuse to fall back to the installed library when the checkout's engine is
# missing: the loader would then resolve libvc4infer.so.1 from /usr/lib and the
# tests would measure an older engine without saying so.
ifeq ($(origin VC4INFER_LIB), undefined)
ifeq ($(filter clean,$(MAKECMDGOALS)),)
ifeq ($(wildcard $(ENGINE_DIR)/libvc4infer.so),)
$(error missing $(ENGINE_DIR)/libvc4infer.so: build the engine first         (make -C infer), or set VC4INFER_LIB for a packaged build)
endif
endif
endif
VC4INFER_LIB ?= -L$(ENGINE_DIR) -lvc4infer -Wl,-rpath,$(ENGINE_DIR)
LDLIBS  = $(VC4INFER_LIB) -lm

BUILD = build
LIB   = $(BUILD)/libpjrt_vc4.so
DECODE = $(BUILD)/pjrt_decode

all: $(LIB)

$(BUILD):
	mkdir -p $(BUILD)

SRC = src/vc4_pjrt.c src/stablehlo.c
HDRS = src/stablehlo.h $(PJRT_INC)/pjrt_c_api.h

$(LIB): $(SRC) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Isrc -shared -Wl,-soname,libpjrt_vc4.so \
	    -o $@ $(SRC) $(LDLIBS)

# The decoder links only the reader and the lowering: no engine, no QPU.
$(DECODE): src/stablehlo.c src/stablehlo.h tools/pjrt_decode.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Isrc -o $@ tools/pjrt_decode.c \
	    src/stablehlo.c

decode: $(DECODE)

# Lowering checks: jax.export plus the decoder, so no kernel runs.
test-lower: $(DECODE)
	$(PYTHON) tests/test_lower.py

# End-to-end checks.  These launch kernels, so run them on an idle board.
test: $(LIB)
	$(PYTHON) tests/test_pjrt.py

clean:
	rm -rf $(BUILD)

.PHONY: all decode test test-lower clean
