CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
# The PJRT C API header is vendored: the jaxlib wheel ships only xla/ffi.
PJRT_INC ?= third_party
CPPFLAGS = -I$(PJRT_INC)
# Prefer the engine built from this checkout (infer/build), which carries the
# ungated gather/reduce/select3 kernels the lowering uses; fall back to the
# installed libvc4infer.  Override VC4INFER_LIB for a packaged build.
ENGINE_DIR := $(abspath $(CURDIR)/../infer/build)
ifeq ($(wildcard $(ENGINE_DIR)/libvc4infer.so),)
VC4INFER_LIB ?= -lvc4infer
else
VC4INFER_LIB ?= -L$(ENGINE_DIR) -lvc4infer -Wl,-rpath,$(ENGINE_DIR)
endif
LDLIBS  = $(VC4INFER_LIB) -lm

BUILD = build
LIB   = $(BUILD)/libpjrt_vc4.so

all: $(LIB)

$(BUILD):
	mkdir -p $(BUILD)

SRC = src/vc4_pjrt.c src/stablehlo.c
HDRS = src/stablehlo.h $(PJRT_INC)/pjrt_c_api.h

$(LIB): $(SRC) $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -Isrc -shared -Wl,-soname,libpjrt_vc4.so \
	    -o $@ $(SRC) $(LDLIBS)

test: $(LIB)
	python3 tests/test_pjrt.py

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
