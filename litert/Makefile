CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
LITERT_INC := third_party/litert
ENGINE_INC := $(abspath $(CURDIR)/../infer/include)
ENGINE_DIR := $(abspath $(CURDIR)/../infer/build)
CPPFLAGS := -I$(LITERT_INC) -I$(ENGINE_INC) -Isrc

# Prefer the engine built in this checkout.  A packaged delegate build can
# override VC4INFER_LIB (for example -lvc4infer from the vc4infer-dev deb).
ifeq ($(wildcard $(ENGINE_DIR)/libvc4infer.so),)
VC4INFER_LIB ?= -lvc4infer
else
VC4INFER_LIB ?= -L$(ENGINE_DIR) -lvc4infer -Wl,-rpath,$(ENGINE_DIR)
endif
LDLIBS := $(VC4INFER_LIB) -lm

# ai-edge-litert lives in a venv; point PYTHON at it.
PYTHON ?= python3

BUILD := build
LIB   := $(BUILD)/libvc4_delegate.so
UNIT  := $(BUILD)/test_partition

all: $(LIB)

$(BUILD):
	mkdir -p $(BUILD)

SRC := src/vc4_delegate.c src/vc4_plan.c
HDR := src/vc4_plan.h

$(LIB): $(SRC) $(HDR) | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -shared -Wl,-soname,libvc4_delegate.so \
	    -o $@ $(SRC) $(LDLIBS)

# The claim table links neither the interpreter nor the engine: it is a pure
# function of the TFLite descriptors.
$(UNIT): tests/test_partition.c src/vc4_plan.c src/vc4_plan.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ tests/test_partition.c src/vc4_plan.c -lm

# No target below launches a QPU workload unless the caller opts in:
# test_partition and the plan/off/decline modes of test_litert only exercise
# the CPU path.  VC4_LITERT_GPU=1 adds the on-device comparison.
test: all $(UNIT)
	$(UNIT)
	VC4_DELEGATE_SO=$(abspath $(LIB)) $(PYTHON) tests/test_litert.py

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
