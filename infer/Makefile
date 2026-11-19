CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
# vc4d ships libvc4client.a and the wire headers; VC4INFER_DAEMON=1 links it.
VC4D_DIR ?= /home/raspi/work/vc4d
CPPFLAGS = -Iinclude -Isrc -Itests -I../training/src -I$(VC4D_DIR)/include
LDLIBS  = -lm $(VC4D_DIR)/build/libvc4client.a

BUILD = build
LIB   = $(BUILD)/libvc4infer.a
# The backward/training kernels live in ../training/src but are part of the
# same library: the op registry dispatches to them.
SRC   = src/vc4_dev.c src/vc4_dev_remote.c src/vc4_kern.c src/vc4_gemm.c \
        src/vc4_ew.c src/vc4_conv.c src/vc4_nhwc.c src/vc4_api.c \
        ../training/src/vc4_convbwd.c ../training/src/vc4_train.c
OBJ   = $(addprefix $(BUILD)/,$(notdir $(SRC:.c=.o)))

TESTS = $(BUILD)/test_matmul $(BUILD)/test_conv $(BUILD)/test_ew $(BUILD)/test_dispatch $(BUILD)/test_softmax_probe $(BUILD)/test_softmax_count $(BUILD)/test_onnxops $(BUILD)/test_nhwc $(BUILD)/bench_matmul
TRAIN_TESTS = $(BUILD)/test_train_ops $(BUILD)/test_grad $(BUILD)/test_train $(BUILD)/test_convgrad $(BUILD)/test_convnet

all: $(LIB) $(BUILD)/libvc4infer.so $(TESTS) $(TRAIN_TESTS)

$(BUILD):
	mkdir -p $(BUILD) kcache

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -DVC4_KGEN_DEFAULT='"$(CURDIR)/tools/kgen.py"' -c $< -o $@

$(BUILD)/%.o: ../training/src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -DVC4_KGEN_DEFAULT='"$(CURDIR)/tools/kgen.py"' -c $< -o $@

$(LIB): $(OBJ) $(VC4D_DIR)/build/libvc4client.a
	$(AR) rcs $@ $^

$(BUILD)/libvc4infer.so: $(OBJ) $(VC4D_DIR)/build/libvc4client.a
	$(CC) -shared -Wl,-soname,libvc4infer.so.1 -o $@ $^

$(VC4D_DIR)/build/libvc4client.a:
	$(MAKE) -C $(VC4D_DIR) build/libvc4client.a

$(BUILD)/test_%: tests/test_%.c $(LIB) tests/util.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDLIBS)

$(BUILD)/test_%: ../training/tests/test_%.c $(LIB) tests/util.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDLIBS)

$(BUILD)/bench_matmul: tests/bench_matmul.c $(LIB) tests/util.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDLIBS)

# GPU-free verification of the formulas the kernels implement
check-math:
	python3 tools/check_math.py

clean:
	rm -rf $(BUILD)

.PHONY: all clean check-math
