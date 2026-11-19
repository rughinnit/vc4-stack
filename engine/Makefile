CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
CPPFLAGS = -Iinclude
LDLIBS  = -lm

BUILD = build
LIB   = $(BUILD)/libvc4infer.a
SRC   = src/vc4_dev.c src/vc4_kern.c src/vc4_gemm.c src/vc4_ew.c \
        src/vc4_conv.c src/vc4_api.c src/vc4_train.c
OBJ   = $(SRC:src/%.c=$(BUILD)/%.o)

TESTS = $(BUILD)/test_matmul $(BUILD)/test_conv $(BUILD)/test_ew $(BUILD)/test_dispatch $(BUILD)/test_train_ops $(BUILD)/test_softmax_probe $(BUILD)/test_softmax_count $(BUILD)/test_grad $(BUILD)/test_train

all: $(LIB) $(BUILD)/libvc4infer.so $(TESTS)

$(BUILD):
	mkdir -p $(BUILD) kcache

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -DVC4_KGEN_DEFAULT='"$(CURDIR)/tools/kgen.py"' -c $< -o $@

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(BUILD)/libvc4infer.so: $(OBJ)
	$(CC) -shared -o $@ $^

$(BUILD)/test_%: tests/test_%.c $(LIB) tests/util.h | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDLIBS)

# GPU-free verification of the formulas the kernels implement
check-math:
	python3 tools/check_math.py

clean:
	rm -rf $(BUILD)

.PHONY: all clean check-math
