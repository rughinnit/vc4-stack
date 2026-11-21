# TensorFlow PluggableDevice plugin for the VideoCore IV QPUs.
#
#   make                 build/libtf_vc4_plugin.so
#   make test-cpu        registration checks; no QPU work
#   make clean
#
# The plugin is loaded into a TensorFlow process with
# tensorflow.python.framework.load_library.load_pluggable_device_library().
# It links against the TensorFlow C API (libtensorflow_framework.so.2 from the
# wheel) and libvc4infer.
#
# TF_DIR may be set explicitly; otherwise it is read from the tensorflow
# package that PYTHON imports.  It is expanded inside the link recipe only, so
# `make clean` does not pay for an `import tensorflow` (seconds on an idle
# board, minutes under load).
#
# DEVICE_TYPE defaults to VC because TensorFlow 2.21 rejects a digit in a
# StreamExecutor platform name/type; see README.md.

PYTHON  ?= python3
TF_DIR  ?= $(shell $(PYTHON) -c 'import os, tensorflow; print(os.path.dirname(tensorflow.__file__))')
TF_INC   = $(TF_DIR)/include
TF_LIB   = $(TF_DIR)/libtensorflow_framework.so.2

# The engine header comes from this checkout (infer/include); the library is
# the installed one unless VC4INFER_LIB is given, e.g.
#   make VC4INFER_LIB="-L$(CURDIR)/../infer/build -lvc4infer -Wl,-rpath,$(CURDIR)/../infer/build"
VC4_CFLAGS ?= -I$(abspath $(CURDIR)/../infer/include)
VC4_LIBS   ?= $(shell pkg-config --libs vc4infer)

DEVICE_TYPE ?= VC
CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -fPIC
CPPFLAGS = -I$(TF_INC) $(VC4_CFLAGS) -DVC4_TF_DEVICE_TYPE='"$(DEVICE_TYPE)"'

BUILD = build
LIB   = $(BUILD)/libtf_vc4_plugin.so
SRC   = src/vc4_plugin.c

all: $(LIB)

$(BUILD):
	mkdir -p $(BUILD)

$(LIB): $(SRC) | $(BUILD)
	@test -f "$(TF_LIB)" || { \
	    echo "$(TF_LIB) not found: point PYTHON at an environment with tensorflow, or set TF_DIR"; \
	    exit 1; }
	$(CC) $(CFLAGS) $(CPPFLAGS) -shared -Wl,-soname,libtf_vc4_plugin.so \
	    -o $@ $(SRC) $(VC4_LIBS) "$(TF_LIB)" -lpthread

test-cpu: $(LIB)
	$(PYTHON) tests/test_plugin_cpu.py

clean:
	rm -rf $(BUILD)

.PHONY: all test-cpu clean
