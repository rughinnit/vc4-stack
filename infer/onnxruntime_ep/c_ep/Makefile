# Build the capability-only VC4 execution-provider plugin.
#
#   make -C infer/onnxruntime_ep/c_ep
#   make -C infer/onnxruntime_ep/c_ep clean
#
# The ORT headers are vendored for the runtime version this board has; see
# ../third_party/onnxruntime-1.32.0/PROVENANCE.md. Nothing is linked against
# libonnxruntime: a plugin receives OrtApiBase from the host process at
# CreateEpFactories time, which is also why a version mismatch cannot be
# papered over at link time.

CXX ?= g++
CXXFLAGS ?= -O2 -fPIC -std=c++17 -Wall -Wextra
ORT_INCLUDE := ../third_party/onnxruntime-1.32.0/include/onnxruntime
CPPFLAGS += -I$(ORT_INCLUDE)
LDFLAGS += -shared

all: libvc4_ep.so

libvc4_ep.so: vc4_ep.cc $(ORT_INCLUDE)/onnxruntime_ep_c_api.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f libvc4_ep.so

.PHONY: all clean
