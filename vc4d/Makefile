CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -std=gnu11 -Wno-unused-parameter
CPPFLAGS = -Iproto -Iinclude
B       = build

ENGINE  ?= /home/raspi/work/engine-fast2

DAEMON  = $(B)/vc4d
LIB     = $(B)/libvc4client.a
SOLIB   = $(B)/libvc4client.so
DEMO    = $(B)/demo_gemm
HOSTILE = $(B)/test_hostile
BENCH   = $(B)/bench_direct
# G2/G3/G4 regression tests (G2's shell driver lives in infer/tests)
LRU     = $(B)/test_prog_lru
UNIMASK = $(B)/test_uni_mask
CMAHOLD = $(B)/cmahold
ARENASZ = $(B)/arena_size

all: $(DAEMON) $(LIB) $(SOLIB) $(DEMO) $(HOSTILE) $(LRU) $(UNIMASK) $(CMAHOLD) $(ARENASZ)

$(B):
	mkdir -p $(B)

DAEMON_SRC = src/vc4d.c src/vc4d_arena.c src/vc4d_gpu.c src/vc4d_sha256.c

$(DAEMON): $(DAEMON_SRC) src/vc4d.h proto/vc4d_proto.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $(DAEMON_SRC)

$(B)/libvc4client.o: src/libvc4client.c include/vc4client.h proto/vc4d_proto.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@

$(LIB): $(B)/libvc4client.o
	$(AR) rcs $@ $^

$(SOLIB): $(B)/libvc4client.o
	$(CC) -shared -o $@ $^

$(DEMO): demo/demo_gemm.c $(LIB) include/vc4client.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ -lm

$(HOSTILE): test/hostile.c proto/vc4d_proto.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

$(LRU): test/test_prog_lru.c $(LIB) include/vc4client.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@

$(UNIMASK): test/test_uni_mask.c $(LIB) include/vc4client.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@

$(ARENASZ): test/arena_size.c $(LIB) include/vc4client.h | $(B)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@

$(CMAHOLD): test/cmahold.c | $(B)
	$(CC) $(CFLAGS) $< -o $@

# the direct-root baseline links the engine's existing static library
$(BENCH): bench/bench_direct.c | $(B)
	@test -f $(ENGINE)/build/libvc4infer.a || (echo "missing $(ENGINE)/build/libvc4infer.a: run make in the engine first"; exit 1)
	$(CC) $(CFLAGS) -I$(ENGINE)/include $< $(ENGINE)/build/libvc4infer.a -o $@ -lm

bench: $(BENCH)

blobs:
	mkdir -p blobs

install: all
	install -m 0755 $(DAEMON) /usr/local/sbin/vc4d
	install -m 0644 systemd/vc4d.service /etc/systemd/system/vc4d.service
	systemctl daemon-reload

clean:
	rm -rf $(B)

.PHONY: all clean bench install
