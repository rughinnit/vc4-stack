#!/usr/bin/env python3
"""A protocol-level stand-in for vc4d, for CPU-only tests of the client side.

It executes nothing.  It speaks HELLO / ALLOC / UPLOAD / LAUNCH /
LAUNCH_BATCH / SUBMIT / FREE / SYNC / STATS / PING / CLOSE, hands out a
memfd-backed arena over SCM_RIGHTS exactly as the daemon hands out its
dma-buf, and writes a JSONL record of every request and every launch entry.
That is enough to drive the whole engine through a graph on a CPU and to check
the properties the pipelined path has to keep:

  * which launches went out, in what order, in which request;
  * whether a request's uniform shadow changed while the request was
    outstanding (it must not: the two in-flight requests need two disjoint
    sets);
  * whether the bytes a launch names changed while it was outstanding (the
    pool must not hand a block to a new node while a request that reads it is
    still running);
  * whether a second request was already in the socket when the first
    completed (the client is allowed to work while the daemon executes).

usage: fake_daemon.py SOCKET [--log FILE] [--delay-ms N] [--arena-mb N]

Environment (kept because the earlier CPU work used them):
  FAKE_VERSION       protocol version to serve (default 5)
  FAKE_HELLO_MAX     highest HELLO version accepted (default = FAKE_VERSION)
  FAKE_NO_BATCH      do not advertise LAUNCH_BATCH
  FAKE_NO_SUBMIT     do not advertise SUBMIT / protocol 4
  FAKE_STRICT_VERSION=0  accept requests whose header version differs
  FAKE_DELAY_MS      milliseconds to "execute" each request
"""
import array
import hashlib
import json
import os
import socket
import struct
import sys
import time

MAGIC = 0x44344356
VERSION = int(os.environ.get("FAKE_VERSION", "5"))
HELLO_MAX = int(os.environ.get("FAKE_HELLO_MAX", str(VERSION)))
ADVERTISE_BATCH = VERSION >= 3 and not os.environ.get("FAKE_NO_BATCH")
ADVERTISE_SUBMIT = (VERSION >= 4 and not os.environ.get("FAKE_NO_SUBMIT")
                    and ADVERTISE_BATCH)
STRICT_REQUEST_VERSION = os.environ.get("FAKE_STRICT_VERSION", "1") != "0"
DELAY_MS = float(os.environ.get("FAKE_DELAY_MS", "0"))

HDR = struct.Struct("<IIHHIIII")          # magic len op ver seq status d0 d1
assert HDR.size == 28
HELLO_REQ = struct.Struct("<II")
HELLO_RSP = struct.Struct("<IIIIIIIIIIIIIII")  # 15 words: version..reserved
ALLOC_REQ = struct.Struct("<II")
ALLOC_RSP = struct.Struct("<IIII")
UPLOAD_REQ = struct.Struct("<IIII")
UPLOAD_RSP = struct.Struct("<IIII")
LAUNCH_REQ = struct.Struct("<IIIIIIII")
LAUNCH_RSP = struct.Struct("<IIII")
WAIT_RSP = struct.Struct("<IIQII")
BATCH_REQ = struct.Struct("<IIII")
BATCH_ENT = struct.Struct("<IIIIIIIIIII")     # 44 bytes
BATCH_ITEM = struct.Struct("<IIIIQII")        # 32 bytes
STATS_RSP = struct.Struct("<QQQQQQQQIIIIQII")

OP_HELLO, OP_ALLOC, OP_FREE, OP_UPLOAD, OP_LAUNCH = 1, 2, 3, 4, 5
OP_WAIT, OP_READ_REGS, OP_WRITEREG, OP_STATS, OP_SYNC = 6, 7, 8, 9, 10
OP_CLOSE, OP_PING, OP_LAUNCH_BATCH, OP_SUBMIT = 11, 12, 13, 14

E_OK, E_PROTO, E_VERSION, E_TOOLARGE, E_INVAL = 0, 1, 2, 3, 4
E_NOMEM, E_NOENT, E_BUSY, E_TIMEOUT, E_GPU = 5, 6, 7, 8, 9
E_NOTRUN = 13

# The map is the protocol version: 5 is slots256 (heap at 6 MiB), 2..4 is
# slots128 (heap at 3 MiB).  The client picks the same values from the version
# it negotiated, so a FAKE_VERSION=4 run still models an old daemon exactly.
DATA_BASE = 0x600000 if VERSION >= 5 else 0x300000
ARENA = 128 << 20
LAYOUT = ("slots256" if VERSION >= 5 else "slots128").encode()
LAYOUT_WORDS = struct.unpack("<III", LAYOUT.ljust(12, b"\0")[:12])
BUS0 = 0xE0000000

# Address words are checked and probed as "bus + arena offset".  The probe
# reads a window at every address a launch names, so a block reused under a
# running request shows up as a changed hash.
PROBE = 4096
PROBE_MAX = 24


def now_ns():
    return time.monotonic_ns()


def reply(hdr, status, payload=b"", d0=0, d1=0):
    magic, ln, op, ver, seq, _, _, _ = hdr
    return HDR.pack(MAGIC, HDR.size + len(payload), op, VERSION, seq,
                    status, d0, d1) + payload


class Arena:
    """One memfd per ALLOC, mapped here so the fake can look inside it."""

    def __init__(self, fd, base, length):
        self.fd = fd
        self.base = base
        self.length = length
        import mmap
        self.map = mmap.mmap(fd, DATA_BASE + length, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE)
        self.bus = BUS0 + base

    def addr_ok(self, w):
        return self.bus <= w < self.bus + self.length

    def probe(self, off):
        """sha1 of PROBE bytes at a region-relative offset, bounded.

        The memfd starts at the arena base and the client's region starts at
        DATA_BASE, so a region-relative offset is DATA_BASE further in.
        """
        start = max(0, min(DATA_BASE + off, DATA_BASE + self.length - 1))
        end = min(DATA_BASE + self.length, start + PROBE)
        if end <= start:
            return "0" * 8
        return hashlib.sha1(self.map[start:end]).hexdigest()


def main():
    path = sys.argv[1]
    log_path = None
    delay_ms = DELAY_MS
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--log":
            log_path = args[i + 1]
            i += 2
        elif args[i] == "--delay-ms":
            delay_ms = float(args[i + 1])
            i += 2
        elif args[i] == "--arena-mb":
            global ARENA
            ARENA = int(float(args[i + 1]) * (1 << 20))
            i += 2
        else:
            i += 1

    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    srv.bind(path)
    os.chmod(path, 0o660)
    srv.listen(4)
    print("fake_daemon: listening on %s (v%d%s%s%s%s)"
          % (path, VERSION,
             " batch" if ADVERTISE_BATCH else "",
             " submit" if ADVERTISE_SUBMIT else "",
             " delay=%gms" % delay_ms if delay_ms > 0 else "",
             " log=%s" % log_path if log_path else ""), flush=True)

    logf = open(log_path, "a", buffering=1) if log_path else None
    state = {"hello": 0, "alloc": 0, "upload": 0, "launch": 0, "batch": 0,
             "submit": 0, "entries": 0, "max_entries": 0, "bad": 0,
             "uniform_changed": 0, "data_changed": 0, "clients": 0}
    progs = {}
    prog_names = {}
    next_prog = 1
    arenas = {}
    next_region = 1
    req_no = 0

    def emit(rec):
        if logf:
            rec["t"] = now_ns()
            logf.write(json.dumps(rec, sort_keys=True) + "\n")

    def counters():
        if not logf:
            return
        logf.flush()
        with open(log_path + ".counts", "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def batch_like(hdr, pl, opname):
        nonlocal req_no
        count, flags, _r0, _r1 = BATCH_REQ.unpack_from(pl)
        req_no += 1
        if count < 1 or count > 64 or len(pl) != BATCH_REQ.size + \
                BATCH_ENT.size * count:
            state["bad"] += 1
            return E_PROTO, b"", 0, 0
        # Snapshot everything the request will read before the "execution"
        # delay, then again after it.  A difference means the client changed
        # memory under a request that was still running.
        def snap():
            words, hashes = [], []
            for k in range(count):
                e = BATCH_ENT.unpack_from(pl, BATCH_REQ.size + BATCH_ENT.size * k)
                (kind, prog, tmo, nqpu, nuni, fl, uh, uo, ua, fh, fdir) = e
                if kind != 0 or uh not in arenas:
                    continue
                a = arenas[uh]
                w = a.map[DATA_BASE + uo:DATA_BASE + uo + 64 * min(nqpu, 12)]
                words.append((k, hashlib.sha1(w).hexdigest()[:16]))
                seen = 0
                for u in range(min(nuni, 16)):
                    if not (ua >> u) & 1:
                        continue
                    for q in range(min(nqpu, 12)):
                        off = DATA_BASE + uo + 64 * q + 4 * u
                        addr = struct.unpack_from("<I", a.map, off)[0]
                        if not a.addr_ok(addr) or seen >= PROBE_MAX:
                            continue
                        hashes.append((k, a.probe(addr - a.bus)))
                        seen += 1
            return words, hashes

        w0, h0 = snap()
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        w1, h1 = snap()
        changed = w0 != w1
        dchanged = h0 != h1
        if changed:
            state["uniform_changed"] += 1
        if dchanged:
            state["data_changed"] += 1

        items = []
        st = E_OK
        fail_idx = 0xffffffff
        for k in range(count):
            e = BATCH_ENT.unpack_from(pl, BATCH_REQ.size + BATCH_ENT.size * k)
            (kind, prog, tmo, nqpu, nuni, fl, uh, uo, ua, fh, fdir) = e
            if kind == 1:                       # fence
                items.append(BATCH_ITEM.pack(E_OK, 0, 0, 0, 0, 0, 0))
                continue
            if kind != 0 or not (fl & 1) or uh not in arenas:
                st, fail_idx = E_INVAL, k
                items.append(BATCH_ITEM.pack(E_INVAL, 0, 0, 0, 0, 0, 0))
                break
            a = arenas[uh]
            if prog not in progs or progs[prog] != nuni or \
                    uo + 64 * nqpu > a.length:
                st, fail_idx = E_NOENT, k
                items.append(BATCH_ITEM.pack(E_NOENT, 0, 0, 0, 0, 0, 0))
                break
            for u in range(min(nuni, 16)):
                for q in range(min(nqpu, 12)):
                    off = DATA_BASE + uo + 64 * q + 4 * u
                    addr = struct.unpack_from("<I", a.map, off)[0]
                    if (ua >> u) & 1 and not a.addr_ok(addr):
                        st, fail_idx = E_INVAL, k
                        break
            if st != E_OK:
                items.append(BATCH_ITEM.pack(E_INVAL, 0, 0, 0, 0, 0, 0))
                break
            items.append(BATCH_ITEM.pack(E_OK, k + 1, 0, 0, 1000, 1, 0))
        done = len(items)
        while len(items) < count:
            items.append(BATCH_ITEM.pack(E_NOTRUN, 0, 0, 0, 0, 0, 0))
        payload = b"".join(items)

        ent_log = []
        for k in range(count):
            e = BATCH_ENT.unpack_from(pl, BATCH_REQ.size + BATCH_ENT.size * k)
            (kind, prog, tmo, nqpu, nuni, fl, uh, uo, ua, fh, fdir) = e
            rec = {"prog": prog, "name": prog_names.get(prog, ""),
                   "nqpu": nqpu, "nuni": nuni, "kind": kind,
                   "uni_off": uo, "mask": ua}
            if kind == 0 and uh in arenas:
                a = arenas[uh]
                rec["words"] = list(struct.unpack_from(
                    "<%dI" % (4 * min(nqpu, 12)), a.map, DATA_BASE + uo))
                rec["probe"] = [h for (kk, h) in h0 if kk == k]
            ent_log.append(rec)
        emit({"op": opname, "seq": hdr[4], "req": req_no, "count": count,
              "done": done, "status": st, "uniform_changed": changed,
              "data_changed": dchanged, "entries": ent_log})
        state["batch" if opname == "batch" else "submit"] += 1
        state["entries"] += count
        state["max_entries"] = max(state["max_entries"], count)
        counters()
        return st, payload, done, fail_idx

    def serve(conn):
        nonlocal next_prog, next_region
        state["clients"] += 1
        while True:
            msg = conn.recv(65536)
            if not msg:
                return
            if len(msg) < HDR.size:
                state["bad"] += 1
                return
            hdr = HDR.unpack_from(msg)
            magic, ln, op, ver, seq, st_in, _, _ = hdr
            if magic != MAGIC or ln != len(msg):
                state["bad"] += 1
                return
            if ver < 2 or ver > VERSION:
                # A daemon older than the client answers E_VERSION and keeps
                # the connection: that is how the client's HELLO ladder finds
                # the version it can use.
                conn.sendmsg([reply(hdr, E_VERSION)])
                continue
            if op != OP_HELLO and STRICT_REQUEST_VERSION and ver != VERSION:
                state["bad"] += 1
                return
            pl = msg[HDR.size:]
            st, out, d0, d1, fds = E_OK, b"", 0, 0, []
            if op == OP_HELLO:
                state["hello"] += 1
                reqver = HELLO_REQ.unpack_from(pl)[0]
                if reqver < 2 or reqver > HELLO_MAX:
                    st = E_VERSION
                else:
                    flags = (1 << 2) | (1 << 3) | (1 << 4)
                    if ADVERTISE_BATCH:
                        flags |= (1 << 5)
                    if ADVERTISE_SUBMIT:
                        flags |= (1 << 6)
                    out = HELLO_RSP.pack(VERSION, ARENA, DATA_BASE, 4096,
                                        65536, 12, 16, 16384, flags,
                                       0x02443356, 0xC1102431,
                                       *LAYOUT_WORDS, 0)
                emit({"op": "hello", "version": reqver if pl else 0,
                      "status": st})
            elif op == OP_ALLOC:
                state["alloc"] += 1
                bytes_ = ALLOC_REQ.unpack_from(pl)[0]
                fd = os.memfd_create("fakearena%d" % next_region, 0)
                os.ftruncate(fd, DATA_BASE + bytes_)
                handle = next_region
                next_region += 1
                arenas[handle] = Arena(fd, DATA_BASE, bytes_)
                fds = [fd]
                out = ALLOC_RSP.pack(handle, DATA_BASE, bytes_, BUS0 + DATA_BASE)
                emit({"op": "alloc", "handle": handle, "bytes": bytes_})
            elif op == OP_UPLOAD:
                state["upload"] += 1
                bytes_, max_qpu, nuni, flags = UPLOAD_REQ.unpack_from(pl)
                prog = next_prog
                next_prog += 1
                name = ""
                if (flags & 1) and len(pl) > UPLOAD_REQ.size + bytes_:
                    name = pl[UPLOAD_REQ.size + bytes_:].decode(
                        "ascii", "replace")
                prog_names[prog] = name
                progs[prog] = nuni
                out = UPLOAD_RSP.pack(prog, 0x1000, bytes_ // 8, 0)
            elif op == OP_LAUNCH:
                state["launch"] += 1
                prog, tmo, nqpu, nuni, flags, uh, uo, ua = \
                    LAUNCH_REQ.unpack_from(pl)
                if prog not in progs or progs[prog] != nuni:
                    st = E_NOENT
                elif flags & 2:      # WAIT
                    out = WAIT_RSP.pack(seq, 0, 1000, 1, 0)
                else:
                    out = LAUNCH_RSP.pack(seq, nqpu, 0, 0)
            elif op == OP_LAUNCH_BATCH:
                st, out, d0, d1 = batch_like(hdr, pl, "batch")
            elif op == OP_SUBMIT:
                st, out, d0, d1 = batch_like(hdr, pl, "submit")
            elif op == OP_FREE:
                pass
            elif op == OP_SYNC:
                pass
            elif op == OP_STATS:
                out = STATS_RSP.pack(0, 0, 0, 0, 0, ARENA, 0, 0, 0, 1,
                                     next_prog - 1, 0, ARENA, 0)
            elif op == OP_CLOSE:
                conn.sendmsg([reply(hdr, E_OK)])
                return
            elif op == OP_PING:
                d0 = 1
            else:
                st = E_INVAL
            conn.sendmsg([reply(hdr, st, out, d0, d1)],
                         [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                           array.array("i", fds))] if fds else [])
            for fd in fds:
                os.close(fd)

    while True:
        conn, _ = srv.accept()
        try:
            serve(conn)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            counters()
            print("fake_daemon: " + json.dumps(state, sort_keys=True),
                  flush=True)


if __name__ == "__main__":
    main()
