#!/usr/bin/env python3
"""baseline.py -- re-take the CPU baseline on an idle board.

The plan's on/off delta (docs/CPU-BREAKDOWN.md 6.5) is measured inside one
environment, so it survives a busy board.  The absolute client CPU figure does
not: a shared daemon and a few idle Python processes on the ARM took the 160
baseline from 0.241 s to 0.2875 s, which is the number a reader compares with
ncnn.  This tool refuses to start until the board is idle and then measures
both paths, interleaved, the way the document's protocol describes.

    tools/cost/baseline.py --check          # preconditions only, no device
    tools/cost/baseline.py --out ~/scratch/logs/baseline.json

Checks: one ARM clock, no other Python client, load average, temperature,
throttling, the packaged vc4d and its arena, and the model files.  The run is
three rounds per size, plan off then on, eight timed frames after three
warm-ups per cell, each cell under tools/thermal-guard.sh --max 75.  Medians
of the three rounds are printed and written to --out.

The daemon is not restarted.  The document's protocol starts a fresh vc4d per
cell so the client gets the whole arena; if the daemon has been up for a long
time this tool prints that, and restarting it is the window owner's call
(``sudo systemctl restart vc4d``), not something to do behind their back.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODELS = os.path.expanduser("~/work/onnx/models")
SOCKET = "/run/vc4d.sock"
HZ = os.sysconf("SC_CLK_TCK")

THERMAL = os.path.join(REPO, "tools", "thermal-guard.sh")
BREAKDOWN = os.path.join(HERE, "breakdown.py")


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def temp_c():
    try:
        return float(read("/sys/class/thermal/thermal_zone0/temp")) / 1000.0
    except ValueError:
        return None


def proc_cmd(pid):
    try:
        raw = open("/proc/%d/cmdline" % pid, "rb").read()
    except OSError:
        return ""
    return " ".join(a.decode("utf-8", "replace") for a in raw.split(b"\0") if a)


def python_processes():
    """(pid, cmdline) for Python processes other than this one and its parent."""
    skip = {os.getpid(), os.getppid()}
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in skip:
            continue
        comm = read("/proc/%d/comm" % pid)
        if not comm.startswith("python"):
            continue
        cmd = proc_cmd(pid)
        if not cmd:
            continue
        out.append((pid, cmd))
    return out


def vc4d_pids():
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        cmd = proc_cmd(pid)
        if cmd.split(" ")[0].endswith("/vc4d") or cmd.startswith("vc4d "):
            out.append((pid, cmd))
    return out


def daemon_uptime(pid):
    """Seconds since the daemon started, from btime and its starttime."""
    try:
        stat = read("/proc/%d/stat" % pid).rsplit(")", 1)[1].split()
        start_ticks = int(stat[19])
        btime = None
        for line in read("/proc/stat").splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
        if btime is None:
            return None
        age = time.time() - (btime + start_ticks / float(HZ))
        return age
    except (OSError, IndexError, ValueError):
        return None


def check(sizes, stream=sys.stdout):
    """Print one PASS/FAIL line per precondition; return the failing ones."""
    bad = []

    def report(ok, what, detail=""):
        print("%-4s %s%s" % ("PASS" if ok else "FAIL", what,
                             ("  [%s]" % detail) if detail else ""),
              file=stream)
        if not ok:
            bad.append(what)

    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    mn, mx = read(base + "/scaling_min_freq"), read(base + "/scaling_max_freq")
    report(mn == mx == "1000000", "ARM pinned at 1.0 GHz",
           "min %s max %s (echo 1000000 | sudo tee %s/scaling_max_freq)"
           % (mn, mx, base))

    load = os.getloadavg()[0]
    report(load <= 0.20, "load average %.2f <= 0.20" % load)

    procs = python_processes()
    report(not procs, "no other Python process",
           "; ".join("%d %s" % (p, c[:60]) for p, c in procs[:4]))

    t = temp_c()
    report(t is not None and t <= 55.0, "temperature %s C <= 55" % t)

    throttled = ""
    try:
        from subprocess import run
        throttled = run(["vcgencmd", "get_throttled"], capture_output=True,
                        text=True).stdout.strip().split("=")[-1]
    except OSError:
        pass
    report(throttled in ("0x0", "", "0"), "throttled %s" % (throttled or "?"))

    pids = vc4d_pids()
    ours = [(pid, cmd) for pid, cmd in pids if SOCKET in cmd]
    others = [(pid, cmd) for pid, cmd in pids if SOCKET not in cmd]
    report(len(ours) == 1, "one packaged vc4d on %s" % SOCKET,
           "%d found" % len(ours))
    report(not others, "no second daemon on another socket",
           "; ".join(c[:70] for _, c in others))
    if len(ours) == 1:
        cmd = ours[0][1]
        age = daemon_uptime(ours[0][0])
        if age is not None:
            print("%-4s vc4d up %.0f s%s" % ("INFO", age,
                  "" if age < 900 else " (restart it for a fresh arena)"))
        arena = ""
        words = cmd.split()
        for word, nxt in zip(words, words[1:]):
            if word == "--arena-mb":
                arena = nxt
        report(arena.isdigit() and int(arena) >= 64,
               "daemon arena %s MiB >= 64" % (arena or "?"))

    for size in sizes:
        p = os.path.join(MODELS, "yolov8n_%d.onnx" % size)
        report(os.path.isfile(p), "model %s" % p)

    report(os.path.isfile(THERMAL), "thermal guard %s" % THERMAL)
    return bad


def cell(size, plan, runs):
    env = dict(os.environ)
    env["VC4INFER_PROF"] = "1"
    env["TMPDIR"] = os.path.expanduser("~/scratch/tmp")
    # The checkout's own build, not whatever VC4INFER_LIB the caller had.
    env.pop("VC4INFER_LIB", None)
    if plan:
        env["VC4_CPLAN"] = "1"
        env.pop("VC4_NO_CPLAN", None)
    else:
        env["VC4_NO_CPLAN"] = "1"
        env.pop("VC4_CPLAN", None)
    cmd = [THERMAL, "--max", "75", "--", sys.executable, BREAKDOWN,
           "--size", str(size), "--runs", str(runs)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=REPO)
    if p.returncode == 99:
        raise SystemExit("thermal guard stopped the %d cell (%s); the run is "
                         "not a valid measurement" % (size,
                                                      "plan" if plan else
                                                      "python"))
    if p.returncode != 0:
        raise SystemExit("breakdown failed for %d %s (%d): %s"
                         % (size, "plan" if plan else "python",
                            p.returncode, p.stderr[-600:]))
    for line in p.stdout.splitlines():
        if line.startswith("BREAKDOWN_JSON "):
            return json.loads(line[len("BREAKDOWN_JSON "):])
    raise SystemExit("no BREAKDOWN_JSON from the %d cell" % size)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="preconditions only; opens no device")
    ap.add_argument("--sizes", default="96,128,160")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    bad = check(sizes)
    if bad:
        print("\nnot idle: %s" % ", ".join(bad), file=sys.stderr)
        return 1
    if args.check:
        print("\nidle: the baseline can be taken")
        return 0

    rows = {}
    for size in sizes:
        for rnd in range(1, args.rounds + 1):
            for plan in (False, True):
                r = cell(size, plan, args.runs)
                rows.setdefault((size, plan), []).append(r)
                pf = r["per_frame"]
                print("%3d %-6s r%d wall %.1f ms client %.4f daemon %.4f "
                      "total %.4f launches %.0f"
                      % (size, "plan" if plan else "python", rnd,
                         r["wall_ms_median"], pf["client_cpu_s"],
                         pf["daemon_cpu_s"], pf["total_cpu_s"],
                         r["remote"]["launches_per_frame"]), flush=True)

    summary = {}
    print("\n%3s %-6s %8s %8s %8s %8s %8s %8s"
          % ("sz", "mode", "wall ms", "client", "daemon", "total",
             "launch", "py"))
    for size in sizes:
        for plan in (False, True):
            rs = rows[(size, plan)]
            s = summary["%d-%s" % (size, "plan" if plan else "python")] = {
                "wall_ms": statistics.median(r["wall_ms_median"] for r in rs),
                "client_cpu_s": statistics.median(
                    r["per_frame"]["client_cpu_s"] for r in rs),
                "daemon_cpu_s": statistics.median(
                    r["per_frame"]["daemon_cpu_s"] for r in rs),
                "total_cpu_s": statistics.median(
                    r["per_frame"]["total_cpu_s"] for r in rs),
                "launches": statistics.median(
                    r["remote"]["launches_per_frame"] for r in rs),
                "python_s": statistics.median(
                    r["client"]["python_s"] for r in rs),
                "rounds": len(rs),
            }
            print("%3d %-6s %8.1f %8.4f %8.4f %8.4f %8.0f %8.4f"
                  % (size, "plan" if plan else "python", s["wall_ms"],
                     s["client_cpu_s"], s["daemon_cpu_s"], s["total_cpu_s"],
                     s["launches"], s["python_s"]))
        a = summary["%d-python" % size]
        b = summary["%d-plan" % size]
        print("    delta  wall %+.1f ms  client %+.4f  total %+.4f (%+.1f%%)"
              % (b["wall_ms"] - a["wall_ms"],
                 b["client_cpu_s"] - a["client_cpu_s"],
                 b["total_cpu_s"] - a["total_cpu_s"],
                 100.0 * (b["total_cpu_s"] - a["total_cpu_s"]) /
                 a["total_cpu_s"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary,
                       "rounds": args.rounds, "runs": args.runs,
                       "sizes": sizes,
                       "arm_hz": 1000000}, f, indent=1, sort_keys=True)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
