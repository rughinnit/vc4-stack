"""Shared readings for the cost measurements: CPU time, RSS, temperature,
system idle, model paths and the two runtimes."""
import os
import statistics
import subprocess
import threading
import time

HZ = os.sysconf("SC_CLK_TCK")
CORES = os.cpu_count() or 4
MODELS = os.path.expanduser("~/work/onnx/models")


def proc_cpu_s(pid=None):
    """utime+stime in seconds for a process (default: this one)."""
    pid = os.getpid() if pid is None else pid
    try:
        data = open("/proc/%d/stat" % pid).read()
    except OSError:
        return 0.0
    tail = data[data.rfind(")") + 2:].split()
    return (int(tail[11]) + int(tail[12])) / float(HZ)


def proc_rss_kb(pid=None):
    """(VmRSS, VmHWM) in KiB for a process (default: this one)."""
    pid = os.getpid() if pid is None else pid
    vals = {}
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith(("VmRSS:", "VmHWM:")):
                vals[line.split(":")[0]] = int(line.split()[1])
    except OSError:
        pass
    return vals.get("VmRSS", 0), vals.get("VmHWM", 0)


def vc4d_pid():
    """The vc4d serving /run/vc4d.sock (the TV daemon has another socket)."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            cmd = open("/proc/%s/cmdline" % name, "rb").read().split(b"\0")
        except OSError:
            continue
        if not cmd or not cmd[0].endswith(b"vc4d"):
            continue
        args = [a.decode("utf-8", "replace") for a in cmd]
        if "/run/vc4d.sock" in args:
            return int(name)
    return None


def temp_c():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except OSError:
        out = subprocess.run(["vcgencmd", "measure_temp"],
                             capture_output=True, text=True).stdout
        return float(out.strip().replace("temp=", "").replace("'C", ""))


def arm_hz():
    out = subprocess.run(["vcgencmd", "measure_clock", "arm"],
                         capture_output=True, text=True).stdout
    try:
        return int(out.strip().split("=")[1])
    except (IndexError, ValueError):
        return 0


def core_hz():
    out = subprocess.run(["vcgencmd", "measure_clock", "core"],
                         capture_output=True, text=True).stdout
    try:
        return int(out.strip().split("=")[1])
    except (IndexError, ValueError):
        return 0


def throttled():
    out = subprocess.run(["vcgencmd", "get_throttled"],
                         capture_output=True, text=True).stdout
    return out.strip().split("=")[-1] if "=" in out else out.strip()


def lcrit_alarm():
    try:
        return int(open("/sys/class/hwmon/hwmon1/in0_lcrit_alarm").read())
    except OSError:
        return None


def cpu_totals():
    parts = open("/proc/stat").readline().split()
    vals = [int(v) for v in parts[1:]]
    idle = vals[3] + vals[4]              # idle + iowait
    return idle, sum(vals)


class Sampler(threading.Thread):
    """Sample system CPU, temperature and clock while a cell runs.

    The thread itself costs a fraction of one percent of a core; it is part of
    the process CPU the cell reports and is noted in the document.
    """

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.samples = []

    def run(self):
        last_idle, last_total = cpu_totals()
        while not self.stop_flag.wait(self.interval):
            idle, total = cpu_totals()
            self.samples.append({
                "idle": idle - last_idle,
                "total": total - last_total,
                "temp_c": temp_c(),
                "arm_hz": arm_hz(),
            })
            last_idle, last_total = idle, total

    def stop(self):
        self.stop_flag.set()
        self.join()

    def summary(self):
        if not self.samples:
            return {}
        busy = sum(s["total"] - s["idle"] for s in self.samples)
        total = sum(s["total"] for s in self.samples)
        inst = [(s["total"] - s["idle"]) / float(s["total"])
                for s in self.samples if s["total"]]
        temps = [s["temp_c"] for s in self.samples]
        hz = [s["arm_hz"] for s in self.samples if s["arm_hz"]]
        return {
            "sys_busy_pct": 100.0 * busy / total if total else None,
            "sys_idle_pct": 100.0 * (1.0 - busy / total) if total else None,
            "sys_busy_pct_min": 100.0 * min(inst) if inst else None,
            "sys_busy_pct_max": 100.0 * max(inst) if inst else None,
            "temp_start_c": temps[0] if temps else None,
            "temp_peak_c": max(temps) if temps else None,
            "temp_end_c": temps[-1] if temps else None,
            "arm_hz_min": min(hz) if hz else None,
            "arm_hz_max": max(hz) if hz else None,
        }


def vc4_model(size):
    return os.path.join(MODELS, "yolov8n_%d.onnx" % size)


def ncnn_model(size):
    if size == 160:
        return os.path.join(MODELS, "yolov8n_ncnn_model")
    return os.path.join(MODELS, "yolov8n_ncnn_%d" % size)


def expected_elems(size):
    """YOLOv8n detection head: 84 channels x (P3+P4+P5) cells."""
    s = size // 8
    return 84 * (s * s + (s // 2) * (s // 2) + (s // 4) * (s // 4))


def load_ncnn(size, threads=4):
    import ncnn
    import numpy as np
    d = ncnn_model(size)
    net = ncnn.Net()
    net.opt.num_threads = threads
    net.load_param(os.path.join(d, "model.ncnn.param"))
    net.load_model(os.path.join(d, "model.ncnn.bin"))
    x = np.random.default_rng(0).standard_normal(
        (1, 3, size, size)).astype(np.float32)
    mat = ncnn.Mat(x[0]).clone()

    def run():
        ex = net.create_extractor()
        ex.input("in0", mat)
        rc, out = ex.extract("out0")
        if rc != 0:
            raise RuntimeError("ncnn extract rc=%d" % rc)
        elems = out.w * out.h * out.c
        del out, ex
        return {"elems": elems}

    return {
        "run": run,
        "input": x,
        "model_bytes": os.path.getsize(os.path.join(d, "model.ncnn.param")) +
                       os.path.getsize(os.path.join(d, "model.ncnn.bin")),
    }


def load_vc4(size, arena_mb=64):
    import numpy as np
    import vc4infer
    engine = vc4infer.Engine(vc4_model(size), arena_mb=arena_mb,
                             passes="all")
    x = np.random.default_rng(0).standard_normal(
        (1, 3, size, size)).astype(np.float32)

    def run():
        out = engine.run(x)
        arr = list(out.values())[0]
        return {"elems": int(arr.size)}

    return {
        "run": run,
        "input": x,
        "engine": engine,
        "model_bytes": os.path.getsize(vc4_model(size)),
    }
