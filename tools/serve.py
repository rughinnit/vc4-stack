#!/usr/bin/env python3
"""serve.py - keep one engine warm, switch models, answer detections over HTTP.

    tools/serve.py [DIR] [--host 127.0.0.1] [--port 8000] [--token SECRET]
                   [--conf 0.25] [--iou 0.45] [--max-det 300]

DIR is scanned for `.vc4x` artifacts and `.onnx` graphs (default ~/models); the
page lists them, loads one on demand and keeps it warm until you pick another.
Nothing here is tied to a model: the class count comes from the head, COCO-80 is
assumed when the head has 80 classes, a sidecar `<model>.names.txt` overrides
both, and any other count becomes "class N".

Only one model is loaded at a time, because the engine takes the board's single
client arena; switching closes the current engine and opens the next, which
costs one load (about 0.5 s from an artifact, 3 s from ONNX) and one warm-up
frame.

Reaching it:
    board only        tools/serve.py                       (127.0.0.1)
    from a laptop     ssh -L 8000:127.0.0.1:8000 raspberrypi
    same LAN          tools/serve.py --host 0.0.0.0 --token S3CRET
                      then http://<board-ip>:8000/

Endpoints:
    GET  /              the page: pick a model, pick an image, see the boxes
    GET  /models        {"models": [...], "current": "name or null"}
    GET  /health        the loaded model, its size and class count
    POST /select        {"model": "name"} - load that model
    POST /detect        image bytes in, JSON out

Connections are handled in threads so an idle browser preconnect cannot own the
server, but every GPU call takes one lock: the QPU is a single resource, and two
inferences at once would measure each other.
"""
import argparse
import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from vc4infer import Engine
from vc4infer.postprocess import COCO80
from vc4infer.yolo import Detector

LOGIN = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>vc4infer</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee }
 input, button { font-size: 1rem; margin-right: .5rem }
</style></head><body>
<h1>vc4infer</h1>
<p>This server needs its token. It is the one the server was started with
(<code>--token</code>), and it is remembered in this browser afterwards.</p>
<input id="t" type="password" placeholder="token" autofocus>
<button id="go">Open</button>
<script>
document.getElementById('t').value = localStorage.getItem('vc4token') || '';
const go = () => {
  const v = document.getElementById('t').value.trim();
  if (!v) { return; }
  localStorage.setItem('vc4token', v);
  location.href = '/?token=' + encodeURIComponent(v);
};
document.getElementById('go').onclick = go;
document.getElementById('t').onkeydown = e => { if (e.key === 'Enter') go(); };
</script></body></html>
""".encode()

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>vc4infer</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee }
 input, button, select { font-size: 1rem; margin-right: .5rem }
 #row { margin-bottom: 1rem }
 #meta { color: #888; margin-bottom: 1rem }
 #wrap { position: relative; display: inline-block; margin-top: 1rem }
 #wrap canvas { position: absolute; left: 0; top: 0 }
 #out { margin-top: 1rem; white-space: pre; font-family: ui-monospace, monospace }
</style></head><body>
<h1>vc4infer</h1>
<div id="row">
  <select id="m"></select><button id="load">Load</button>
  <input id="f" type="file" accept="image/*"><button id="go">Detect</button>
  <label>conf <input id="conf" type="number" value="0.25" step="0.05" min="0" max="1" style="width:5rem"></label>
  <label><input id="ref" type="checkbox"> compare CPU (ncnn)</label>
</div>
<p id="meta">loading model list...</p>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<div id="out"></div>
<script>
const token = new URLSearchParams(location.search).get('token') || '';
if (token) { localStorage.setItem('vc4token', token); }
else {
  const saved = localStorage.getItem('vc4token');
  if (saved) { location.replace('/?token=' + encodeURIComponent(saved)); }
}
const q = token ? ('?token=' + encodeURIComponent(token)) : '';
const sel = document.getElementById('m'), meta = document.getElementById('meta');
const out = document.getElementById('out');
const img = document.getElementById('img'), cv = document.getElementById('cv');

function show(h) {
  meta.textContent = h.model
    ? h.model + '  at ' + h.imgsz + 'x' + h.imgsz + '  ' + h.classes + ' classes'
    : 'no model loaded';
}

async function refresh() {
  let d;
  try {
    const r = await fetch('/models' + q, {cache: 'no-store'});
    if (r.status === 401) { location.replace('/'); return; }
    d = await r.json();
    if (!r.ok) { meta.textContent = 'error: ' + (d.error || r.status); return; }
  } catch (e) {
    meta.textContent = 'cannot reach the server: ' + e;
    return;
  }
  if (!Array.isArray(d.models)) {
    meta.textContent = 'unexpected reply from /models';
    return;
  }
  sel.innerHTML = '';
  d.models.forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    if (m === d.current) { o.selected = true; }
    sel.appendChild(o);
  });
  if (d.current) { const h = await (await fetch('/health' + q)).json(); show(h); }
  else if (d.models.length) { await load(); }
  else { meta.textContent = 'no models in the served directory'; }
}

async function load() {
  if (!sel.value) { return; }
  meta.textContent = 'loading ' + sel.value + ' ...';
  out.textContent = '';
  let h;
  try {
    const r = await fetch('/select' + q, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: sel.value})});
    h = await r.json();
    if (r.status === 401) { location.replace('/'); return; }
  } catch (e) {
    meta.textContent = 'cannot reach the server: ' + e;
    return;
  }
  if (!h || h.error) { meta.textContent = 'error: ' + ((h || {}).error || 'load failed'); return; }
  show(h);
}

document.getElementById('load').onclick = load;
sel.onchange = load;
document.getElementById('go').onclick = async () => {
  const file = document.getElementById('f').files[0];
  if (!file) { out.textContent = 'choose an image first'; return; }
  out.textContent = 'running...';
  const t0 = performance.now();
  const conf = document.getElementById('conf').value || '0.25';
  let d;
  try {
    const r = await fetch('/detect' + q + '&conf=' + encodeURIComponent(conf),
                          { method: 'POST', body: file });
    if (r.status === 401) { location.replace('/'); return; }
    d = await r.json();
  } catch (e) {
    out.textContent = 'cannot reach the server: ' + e;
    return;
  }
  const total = performance.now() - t0;
  if (!d || d.error) { out.textContent = 'error: ' + ((d || {}).error || 'detect failed'); return; }
  img.src = URL.createObjectURL(file);
  await img.decode();
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  cv.style.width = img.clientWidth + 'px'; cv.style.height = img.clientHeight + 'px';
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.lineWidth = Math.max(2, cv.width / 300);
  ctx.font = Math.max(12, cv.width / 45) + 'px sans-serif';
  d.detections.forEach(b => {
    ctx.strokeStyle = '#3f6'; ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    ctx.fillStyle = '#3f6';
    ctx.fillText(b.name + ' ' + b.score.toFixed(2), b.x1 + 2, Math.max(b.y1 - 4, 12));
  });
  if (document.getElementById('ref').checked) {
    try {
      const rr = await fetch('/reference' + q, { method: 'POST', body: file });
      const rd = await rr.json();
      if (!rd.error) {
        ctx.lineWidth = Math.max(1, cv.width / 500);
        rd.detections.forEach(b => {
          ctx.strokeStyle = '#f80';
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        });
        out.textContent = d.detections.length + ' on VC4, ' + rd.count +
          ' on CPU (ncnn, ' + rd.ms.toFixed(0) + ' ms)';
      } else {
        out.textContent = 'CPU reference: ' + rd.error;
      }
    } catch (e) {
      out.textContent = 'CPU reference failed: ' + e;
    }
    out.textContent += ' | ';
  }
  out.textContent += d.detections.length + ' detections  ' +
    d.timing.server_ms.toFixed(0) + ' ms server  ' +
    total.toFixed(0) + ' ms round trip' +
    (d.detections.length ? '' :
     ' no detections: this model and this image may not match, ' +
     ' or the confidence is too high - try 0.10');
};
refresh();
</script></body></html>
""".encode()


def class_count(engine):
    """Classes in a YOLO detect head: [1, 4+nc, A] or [1, A, 4+nc]."""
    shape = [int(v) for v in engine.outputs[0].shape]
    if len(shape) != 3:
        return None
    a, b = shape[1], shape[2]
    small = a if a < b else b
    return small - 4 if small > 4 else None


def resolve_names(path, nc):
    sidecar = os.path.splitext(path)[0] + ".names.txt"
    if os.path.isfile(sidecar):
        with open(sidecar, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if nc is not None and len(names) != nc:
            sys.stderr.write("serve: warning: %s has %d names for a %d-class head\n"
                             % (os.path.basename(sidecar), len(names), nc))
        return names
    if nc == 80:
        return list(COCO80)
    if nc and nc > 0:
        return ["class %d" % i for i in range(nc)]
    return ["class 0"]


def open_model(path):
    engine = Engine(path)
    nc = class_count(engine)
    names = resolve_names(path, nc)
    det = Detector(engine, names=names)
    return det, names, engine


def close_current():
    """Release the arena before another model is opened."""
    if Handler.detector is not None:
        Handler.detector.close()
        Handler.detector = None
    if Handler.engine is not None:
        Handler.engine.close()
        Handler.engine = None
    Handler.current = None
    Handler.names = None
    gc.collect()


def list_models(directory):
    """Artifacts first, then ONNX, then .pt checkpoints that can be converted."""
    order, stems = [], {}
    for name in sorted(os.listdir(directory)):
        if not os.path.isfile(os.path.join(directory, name)):
            continue
        stem, ext = os.path.splitext(name)
        if ext in (".vc4x", ".onnx", ".pt"):
            stems.setdefault(stem, []).append((ext, name))
    for stem in sorted(stems):
        entries = stems[stem]
        for want in (".vc4x", ".onnx", ".pt"):
            for ext, name in entries:
                if ext == want and name not in order:
                    order.append(name)
                    break
    return order


MODEL_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    # a client that connects and sends nothing must not hold the server: the
    # handler times out and closes instead of waiting for a first line forever
    timeout = 15

    directory = ""
    models = []
    detector = None
    engine = None
    names = None
    current = None
    conf = 0.25
    iou = 0.45
    max_det = 300
    token = None
    pt_imgsz = 320
    convert_python = os.path.expanduser("~/venvs/yolo/bin/python")
    ncnn_dir = None
    ncnn_python = os.path.expanduser("~/venvs/ncnn/bin/python")
    ncnn_imgsz = 160
    ncnn_threads = 2

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: %s\n" % (fmt % args))

    def _route(self):
        return urlparse(self.path).path

    def _authorised(self):
        if not self.token:
            return True
        if self.headers.get("X-Token") == self.token:
            return True
        given = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return given == self.token

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _info(self):
        det = Handler.detector
        if det is None:
            return {"ok": True, "model": None, "imgsz": 0, "classes": 0}
        return {"ok": True, "model": Handler.current, "imgsz": det.imgsz,
                "classes": len(Handler.names),
                "secured": bool(Handler.token)}

    def do_GET(self):
        route = self._route()
        if not self._authorised():
            if route in ("/", "/index.html"):
                self._send(200, LOGIN, "text/html; charset=utf-8")
            else:
                self._json(401, {"error": "missing or wrong token"})
            return
        if route in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif route == "/models":
            # rescan: models are copied into the directory while it runs
            Handler.models = list_models(Handler.directory)
            self._json(200, {"models": Handler.models,
                             "current": Handler.current})
        elif route == "/health":
            self._json(200, self._info())
        else:
            self._json(404, {"error": "no such path"})

    def do_POST(self):
        if not self._authorised():
            self._json(401, {"error": "missing or wrong token"})
            return
        route = self._route()
        if route == "/select":
            self._select()
        elif route == "/detect":
            self._detect()
        elif route == "/reference":
            self._reference()
        else:
            self._json(404, {"error": "no such path"})

    def _select(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            wanted = json.loads(self.rfile.read(n) or b"{}").get("model")
        except Exception:
            self._json(400, {"error": "body must be {\"model\": \"name\"}"})
            return
        Handler.models = list_models(Handler.directory)
        if wanted not in Handler.models:
            self._json(400, {"error": "no such model: %s" % wanted})
            return
        path = os.path.join(Handler.directory, wanted)
        previous = Handler.current
        converted_ms = None

        if wanted.endswith(".pt"):
            # the engine cannot read a checkpoint, and the conversion needs the
            # one client arena, so the current model is closed first
            stem = os.path.splitext(wanted)[0]
            artifact = os.path.join(Handler.directory,
                                    "%s-pt%d.vc4x" % (stem, Handler.pt_imgsz))
            if not os.path.isfile(artifact):
                close_current()
                t0 = time.perf_counter()
                cmd = [Handler.convert_python, "-m", "vc4infer.cli", "build",
                       path, "--imgsz", str(Handler.pt_imgsz), "-o", artifact]
                env = dict(os.environ, VC4_OPT="all")
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if r.returncode != 0:
                    self._json(500, {"error": "conversion failed: "
                                     + (r.stderr or r.stdout).strip().splitlines()[-1]
                                     if (r.stderr or r.stdout) else "conversion failed"})
                    return
                converted_ms = (time.perf_counter() - t0) * 1e3
            wanted_path = artifact
            label = wanted
        else:
            wanted_path = path
            label = wanted
        try:
            with MODEL_LOCK:
                close_current()       # the arena is released before the next open
                t0 = time.perf_counter()
                det, names, engine = open_model(wanted_path)
        except Exception as exc:
            close_current()
            if previous:
                try:
                    det, names, engine = open_model(
                        os.path.join(Handler.directory, previous))
                    Handler.detector, Handler.names = det, names
                    Handler.engine, Handler.current = engine, previous
                except Exception:
                    pass
            self._json(500, {"error": "cannot load %s: %s" % (wanted, exc)})
            return
        Handler.detector, Handler.names = det, names
        Handler.engine, Handler.current = engine, label
        info = {**self._info(), "load_ms": (time.perf_counter() - t0) * 1e3}
        if converted_ms is not None:
            info["converted_ms"] = converted_ms
            info["from"] = wanted
        self._json(200, info)

    def _reference(self):
        """The same image through ncnn on the CPU, as an independent baseline."""
        if not Handler.ncnn_dir:
            self._json(400, {"error": "no CPU reference configured "
                                      "(start serve.py with --ncnn-dir)"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
        except Exception as exc:
            self._json(400, {"error": str(exc)})
            return
        fd, path = tempfile.mkstemp(suffix=".jpg", dir=os.environ.get("TMPDIR"))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
            cmd = [Handler.ncnn_python, Handler.ncnn_script, path,
                   Handler.ncnn_dir, "--imgsz", str(Handler.ncnn_imgsz),
                   "--threads", str(Handler.ncnn_threads), "--json"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                self._json(500, {"error": "ncnn failed: "
                                 + (r.stderr or r.stdout).strip().splitlines()[-1]})
                return
            self._json(200, json.loads(r.stdout))
        except Exception as exc:
            self._json(500, {"error": "ncnn failed: %s" % exc})
        finally:
            os.unlink(path)

    def _detect(self):
        det = Handler.detector
        if det is None:
            self._json(409, {"error": "no model loaded; POST /select first"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            image = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as exc:
            self._json(400, {"error": "not an image: %s" % exc})
            return
        query = parse_qs(urlparse(self.path).query)
        try:
            conf = float(query.get("conf", [Handler.conf])[0])
            iou = float(query.get("iou", [Handler.iou])[0])
        except ValueError:
            self._json(400, {"error": "conf and iou must be numbers"})
            return
        t0 = time.perf_counter()
        try:
            with MODEL_LOCK:
                d = det.predict(image, conf=conf, iou=iou,
                                max_det=Handler.max_det)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        ms = (time.perf_counter() - t0) * 1e3
        boxes = np.asarray(d.boxes, dtype=float)
        names = Handler.names
        self._json(200, {
            "model": Handler.current,
            "detections": [
                {"x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                 "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2),
                 "score": round(float(s), 4), "class": int(c),
                 "name": names[int(c)] if int(c) < len(names) else "class %d" % int(c)}
                for b, s, c in zip(boxes, d.scores, d.classes)],
            "count": int(len(boxes)),
            "timing": {"server_ms": ms, **d.speed_ms},
        })


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?",
                    default=os.path.expanduser("~/models"),
                    help="directory to scan for .vc4x and .onnx (default ~/models)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", help="require ?token=SECRET or an X-Token header")
    ap.add_argument("--pt-imgsz", type=int, default=320,
                    help="export size used when a .pt is converted (default 320)")
    ap.add_argument("--convert-python",
                    default=os.path.expanduser("~/venvs/yolo/bin/python"),
                    help="interpreter with Ultralytics, for .pt conversion")
    ap.add_argument("--ncnn-dir",
                    help="an Ultralytics ncnn export; enables POST /reference")
    ap.add_argument("--ncnn-python",
                    default=os.path.expanduser("~/venvs/ncnn/bin/python"))
    ap.add_argument("--ncnn-imgsz", type=int, default=160)
    ap.add_argument("--ncnn-threads", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--load", help="model name to load at start")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.directory):
        sys.exit("not a directory: %s" % args.directory)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        sys.stderr.write("serve: warning: binding %s without --token; anyone on "
                         "that network can drive the GPU and hold the arena\n"
                         % args.host)

    Handler.directory = args.directory
    Handler.models = list_models(args.directory)
    Handler.conf = args.conf
    Handler.iou = args.iou
    Handler.max_det = args.max_det
    Handler.token = args.token
    Handler.pt_imgsz = args.pt_imgsz
    Handler.convert_python = args.convert_python
    Handler.ncnn_dir = args.ncnn_dir
    Handler.ncnn_python = args.ncnn_python
    Handler.ncnn_imgsz = args.ncnn_imgsz
    Handler.ncnn_threads = args.ncnn_threads
    Handler.ncnn_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "ncnn_reference.py")

    start = args.load
    if start is None:
        for name in Handler.models:
            if name.endswith(".vc4x"):
                start = name
                break
    if start:
        t0 = time.perf_counter()
        det, names, engine = open_model(os.path.join(args.directory, start))
        Handler.detector, Handler.names = det, names
        Handler.engine, Handler.current = engine, start
        sys.stderr.write("serve: loaded %s, %dx%d, %d classes in %.0f ms\n"
                         % (start, det.imgsz, det.imgsz, len(names),
                            (time.perf_counter() - t0) * 1e3))

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    sys.stderr.write("serve: %d model(s) in %s\n" % (len(Handler.models), args.directory))
    sys.stderr.write("serve: http://%s:%d/%s\n"
                     % (args.host, args.port,
                        "?token=SECRET" if args.token else ""))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        close_current()
    return 0


if __name__ == "__main__":
    sys.exit(main())
