#!/usr/bin/env python3
"""serve.py - keep one engine warm and answer detections over HTTP.

    tools/serve.py MODEL[.vc4x] [--host 127.0.0.1] [--port 8000]
                   [--names FILE] [--conf 0.25] [--iou 0.45] [--max-det 300]
                   [--token SECRET]

Any YOLO-style detection export works: a `.vc4x` artifact (recommended) or the
`.onnx` it was built from, at whatever square size the model was exported for.
Class names come from `--names` if given, from COCO-80 when the head has 80
classes, and from the class index otherwise.

Reaching it:
    board only        tools/serve.py m.vc4x                       (127.0.0.1)
    from a laptop     ssh -L 8000:127.0.0.1:8000 raspberrypi
    same LAN          tools/serve.py m.vc4x --host 0.0.0.0 --token S3CRET
                      then http://<board-ip>:8000/?token=S3CRET

`--token` is the only access control there is: without it, anyone who can reach
the port can use the GPU, and while the server runs it holds the one client
arena, so nothing else on the board can.  Use the tunnel when you can, and a
token when you cannot.

Endpoints:
    GET  /              a page that takes a file and draws the boxes
    GET  /health        {"ok": true, "model": ..., "imgsz": ..., "classes": n}
    POST /detect        image bytes in, JSON out

Single-threaded on purpose: the QPU is one resource, and two requests in flight
would measure each other.
"""
import argparse
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from vc4infer import Engine
from vc4infer.postprocess import COCO80
from vc4infer.yolo import Detector

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>vc4infer</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee }
 input, button { font-size: 1rem; margin-right: .5rem }
 #meta { color: #888; margin-bottom: 1rem }
 #wrap { position: relative; display: inline-block; margin-top: 1rem }
 #wrap canvas { position: absolute; left: 0; top: 0 }
 #out { margin-top: 1rem; white-space: pre; font-family: ui-monospace, monospace }
</style></head><body>
<h1>vc4infer</h1>
<p id="meta"></p>
<p>Pick an image. The boxes come from the QPUs.</p>
<input id="f" type="file" accept="image/*">
<button id="go">Detect</button>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<div id="out"></div>
<script>
const token = new URLSearchParams(location.search).get('token') || '';
const q = token ? ('?token=' + encodeURIComponent(token)) : '';
const img = document.getElementById('img'), cv = document.getElementById('cv');
const out = document.getElementById('out'), meta = document.getElementById('meta');
fetch('/health' + q).then(r => r.json()).then(h => {
  meta.textContent = h.ok
    ? h.model + '  modelled at ' + h.imgsz + 'x' + h.imgsz + '  ' + h.classes + ' classes'
    : 'health: ' + (h.error || 'unknown error');
});
document.getElementById('go').onclick = async () => {
  const file = document.getElementById('f').files[0];
  if (!file) { out.textContent = 'choose an image first'; return; }
  out.textContent = 'running...';
  const t0 = performance.now();
  const r = await fetch('/detect' + q, { method: 'POST', body: file });
  const d = await r.json();
  const total = performance.now() - t0;
  if (d.error) { out.textContent = 'error: ' + d.error; return; }
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
  out.textContent = d.detections.length + ' detections  ' +
    d.timing.server_ms.toFixed(0) + ' ms server  ' +
    total.toFixed(0) + ' ms round trip';
};
</script></body></html>
""".encode()


class Handler(BaseHTTPRequestHandler):
    detector = None
    names = None
    conf = 0.25
    iou = 0.45
    max_det = 300
    model = ""
    token = None

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: %s\n" % (fmt % args))

    def _route(self):
        """Path with the query string stripped."""
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

    def do_GET(self):
        if not self._authorised():
            self._json(401, {"error": "missing or wrong token"})
            return
        route = self._route()
        if route in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif route == "/health":
            det = self.detector
            self._json(200, {"ok": True, "model": os.path.basename(self.model),
                             "imgsz": det.imgsz, "classes": len(self.names),
                             "secured": bool(self.token)})
        else:
            self._json(404, {"error": "no such path"})

    def do_POST(self):
        if not self._authorised():
            self._json(401, {"error": "missing or wrong token"})
            return
        if self._route() != "/detect":
            self._json(404, {"error": "no such path"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            image = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as exc:
            self._json(400, {"error": "not an image: %s" % exc})
            return
        t0 = time.perf_counter()
        try:
            det = self.detector.predict(image, conf=self.conf, iou=self.iou,
                                        max_det=self.max_det)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        ms = (time.perf_counter() - t0) * 1e3
        boxes = np.asarray(det.boxes, dtype=float)
        self._json(200, {
            "detections": [
                {"x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                 "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2),
                 "score": round(float(s), 4), "class": int(c),
                 "name": self.names[int(c)] if int(c) < len(self.names)
                         else "class %d" % int(c)}
                for b, s, c in zip(boxes, det.scores, det.classes)],
            "count": int(len(boxes)),
            "timing": {"server_ms": ms, **det.speed_ms},
        })


def class_count(engine):
    """Classes in a YOLO detect head: [1, 4+nc, A] or [1, A, 4+nc]."""
    shape = [int(v) for v in engine.outputs[0].shape]
    if len(shape) != 3:
        return None
    a, b = shape[1], shape[2]
    small = a if a < b else b
    return small - 4 if small > 4 else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .vc4x artifact (recommended) or an ONNX graph")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; 0.0.0.0 for the LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--names", help="class names, one per line; omit for COCO-80 "
                                    "when the head has 80 classes")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--token", help="require ?token=SECRET or an X-Token header; "
                                    "the only access control there is")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.model):
        sys.exit("no such model: %s" % args.model)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        sys.stderr.write("serve: warning: binding %s without --token; anyone on "
                         "that network can drive the GPU and hold the arena\n"
                         % args.host)

    engine = Engine(args.model)
    nc = class_count(engine)

    if args.names:
        with open(args.names, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if nc is not None and len(names) != nc:
            sys.stderr.write("serve: warning: %d names for a %d-class head\n"
                             % (len(names), nc))
    elif nc == 80:
        names = list(COCO80)
    else:
        n = nc if nc and nc > 0 else 1
        names = ["class %d" % i for i in range(n)]
        if nc is None:
            sys.stderr.write("serve: cannot tell the class count from the head; "
                             "pass --names\n")

    t0 = time.perf_counter()
    detector = Detector(engine, names=names)
    load_ms = (time.perf_counter() - t0) * 1e3

    Handler.detector = detector
    Handler.names = names
    Handler.conf = args.conf
    Handler.iou = args.iou
    Handler.max_det = args.max_det
    Handler.model = args.model
    Handler.token = args.token

    srv = HTTPServer((args.host, args.port), Handler)
    sys.stderr.write("serve: %s, %dx%d, %d classes, opened in %.0f ms%s\n"
                     % (os.path.basename(args.model), detector.imgsz,
                        detector.imgsz, len(names), load_ms,
                        ", token required" if args.token else ""))
    sys.stderr.write("serve: http://%s:%d/%s\n"
                     % (args.host, args.port,
                        "?token=SECRET" if args.token else ""))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
