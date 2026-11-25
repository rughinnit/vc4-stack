#!/usr/bin/env python3
"""serve.py - keep one engine warm and answer detections over HTTP.

    tools/serve.py MODEL[.vc4x] [--host 127.0.0.1] [--port 8000]
                   [--conf 0.25] [--iou 0.45] [--names FILE]

The overhead this removes is the one that dominates a per-image CLI loop: every
`vc4infer run` pays the model load (0.5 s from an artifact, 3 s from ONNX), the
kernel cache check and the first-frame packing.  Here the engine is opened once
and every request pays only letterbox, inference and decode - 519 ms for
YOLO11n-160, 25 ms for the 8-conv student at 96.

Endpoints:
    GET  /              a page with a file picker that draws the boxes
    GET  /health        {"ok": true, "model": ..., "imgsz": ...}
    POST /detect        image bytes in, JSON out

The server is single-threaded on purpose: the QPU is one resource, and two
requests in flight would measure each other.  It binds to 127.0.0.1 unless told
otherwise, and it has no authentication - do not expose it beyond the board.
"""
import argparse
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

import vc4infer
from vc4infer.yolo import Detector

PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>vc4infer</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee }
 input, button { font-size: 1rem; margin-right: .5rem }
 #wrap { position: relative; display: inline-block; margin-top: 1rem }
 #wrap canvas { position: absolute; left: 0; top: 0 }
 #out { margin-top: 1rem; white-space: pre; font-family: ui-monospace, monospace }
</style></head><body>
<h1>vc4infer</h1>
<p>Pick an image. Boxes come back from the engine on the QPUs.</p>
<input id="f" type="file" accept="image/*">
<button id="go">Detect</button>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<div id="out"></div>
<script>
const img = document.getElementById('img'), cv = document.getElementById('cv');
const out = document.getElementById('out');
document.getElementById('go').onclick = async () => {
  const file = document.getElementById('f').files[0];
  if (!file) { out.textContent = 'choose an image first'; return; }
  const t0 = performance.now();
  const r = await fetch('/detect', { method: 'POST', body: file });
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
    ctx.fillText(b.name + ' ' + b.score.toFixed(2), b.x1 + 2, b.y1 - 4);
  });
  out.textContent = d.detections.length + ' detections  ' +
    d.timing.server_ms.toFixed(0) + ' ms server  ' +
    total.toFixed(0) + ' ms round trip';
};
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    detector = None
    names = None
    conf = 0.25
    iou = 0.45
    model = ""

    def log_message(self, fmt, *args):
        sys.stderr.write("serve: %s\n" % (fmt % args))

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"ok": True, "model": self.model,
                             "imgsz": self.detector.imgsz})
        else:
            self._json(404, {"error": "no such path"})

    def do_POST(self):
        if self.path != "/detect":
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
            det = self.detector.predict(image, conf=self.conf, iou=self.iou)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        ms = (time.perf_counter() - t0) * 1e3
        boxes = np.asarray(det.boxes, dtype=float)
        self._json(200, {
            "detections": [
                {"x1": round(float(b[0]), 2), "y1": round(float(b[1]), 2),
                 "x2": round(float(b[2]), 2), "y2": round(float(b[3]), 2),
                 "score": round(float(s), 4),
                 "class": int(c), "name": self.names[int(c)]}
                for b, s, c in zip(boxes, det.scores, det.classes)],
            "count": int(len(boxes)),
            "timing": {"server_ms": ms, **det.speed_ms},
        })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .vc4x artifact (recommended) or an ONNX graph")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--names", help="class names, one per line")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.model):
        sys.exit("no such model: %s" % args.model)
    names = None
    if args.names:
        with open(args.names, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]

    t0 = time.perf_counter()
    detector = Detector(args.model, names=names)
    load_ms = (time.perf_counter() - t0) * 1e3

    Handler.detector = detector
    Handler.names = detector.names
    Handler.conf = args.conf
    Handler.iou = args.iou
    Handler.model = args.model

    srv = HTTPServer((args.host, args.port), Handler)
    sys.stderr.write("serve: %s loaded in %.0f ms, %dx%d, %s:%d\n"
                     % (args.model, load_ms, detector.imgsz, detector.imgsz,
                        args.host, args.port))
    sys.stderr.write("serve: open http://%s:%d/ or POST an image to /detect\n"
                     % (args.host, args.port))
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
