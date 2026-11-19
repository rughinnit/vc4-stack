#!/usr/bin/env python3
"""Run the stock Ultralytics pipeline with its convolutions on the VC4 QPUs.

    python3 infer/examples/ultralytics_predict.py --model yolov8n.onnx --image bus.jpg

The only vc4-specific line is ``vc4infer.contrib.ultralytics.register()``: it
points Ultralytics' ``.onnx`` backend at :class:`vc4infer.Engine`.  Everything
after it -- ``YOLO(path)``, the letterbox, ``non_max_suppression``,
``Results.plot()`` -- is stock Ultralytics.

Ultralytics has no entry-point plugin API in 8.4.x; the format is derived from
the file suffix before the backend map is consulted, so ``format="vc4"`` would
need a fork of ``guess_model_type``/``export_formats``.  The registry entry is
the supported minimal patch and is undone by ``unregister()``.
"""
import argparse
import os
import sys

try:                                    # in-tree checkout: package in infer/python
    import vc4infer  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8n.onnx")
    ap.add_argument("--image", default="bus.jpg")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="bus_ultralytics_vc4.jpg")
    ap.add_argument("--restore", action="store_true",
                    help="unregister the backend when done")
    a = ap.parse_args()

    try:
        import vc4infer.contrib.ultralytics as vu
    except ImportError as e:
        raise SystemExit("ultralytics integration unavailable: %s" % e)

    previous = vu.register()
    print("registered vc4 for .onnx (replaced %s)"
          % getattr(previous, "__name__", previous))
    try:
        from ultralytics import YOLO
        model = YOLO(a.model)
        res = model.predict(a.image, imgsz=a.imgsz, conf=a.conf, verbose=False)
        r = res[0]
        print("backend:", type(model.predictor.model.backend).__name__)
        print("detections: %d" % len(r.boxes))
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                  r.boxes.cls.tolist()):
            print("  %-12s %.3f  %s" % (model.names[int(cls)], conf,
                                        [round(v, 1) for v in box]))
        r.save(filename=a.out)
        print("saved", a.out)
    finally:
        if a.restore:
            vu.unregister(previous)
            print("restored", getattr(previous, "__name__", previous))


if __name__ == "__main__":
    main()
