#!/bin/sh
# demo.sh - the five steps of running a model on the QPUs, with their output.
#
#   tools/demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]
#
#   1. nodes   what the graph is and whether anything is refused
#   2. build   the .vc4x conversion: optimiser, prepacked weights, kernels, plans
#   3. verify  every section and key against this engine
#   4. run     detections on one image
#   5. bench   timed, five runs
#
# The artifact goes to $TMPDIR (default ~/scratch/tmp), never into the tree.
# One client at a time: the steps run in sequence, and nothing else may hold the
# GPU while this runs.
set -eu

model=${1:?usage: demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]}
image=${2:?usage: demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]}
imgsz=${3:-160}
classes=${4:-}

[ -f "$model" ] || { echo "no such model: $model" >&2; exit 1; }
[ -f "$image" ] || { echo "no such image: $image" >&2; exit 1; }
command -v vc4infer >/dev/null 2>&1 || { echo "vc4infer not on PATH" >&2; exit 1; }
vc4infer --version >/dev/null 2>&1 || { echo "the engine does not load" >&2; exit 1; }

out=${TMPDIR:-$HOME/scratch/tmp}/demo
mkdir -p "$out"
name=$(basename "$model" .onnx)
art=$out/$name-$imgsz.vc4x
written=$out/$name-$imgsz.jpg

say() { printf '\n=== %s\n' "$1"; }

say "1. nodes: $model"
vc4infer nodes "$model" | grep -E "nodes:|tiers:|unsupported|host debt" || true

say "2. build: $art"
VC4_OPT=all vc4infer build "$model" -o "$art"

say "3. verify"
vc4infer verify "$art" | tail -2

say "4. run: $image at $imgsz"
if [ -n "$classes" ]; then
	vc4infer run "$art" --imgsz "$imgsz" --input "$image" \
		--classes "$classes" --out "$written"
else
	vc4infer run "$art" --imgsz "$imgsz" --input "$image" \
		--out "$written"
fi
echo "annotated image: $written"

say "5. bench, five runs"
vc4infer bench "$art" --imgsz "$imgsz" --reps 5 | grep -E "ms/image|FPS" || true

printf '\nartifact: %s (%s)\n' "$art" "$(du -h "$art" | cut -f1)"
