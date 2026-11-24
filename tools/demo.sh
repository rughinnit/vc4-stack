#!/bin/sh
# demo.sh - the five steps of running a model on the QPUs, with their output.
#
#   tools/demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]
#
# Example:
#   tools/demo.sh ~/models/bccd_v11n320ft_160.onnx ~/models/BloodImage_00007.jpg \
#       160 ~/models/bccd_names.txt
#
# The script takes the slow way on purpose: it converts the ONNX to an artifact
# and then runs the artifact, because that is the path you would deploy.  Every
# step prints what it did, so the output can be read rather than trusted.
set -eu
# set -e: stop at the first failing command.  Without it a refused operator or a
#         missing arena would scroll past and the next step would report on
#         something that was never built.
# set -u: a misspelt variable is an error, not an empty string.

# ---- arguments --------------------------------------------------------------
# $1 the model as ONNX.  A .vc4x is deliberately not accepted: this script is
#    about the conversion, so it builds one.
# $2 an image to run it on.  Any JPEG or PNG the model's task can consume.
# $3 the letterbox size.  It must equal the size the model was exported at;
#    a mismatch changes the anchors and the boxes come out wrong.  Default 160.
# $4 class names, one per line, in the model's class order.  Without it the
#    output says "class 1" instead of "WBC".  Optional.
model=${1:?usage: demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]}
image=${2:?usage: demo.sh MODEL.onnx IMAGE.jpg [IMGSZ] [CLASSES.txt]}
imgsz=${3:-160}
classes=${4:-}

# ---- preconditions ----------------------------------------------------------
# Three failures account for almost every "it did not work": a path typo, the
# tool not installed, and the daemon not running.  Name each one here rather
# than letting it surface three steps later.
[ -f "$model" ] || { echo "no such model: $model" >&2; exit 1; }
[ -f "$image" ] || { echo "no such image: $image" >&2; exit 1; }
command -v vc4infer >/dev/null 2>&1 || { echo "vc4infer not on PATH" >&2; exit 1; }
# --version prints the package, the engine API and the live backend; it fails
# when libvc4infer does not load or vc4d is unreachable.  Cheapest way to know
# that a GPU client could work at all.
vc4infer --version >/dev/null 2>&1 || { echo "the engine does not load" >&2; exit 1; }

# ---- where the outputs go ---------------------------------------------------
# Never into the repository: the artifact is ~25 MiB and the annotated image is
# another file.  $TMPDIR matters on this board because /tmp is tmpfs (RAM) and
# the machine has 1 GB; ~/scratch/tmp is on the SD card.
out=${TMPDIR:-$HOME/scratch/tmp}/demo
mkdir -p "$out"
# Name the artifact after the model and the size, so two sizes of the same
# model do not overwrite each other.
name=$(basename "$model" .onnx)
art=$out/$name-$imgsz.vc4x
written=$out/$name-$imgsz.jpg

say() { printf '\n=== %s\n' "$1"; }

# ---- step 1: what is the graph? ---------------------------------------------
# nodes is a static walk of the graph.  Nothing is executed, no GPU is touched,
# so it costs a second and tells you whether the model is runnable at all.
#
# Read the output as:
#   "N nodes: X supported, Y unsupported"  - Y > 0 means an operator, shape or
#       attribute outside the table; the message names the node.
#   "tiers: vc4 / folded / host-*"          - folded nodes are evaluated at
#       build time (reshapes, constants); host-* would be host work, which the
#       default device policy forbids.
#   "host debt: none"                       - nothing numeric would run on ARM.
say "1. nodes: $model"
vc4infer nodes "$model" | grep -E "nodes:|tiers:|unsupported|host debt" || true
# "|| true" because grep exits 1 when it matches nothing, and set -e would then
# abort a run that actually succeeded.

# ---- step 2: the .vc4x conversion -------------------------------------------
# This is the step that makes deployment cheap.  It runs the optimiser passes,
# packs every weight into the layout the QPU kernel reads, generates and hashes
# the kernel blobs, and freezes both the allocation plan and the execution plan.
#
# VC4_OPT=all is inline because the choice is baked into the artifact, not read
# at run time.  Without it the graph runs unfolded: same answers, many more
# launches, roughly 1.7x slower on this board.
#
# Read the output as:
#   graph:   344 nodes, opt=all (passes: ...)   - the graph after optimisation
#   weights: 91 prepacked buffers, 14.57 MiB    - weights in QPU read order
#   kernels: 133 blobs, 0.64 MiB                - the QPU programs; this is the
#               number that must fit the daemon's program table
#   plan:    peak ... (permanent + temp + live, reused ...)  - memory layout
#   pool / scratch / arena                      - graph segment, scratch beside
#               it, and the whole client arena it was planned against
#   build:   16.4 s (optimise, panels, plan)    - where the time went
#   keys:    generator, abi, board              - what verify checks later
say "2. build: $art"
VC4_OPT=all vc4infer build "$model" -o "$art"

# ---- step 3: is the artifact valid here? ------------------------------------
# Reads the artifact back and checks every section and key against this engine,
# this board and the kernel ABI, without running it.  A package upgrade changes
# the engine digest, so an artifact built before the upgrade is refused here
# instead of running against offsets that moved.
say "3. verify"
vc4infer verify "$art" | tail -2

# ---- step 4: one real inference ---------------------------------------------
# The decode line reads:
#   "525 anchors, 3 classes, channels_first"  - raw predictions before filtering
#   "87 above conf 0.25 -> 14 after NMS"      - confidence filter, then overlap
#       removal at IoU 0.45; the number after NMS is what you would count
#   "boxes in original image pixels"          - mapped back out of the 160x160
#       letterboxed square, so the coordinates fit the input file
# --out sends the annotated image to scratch; without it the tool writes
# <input>_vc4infer.jpg next to the input.
say "4. run: $image at $imgsz"
if [ -n "$classes" ]; then
	vc4infer run "$art" --imgsz "$imgsz" --input "$image" \
		--classes "$classes" --out "$written"
else
	vc4infer run "$art" --imgsz "$imgsz" --input "$image" \
		--out "$written"
fi
echo "annotated image: $written"

# ---- step 5: speed ----------------------------------------------------------
# Five runs after a warm-up; the median is the number to quote.
# bench also prints one util: line, which is where a slow frame is explained:
#   launches=438             - GPU dispatches per frame
#   qpu_duty=54%             - share of the frame the QPUs were computing
#   launch path 239 ms (46%) - client wall time inside launch calls; when this
#       dominates, the daemon or the launch count is the problem, not the maths
say "5. bench, five runs"
vc4infer bench "$art" --imgsz "$imgsz" --reps 5 | grep -E "ms/image|FPS" || true

printf '\nartifact: %s (%s)\n' "$art" "$(du -h "$art" | cut -f1)"
