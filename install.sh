#!/bin/sh
# install.sh - install the whole VC4 stack on a Raspberry Pi (VideoCore IV).
#
#   curl -fsSL https://rughinnit.github.io/vc4-apt/install.sh | sudo sh
#   sudo sh install.sh --fast          # speed-tuned build, prewarmed kernels
#   sudo sh install.sh --pin-clocks    # floor the core clock at its stock 400 MHz
#   sudo sh install.sh --dry-run       # print what would happen
#   sudo sh install.sh --uninstall     # remove everything it installed
#
# What it installs: the signed apt repository, then the meta-package
# (vc4-toolkit or vc4-toolkit-fast), which brings vc4d, libvc4infer/vc4infer,
# the Python package, vc4-query, vc4-smi, vc4train, vc4top and the vc4mon DKMS
# module.  It then verifies the install and prints the measured state.
#
# It does NOT change voltage or overclock.  --pin-clocks sets core_freq and
# core_freq_min to 400, which is this board's stock maximum: the firmware's
# DVFS cannot see QPU work issued through vc4d and otherwise downclocks to
# 250 MHz mid-run (docs/CLOCK-VARIANCE.md).

set -eu

BASE_URL=${VC4_BASE_URL:-https://rughinnit.github.io/vc4-apt}
KEYRING=/usr/share/keyrings/vc4top-archive-keyring.gpg
LIST=/etc/apt/sources.list.d/vc4-apt.list
CONFIG=/boot/firmware/config.txt
LOG=/var/log/vc4-install.log

MODE=standard
PIN_CLOCKS=0
WITH_TORCH=0
DO_VERIFY=1
ASSUME_YES=0
DRY_RUN=0
UNINSTALL=0

usage() {
	cat <<'EOF'
install.sh - install the VC4 stack (Raspberry Pi VideoCore IV)

  --fast          install vc4-toolkit-fast (source-built, prewarmed kernel cache)
  --torch         also install the torch_vc4 wheel (requires torch already present)
  --pin-clocks    set core_freq=400 and core_freq_min=400 (stock maximum, not OC)
  --no-verify     skip the post-install selftest
  --uninstall     purge the packages and remove the apt source
  --dry-run       print the commands without running them
  -y, --yes       do not ask before removing/replacing packages

Environment: VC4_BASE_URL overrides the repository URL (for a mirror).
EOF
}

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
	if [ "$DRY_RUN" = 1 ]; then
		printf '   [dry-run] %s\n' "$*"
		return 0
	fi
	printf '   %s\n' "$*" >>"$LOG" 2>/dev/null || true
	"$@"
}

# Run as root, but remember who asked so the user can be added to the group.
if [ "$(id -u)" -ne 0 ]; then
	[ -x "$(command -v sudo)" ] || die "run as root (sudo not found)"
	INVOKER=${USER:-$(id -un)}
	exec sudo -E sh "$0" "$@"
fi
INVOKER=${SUDO_USER:-${INVOKER:-root}}

while [ $# -gt 0 ]; do
	case $1 in
	--fast)       MODE=fast ;;
	--torch)      WITH_TORCH=1 ;;
	--pin-clocks) PIN_CLOCKS=1 ;;
	--no-verify)  DO_VERIFY=0 ;;
	--uninstall)  UNINSTALL=1 ;;
	--dry-run)    DRY_RUN=1 ;;
	-y|--yes)     ASSUME_YES=1 ;;
	-h|--help)    usage; exit 0 ;;
	*)            usage >&2; die "unknown option: $1" ;;
	esac
	shift
done

if [ "$MODE" = fast ]; then
	META=vc4-toolkit-fast
else
	META=vc4-toolkit
fi

# ---------------------------------------------------------------- preflight
say "checking this machine"

[ "$(uname -m)" = aarch64 ] || warn "architecture is $(uname -m), not aarch64; the packages are arm64"

MODEL=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)
case $MODEL in
*"Raspberry Pi"*) : ;;
*) warn "not detected as a Raspberry Pi (model: $MODEL)" ;;
esac

case $MODEL in
*"Raspberry Pi 4"*|*"Raspberry Pi 400"*|*"Compute Module 4"*)
	warn "this is a VideoCore VI board; the stack targets VideoCore IV (Pi 1/2/3, Zero, CM1/CM3)" ;;
esac

if ! command -v vcgencmd >/dev/null 2>&1; then
	warn "vcgencmd not found (raspberrypi-userland tools); telemetry will be limited"
fi

if [ -r /proc/meminfo ]; then
	cma_kb=$(awk '/^CmaTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
	[ -n "$cma_kb" ] || cma_kb=0
	if [ "$cma_kb" -lt 131072 ] 2>/dev/null; then
		warn "CMA is $((cma_kb / 1024)) MiB; 320x320 inference wants >= 128 MiB (dtoverlay=vc4-kms-v3d,cma-256)"
	fi
fi

if [ -d /sys/module/vc4 ] || [ -e /sys/class/drm/card0 ]; then
	: # KMS/vc4 present
else
	warn "vc4/KMS does not look loaded; the QPUs are reached through vc4d, which needs the V3D device"
fi

AVAIL_KB=$(df -Pk /usr 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${AVAIL_KB:-}" ] && [ "$AVAIL_KB" -lt 262144 ] 2>/dev/null; then
	warn "less than 256 MiB free on /usr; the packages need room"
fi

say "model: $MODEL"
say "mode:  $MODE ($META), clocks: $([ "$PIN_CLOCKS" = 1 ] && echo 'pin to stock 400 MHz' || echo 'unchanged')"

# ---------------------------------------------------------------- uninstall
if [ "$UNINSTALL" = 1 ]; then
	say "uninstalling"
	pkgs="vc4-toolkit-fast vc4-toolkit vc4infer vc4infer-dev vc4-query vc4-smi vc4train vc4top vc4top-dkms vc4d"
	if [ "$ASSUME_YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
		printf 'Purge %s? [y/N] ' "$pkgs"
		read -r ans
		case $ans in y|Y|yes|YES) : ;; *) die "aborted" ;; esac
	fi
	run apt-get purge -y $pkgs || true
	run apt-get autoremove -y || true
	run rm -f "$LIST"
	run rm -f "$KEYRING"
	warn "the torch_vc4 wheel, if pip-installed, is left in place (pip uninstall torch_vc4)"
	say "done"
	exit 0
fi

# ---------------------------------------------------------------- repository
say "installing the signed apt repository"
run apt-get update -qq || true
if ! command -v curl >/dev/null 2>&1; then
	run apt-get install -y ca-certificates curl
fi

if [ -f "$KEYRING" ] && [ -f "$LIST" ]; then
	say "repository already configured"
else
	tmp=$(mktemp -d "${TMPDIR:-/tmp}/vc4-install.XXXXXX")
	trap 'rm -rf "$tmp"' EXIT
	run curl -fsSL "$BASE_URL/key.asc" -o "$tmp/key.asc"
	run gpg --dearmor -o "$tmp/keyring.gpg" "$tmp/key.asc"
	run install -m 0644 "$tmp/keyring.gpg" "$KEYRING"
	run sh -c "printf '%s\n' 'deb [signed-by=$KEYRING] $BASE_URL stable main' >'$LIST'"
	say "key fingerprint: $(gpg --show-keys --with-fingerprint "$tmp/key.asc" 2>/dev/null | sed -n '2p' | tr -s ' ')"
fi
run apt-get update -qq

# ---------------------------------------------------------------- packages
say "installing $META"
if [ "$DRY_RUN" != 1 ]; then
	apt-get install -y "$META" >>"$LOG" 2>&1 || {
		tail -20 "$LOG" >&2
		die "package install failed (log: $LOG)"
	}
else
	printf '   [dry-run] apt-get install -y %s\n' "$META"
fi

say "enabling the privileged helper"
run systemctl enable vc4d
# Do not restart a healthy daemon: a restart mid-run kills every client's
# launch, and other work on the board may be in flight.
if ! systemctl is-active --quiet vc4d; then
	run systemctl start vc4d
fi

if [ "$INVOKER" != root ] && id "$INVOKER" >/dev/null 2>&1; then
	if ! id -nG "$INVOKER" | tr ' ' '\n' | grep -qx vc4; then
		say "adding $INVOKER to the vc4 group (log out and back in to use it)"
		run usermod -aG vc4 "$INVOKER"
	fi
fi

# ---------------------------------------------------------------- clocks
if [ "$PIN_CLOCKS" = 1 ]; then
	say "pinning the core clock to its stock 400 MHz"
	if grep -q '^core_freq_min=' "$CONFIG" 2>/dev/null; then
		say "already pinned"
	else
		run sh -c "cat >>'$CONFIG' <<'EOF'

# Pin the core (QPU) clock to its stock 400 MHz.  The firmware DVFS cannot see
# QPU work issued through vc4d and otherwise drops to 250 MHz mid-run.  This is
# the stock maximum: no voltage change, no overclock.
core_freq=400
core_freq_min=400
EOF"
		warn "reboot required for the clock floor to take effect"
	fi
fi

# ---------------------------------------------------------------- torch
if [ "$WITH_TORCH" = 1 ]; then
	say "installing torch_vc4"
	if ! python3 -c 'import torch' >/dev/null 2>&1; then
		warn "torch is not importable; install it first, then:"
		warn "  pip install --extra-index-url $BASE_URL/pip/simple/ torch-vc4"
	else
		run python3 -m pip install --extra-index-url "$BASE_URL/pip/simple/" torch-vc4
	fi
fi

# ---------------------------------------------------------------- verify
if [ "$DO_VERIFY" = 1 ] && [ "$DRY_RUN" != 1 ]; then
	say "verifying"
	vc4-query 2>&1 | sed 's/^/   /' || warn "vc4-query failed"
	vc4-smi 2>&1 | sed 's/^/   /' || warn "vc4-smi failed"
	vc4infer --version 2>&1 | sed 's/^/   /' || warn "vc4infer --version failed"
	say "running the engine selftest (four models, tolerance 5e-4)"
	if vc4infer selftest >/dev/null 2>&1; then
		echo "   selftest: PASS (4/4 cases, 0 elements over tolerance)"
	else
		warn "selftest FAILED - run 'vc4infer selftest' to see which case"
	fi
	say "training demo (64-32-10 MLP, 25 steps)"
	if vc4train --steps 25 >/dev/null 2>&1; then
		echo "   vc4train: PASS"
	else
		warn "vc4train failed - run 'vc4train' to see why"
	fi
fi

# ---------------------------------------------------------------- summary
cat <<EOF

$(say "installed")
   packages    $META
   daemon      $(systemctl is-active vc4d 2>/dev/null || echo unknown)
   python      $(vc4infer --version 2>/dev/null | head -1 || echo 'vc4infer not runnable')
   log         $LOG

Run-time backend (same kernels, same results):
   vc4infer --backend managed ...   unprivileged through vc4d, ~0.1 ms/launch (default)
   sudo -E vc4infer --backend direct ...   root + /dev/mem, no launch overhead
   Measured on YOLOv8n 160: 492 ms (managed) vs 412 ms (direct); see docs/BACKENDS.md

Next steps:
   vc4infer run model.onnx --input image.jpg --imgsz 160
   vc4infer util model.onnx --reps 5
   vc4infer selftest
   vc4train
   vc4top
   vc4-smi

Documentation (from a checkout of the stack):
   docs/TUTORIAL.md   install -> first model, PyTorch and YOLO
   docs/INSTALL.md    both install paths in detail
   docs/ECOSYSTEM.md  what each repository owns
EOF
