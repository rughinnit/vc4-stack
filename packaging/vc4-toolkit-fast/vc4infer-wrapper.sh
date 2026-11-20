#!/bin/sh
# vc4infer for vc4-toolkit-fast: seed the per-user kernel cache from the
# read-only, prewarmed /usr/lib/vc4infer/kcache, then run the packaged CLI.
#
# The engine has one writable cache directory.  The fast package ships every
# blob the bundled models need in a read-only system cache; copying them into
# the user's cache on first use means a fresh install runs the bundled models
# without JIT (no python3 kgen.py spawn) while a non-bundled model can still
# JIT into the user's own directory.
#
# VC4INFER_CACHE set explicitly disables the seeding: setting it explicitly is the supported way to disable it
# directory, so it is left alone.
set -u

sys=/usr/lib/vc4infer/kcache
if [ -z "${VC4INFER_CACHE:-}" ] && [ -d "$sys" ]; then
	base=${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}
	usr=$base/vc4infer/kcache
	# the marker carries the system cache's identity, not just the package
	# version: a reinstall of the same version can ship a different blob set
	stamp=$(stat -c '%Y-%s' "$sys/MANIFEST" 2>/dev/null || echo none)
	marker=$usr/.seeded-@VERSION@-$stamp
	if [ ! -e "$marker" ]; then
		if mkdir -p "$usr" 2>/dev/null; then
			cp -a "$sys"/. "$usr"/ 2>/dev/null || true
			chmod -R u+w "$usr" 2>/dev/null || true
			: > "$marker" 2>/dev/null || true
		fi
	fi
fi

exec python3 /usr/lib/vc4infer/vc4infer-cli "$@"
