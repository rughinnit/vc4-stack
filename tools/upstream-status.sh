#!/bin/sh
# upstream-status.sh - how far each forked project is from the tree it forks.
#
#   tools/upstream-status.sh            report from what is already fetched
#   tools/upstream-status.sh --fetch    fetch upstream first (network)
#
# The convention these repositories follow, and docs/UPSTREAM.md explains:
# `origin` is our fork, `upstream` is the project we forked, and the local
# branch `upstream-master` is a read-only mirror of `upstream/master` that is
# never committed to.  Work happens on `master` or on feature branches, and it
# is merged with upstream, never written over it.
#
# Exit 0 when every checked repository is in sync or ahead only, 1 when any is
# behind upstream, so a cron entry or a `set -e` script can use it.
set -eu

fetch=0
[ "${1:-}" = "--fetch" ] && fetch=1

repos=${VC4_UPSTREAM_REPOS:-"VC4CL VC4C VC4CLStdLib"}
root=${VC4_UPSTREAM_ROOT:-$HOME/vc4cl}
behind_any=0

printf '%-12s %7s %6s  %-26s %s\n' repo behind ahead branch note
printf '%-12s %7s %6s  %-26s %s\n' ---- ------ ----- ------ ----

for name in $repos; do
	dir=$root/$name
	if [ ! -d "$dir/.git" ]; then
		printf '%-12s %7s %6s  %-26s %s\n' "$name" - - - "not cloned"
		continue
	fi
	if [ "$fetch" = 1 ]; then
		git -C "$dir" fetch -q upstream 2>/dev/null || \
			git -C "$dir" fetch -q upstream 2>&1 | tail -1
	fi
	if ! git -C "$dir" rev-parse -q --verify upstream/master >/dev/null; then
		printf '%-12s %7s %6s  %-26s %s\n' "$name" - - - \
			"no upstream/master; run: git -C $dir fetch upstream"
		behind_any=1
		continue
	fi
	counts=$(git -C "$dir" rev-list --left-right --count \
		upstream/master...HEAD 2>/dev/null || echo "? ?")
	behind=$(printf '%s' "$counts" | cut -f1)
	ahead=$(printf '%s' "$counts" | cut -f2)
	branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
	dirty=$(git -C "$dir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')

	note="in sync"
	[ "$ahead" != 0 ] && note="fork carries local work"
	[ "$behind" != 0 ] && note="merge upstream/master into this branch"
	[ "$dirty" != 0 ] && note="$note; $dirty uncommitted file(s)"
	[ "$behind" != 0 ] && behind_any=1

	printf '%-12s %7s %6s  %-26s %s\n' "$name" "$behind" "$ahead" "$branch" "$note"
done

cat <<'EOF'

own projects, no upstream: vc4-stack (engine and this tooling), vc4top.
A fork is released only after `--fetch` shows behind=0.
EOF
exit "$behind_any"
