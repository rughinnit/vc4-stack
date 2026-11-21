#!/bin/sh
# Publish the vc4-stack packages and wheels into the signed vc4-apt repository.
#
#   packaging/publish_apt.sh [DEB...]
#
# With no arguments every vc4* .deb in ../dist (the newest file per package
# name) is published, together with the newest vc4infer wheel and, when it has
# been built, the newest torch_vc4 wheel.  Named .debs are published instead
# when given.
#
# The repository is the main branch (root) of github.com:rughinnit/vc4-apt,
# served at https://rughinnit.github.io/vc4-apt and signed with the vc4top
# archive key (~/.gnupg-vc4top, fingerprint E4CB3BA0...).  Layout:
#
#   pool/main/<letter>/<package>/<package>_<version>_<arch>.deb
#   pip/<project>-<version>-py3-none-any.whl
#   pip/simple/<project>/index.html        (PEP 503; one directory per project,
#                                           named by the normalised project name
#                                           and carrying each wheel's sha256)
#   dists/stable/{Release,Release.gpg,InRelease}
#
# Packages/Release/InRelease are regenerated over the whole pool, so adding a
# package never drops an existing one.  The pip index is regenerated from the
# wheel files actually present, so a project another change published
# (torch_vc4, vc4infer-ort, ...) stays listed without this script knowing it.
#
# The old gh-pages tree of github.com:rughinnit/vc4top (served at
# https://rughinnit.github.io/vc4top) is deprecated and deliberately not
# updated; it stays online as a frozen fallback.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
wt=${VC4APT_DIR:-$HOME/vc4-apt}
src=${VC4APT_REPO:-git@github.com:rughinnit/vc4-apt.git}
branch=main
arch=arm64
suite=stable
dist=$root/dist

: "${GNUPGHOME:=$HOME/.gnupg-vc4top}"
export GNUPGHOME

# ---- choose the .debs -------------------------------------------------------
if [ "$#" -gt 0 ]; then
	debs=$*
else
	debs=
	for p in $(ls "$dist"/vc4*_"$arch".deb 2>/dev/null |
		sed 's|.*/||; s|_[^_]*_'$arch'\.deb$||' | sort -u); do
		f=$(ls -t "$dist/${p}"_*_"$arch".deb 2>/dev/null | head -1)
		[ -n "$f" ] && debs="$debs $f"
	done
fi
[ -n "$debs" ] || { echo "no .deb files to publish in $dist" >&2; exit 1; }

wheel=$(ls -t "$dist"/vc4infer-*-py3-none-any.whl 2>/dev/null | head -1 || true)
# The PyTorch backend is built from torch/ by packaging/build_torch_wheel.sh.
# Its filename is the published interface, so copy it unchanged (and leave the
# existing one in place when this run did not rebuild it).
torchwheel=$(ls -t "$dist"/torch_vc4-*.whl 2>/dev/null | head -1 || true)

# ---- vc4-apt checkout -------------------------------------------------------
if [ -e "$wt/.git" ]; then
	git -C "$wt" checkout -q "$branch"
	git -C "$wt" pull -q --ff-only origin "$branch"
else
	git clone -q "$src" "$wt"
fi

# ---- debs -------------------------------------------------------------------
for deb in $debs; do
	[ -f "$deb" ] || { echo "missing $deb" >&2; exit 1; }
	pkg=$(dpkg-deb -f "$deb" Package)
	ver=$(dpkg-deb -f "$deb" Version)
	a=$(dpkg-deb -f "$deb" Architecture)
	letter=$(printf '%s' "$pkg" | cut -c1)
	dir=$wt/pool/main/$letter/$pkg
	install -d "$dir"
	# the pool keeps the newest version only, like the vc4top/tc4infer trees
	find "$dir" -maxdepth 1 -name "${pkg}_*_${a}.deb" \
		! -name "$(basename "$deb")" -delete
	cp -f "$deb" "$dir/"
	echo "deb: $pkg $ver ($a)"
done

# ---- pip wheels + PEP 503 indexes ------------------------------------------
install -d "$wt/pip"
if [ -n "${wheel:-}" ] && [ -f "$wheel" ]; then
	cp -f "$wheel" "$wt/pip/"
fi
if [ -n "${torchwheel:-}" ] && [ -f "$torchwheel" ]; then
	cp -f "$torchwheel" "$wt/pip/"
	echo "wheel: $(basename "$torchwheel")"
fi

python3 - "$wt/pip" <<'PY'
import hashlib, os, re, shutil, sys

pip = sys.argv[1]
wheels = sorted(f for f in os.listdir(pip) if f.endswith(".whl"))


def normalise(name):
    """PEP 503 project name: lowercase, runs of -_. become one -."""
    return re.sub(r"[-_.]+", "-", name).lower()


projects = {}
for f in wheels:
    parts = f.split("-")
    cut = next((i for i, p in enumerate(parts) if p and p[0].isdigit()), None)
    if cut is None:
        continue
    projects.setdefault(normalise("-".join(parts[:cut])), []).append(f)

simple = os.path.join(pip, "simple")
os.makedirs(simple, exist_ok=True)
# A directory under a non-canonical spelling of a project we publish is stale
# (pip only ever asks for the normalised name): drop it so the index has one
# entry point per project.
for entry in os.listdir(simple):
    path = os.path.join(simple, entry)
    if os.path.isdir(path) and normalise(entry) in projects \
            and normalise(entry) != entry:
        shutil.rmtree(path)
for proj, files in sorted(projects.items()):
    d = os.path.join(simple, proj)
    os.makedirs(d, exist_ok=True)
    lines = [
        "<!DOCTYPE html>",
        '<html><head><meta name="pypi:repository-version" content="1.0">'
        "</head><body>",
    ]
    for f in sorted(files):
        with open(os.path.join(pip, f), "rb") as handle:
            sha = hashlib.sha256(handle.read()).hexdigest()
        lines.append('<a href="../../%s#sha256=%s">%s</a><br>' % (f, sha, f))
    lines.append("</body></html>")
    open(os.path.join(d, "index.html"), "w").write("\n".join(lines) + "\n")

lines = [
    "<!DOCTYPE html>",
    '<html><head><meta charset="utf-8"><title>vc4-apt wheels</title>'
    "</head><body>",
    "<h1>VideoCore IV wheels</h1>",
    "<p>Pure-Python packages for the VideoCore IV QPU stack.  The engine "
    "library (<code>libvc4infer.so.1</code>) and the daemon come from the "
    "apt repository.</p>",
    "<ul>",
]
for f in sorted(wheels):
    lines.append('<li><a href="%s">%s</a></li>' % (f, f))
lines += [
    "</ul>",
    "<h2>Install</h2>",
    "<pre>",
    "sudo apt install vc4-toolkit          # the whole stack",
    "pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ vc4infer",
    "pip install --extra-index-url https://rughinnit.github.io/vc4-apt/pip/simple/ torch-vc4",
    "</pre>",
    "<p>Projects: " + ", ".join(
        '<a href="simple/%s/">%s</a>' % (p, p) for p in sorted(projects)) +
    "</p>",
    "</body></html>",
]
open(os.path.join(pip, "index.html"), "w").write("\n".join(lines) + "\n")
print("pip: %d wheel(s), %d project(s): %s"
      % (len(wheels), len(projects), ", ".join(sorted(projects))))
PY

# ---- Packages (whole pool) --------------------------------------------------
cd "$wt"
install -d "dists/$suite/main/binary-$arch"
apt-ftparchive packages pool > "dists/$suite/main/binary-$arch/Packages"
gzip -9 -c "dists/$suite/main/binary-$arch/Packages" \
	> "dists/$suite/main/binary-$arch/Packages.gz"

# Release must not exist while apt-ftparchive scans the tree (it would hash the
# previous Release into the new one).
rm -f "dists/$suite/Release" "dists/$suite/Release.gpg" "dists/$suite/InRelease"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
apt-ftparchive \
	-o APT::FTPArchive::Release::Origin=vc4top \
	-o APT::FTPArchive::Release::Label=vc4top \
	-o APT::FTPArchive::Release::Suite="$suite" \
	-o APT::FTPArchive::Release::Codename="$suite" \
	-o APT::FTPArchive::Release::Architectures="$arch" \
	-o APT::FTPArchive::Release::Components=main \
	-o APT::FTPArchive::Release::Description="VideoCore IV stack: vc4d, vc4infer, vc4-query, vc4-smi, vc4train, vc4top" \
	release "dists/$suite" > "$tmp"
chmod 0644 "$tmp"
mv "$tmp" "dists/$suite/Release"

gpg --batch --yes --armor --digest-algo SHA512 \
	--detach-sign -o "dists/$suite/Release.gpg" "dists/$suite/Release"
gpg --batch --yes --clearsign --digest-algo SHA512 \
	-o "dists/$suite/InRelease" "dists/$suite/Release"
gpgv --keyring "$wt/vc4top-archive-keyring.gpg" "dists/$suite/InRelease" \
	2>&1 | sed 's/^/  /'

# ---- checksums --------------------------------------------------------------
{
	find pool -name '*.deb' -type f -print0 | sort -z | xargs -0 sha256sum
	find pip -maxdepth 1 -name '*.whl' -type f -print0 | sort -z | xargs -0 sha256sum
	sha256sum key.asc vc4top-archive-keyring.gpg
} > SHA256SUMS.txt

# ---- commit + push (retry once if another change published meanwhile) -------
publish_commit() {
	git add -A
	if git diff --quiet --cached; then
		echo "repository already up to date (nothing to commit)"
		return 1
	fi
	git commit -q -m "publish: $commit_msg"
	git push -q origin "$branch" && return 0
	return 2
}

commit_msg=$(printf '%s' "$debs" | sed 's|.*/||' | sed 's/_[^_]*_'"$arch"'\.deb//' |
	tr '\n' ' ')
commit_msg="vc4 stack ($commit_msg)"
set +e
publish_commit
rc=$?
set -e
if [ "$rc" = 2 ]; then
	echo "push rejected; rebasing on origin/$branch and retrying"
	git pull -q --rebase origin "$branch"
	publish_commit || true
fi

echo
echo "repo:  https://rughinnit.github.io/vc4-apt"
echo "index: https://rughinnit.github.io/vc4-apt/pip/simple/"
echo "signed by: $(gpg --batch --with-colons --list-keys | awk -F: '/^fpr:/{print $10; exit}')"
