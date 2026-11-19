#!/bin/sh
# Publish vc4infer into the signed vc4top apt repository on gh-pages.
#
#   infer/packaging/publish_apt.sh [DEB] [VERSION] [WHEEL]
#
# The repository is the gh-pages branch of github.com:rughinnit/vc4top, served
# at https://rughinnit.github.io/vc4top, signed with the vc4top archive key
# (~/.gnupg-vc4top).  vc4infer is added next to vc4top:
#
#   pool/main/v/vc4top/...                  (untouched)
#   pool/main/v/vc4infer/....deb            (this package)
#   pip/vc4infer-<ver>-py3-none-any.whl     (the wheel)
#   pip/simple/vc4infer/index.html          (a PEP 503 simple index)
#
# Packages/Release/InRelease are regenerated over the *whole* pool, so adding a
# package never drops an existing one.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
wt=${VC4TOP_GH_PAGES:-$HOME/vc4top-gh-pages}
src=${VC4TOP_REPO:-$HOME/vc4top}
arch=arm64
suite=stable

deb=${1:-$(ls "$root"/../dist/vc4infer_*_"$arch".deb 2>/dev/null | sort | tail -1)}
[ -n "${deb:-}" ] && [ -f "$deb" ] || { echo "no vc4infer .deb (build it first)" >&2; exit 1; }
ver=${2:-$(dpkg-deb -f "$deb" Version)}
wheel=${3:-$(ls "$root"/../dist/vc4infer-"$ver"-py3-none-any.whl 2>/dev/null | tail -1)}
[ -n "${wheel:-}" ] && [ -f "$wheel" ] || { echo "no wheel for $ver" >&2; exit 1; }

: "${GNUPGHOME:=$HOME/.gnupg-vc4top}"
export GNUPGHOME

# ---- gh-pages worktree ------------------------------------------------------
if [ -e "$wt/.git" ]; then
	git -C "$wt" checkout -q gh-pages
	git -C "$wt" pull -q --ff-only origin gh-pages
else
	git -C "$src" worktree add -q "$wt" gh-pages
fi

install -d "$wt/pool/main/v/vc4infer" "$wt/pip/simple/vc4infer"
rm -f "$wt"/pool/main/v/vc4infer/vc4infer_*_"$arch".deb
cp "$deb" "$wt/pool/main/v/vc4infer/"
rm -f "$wt"/pip/vc4infer-*.whl
cp "$wheel" "$wt/pip/"
whl=$(basename "$wheel")
sha=$(sha256sum "$wheel" | cut -d' ' -f1)

# ---- PEP 503 simple index ---------------------------------------------------
cat > "$wt/pip/simple/vc4infer/index.html" <<EOF
<!DOCTYPE html>
<html><head><meta name="pypi:repository-version" content="1.0"></head><body>
<a href="../../$whl#sha256=$sha">$whl</a>
</body></html>
EOF

cat > "$wt/pip/index.html" <<EOF
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>vc4infer wheels</title></head><body>
<h1>vc4infer $ver</h1>
<p>Pure-Python API for ONNX inference on the VideoCore IV QPUs.  The engine
library (<code>libvc4infer.so.1</code>) comes from the <code>vc4infer</code>
apt package.</p>
<ul><li><a href="$whl">$whl</a></li></ul>
<h2>Install</h2>
<pre>
sudo apt install vc4infer
pip install https://rughinnit.github.io/vc4top/pip/$whl
# or as a pip index:
pip install --extra-index-url https://rughinnit.github.io/vc4top/pip/simple/ vc4infer
</pre>
<p>sha256: <code>$sha</code></p>
</body></html>
EOF

# ---- Packages (whole pool) --------------------------------------------------
cd "$wt"
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
	-o APT::FTPArchive::Release::Description="VideoCore IV packages (vc4top telemetry, vc4infer ONNX inference)" \
	release "dists/$suite" > "$tmp"
chmod 0644 "$tmp"
mv "$tmp" "dists/$suite/Release"

gpg --batch --yes --armor --digest-algo SHA512 \
	--detach-sign -o "dists/$suite/Release.gpg" "dists/$suite/Release"
gpg --batch --yes --clearsign --digest-algo SHA512 \
	-o "dists/$suite/InRelease" "dists/$suite/Release"

gpgv --keyring "$wt/vc4top-archive-keyring.gpg" "dists/$suite/InRelease"

sha256sum "pool/main/v/vc4infer/vc4infer_${ver}_${arch}.deb" \
	"pip/$whl" key.asc vc4top-archive-keyring.gpg > SHA256SUMS.txt
for f in pool/main/v/vc4top/*.deb; do
	[ -f "$f" ] && sha256sum "$f" >> SHA256SUMS.txt
done

git add -A
if git diff --quiet --cached; then
	echo "repository already contains vc4infer $ver (nothing to commit)"
else
	git commit -q -m "publish: vc4infer $ver (deb + wheel)"
	git push -q origin gh-pages
	echo "pushed gh-pages: vc4infer $ver"
fi
echo "deb:   https://rughinnit.github.io/vc4top/pool/main/v/vc4infer/vc4infer_${ver}_${arch}.deb"
echo "wheel: https://rughinnit.github.io/vc4top/pip/$whl"
echo "index: https://rughinnit.github.io/vc4top/pip/simple/"
