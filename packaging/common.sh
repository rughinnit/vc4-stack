# common.sh - shared .deb assembly for the vc4-stack install packages.
#
# The packages are assembled by hand with dpkg-deb (no debhelper) for the same
# reason vc4infer is: the payloads are small and fully known, and the build has
# to run on the Pi itself, where pulling a full debhelper stack is not free.
#
# Every builder sources this file, sets PKG/VER/ARCH/DEPENDS/SECTION/SHORT/
# LONG (and optionally EXTRA and CONFFILES), then calls deb_finish.
#
#   . ../common.sh
#   deb_finish "$stage" "$out"
#
set -eu

: "${PKG:?set PKG}"
: "${VER:?set VER}"
: "${ARCH:=arm64}"
: "${DEPENDS:=}"
: "${SECTION:=misc}"
: "${PRIORITY:=optional}"
: "${SHORT:?set SHORT}"
: "${LONG:=}"
: "${EXTRA:=}"          # extra control fields, one per line
: "${HOMEPAGE:=https://github.com/rughinnit/vc4-stack}"
: "${MAINT:=rughinnit <rughinnit@users.noreply.github.com>}"

# deb_finish STAGE OUT
#   writes DEBIAN/{control,md5sums,conffiles}, fixes ownership and builds
#   OUT/<pkg>_<ver>_<arch>.deb
deb_finish() {
	stage=${1:?stage}
	out=${2:?out}
	mkdir -p "$stage/DEBIAN" "$out"

	# every package gets a changelog and a copyright, even the tiny metas
	doc=$stage/usr/share/doc/$PKG
	mkdir -p "$doc"
	if [ ! -f "$doc/copyright" ]; then
		cp "$here/../copyright" "$doc/copyright"
	fi
	if [ ! -f "$doc/changelog.Debian.gz" ]; then
		printf '%s (%s) stable; urgency=medium\n\n  * Install-path packaging for the vc4-stack.\n\n -- %s  %s\n' \
			"$PKG" "$VER" "$MAINT" "$(date -R)" |
			gzip -9n > "$doc/changelog.Debian.gz"
	fi
	chmod 0644 "$doc/copyright" "$doc/changelog.Debian.gz"

	size=$(du -sk "$stage/usr" | cut -f1)
	{
		printf 'Package: %s\n' "$PKG"
		printf 'Version: %s\n' "$VER"
		printf 'Architecture: %s\n' "$ARCH"
		printf 'Maintainer: %s\n' "$MAINT"
		printf 'Installed-Size: %s\n' "$size"
		[ -n "$DEPENDS" ] && printf 'Depends: %s\n' "$DEPENDS"
		printf 'Section: %s\n' "$SECTION"
		printf 'Priority: %s\n' "$PRIORITY"
		printf 'Homepage: %s\n' "$HOMEPAGE"
		[ -n "$EXTRA" ] && printf '%s\n' "$EXTRA"
		printf 'Description: %s\n' "$SHORT"
		if [ -n "$LONG" ]; then
			printf '%s\n' "$LONG" | sed 's/^/ /'
		fi
	} > "$stage/DEBIAN/control"
	chmod 0644 "$stage/DEBIAN/control"

	if [ -n "${CONFFILES:-}" ]; then
		printf '%s\n' "$CONFFILES" | sed '/^$/d' > "$stage/DEBIAN/conffiles"
		chmod 0644 "$stage/DEBIAN/conffiles"
	fi

	for script in postinst preinst prerm postrm; do
		if [ -f "$here/$PKG.$script" ]; then
			install -m 0755 "$here/$PKG.$script" "$stage/DEBIAN/$script"
		fi
	done

	# no byte-compiled python: the payload must work with any >= 3.11
	find "$stage" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

	( cd "$stage" && find . -type f ! -path './DEBIAN/*' -printf '%P\0' |
		xargs -0 md5sum > DEBIAN/md5sums )
	chmod 0644 "$stage/DEBIAN/md5sums"

	rm -f "$out/${PKG}_${VER}_${ARCH}.deb"
	dpkg-deb --root-owner-group -Zxz --build "$stage" \
		"$out/${PKG}_${VER}_${ARCH}.deb" >/dev/null
	echo "built $out/${PKG}_${VER}_${ARCH}.deb ($(du -h "$out/${PKG}_${VER}_${ARCH}.deb" | cut -f1))"
	dpkg-deb -f "$out/${PKG}_${VER}_${ARCH}.deb" Package Version Depends |
		sed 's/^/  /'
}
