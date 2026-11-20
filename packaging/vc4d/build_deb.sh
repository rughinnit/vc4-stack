#!/bin/sh
# Build the vc4d package: the privileged helper, its unit, the vc4 group and
# the /run socket policy.
#
#   packaging/vc4d/build_deb.sh [VERSION]
#
# The binary is the same one the source tree builds (`make -C vc4d`); the
# hand-installed copy at /usr/local/sbin/vc4d is not used by the unit.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ver=${1:-1.2.2}
out=$root/dist
stage=$root/build/vc4d-deb

make -C "$root/vc4d" build/vc4d >/dev/null

rm -rf "$stage"
install -d "$stage/usr/sbin" \
	"$stage/lib/systemd/system" \
	"$stage/etc/default" \
	"$stage/etc/profile.d" \
	"$stage/usr/share/man/man8" \
	"$stage/usr/share/doc/vc4d"

install -m 0755 "$root/vc4d/build/vc4d" "$stage/usr/sbin/vc4d"
strip --strip-unneeded "$stage/usr/sbin/vc4d"

install -m 0644 "$here/vc4d.service" "$stage/lib/systemd/system/vc4d.service"
install -m 0644 "$here/vc4d.default" "$stage/etc/default/vc4d"
install -m 0644 "$here/vc4.sh" "$stage/etc/profile.d/vc4.sh"
install -d "$stage/etc/systemd/system"
# enable by shipping the wants link?  The postinst enables it with systemctl;
# a static link is what dh_installsystemd produces and survives a daemon-reload
install -d "$stage/lib/systemd/system/multi-user.target.wants"
ln -s ../vc4d.service "$stage/lib/systemd/system/multi-user.target.wants/vc4d.service"

gzip -9n -c "$here/vc4d.8" > "$stage/usr/share/man/man8/vc4d.8.gz"
chmod 0644 "$stage/usr/share/man/man8/vc4d.8.gz"

install -m 0644 "$root/vc4d/README.md" "$stage/usr/share/doc/vc4d/README.md"

PKG=vc4d
VER=$ver
ARCH=arm64
DEPENDS="adduser"
SECTION=admin
SHORT="VideoCore IV QPU helper daemon (privileged half of the vc4 stack)"
LONG="vc4d owns /dev/mem and the V3D register window so the engine can run
unprivileged.  It allocates one client arena from CMA, publishes it over a
Unix socket in /run (mode 0660, group vc4), drops to an unprivileged uid with
an empty capability set, and supervises GPU launches for its clients.
 .
 Install this package alone on a headless box; vc4-toolkit pulls in the
 engine, the device query, the monitor and the -dev headers."
EXTRA="Recommends: vc4infer
Suggests: vc4top"
CONFFILES="/etc/default/vc4d
/etc/profile.d/vc4.sh"

. "$here/../common.sh"
deb_finish "$stage" "$out"
