#!/bin/sh
# install.sh - install vc4d, its unit and the program allowlist.
#
# Run as root from the vc4d source tree.
set -e
cd "$(dirname "$0")/.."

make
groupadd -f vc4
install -m 0755 build/vc4d /usr/local/sbin/vc4d
install -m 0644 systemd/vc4d.service /etc/systemd/system/vc4d.service
mkdir -p /etc/vc4d /usr/local/share/doc/vc4d
install -m 0644 README.md /usr/local/share/doc/vc4d/README.md 2>/dev/null || true

# allowlist of the kernels that ship with this tree (used by --strict)
if [ -d blobs ]; then
    ( cd blobs && sha256sum *.bin 2>/dev/null ) > /etc/vc4d/programs.sha256 || true
    echo "wrote /etc/vc4d/programs.sha256:"
    sed 's/^/  /' /etc/vc4d/programs.sha256
fi

systemctl daemon-reload
systemctl enable --now vc4d
sleep 0.3
systemctl --no-pager --lines=0 status vc4d || true
ls -l /run/vc4d.sock

cat <<'EOF'

Add users that should be able to use the GPU to the vc4 group:

    sudo usermod -aG vc4 <user>

They must log in again (or use `sg vc4 -c ...`) for the group to apply.
EOF
