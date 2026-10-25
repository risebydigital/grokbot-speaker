#!/bin/bash
# Pull the latest runtime and restart enabled units.
# Run the copy in the operator checkout, not the old one under /opt/gva/app:
#   cd ~/milo && git pull --ff-only && sudo bash deploy/update.sh
# If /home/$SUDO_USER/milo exists, that checkout is the source (no GitHub credentials).

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

APP=/opt/gva/app
# Use the scripts next to this file. The copy under /opt/gva/app is the old
# tree until sync_app_checkout finishes, and it may not contain these helpers yet.
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/git-sync.sh"
# shellcheck disable=SC1091
source "$HERE/voice-packages.sh"
# shellcheck disable=SC1091
source "$HERE/permissions.sh"
sync_app_checkout "$APP"
if id gva >/dev/null 2>&1; then
  chown -R gva:gva "$APP"
fi
install_voice_python /opt/gva/venv/bin/pip "$APP"
if id gva >/dev/null 2>&1 && getent group audio >/dev/null; then
  usermod -aG audio gva
fi
ensure_runtime_permissions
for unit in gva-voice.service gva-backend.service gva-sync.service gva-retention.service gva-retention.timer gva-backup.service gva-backup.timer; do
  install -m 644 "$APP/deploy/systemd/$unit" "/etc/systemd/system/$unit"
done
systemctl daemon-reload
restart_enabled gva-voice.service gva-backend.service gva-sync.service
systemctl --no-pager --lines=20 status gva-voice.service || true
echo "Updated. If the voice unit is failing, run: sudo gva doctor"
