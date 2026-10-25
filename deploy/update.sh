#!/bin/bash
# Pull the latest runtime and restart it. Run on the Pi: sudo bash deploy/update.sh

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

APP=/opt/gva/app
cd "$APP"
git pull --ff-only
# shellcheck disable=SC1091
source "$APP/deploy/voice-packages.sh"
# shellcheck disable=SC1091
source "$APP/deploy/permissions.sh"
install_voice_python /opt/gva/venv/bin/pip "$APP"
if id gva >/dev/null 2>&1 && getent group audio >/dev/null; then
  usermod -aG audio gva
fi
ensure_runtime_permissions
systemctl restart gva-voice.service gva-backend.service gva-sync.service
systemctl --no-pager --lines=20 status gva-voice.service || true
echo "Updated. If the voice unit is failing, run: sudo gva doctor"
