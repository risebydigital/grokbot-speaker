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
if ! /opt/gva/venv/bin/pip install -e "$APP[voice]"; then
  /opt/gva/venv/bin/pip install -e "$APP"
fi
systemctl restart gva-voice.service gva-backend.service gva-sync.service
systemctl --no-pager --lines=20 status gva-voice.service || true
echo "Updated. If the voice unit is failing, run: sudo gva doctor"
