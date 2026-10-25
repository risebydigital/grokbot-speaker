#!/bin/bash
# Install GVA on Raspberry Pi OS Lite (64-bit, Bookworm or Trixie).
# Run from a checkout: sudo bash deploy/install.sh
# This does not enable the firewall, and it does not start services until
# `gva doctor` passes (which needs real secrets in /etc/gva/gva.env).

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP=/opt/gva/app
VENV=/opt/gva/venv

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  git \
  alsa-utils \
  chrony \
  libportaudio2 \
  rsync

if ! id gva >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /var/lib/gva --shell /usr/sbin/nologin gva
fi
usermod -aG audio gva
getent group gva-dev >/dev/null || groupadd gva-dev

install -d -o gva -g gva -m 750 \
  /var/lib/gva \
  /var/log/gva
install -d -o gva -g gva -m 755 \
  /etc/gva \
  /usr/share/gva \
  /usr/share/gva/wake \
  /usr/share/gva/voices \
  /usr/share/gva/fallback

if [[ "$REPO" != "$APP" ]]; then
  install -d /opt/gva
  if [[ -d "$APP/.git" ]]; then
    echo "Using the existing checkout at $APP"
  else
    rsync -a --delete --exclude .venv --exclude .gva "$REPO/" "$APP/"
  fi
fi
chown -R gva:gva "$APP"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$REPO/deploy/voice-packages.sh"
install_voice_python "$VENV/bin/pip" "$APP"
ln -sfn "$VENV/bin/gva" /usr/local/bin/gva

if [[ ! -f /etc/gva/config.yaml ]]; then
  install -m 644 -o gva -g gva "$APP/config/config.example.yaml" /etc/gva/config.yaml
fi
if [[ ! -f /etc/gva/gva.env ]]; then
  install -m 600 -o gva -g gva "$APP/gva.env.example" /etc/gva/gva.env
fi

# shellcheck disable=SC1091
source "$REPO/deploy/permissions.sh"
ensure_runtime_permissions

install -m 644 "$APP/deploy/logrotate/gva" /etc/logrotate.d/gva
for unit in gva-voice.service gva-backend.service gva-sync.service gva-retention.service gva-retention.timer gva-backup.service gva-backup.timer; do
  install -m 644 "$APP/deploy/systemd/$unit" "/etc/systemd/system/$unit"
done
systemctl daemon-reload

# Display timezone. Telemetry timestamps stay in UTC.
timedatectl set-timezone Europe/London || true

echo
echo "Secrets file: /etc/gva/gva.env (mode 0600, owner gva)."
echo "Fill in XAI_API_KEY, GVA_HASH_PEPPER, SUPABASE_URL, and SUPABASE_SERVICE_KEY."
echo "The Supabase key on this Pi is the service-role key. It bypasses row-level security."
echo "Do not put that key in Lovable, a browser, or this git repo."
echo
echo "The firewall is unchanged. Do not enable ufw from this script."
echo
# Load the env file as data. Do not shell-source it: placeholder values contain "<".
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
  key="${line%%=*}"
  value="${line#*=}"
  export "$key=$value"
done < /etc/gva/gva.env

if gva doctor; then
  systemctl enable --now gva-voice.service gva-backend.service gva-sync.service gva-retention.timer gva-backup.timer
  echo "Services enabled."
else
  echo "Doctor failed, so systemd units were installed but not started."
  echo "After the env file is filled in:"
  echo "  sudo gva doctor"
  echo "  sudo gva audio devices"
  echo "  sudo gva models download"
  echo "  sudo systemctl enable --now gva-voice.service gva-backend.service gva-sync.service gva-retention.timer gva-backup.timer"
fi
