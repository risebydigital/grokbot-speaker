#!/bin/bash
# Config is not secret. The env file is.
# /etc/gva must be traversable (0755) or a normal login cannot read config.yaml
# even when that file is 0644. gva.env stays 0600.

ensure_runtime_permissions() {
  chmod 755 /etc/gva
  if [[ -f /etc/gva/config.yaml ]]; then
    chmod 644 /etc/gva/config.yaml
  fi
  if [[ -f /etc/gva/gva.env ]]; then
    chown gva:gva /etc/gva/gva.env
    chmod 600 /etc/gva/gva.env
  fi
  chmod 755 /usr/share/gva /usr/share/gva/wake /usr/share/gva/voices /usr/share/gva/fallback
}
