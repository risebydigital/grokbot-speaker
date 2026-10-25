#!/bin/bash
# Shared by install.sh and update.sh.
# The GitHub repo is private. root and the gva user have no credentials,
# so prefer the operator checkout at /home/$SUDO_USER/milo when it is there.
# safe.directory avoids "detected dubious ownership" on /opt/gva/app.

sync_app_checkout() {
  local app="${1:-/opt/gva/app}"
  local sudo_user="${SUDO_USER:-}"
  local home_root="${GVA_OPERATOR_HOME:-/home}"
  local remote=""
  if [[ -n "$sudo_user" && "$sudo_user" != "root" && -d "${home_root}/${sudo_user}/milo/.git" ]]; then
    remote="${home_root}/${sudo_user}/milo"
  fi
  if [[ -n "$remote" ]]; then
    echo "Updating ${app} from ${remote} main"
    git -C "$app" -c "safe.directory=${app}" -c "safe.directory=${remote}" pull --ff-only "$remote" main
  else
    echo "Updating ${app} from origin"
    git -C "$app" -c "safe.directory=${app}" pull --ff-only
  fi
}

restart_enabled() {
  local unit
  for unit in "$@"; do
    if systemctl is-enabled --quiet "$unit"; then
      systemctl restart "$unit"
    else
      echo "Not restarting ${unit} (not enabled)."
    fi
  done
}

supabase_configured() {
  local url="${SUPABASE_URL:-}"
  local key="${SUPABASE_SERVICE_KEY:-}"
  [[ -n "$url" && -n "$key" ]] || return 1
  # Example values contain angle brackets, including the sample Supabase URL.
  [[ "$url" == *"<"* || "$url" == *">"* || "$key" == *"<"* || "$key" == *">"* ]] && return 1
  return 0
}
