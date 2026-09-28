#!/usr/bin/env bash
# Install (or reinstall) the daily cycle's two launchd agents for the current user:
#   com.infra.prefect-server  - Prefect API/UI + run history (http://127.0.0.1:4200)
#   com.infra.daily-cycle     - serves the Tue-Sat 10:00 UTC schedule
# Templates live in deploy/launchd/. Logs: ~/Library/Logs/infra/.
#   scripts/install_launchd.sh            # install / reload
#   scripts/install_launchd.sh uninstall  # stop and remove
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin"
LOG_DIR="$HOME/Library/Logs/infra"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
NAMES=(com.infra.prefect-server com.infra.daily-cycle)

for name in "${NAMES[@]}"; do
  launchctl bootout "$DOMAIN/$name" 2>/dev/null || true
done
if [[ "${1:-}" == "uninstall" ]]; then
  for name in "${NAMES[@]}"; do rm -f "$AGENTS/$name.plist"; done
  echo "uninstalled"
  exit 0
fi

mkdir -p "$LOG_DIR" "$AGENTS"
for name in "${NAMES[@]}"; do
  sed -e "s#__ENV_BIN__#$ENV_BIN#g" -e "s#__REPO__#$REPO#g" -e "s#__LOG_DIR__#$LOG_DIR#g" \
      "$REPO/deploy/launchd/$name.plist" > "$AGENTS/$name.plist"
  plutil -lint "$AGENTS/$name.plist" >/dev/null
  launchctl bootstrap "$DOMAIN" "$AGENTS/$name.plist"
done
echo "installed ${NAMES[*]}; logs in $LOG_DIR; UI http://127.0.0.1:4200"
