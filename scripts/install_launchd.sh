#!/usr/bin/env bash
# Install (or reinstall) the daily cycle's launchd agents for the current user:
#   com.infra.prefect-server  - Prefect API/UI + run history (http://127.0.0.1:4200)
#   com.infra.daily-cycle     - serves the Tue-Sat 10:45 New York schedule
#   com.infra.daily-cycle-watchdog - hourly missed-run alert (macOS notification)
#   com.infra.daily-cycle-keepawake - 10 min awake after the 10:40 wake, bridging to the run
# Templates live in deploy/launchd/. Logs: ~/Library/Logs/infra/.
#   scripts/install_launchd.sh            # install / reload
#   scripts/install_launchd.sh uninstall  # stop and remove
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin"
LOG_DIR="$HOME/Library/Logs/infra"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
NAMES=(com.infra.prefect-server com.infra.daily-cycle com.infra.daily-cycle-watchdog com.infra.daily-cycle-keepawake)

# bootout is asynchronous: bootstrapping again before launchd has fully removed the old
# instance fails with "Bootstrap failed: 5: Input/output error" (hit 2026-09-29, which left
# every agent unloaded). So wait until each is really gone, and retry the bootstrap.
for name in "${NAMES[@]}"; do
  launchctl bootout "$DOMAIN/$name" 2>/dev/null || true
  for _ in $(seq 1 30); do launchctl print "$DOMAIN/$name" >/dev/null 2>&1 || break; sleep 1; done
done

bootstrap() {
  for attempt in 1 2 3 4 5; do
    launchctl bootstrap "$DOMAIN" "$1" 2>/dev/null && return 0
    sleep 2
  done
  echo "bootstrap failed for $1" >&2
  return 1
}
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
  bootstrap "$AGENTS/$name.plist"
done
echo "installed ${NAMES[*]}; logs in $LOG_DIR; UI http://127.0.0.1:4200"
