#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/project
VENV=/opt/ddos-rl-venv
RESULTS="$PROJECT/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RESULTS/training_${STAMP}.log"
NEWER_MARKER="/tmp/ddos-rl-training-${STAMP}.marker"

info() { printf '[INFO] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

cleanup_network() {
  pkill -f '/network/EntryPoint.py' >/dev/null 2>&1 || true
  mn -c >/dev/null 2>&1 || true
  rm -f "$NEWER_MARKER"
}
trap cleanup_network EXIT

[[ $EUID -eq 0 ]] || error "Training requires root for Mininet and packet capture."
[[ -x "$VENV/bin/python3" ]] || error "Missing Python virtual environment: $VENV"
[[ -f "$PROJECT/reinforcement/Main.py" ]] || error "Missing training entrypoint."

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"
export PYTHON="$VENV/bin/python3"
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2

if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:99
  if ! pgrep -f 'Xvfb :99' >/dev/null 2>&1; then
    info "Starting headless X server on $DISPLAY"
    Xvfb "$DISPLAY" -screen 0 1280x800x24 >/tmp/ddos-rl-xvfb.log 2>&1 &
    sleep 2
  fi
fi

install -d "$RESULTS"
cleanup_network
printf 'RUNNING\n' > "$RESULTS/status.txt"
touch "$NEWER_MARKER"

train_args=()
if [[ -n "${TRAIN_ARGS:-}" ]]; then
  read -r -a train_args <<< "$TRAIN_ARGS"
fi
train_args+=("$@")

info "Starting training"
info "TRAIN_ARGS: ${TRAIN_ARGS:-<project defaults>}"
info "Positional args: ${*:-<none>}"
info "Live log: $LOG"
cd "$PROJECT"

set +e
"$VENV/bin/python3" reinforcement/Main.py "${train_args[@]}" 2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}
set -e

run_output="$(find "$RESULTS" -mindepth 1 -maxdepth 1 -type d -name 'train_*' -newer "$NEWER_MARKER" -print | sort | tail -n 1)"
status_dir="$RESULTS"
if [[ -n "$run_output" ]]; then
  status_dir="$run_output"
  mv "$LOG" "$run_output/training.log"
  LOG="$run_output/training.log"
fi

if (( exit_code == 0 )); then
  printf 'SUCCEEDED\n' > "$status_dir/status.txt"
else
  printf 'FAILED\n' > "$status_dir/status.txt"
fi
printf '%s\n' "$exit_code" > "$status_dir/exit_code.txt"

if [[ "$status_dir" != "$RESULTS" ]]; then
  rm -f "$RESULTS/status.txt"
fi

info "Training log: $LOG"
info "Status files: $status_dir/status.txt and $status_dir/exit_code.txt"
exit "$exit_code"
