#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/project
RUNTIME_DIR="$PROJECT/kadeploy_runtime"

if (( $# < 1 )); then
  printf 'Usage: %s EXPERIMENT_CONFIG [MAIN.PY OVERRIDES...]\n' "$0" >&2
  exit 2
fi

exec "$RUNTIME_DIR/run_experiment.sh" "$@" \
  --episodes 1 \
  --steps 9 \
  --checkpoint-every 1 \
  --keep-last-checkpoints 1
