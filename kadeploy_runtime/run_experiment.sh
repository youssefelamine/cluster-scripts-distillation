#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/project
RUNTIME_DIR="$PROJECT/kadeploy_runtime"

if (( $# < 1 )); then
  printf 'Usage: %s EXPERIMENT_CONFIG [MAIN.PY OVERRIDES...]\n' "$0" >&2
  exit 2
fi

experiment_config=$1
shift
if [[ "$experiment_config" != /* ]]; then
  experiment_config="$PROJECT/$experiment_config"
fi

if [[ ! -f "$experiment_config" ]]; then
  printf '[ERROR] Experiment configuration not found: %s\n' "$experiment_config" >&2
  exit 1
fi

exec "$RUNTIME_DIR/start_training.sh" --experiment-config "$experiment_config" "$@"
