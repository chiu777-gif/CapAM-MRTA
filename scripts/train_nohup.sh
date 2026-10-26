#!/usr/bin/env bash
# Run from any directory. All run.py arguments are forwarded without eval.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Activate mrta_c first, or set PYTHON_BIN to its Python interpreter.\n' >&2
  exit 1
fi

TASK_NAME=train
METRICS_PATH=
EXPECT=
for ARG in "$@"; do
  if [[ "$EXPECT" == name ]]; then
    TASK_NAME="$ARG"
    EXPECT=
  elif [[ "$EXPECT" == metrics ]]; then
    METRICS_PATH="$ARG"
    EXPECT=
  else
    case "$ARG" in
      --run_name) EXPECT=name ;;
      --run_name=*) TASK_NAME="${ARG#*=}" ;;
      --metrics_dir) EXPECT=metrics ;;
      --metrics_dir=*) METRICS_PATH="${ARG#*=}" ;;
    esac
  fi
done

TASK_NAME="$(printf '%s' "$TASK_NAME" | tr -c '[:alnum:]_.-' '_' | cut -c1-80)"
TASK_TAG="${TASK_NAME}_$(date +%Y%m%dT%H%M%S)_$$"
CONSOLE_DIR="${CLOG_DIR:-clog}"
CONSOLE_PATH="$CONSOLE_DIR/$TASK_TAG.log"
PID_PATH="$CONSOLE_DIR/$TASK_TAG.pid"
METRICS_PATH="${METRICS_PATH:-${MLOG_DIR:-mlog}/$TASK_TAG}"
mkdir -p -- "$CONSOLE_DIR" "$METRICS_PATH"

nohup "$PYTHON_BIN" -u "$PROJECT_ROOT/run.py" \
  --no_progress_bar --metrics_dir "$METRICS_PATH" "$@" \
  >"$CONSOLE_PATH" 2>&1 </dev/null &
TASK_PID=$!
printf '%s\n' "$TASK_PID" >"$PID_PATH"
printf 'PID: %s\nConsole: %s\nMetrics: %s\nPID file: %s\n' \
  "$TASK_PID" "$CONSOLE_PATH" "$METRICS_PATH" "$PID_PATH"
printf 'Monitor: tail -f "%s"\n' "$CONSOLE_PATH"
