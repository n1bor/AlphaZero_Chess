#!/usr/bin/env bash
# Run N parallel generate.py workers, each looping forever until this script
# is killed.  Each worker writes its own timestamped .gz file so there are no
# collisions.
#
# Usage:
#   ./generate_games.sh [num_workers]   # default: 8
#
# Stop with:  Ctrl-C  (or kill the PID printed at startup)

set -euo pipefail

WORKERS=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/src"
GAME_SECS=3600   # runtime passed to each generate.py invocation

echo "Starting $WORKERS generate.py workers (Ctrl-C to stop)..."
echo "PID: $$"

# Trap Ctrl-C / SIGTERM and kill all child processes cleanly
cleanup() {
    echo ""
    echo "Stopping all workers..."
    kill 0
    exit 0
}
trap cleanup INT TERM

worker() {
    local worker_id=$1
    while true; do
        run_id="w${worker_id}_$(date +%s)"
        echo "[worker $worker_id] starting run $run_id"
        uv run --project "$SCRIPT_DIR/.." "$SCRIPT_DIR/generate.py" "$run_id" "$GAME_SECS"
    done
}

# Start all workers in background
for i in $(seq 1 "$WORKERS"); do
    worker "$i" &
done

# Wait for all background jobs (runs until Ctrl-C triggers cleanup)
wait
