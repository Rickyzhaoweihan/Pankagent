#!/usr/bin/env bash
# Pinned launcher for the agent server on :8001.
# Safe to run by hand: kills any existing server.py process, rotates the log, spawns fresh.
set -euo pipefail

PYTHON=/db/usr/rickyhan/envs/agent/bin/python
BASE=/db/usr/rickyhan/PanKLLM_implementation
PID_FILE="$BASE/server.pid"
LOG_FILE="$BASE/server.log"

TS=$(date -u +"%Y%m%dT%H%M%SZ")

# -- Kill existing process (with cmdline check to avoid killing random PIDs) --
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        CMD=$(cat "/proc/$OLD_PID/cmdline" 2>/dev/null | tr '\0' ' ' || true)
        if echo "$CMD" | grep -q "server.py"; then
            echo "[restart_server.sh] Sending SIGTERM to PID $OLD_PID"
            kill -TERM "$OLD_PID" 2>/dev/null || true
            for _ in $(seq 1 10); do
                sleep 1
                kill -0 "$OLD_PID" 2>/dev/null || break
            done
            # force-kill if still alive
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "[restart_server.sh] SIGKILL to PID $OLD_PID"
                kill -9 "$OLD_PID" 2>/dev/null || true
            fi
        else
            echo "[restart_server.sh] PID $OLD_PID cmdline does not contain server.py — skipping kill"
        fi
    fi
fi

# -- Rotate existing log --
if [[ -f "$LOG_FILE" ]]; then
    mv "$LOG_FILE" "${LOG_FILE}.${TS}"
    echo "[restart_server.sh] Rotated server.log → server.log.${TS}"
fi

# -- Spawn --
cd "$BASE"
setsid nohup "$PYTHON" "$BASE/server.py" 8001 \
    > "$LOG_FILE" 2>&1 < /dev/null &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "[restart_server.sh] Spawned server PID $NEW_PID"
