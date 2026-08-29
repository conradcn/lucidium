#!/usr/bin/env bash
# ONE packaged-app launch on Linux; probe while it runs; guaranteed teardown.
cd /mnt/c/Users/conra/source/repos/lucidium || exit 1
BIN="$PWD/frontend/release/linux-unpacked/lucidium-frontend"
DATA="$HOME/lucidium-smoke-data"
rm -rf "$DATA"; mkdir -p "$DATA"
LOG=$(mktemp)

cleanup() {
  pkill -TERM -f 'linux-unpacked/lucidium-frontend' 2>/dev/null
  pkill -TERM -f 'lucidium-backend' 2>/dev/null
  sleep 3
  pkill -KILL -f 'linux-unpacked/lucidium-frontend' 2>/dev/null
  pkill -KILL -f 'lucidium-backend' 2>/dev/null
  echo "== leftover procs: $(pgrep -fc 'lucidium-frontend|lucidium-backend' 2>/dev/null || echo 0) =="
}
trap cleanup EXIT

LUCIDIUM_APP_DATA="$DATA" setsid "$BIN" --no-sandbox >"$LOG" 2>&1 &
for i in $(seq 1 24); do
  sleep 5
  if grep -qa 'listening\|WebSocket server' "$DATA/session.log" 2>/dev/null; then break; fi
done
echo "== renderer/main procs: $(pgrep -fc 'linux-unpacked/lucidium-frontend') =="
echo "== backend procs: $(pgrep -fc 'lucidium-backend') =="
echo "== listening sockets =="
ss -ltn 2>/dev/null | grep -E '8765|127.0.0.1' | head -5
echo "== session.log =="
tail -25 "$DATA/session.log" 2>&1
echo "== stdout/stderr =="
grep -aiE 'backend|asset-protocol|cors|error|Traceback' "$LOG" | head -25
tail -10 "$LOG"
