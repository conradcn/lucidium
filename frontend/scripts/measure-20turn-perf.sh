#!/usr/bin/env bash
#
# Run the live 20-turn e2e and emit the perceived-wait metric.
#
# Metric: sum of max(0, turn_ms - 10000) across all 20 turns.
# Lower is better. Assumes a 10-second "reading" window per beat —
# anything that lands within 10s costs the player nothing; everything
# beyond 10s is felt as a wait.
#
# If the test fails (any turn missing or "1 passed" not present),
# the script emits 999999 so the autoresearch loop treats it as a
# regression rather than mis-attributing a bogus low metric to the
# experiment.
#
# Output: a single integer (ms) on stdout. All test diagnostics go
# to stderr so a stdout-capturing harness gets only the number.

set -uo pipefail

cd "$(dirname "$0")/.."

# Stream output to stderr LIVE (via tee) so progress is visible
# in the autoresearch logs, while also capturing the full text into
# a temp file for the metric extraction below. Without the tee,
# the harness sees nothing until the test ends 5+ minutes later.
TMP_OUTPUT="$(mktemp 2>/dev/null || echo "/tmp/measure-20turn-perf.$$.log")"
trap 'rm -f "$TMP_OUTPUT"' EXIT
npx playwright test live-20-move-playthrough.spec.ts --reporter=list 2>&1 \
  | tee "$TMP_OUTPUT" >&2
EXIT_CODE=${PIPESTATUS[0]}

if ! grep -q "1 passed" "$TMP_OUTPUT"; then
  echo "999999"
  exit 0
fi

# Extract every "turn N ok elapsed=Mms" line, sum (M - 10000) for
# M > 10000. ``+0`` forces awk to print 0 (not blank) when no turn
# exceeds the budget.
grep -oE "turn [0-9]+ ok elapsed=[0-9]+ms" "$TMP_OUTPUT" \
  | grep -oE "elapsed=[0-9]+" \
  | grep -oE "[0-9]+" \
  | awk '{ if ($1 > 10000) sum += $1 - 10000 } END { print sum + 0 }'
