#!/usr/bin/env bash
# run_many_large.sh NWORKERS TABLE... : 5-piece (large-mode) runs, lock-based queue.
set -uo pipefail
cd "$(dirname "$0")/.."
N=$1; shift
PY=${PY:-python}
LOCKS=logs/locks/large
mkdir -p "$LOCKS"
worker() {
  for t in "$@"; do
    [ -f "results/tables/$t.json" ] && continue
    mkdir "$LOCKS/$t" 2>/dev/null || continue
    $PY -u scripts/run_large.py "$t" ${RUN_ARGS:-} >> "logs/large_$t.log" 2>&1 || echo "FAILED $t"
    rm -rf "$LOCKS/$t"
  done
}
for i in $(seq "$N"); do worker "$@" & done
wait
