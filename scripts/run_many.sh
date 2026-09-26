#!/usr/bin/env bash
# run_many.sh NWORKERS TABLE... : per-table experiments, NWORKERS processes with
# one torch thread each. Workers pull from a shared queue (mkdir locks), so more
# workers can be started later on the same list. Extra args via RUN_ARGS,
# output dir via OUT (default results/tables).
set -uo pipefail
cd "$(dirname "$0")/.."
N=$1; shift
PY=${PY:-python}
OUT=${OUT:-results/tables}
LOCKS=${LOCKS:-logs/locks/$(basename "$OUT")}
mkdir -p logs "$LOCKS" "$OUT"
worker() {
  for t in "$@"; do
    [ -f "$OUT/$t.json" ] && continue
    mkdir "$LOCKS/$t" 2>/dev/null || continue
    $PY -u scripts/run_table.py "$t" --out "$OUT" ${RUN_ARGS:-} > "logs/$(basename "$OUT")_$t.log" 2>&1 \
      || { echo "FAILED $t"; rm -rf "$LOCKS/$t"; }
  done
}
for i in $(seq "$N"); do worker "$@" & done
wait
