#!/usr/bin/env bash
# One-command reproduction of every number and figure in this repository.
#
#   scripts/run_all.sh
#
# Environment variables (defaults in brackets):
#   TB_DIR   [~/tb]    tablebase files (generated here; never committed)
#   TB_DATA  [~/data]  enumerated positions/labels and encoded containers
#   EXT      [~/ext]   third-party sources (syzygy1/tb, Fathom)
#   WORKERS  [nproc]   parallel per-table workers (1 torch thread each)
#   PY       [python]  python interpreter with requirements.txt installed
#
# Wall time on the 4-core machine described in docs/ENVIRONMENT.md: roughly
# 1 h for table generation, several hours for the 4-piece sweep and several
# more for the 5-piece subset.
set -euo pipefail
cd "$(dirname "$0")/.."
export TB_DIR=${TB_DIR:-$HOME/tb} TB_DATA=${TB_DATA:-$HOME/data} EXT=${EXT:-$HOME/ext}
export PY=${PY:-python}
WORKERS=${WORKERS:-$(nproc)}

# 5-piece subset (see docs/METHOD.md, "5-piece subset")
SUBSET5="KQRBvK KQRvKR KRBvKR KQBvKQ KBBvKN KBNvKP KBPvKB KRPvKR KQPvKQ KPPvKR"
# pawnful 5-piece tables needed by the subset (sub-tables by promotion), in order
PAWNFUL5="KBNvKP KBPvKB KRPvKR KQPvKQ KQPvKR KBPvKR KNPvKR KPPvKR"

# 1. third-party code
mkdir -p "$EXT"
[ -d "$EXT/Fathom" ] || git clone -q https://github.com/jdart1/Fathom "$EXT/Fathom"
git -C "$EXT/Fathom" checkout -q c9c6fef0dddc05d2e242c183acf5833149ab676d
make -C src/enumerate FATHOM="$EXT/Fathom/src" >/dev/null

# 2. tables: all 3-4 piece, all pawnless 5-piece, the pawnful 5-piece subset
scripts/gen_tables.sh 4 "$TB_DIR" "$EXT"
PAWNLESS5=$(awk -F'[.:]' '{print $1}' "$EXT/tb/checksums/wdl345.txt" | awk 'length($0)==6 && !/P/')
ONLY="$PAWNLESS5 $PAWNFUL5" scripts/gen_tables.sh 5 "$TB_DIR" "$EXT"

# 3. enumeration (positions, features, labels)
ALL34=$(awk 'length($0)<=5' "$TB_DIR/tables.lst")
for t in $ALL34 $SUBSET5; do
  [ -f "$TB_DATA/$t/labels.u8" ] || src/enumerate/tbenum --tb "$TB_DIR" --out "$TB_DATA/$t" "$t"
done

# 4. per-table experiments (3-4 pieces exhaustive)
RUN_ARGS="--budget 12000 --cfgs 0,1,2,3,4,5,6,7" scripts/run_many.sh "$WORKERS" $ALL34
# 3-class labels
OUT=results/tables3 RUN_ARGS="--classes 3 --budget 12000 --cfgs 0,1,2,3,4,5,6,7" scripts/run_many.sh "$WORKERS" $ALL34
# ablation without move-generator features (subset)
OUT=results/ablation_nomovegen RUN_ARGS="--no-movegen --budget 12000 --cfgs 0,1,2,3,4,5,6,7" \
  scripts/run_many.sh "$WORKERS" KQvK KRvK KPvK KQvKR KRvKP KPvKP KBNvK KQvKQ

# 5. joint models per piece count
$PY scripts/run_joint.py 3 --threads "$WORKERS"
$PY scripts/run_joint.py 4 --threads "$WORKERS"

# 6. 5-piece subset (sample-based size sweep, exact + verified final encoding)
printf "%s\n" $SUBSET5 | xargs -P 2 -I{} sh -c "$PY -u scripts/run_large.py {} > logs/large_{}.log 2>&1"

# 7. summary, figures, error analysis, tests
$PY scripts/summarize.py
$PY scripts/error_analysis.py
$PY -m pytest -q tests/
