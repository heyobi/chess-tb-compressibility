#!/usr/bin/env bash
# Idempotent (re)start of the long-running jobs of this session after an
# interruption: 5-piece generation of the subset, enumeration, per-table runs.
cd "$(dirname "$0")/.."
export TB_DIR=/root/tb TB_DATA=/root/data EXT=/root/ext PY=/home/user/venv/bin/python
rm -rf logs/locks; mkdir -p logs
SUBSET5="KQRBvK KQRvKR KRBvKR KQBvKQ KBBvKN KBNvKP KBPvKB KRPvKR KQPvKQ KPPvKR"
W34=${W34:-2}
(
  ONLY="KRPvKR KQPvKQ KQPvKR KBPvKR KNPvKR KPPvKR" THREADS=2 TBDIR=$TB_DIR scripts/gen_tables.sh 5 >> /root/gen5b.log 2>&1
  for t in $SUBSET5; do
    [ -f $TB_DATA/$t/meta.json ] && [ -f $TB_DATA/$t/labels.u8 ] && continue
    rm -rf $TB_DATA/$t
    OMP_NUM_THREADS=2 src/enumerate/tbenum --tb $TB_DIR --out $TB_DATA/$t $t >> /root/enum5.log 2>&1
  done
  echo "ENUM5 DONE" >> /root/enum5.log
  # 5-piece subset, W5 workers x 1 thread, largest-mode script
  printf "%s\n" $SUBSET5 | xargs -P ${W5:-2} -I{} sh -c "[ -f results/tables/{}.json ] || $PY -u scripts/run_large.py {} > logs/large_{}.log 2>&1"
) &
RUN_ARGS="--budget 12000 --cfgs 0,1,2,3,4,5,6,7" scripts/run_many.sh $W34 $(cat logs/list34.txt) >> logs/batch34.log 2>&1 &
wait
