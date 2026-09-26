#!/usr/bin/env bash
# Build the syzygy1/tb generator and generate Syzygy WDL(+DTZ) tables for
# up to MAXPCS pieces, then verify every generated .rtbw against the official
# embedded checksums shipped with the generator (checksums/wdl345.txt).
#
# DTZ files are generated too because the generator needs the DTZ of
# sub-tables (captures / promotions) to build larger tables. They are not
# used in the experiment otherwise.
#
# Usage: scripts/gen_tables.sh [MAXPCS=5] [TBDIR=$HOME/tb] [EXT=$HOME/ext]
set -euo pipefail
MAXPCS=${1:-5}
TBDIR=${2:-${TBDIR:-$HOME/tb}}
EXT=${3:-${EXT:-$HOME/ext}}
THREADS=${THREADS:-$(nproc)}
mkdir -p "$TBDIR" "$EXT"

if [ ! -x "$EXT/tb/src/rtbgen" ]; then
  [ -d "$EXT/tb" ] || git clone -q https://github.com/syzygy1/tb "$EXT/tb"
  # pinned commit for reproducibility
  git -C "$EXT/tb" checkout -q "${TB_COMMIT:-0bb8aeee525f364bb750f96df312a1a7c9b54398}"
  # -flto + parallel make triggers an lto1 ICE with gcc 13; build sequentially.
  make -C "$EXT/tb/src" all >/dev/null
fi
B="$EXT/tb/src"
export RTBWDIR="$TBDIR" RTBZDIR="$TBDIR"
cd "$TBDIR"

# Official table list, ordered by (#pieces, #pawns) so that every sub-table
# reachable by capture or promotion is generated first.
awk -F'[.:]' '{print $1}' "$B/../checksums/wdl345.txt" \
  | awk -v M="$MAXPCS" '{n=length($0)-1; t=$0; p=gsub(/P/,"P",t); if(n<=M) print n, p, $0}' \
  | sort -n -k1,1 -k2,2 -k3,3 | awk '{print $3}' > tables.lst

while read -r t; do
  if [ -f "$t.rtbw" ] && [ -f "$t.rtbz" ]; then continue; fi
  if [[ "$t" == *P* ]]; then gen=rtbgenp; else gen=rtbgen; fi
  echo "[gen] $t"
  "$B/$gen" -t "$THREADS" "$t" > /dev/null
done < tables.lst

# Verify embedded checksums equal the official ones.
fail=0
while read -r t; do
  got=$("$B/tbcheck" --print "$t.rtbw" | awk '{print $2}')
  want=$(grep "^$t.rtbw:" "$B/../checksums/wdl345.txt" | awk '{print $2}')
  if [ "$got" != "$want" ]; then echo "CHECKSUM MISMATCH $t $got $want"; fail=1; fi
done < tables.lst
# Also verify each file's content against its own embedded checksum.
"$B/tbcheck" -t "$THREADS" $(sed 's/$/.rtbw/' tables.lst) | grep -v "OK!" || true
[ $fail = 0 ] && echo "All $(wc -l < tables.lst) WDL tables match official Syzygy checksums."
exit $fail
