# 6-piece tables: not measured (resource limit)

**Result: no 6-piece number is reported — neither exact nor estimated.**
This document records what was tried and why it failed, so the missing point
on the scaling curve is not mistaken for a silent omission.

## Route 1: download the official files — blocked

`tablebase.lichess.ovh` (and the other known mirrors reachable by name) is
denied by the environment's egress policy (HTTP 403 from the proxy, see
`ENVIRONMENT.md`). The 6-piece WDL set is 68.2 GB in total; a few
representative files (tens to hundreds of MB each) would fit on disk, but they
cannot be fetched.

## Route 2: generate them locally — not enough RAM

The generator (`syzygy1/tb`, `rtbgen`) allocates two byte tables (one per side
to move) indexed by 462 king configurations × 64 squares per other piece,
independently of identical pieces:

```
tbgen.c:857   size = 462ULL << (6 * (numpcs-2));      // 6 pieces: 7.75e9
tbgen.c:960   table_w = alloc_huge(2 * alloc_size);   // 15.5 GB
```

The machine has 15.7 GiB of RAM in total, no swap, and the running
experiments needed several GB. An attempt to generate `KRRvKRR` (chosen
because two pairs of identical pieces make its *index* 4× smaller, and all of
its sub-tables were available) failed immediately with
`Could not allocate sufficient memory` under a 7 GB address-space cap, and
the uncapped allocation (15.5 GB + compression buffers) cannot fit. The
generator's `--disk` option only reduces the memory used during
*compression*, not these two tables; its README states 16 GB as the minimum
for 6-piece tables. Pawnful 6-piece tables additionally need all of their
5-piece pawnful promotion sub-tables, most of which were not generated (see
`METHOD.md`, "5-piece subset").

## Route 3: labels without a table — not meaningful

Computing exact WDL values for random 6-piece positions without the table
requires a full search to mate/conversion into 5-piece tables, which is not
tractable for a statistically useful sample on 4 cores and would itself be the
uncertain part of the experiment.

## What is ready if the files become available

The pipeline for a sample-based 6-piece estimate is implemented and tested:

1. put e.g. `KRRvKRR.rtbw`/`.rtbz` (and sub-tables) into `$TB_DIR`;
2. `src/enumerate/tbenum --tb $TB_DIR --out $TB_DATA/KRRvKRR KRRvKRR`
   (~10^10 raw indices, ~2×10^9 positions, ~6 GB of output);
3. `python scripts/run_sampled.py KRRvKRR` — trains on a 16M uniform sample,
   evaluates on 64 blocks of consecutive positions, reports the estimated
   total with a 95% bootstrap interval over blocks, written to
   `results/sampled/` and kept separate from exhaustively verified results.

The accuracy of this block-sample estimator can be judged from the 5-piece
runs, where the same estimate was computed for the chosen configuration and
then compared with the exact full-table number (`est_total_bytes` vs
`total_bytes` in `results/tables/K*v*.json` with `"mode": "large"`).
