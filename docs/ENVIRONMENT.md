# Execution environment

Measured at the start of the session (2026-09-26) in a managed cloud container.

| Resource | Value |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, **4 cores** (1 thread/core), AVX2 + BMI2 + POPCNT |
| RAM | **15.7 GiB**, no swap |
| Disk | ~30 GB writable allowance at start (≈22 GB after venv + tools) |
| GPU | **none** (`nvidia-smi` absent) |
| OS | Linux 6.18, Ubuntu 24.04 userland, gcc 13.3, cmake 3.28, Python 3.11.15 |

## Network access (egress proxy, policy-enforced)

| Host | Reachable? | Consequence |
|---|---|---|
| `tablebase.lichess.ovh` | **No** (proxy 403, policy denial) | Official Syzygy files cannot be downloaded |
| `syzygy-tables.info` | No (403) | – |
| `huggingface.co`, `kaggle.com` | No (403) | – |
| `download.pytorch.org` | No (403) | CPU-only torch wheel index unavailable; PyPI torch (with unused CUDA deps) installed instead |
| `github.com`, `raw.githubusercontent.com` | Yes | `syzygy1/tb` generator and `jdart1/Fathom` cloned |
| `pypi.org`, `files.pythonhosted.org` | Yes | Python dependencies |

## Consequences for the plan

* **Tables are generated locally** with `github.com/syzygy1/tb`
  (commit `0bb8aeee525f364bb750f96df312a1a7c9b54398`). Every generated `.rtbw`
  is checked against the official embedded checksums shipped in that repo
  (`checksums/wdl345.txt`); a match means the file content is identical to the
  official Syzygy file (see `scripts/gen_tables.sh`). The Fathom probe code is
  pinned at `c9c6fef0dddc05d2e242c183acf5833149ab676d`.
* **No GPU**: all models are small MLPs trained with PyTorch on CPU (4 threads).
* **Compute budget**: 3- and 4-piece tables are small (≤ ~10⁷ positions) and are
  handled exhaustively. 5-piece tables have 10⁸–10⁹ positions each; with 4 cores a
  full model-size sweep over all 110 of them is not feasible in one session, so a
  representative subset is used (documented in `docs/METHOD.md`).
* **6-piece**: the official 6-piece WDL set is 68 GB; generating even one 6-piece
  table with `rtbgen` needs ~16 GB RAM (with `--disk`) and all its 5-piece
  sub-tables. See `docs/METHOD.md` for what was attempted.
