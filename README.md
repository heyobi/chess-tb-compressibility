# How compressible is perfect chess play?

**A scaling experiment on Syzygy WDL endgame tablebases: small neural network
+ exception list vs. Syzygy, as a function of the number of pieces.**

> Status: RESULTS_STATUS

## Question

Syzygy WDL tables store, for every legal position of an endgame, whether the
side to move wins, loses or draws (with 50-move-rule nuances: 5 classes).
Encode the same information **losslessly** as

    small MLP (quantised weights)  +  list of positions where the MLP is wrong

and measure the total size relative to the official `.rtbw` file. How does
this ratio change as the number of pieces grows (3 → 4 → 5)? If it improves,
perfect play has structure that grows with the piece count and that a small
learned model captures; if it worsens, the extra information is essentially
irregular. Both outcomes are informative; the result was not assumed in
advance.

## Prior work (not repeated here)

* **University of Reading, 1998** — a neural network plus a residual file of
  misclassified positions for a single endgame (KPK).
* **Compressing Chinese Dark Chess endgame databases with deep learning, 2018.**
* **ACG 2021** — lossless compression of chess endgame data by two-level logic
  minimisation.
* **ECAI 2024, "Comparing Lossless Compression Methods for Chess Endgame
  Data"** — decision trees, decision diagrams and logic minimisation compared
  with Syzygy; Syzygy is the most compact. No neural networks.

Contribution of this repository: a systematic measurement of the **scaling
curve** of NN-based lossless coding with the number of pieces, with an
MDL-style model-size sweep, a fair comparison with Syzygy (two variants,
see below) and a bit-exact lossless verification of every reported number.

RESULTS_BODY

## Method in brief

Full definitions: [`docs/METHOD.md`](docs/METHOD.md). Environment and its
constraints: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

* **Data.** The official download host is blocked in this environment, so all
  3-, 4- and 5-piece tables used were generated with the reference generator
  (`syzygy1/tb`); each generated `.rtbw` carries exactly the official embedded
  checksum (108 of 145 files generated; see "Scope" below).
* **Position set.** Per table: every legal position (side not to move not in
  check, no pawn on rank 1/8), no castling, no en-passant rights (exactly the
  set Syzygy indexes), both sides to move (one side for mirror-material
  tables), one representative per symmetry orbit (8 board symmetries without
  pawns, left-right mirror with pawns; colour flip is implicit in the table
  name). Enumerated in C with the Fathom prober, checked against a brute-force
  python-chess enumeration and python-chess's own Syzygy prober.
* **Labels.** 5-class WDL from the side to move (loss, blessed loss, draw,
  cursed win, win); a 3-class version (50-move rule applied) is also reported.
* **Model.** Categorical features (piece-square, side to move, king
  distances, piece-king distances, a pawn "rule of the square" feature, and
  three move-generator features: in check, #captures, #legal moves) →
  summed embeddings → 0–2 hidden ReLU layers → 5 logits. 8-bit weights (post
  training) and 4-bit weights (short quantisation-aware fine-tuning), integer
  inference that is bit-exact across machines. Model bytes = zstd -19 of the
  serialised integer model.
* **Exceptions.** Sorted indices of mispredicted positions + true labels,
  coded with an adaptive binary range coder (gap bit-length context, label in
  the context of the model's 1st/2nd choice).
* **MDL sweep.** Several sizes (≈1K … several 100K parameters) at 8 and 4
  bits; the minimum of model + exceptions (+ small header) is reported.
* **Two variants.** (a) strict: every position is coded. (b) Syzygy-style:
  decoded value = max(stored, best capture), the best capture found at decode
  time by a 1-ply capture search into smaller tables, exactly as Syzygy
  does, so positions whose value is achieved by a capture are "free" unless
  the model over-predicts them.
* **Baselines.** Syzygy `.rtbw` size; zstd -19 and xz -9e of the label array;
  "no model" (exception coder alone); CART decision tree + the same exception
  coder.
* **Lossless verification.** Every reported encoding is decoded from its bytes
  and compared position by position with the Syzygy values
  (`tests/test_lossless.py`; one mismatch fails the test).

## Reproduce

```
pip install -r requirements.txt
scripts/run_all.sh            # generation, enumeration, all experiments, figures, tests
```

`scripts/run_all.sh` documents the environment variables (tablebase and data
directories, number of workers). Seeds are fixed; library versions are pinned.
