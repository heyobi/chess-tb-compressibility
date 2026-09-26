# Method

This document fixes every definition the numbers depend on. Code references
are given in brackets.

## 1. Data: Syzygy WDL tables

The official download host (`tablebase.lichess.ovh`) is blocked in the
execution environment (see `ENVIRONMENT.md`). All 3-, 4- and 5-piece tables
were therefore generated locally with the reference generator
`github.com/syzygy1/tb` (pinned commit) [`scripts/gen_tables.sh`]. The
generator repository ships the official embedded checksums of every 3–7 piece
table (`checksums/wdl345.txt`). For every generated `.rtbw` the embedded
checksum (`tbcheck --print`) equals the official one and the file content
matches its embedded checksum (`tbcheck`), so the files are the official
Syzygy files. The "Syzygy size" used below is the byte size of these `.rtbw`
files.

Values are read with the Fathom probing code (`github.com/jdart1/Fathom`,
pinned commit), compiled into the enumerator [`src/enumerate/tbenum.c`].
Independent checks with python-chess's own Syzygy prober
[`tests/test_enumeration.py`]: exhaustive for all 3-piece tables, 300 random
positions for every other enumerated table, plus a handful of textbook
positions.

## 2. Position set (what is being compressed)

For a table named e.g. `KRPvKR`:

* **Colours.** "White" is the side listed first. The file also covers the
  colour-flipped material (K+R v K+R+P), which is the same set of positions
  with colours and board flipped, so it is not counted twice.
* **Side to move.** Both sides, except for tables with identical material on
  both sides (`KRvKR`, `KPvKP`, …) where the colour flip maps white-to-move
  onto black-to-move; there only white-to-move positions are included.
* **Legality.** All pieces on distinct squares, no pawn on rank 1 or 8, the
  side not to move is not in check. No castling rights. Positions that are
  legal in this sense but unreachable in a game (e.g. some double checks) are
  included, as in Syzygy.
* **En passant.** Positions carry no en-passant square. This is exactly the
  set Syzygy indexes: the value of a position *with* e.p. rights is not stored
  but computed by the prober from the no-e.p. value and the e.p. capture.
  Our label is the value of the position without e.p. rights.
* **Symmetry.** Pawnless tables: the 8 symmetries of the square; tables with
  pawns: the left-right mirror. Each orbit is represented once, by the member
  whose sequence of square ranks (white king, black king, remaining pieces in
  table order, identical pieces sorted) is lexicographically smallest, where
  the rank of a square puts the white-king fundamental domain (a1-d1-d4
  triangle, resp. files a–d) first. Self-symmetric positions are counted once.
  Identical pieces are unordered (counted once).

The enumerator walks a raw index space (side to move × white king in its
fundamental domain × black king × other pieces, pawns on a2–h7 only) and keeps
exactly the legal canonical positions. **The codec index of a position is its
rank among the valid positions in raw-index order.** The decoder recomputes
this order itself (`tbenum --noprobe`, no tablebase access), so indices cost
nothing to store. The position set is checked against a brute-force
python-chess enumeration of all orbits for the 3-piece tables.

## 3. Labels

`tbenum` stores, per position, Fathom's `probe_wdl` value — Syzygy's 5-class
WDL from the side to move: 0 loss, 1 blessed loss (loss, but drawn under the
50-move rule), 2 draw, 3 cursed win, 4 win. `probe_wdl` includes Syzygy's
capture resolution, so values that Syzygy stores as "don't care" are resolved
to the true value. It also stores the **best capture value** (maximum over
legal captures, including capturing promotions, of minus the child value; the
children are always in smaller tables).

The 3-class version (loss/draw/win, where blessed loss and cursed win count as
draws under the 50-move rule, or as loss/win without it) is reported as a
separate label entropy statistic; the compression experiments use the 5-class
labels (a 5-class lossless code also losslessly gives both 3-class views).

## 4. Two variants

* **(a) strict**: every position's value must be decoded from the code.
* **(b) Syzygy-style**: decoded value = max(stored value, best capture value).
  The best capture value is obtained at decode time by a 1-ply capture search
  into the smaller tables, exactly as Syzygy's prober does. A position whose
  value equals its best capture value ("capture-resolved") may store any
  value ≤ its true value. Consequently a model prediction counts as an error
  only if max(prediction, best capture) ≠ true value. The model is trained
  with the matching set-valued loss −log Σ_{c allowed} p(c).

## 5. Model features [`src/model/features.py`]

All features are categorical; each selects one row of an embedding table and
the first layer is the sum of the selected rows (equivalent to a linear layer
on one-hot inputs).

| Feature | Values |
|---|---|
| piece-square, per piece | (colour, type) × 64 squares; identical pieces share rows |
| side to move | 2 |
| Chebyshev distance wK–bK | 8 |
| per non-king piece: distance to own king, to enemy king | (colour, type) × 8, each |
| per pawn: (ranks to promotion, enemy-king distance to promotion square, pawn side to move) | 7 × 8 × 2 per pawn colour |
| side to move in check | 2 |
| min(3, number of legal captures) | 4 |
| min(15, number of legal moves) | 16 |

The last three come from a move generator (no tablebase access). They are
cheap but powerful: "0 legal moves and not in check" is stalemate, "0 legal
moves and in check" is mate, and "a capture exists" flags hanging pieces. This
makes very small tables almost trivial (a 16-leaf tree codes KQvK with zero
exceptions). The decision-tree baseline gets the same information as integers.
An ablation without the move-generator features is reported separately.

## 6. MLP, quantisation and bit-exact inference [`src/model/net.py`]

Architectures (index stored in the model; the list is part of the decoder):
embedding width h1 followed by 0–2 hidden ReLU layers, 5 outputs. Sizes range
from ~1K to several 100K parameters depending on the table's vocabulary.

Training: Adam, one-cycle LR (max 3e-3), batch 1024/4096, fixed step budget,
fixed seeds. The training set is the whole table (3–4 pieces) or a uniform
random sample (5 pieces). Since the goal is to memorise the table, there is
no held-out set: all errors on the full table are coded as exceptions.

Post-training quantisation: embedding columns and hidden-layer rows get their
own symmetric scale (8-bit: ±127; 4-bit: ±7); the output layer uses a single
scale so that integer argmax = real argmax; activations are uint7 with a
per-layer scale calibrated at the 99.999th percentile on 200K positions;
biases are int32 in accumulator scale. Rows of the embedding that no position
of the table uses are zeroed (they cannot influence any prediction).

Integer inference is exact on any IEEE-754 machine: matrix products are
float32 BLAS on integer-valued tensors whose partial sums are provably
< 2^24 in magnitude (checked per output row when quantising), so every
summation order gives the same exact integer; re-quantisation is one
correctly rounded multiply, one correctly rounded add of 0.5, a floor and a
clamp to 0..127; ties in argmax resolve to the lowest class. Model format
version 1 (used by the first tables processed) does the re-quantisation in
float64, version 2 in float32 (3× faster); the version byte is part of the
model and the decoder follows it.

**Model bytes** = zstd -19 of the serialised model: header (3 B), float32
scales, packed int8/int4 weights, int32 biases.

## 7. Exception coding [`src/codec/rangecoder.c`]

Exceptions are the positions whose decoded value (variant-dependent, §4)
differs from the true value, sorted by index. Stream: an LZMA-style adaptive
binary range coder (11-bit probabilities, shift 5):

* gap to previous exception g ≥ 1: bit length ⌊log2 g⌋ with a 6-bit binary
  tree whose context is the previous gap's bit length; the top 4 mantissa bits
  adaptive (context: bit length, position), the rest raw;
* true label: 3-bit binary tree whose context is (model's first choice,
  model's second choice) — the decoder knows both after running the model.

Container: `varint(len(model)) | model | varint(#exceptions) | range-coded
stream`. **Total bytes** = the length of that container, i.e. everything the
decoder needs beyond the (fixed) decoder program, the table name and — for
variant (b) — the smaller tables.

## 8. Baselines

* **Syzygy**: size of the official `.rtbw`.
* **Generic compressors**: the label array (one byte per position, canonical
  order) compressed with zstd -19 and xz -9e. For variant (b) the free
  positions are filled with the previous non-free value clipped to ≤ the true
  value (keeps runs long; still decodes correctly with max(·, best capture)).
* **No model**: constant prediction + exceptions (the exception coder alone).
* **Decision tree**: scikit-learn CART (best-first, `max_leaf_nodes` sweep
  16 … 262144, ×2 steps) on small-integer features (per piece: square, file,
  rank, square colour; king distances; per pawn: ranks to promotion, enemy-king
  distance to the promotion square and the rule-of-the-square margin; the
  three move-generator features), serialised in pre-order and zstd -19'd;
  same exception coder (label context = the tree's class). In both variants
  the tree is fitted on all positions with the true label (the true label is
  an allowed stored value in variant (b) too). Note: an earlier version fitted
  the (b) tree only on non-capture-resolved positions, which extrapolated
  badly; all reported trees are "tree_version 2".

## 9. Model-size selection (MDL)

For every table and variant, configurations are tried in increasing size, each
at 8 and 4 bits. The sweep stops when two consecutive sizes do not improve the
best total, when the model alone exceeds the best total, or when there are no
exceptions left. The reported number is the minimum total. The same holds for
tree sizes.

## 10. Large (5-piece) tables

5-piece tables have 0.85–5.5 × 10^8 positions. On 4 CPU cores a full
model-size sweep with exhaustive evaluation of every size is not affordable,
so for these tables [`scripts/run_large.py`]:

* models are trained on a uniform random sample of 16M positions (trees: 4M);
* every sweep point (MLP config × {8, 4} bits, tree size) is evaluated on an
  evaluation sample of 4M positions made of 64 random blocks of consecutive
  positions (blocks preserve the index locality of exceptions, which the gap
  coder exploits; a uniform sample over-estimated exception bytes by ~60%,
  blocks come within ~10%). Exception bytes are extrapolated linearly;
  these sweep points are marked `estimated` in the JSON;
* the configuration with the best *estimated* total is then encoded exactly on
  the **full** table and verified on the full table. Only these exact,
  verified numbers are used in the scaling figures.

Consequences: the chosen size may differ from the exact optimum by the
estimation error, and models trained on a sample see only a fraction of the
positions they must encode (they have to generalise), which the 3–4 piece
models do not.

### 5-piece subset

All 110 5-piece tables would take far longer than the session allows
(one 5-piece table: 1–4 CPU-hours for the sweep; one full pass of the
largest model over a 5 × 10^8-position table: ~30 min), and the
pawnful tables need their promotion sub-tables to be generated first. The
subset (10 tables) was chosen before seeing any 5-piece result, to mix
material types and Syzygy sizes (33 KB – 9.6 MB range in the full set):

| table | why |
|---|---|
| KQRBvK | 3 v 0, trivially won (lower extreme) |
| KQRvKR | 2 v 1 pawnless, mostly won |
| KRBvKR | 2 v 1 pawnless, famously hard (mostly drawn, long wins) |
| KQBvKQ | queen endings, very large Syzygy file |
| KBBvKN | long wins, large Syzygy file |
| KBNvKP | pawn on the weaker side (promotion races) |
| KBPvKB | opposite/same-coloured bishops |
| KRPvKR | the most important practical rook endgame |
| KQPvKQ | queen + pawn vs queen |
| KPPvKR | two pawns vs rook (needed 4 extra pawnful sub-tables) |

All 110 pawnless 5-piece tables were generated (they are cheap and needed as
promotion targets); of the pawnful ones only the subset and its promotion
sub-tables (KQPvKR, KBPvKR, KNPvKR) were generated.

## 11. Verification

For the best encoding of every method and variant, the container bytes are
decoded (model deserialised from bytes, exceptions range-decoded, variant-(b)
max applied) and compared with the Syzygy value of every position
[`src/mdl.py: verify_*`]. For 3–4 piece tables the predictions are recomputed
from the deserialised model during verification. For the 5-piece tables the
single full inference pass already uses the model deserialised from the
container's bytes; verification then parses the container, checks the model
part is byte-identical, range-decodes the exceptions, applies them to those
predictions and compares all positions (a second inference pass would only
re-test the determinism of integer inference). `tests/test_lossless.py` repeats this from the files
on disk after regenerating the position set without tablebase access, and
fails on a single mismatch. `verified_mismatches` is stored in each result
JSON.
