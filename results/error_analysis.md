| table | positions | exceptions | rate | in check | capture available | value given by a capture | zugzwang | class=blessed loss | class=cursed win | class=draw | median abs(DTZ) exc / all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| KRvKR | 1,347,906 | 9,854 | 0.73% | 0.69 | 0.16 | 0.15 | – | – | – | 0.65 | 3 / 1 |
| KQvKQ | 1,119,216 | 9,619 | 0.86% | 0.35 | 0.10 | 0.08 | – | – | – | 0.66 | 3 / 1 |
| KQvKB | 2,598,414 | 2,709 | 0.10% | 0.65 | 0.23 | 0.09 | – | – | – | 5.17 | 14 / 8 |

Enrichment = exception rate within the category / overall exception rate of the table (best MLP, variant a). '–' = category empty.

## Example exceptions (FEN, true value, model prediction)

**KRvKR**

- `8/8/8/8/2R5/1k6/8/3K1r2 w - - 0 1` — true: draw, predicted: loss
- `8/8/6k1/8/7R/8/2K5/6r1 w - - 0 1` — true: draw, predicted: win
- `3R4/8/8/8/r7/2K5/8/1k6 w - - 0 1` — true: win, predicted: draw
- `8/8/8/3k4/8/r2K4/8/7R w - - 0 1` — true: draw, predicted: loss

**KQvKQ**

- `4Q3/q7/8/8/3k4/8/8/2K5 w - - 0 1` — true: draw, predicted: win
- `8/8/4q3/7Q/8/k7/2K5/8 w - - 0 1` — true: win, predicted: draw
- `8/8/q7/6Q1/2k5/8/3K4/8 w - - 0 1` — true: draw, predicted: win
- `1q6/8/6Q1/8/8/3K4/8/4k3 w - - 0 1` — true: win, predicted: draw

**KQvKB**

- `8/8/1b6/6Q1/8/8/5k2/2K5 b - - 0 1` — true: draw, predicted: loss
- `5b2/8/8/8/8/8/k2K4/2Q5 b - - 0 1` — true: draw, predicted: loss
- `8/8/1b6/8/8/2K1k3/1Q6/8 b - - 0 1` — true: draw, predicted: loss
- `4b2k/8/8/8/4Q3/3K4/8/8 b - - 0 1` — true: draw, predicted: loss
