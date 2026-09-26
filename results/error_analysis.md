| table | positions | exceptions | rate | in check | capture available | value given by a capture | zugzwang | class=blessed loss | class=cursed win | class=draw | median abs(DTZ) exc / all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| KRvKP | 9,031,524 | 94,069 | 1.04% | 1.44 | 0.59 | 0.52 | 84.01 | – | – | 2.83 | 6 / 3 |
| KNvKP | 9,685,770 | 85,297 | 0.88% | 1.04 | 0.94 | 0.88 | 80.27 | – | – | 0.63 | 1 / 1 |
| KPvKP | 3,718,044 | 81,243 | 2.19% | 0.80 | 0.73 | 0.63 | – | – | – | 1.59 | 1 / 1 |
| KRvKN | 2,915,128 | 69,445 | 2.38% | 1.37 | 0.08 | 0.04 | 25.65 | – | – | 0.59 | 12 / 1 |
| KNPvK | 9,699,457 | 38,166 | 0.39% | 1.43 | 1.99 | 1.86 | 83.02 | – | – | 4.50 | 8 / 2 |
| KQvKP | 8,352,472 | 29,551 | 0.35% | 1.44 | 1.33 | 1.40 | – | – | – | 7.22 | 2 / 3 |
| KQvKR | 2,467,122 | 29,495 | 1.20% | 0.72 | 0.20 | 0.14 | – | – | – | 16.92 | 32 / 4 |
| KRvKB | 2,827,104 | 28,033 | 0.99% | 2.00 | 0.06 | 0.03 | 80.68 | – | – | 0.52 | 2 / 1 |
| KBvKP | 9,427,184 | 24,017 | 0.25% | 2.20 | 1.62 | 1.53 | 196.26 | – | – | 0.38 | 1 / 1 |
| KBPvK | 9,441,347 | 19,918 | 0.21% | 2.11 | 2.72 | 2.66 | 355.51 | – | – | 2.79 | 3 / 2 |
| KPPvK | 3,719,043 | 16,842 | 0.45% | 4.92 | 4.66 | 4.51 | 100.14 | – | – | 14.77 | 2 / 2 |
| KRPvK | 9,063,318 | 16,141 | 0.18% | 4.58 | 9.96 | 9.96 | – | – | – | 26.02 | 2 / 2 |
| KQPvK | 8,398,429 | 11,906 | 0.14% | 4.53 | 10.06 | 10.06 | – | – | – | 20.36 | 2 / 2 |
| KRvKR | 1,347,906 | 9,854 | 0.73% | 0.69 | 0.16 | 0.15 | – | – | – | 0.65 | 3 / 1 |
| KQvKQ | 1,119,216 | 9,619 | 0.86% | 0.35 | 0.10 | 0.08 | – | – | – | 0.66 | 3 / 1 |
| KBNvK | 3,067,466 | 5,246 | 0.17% | 2.65 | 0.00 | 0.00 | – | – | – | 4.54 | 57 / 51 |
| KQvKN | 2,686,438 | 2,828 | 0.11% | 0.45 | 0.01 | 0.01 | – | – | – | 4.04 | 12 / 4 |
| KQvKB | 2,598,414 | 2,709 | 0.10% | 0.65 | 0.23 | 0.09 | – | – | – | 5.17 | 14 / 8 |
| KRNvK | 2,946,334 | 2,319 | 0.08% | 1.69 | 9.89 | 9.89 | – | – | – | 13.14 | 12 / 14 |
| KRBvK | 2,878,883 | 1,089 | 0.04% | 5.34 | 10.44 | 10.44 | – | – | – | 10.45 | 14 / 14 |
| KQNvK | 2,738,690 | 853 | 0.03% | 4.02 | 10.36 | 10.36 | – | – | – | 1.66 | 2 / 8 |
| KNNvK | 1,573,368 | 92 | 0.01% | 1.90 | 0.00 | 0.00 | – | – | – | 0.00 | 1 / 1 |
| KBvKB | 1,479,198 | 52 | 0.00% | 0.00 | 0.00 | 0.00 | – | – | – | 0.00 | 1 / 1 |
| KQBvK | 2,670,227 | 45 | 0.00% | 3.58 | 9.92 | 9.92 | – | – | – | 0.00 | 4 / 7 |
| KBBvK | 1,489,788 | 24 | 0.00% | 0.00 | 0.00 | 0.00 | – | – | – | 1.81 | – |
| KBvKN | 3,046,420 | 9 | 0.00% | 2.04 | 0.56 | 0.00 | – | – | – | 0.00 | 1 / 1 |
| KNvKN | 1,567,222 | 6 | 0.00% | 2.02 | 0.96 | 0.00 | – | – | – | 0.00 | 1 / 1 |

Enrichment = exception rate within the category / overall exception rate of the table (best MLP, variant a). '–' = category empty.

## Example exceptions (FEN, true value, model prediction)

**KRvKP**

- `2K5/8/8/3k4/5R2/8/4p3/8 w - - 0 1` — true: loss, predicted: draw
- `4R3/8/8/8/2K1k1p1/8/8/8 b - - 0 1` — true: loss, predicted: draw
- `8/k7/8/3K1p2/6R1/8/8/8 b - - 0 1` — true: draw, predicted: win
- `8/3K2R1/3p4/8/8/2k5/8/8 b - - 0 1` — true: loss, predicted: draw

**KNvKP**

- `8/K5k1/8/8/N6p/8/8/8 w - - 0 1` — true: draw, predicted: loss
- `8/8/8/4N3/6k1/8/3K1p2/8 b - - 0 1` — true: win, predicted: draw
- `8/8/8/N7/3K4/2p5/k7/8 b - - 0 1` — true: win, predicted: draw
- `8/K7/N7/3k1p2/8/8/8/8 b - - 0 1` — true: win, predicted: draw

**KPvKP**

- `8/6P1/8/8/K7/8/5p2/k7 w - - 0 1` — true: draw, predicted: win
- `8/8/8/3K1p2/8/1P6/6k1/8 w - - 0 1` — true: draw, predicted: win
- `8/8/2K1k3/8/2Pp4/8/8/8 w - - 0 1` — true: loss, predicted: draw
- `K7/8/p7/8/1P2k3/8/8/8 w - - 0 1` — true: win, predicted: draw

**KRvKN**

- `8/2n5/8/8/2k5/4R3/3K4/8 w - - 0 1` — true: win, predicted: draw
- `8/8/8/8/8/8/8/2Knk2R b - - 0 1` — true: loss, predicted: draw
- `8/8/8/8/4n3/k2R4/2K5/8 b - - 0 1` — true: loss, predicted: draw
- `4n3/8/8/8/8/k2K4/8/4R3 b - - 0 1` — true: loss, predicted: draw

**KNPvK**

- `1K6/8/8/3N4/3k4/8/2P5/8 w - - 0 1` — true: draw, predicted: win
- `8/2N4k/8/6P1/8/8/3K4/8 b - - 0 1` — true: draw, predicted: loss
- `4k1N1/8/5P2/8/2K5/8/8/8 b - - 0 1` — true: loss, predicted: draw
- `7N/3K4/8/8/8/8/P1k5/8 b - - 0 1` — true: draw, predicted: loss

**KQvKP**

- `8/8/7p/6Q1/8/8/3K4/5k2 b - - 0 1` — true: win, predicted: draw
- `8/7p/8/5Q2/2K3k1/8/8/8 b - - 0 1` — true: draw, predicted: win
- `8/8/5p2/2K5/8/6k1/6Q1/8 b - - 0 1` — true: draw, predicted: win
- `1Q6/1K6/8/8/8/8/1p6/5k2 b - - 0 1` — true: draw, predicted: loss
