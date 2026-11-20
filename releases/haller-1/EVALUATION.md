# Haller-1 evaluation report

> **Pending.** This report is filled from the evaluation runs of the release
> weights (SHA-256 in `weights.sha256`). Every **TBD** must be replaced, and
> the engine version recorded, before the release is tagged.

## Setup

| | |
|---|---|
| Weights | `haller-1-capitals.npz`, `haller-1-lowercase.npz`, `haller-1-detector.npz` (digests in `SHA256SUMS`) |
| Engine | SeedSigner fork, branch `handwritten-seed`, commit TBD |
| Inference | SeedSigner's C library (16-bit integers), the same on x86 and on the device |
| Date of the runs | TBD |

## Protocol

- Test seeds: random, valid BIP39 seeds of 12 and 24 words, generated for the
  evaluation and never used for funds.
- A photo is one reading of one written seed. No evaluation photo is used for
  training, for the choice of an epoch or for the choice of a threshold.
- Worlds are reported separately and never summed: `seedsigner` (photos taken
  with a SeedSigner camera), `phone` (phone or webcam photos of the
  write-a-seed campaign), `phone` with the SeedSigner camera simulated,
  `web` (public photos), `simulated` (generated pages, evaluation-only fonts
  and EMNIST test letters).
- A blurry photo is retaken at capture time, as a user would: the numbers hold
  for photos judged sharp by the person taking them. TBD: number of retakes.
- Metrics: words read correctly (a page not found or found with the wrong
  number of words counts all its words as wrong); exact seeds; valid and
  exact; false valid (reported valid but wrong). 95 % intervals: Wilson over
  photos; for words, the union of a bootstrap over writers then photos and of
  the Wilson interval over words.

## Results by world

| World | Photos | Writers | Words read correctly [95 % CI] | Exact seeds | Valid and exact | False valid |
|---|---|---|---|---|---|---|
| `seedsigner` | TBD | TBD | TBD | TBD | TBD | TBD |
| `phone` | TBD | TBD | TBD | TBD | TBD | TBD |
| `phone`, SeedSigner camera simulated | TBD | TBD | TBD | TBD | TBD | TBD |
| `web` | TBD | TBD | TBD | TBD | TBD | TBD |
| `simulated` | TBD | — | TBD | TBD | TBD | TBD |

## Results by writing (each world)

TBD: table per world, rows block capitals / lowercase / neat cursive / fast
cursive.

## Results by layout and support (each world)

TBD: official card 12 and 24, list in one and two columns, numbered or not,
blank, ruled and squared paper; pen, felt pen, pencil; daylight and lamp.

## False valid seeds

| Set | Photos | False valid | Upper bound (95 %) |
|---|---|---|---|
| Evaluation photos, all worlds | TBD | TBD | TBD |
| Adversarial pages (column order, faint letters, look-alike words, crossings-out, ruled paper taken for a card) | TBD | TBD | — |

## Calibration

| Measure | Value |
|---|---|
| Temperature, threshold of a sure word | 2.2, 0.95 |
| ECE of the displayed word probability, `seedsigner` | TBD |
| ECE of the displayed word probability, `simulated` (capitals / cursive) | TBD |
| ECE of the seed probability | TBD |

## Speed (Raspberry Pi Zero)

| Step | Time |
|---|---|
| Loading the three networks | TBD |
| 12-word card | TBD |
| 24-word card | TBD |
| 12-word list, capitals | TBD |
| 12-word list, lowercase or cursive | TBD |

## Word strips

| Strips | Capitals | Lowercase | Blend |
|---|---|---|---|
| Simulated evaluation strips (`recipes/haller-1/data/haller-1-eval-simulated.npz`, 9,156) | TBD | TBD | TBD |
| GNHK test split, BIP39 words (2,980) | TBD | TBD | TBD |
