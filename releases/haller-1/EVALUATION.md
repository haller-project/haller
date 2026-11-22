# Haller-1 evaluation report

## Setup

| | |
|---|---|
| Weights | `haller-1-capitals.npz`, `haller-1-lowercase.npz`, `haller-1-detector.npz`, `haller-1-fingerprint.npz` (digests in `weights.sha256`) |
| Engine | SeedSigner fork, branch `handwritten-seed` as of 2026-11-22: `helpers/handwriting/__init__.py` SHA-256 `418babe07467d76c43016f5513c421189e8739675b626551ee2a649d50083a96`, `detector.py` `b84ce2e53957e5fba6daa827894da98edd425d16cef93d3fbb705fbc456887b4`, `fingerprint.py` `f518bb2b29290b72f7626b71a5cb3787c74442afb5ec93ea78e4b43dc5abe134` |
| Inference | SeedSigner's C library (16-bit integers); benches on x86 (Intel Xeon 8375C, Python 3.12, numpy 2.5, Pillow 13.0), timings on a Raspberry Pi Zero |
| Date of the runs | 2026-11-22 |

The benches were run on the engine with the comments of `detector.py` and
`fingerprint.py` as they stood before publication; the code is the same.

## Protocol

- Test seeds: random, valid BIP39 seeds of 12 and 24 words, generated for the
  evaluation and never used for funds.
- Each page is read by `read_seed`, as on the device.
- **Verified**: the fingerprint written with the words is read and matches the
  seed; the only state announced as right. **Wrong verified**: verified but
  not the written seed. **Exact**: every word displayed is the written one,
  whatever the trust level. **Words**: words displayed correctly. **Not
  found**: no list or card of a valid number of words.
- Classes: *test* benches never served to choose a threshold, a rule or an
  epoch; *tuning* benches did. The real photos of the test half were looked at
  during development and serve as a non-regression set.
- A change of the engine is kept only with 0 wrong verified on every bench,
  more exact seeds and words on the real photos, a trust gauge never more
  optimistic than announced, and no loss of more than 1 % of exact seeds by
  writing or by support.

## Results

Verified (wrong) / exact / words / not found:

| Bench | Class | Pages | Result |
|---|---|---:|---|
| Real photos, test half | test | 35 | 0 (0) / 30 / 485 / 1 |
| Blank cards of other makers photographed, writing simulated | test | 70 | 0 (0) / 21 / 417 / 23 |
| Hard real photos taken by the device | test | 10 | 4 (0) / 7 / 99 / 1 |
| Usual simulated photos | test | 1,560 | 0 (0) / 1,402 / 21,048 / 45 |
| Badly framed | test | 1,167 | 0 (0) / 723 / 13,135 / 176 |
| Red team | test | 845 | 0 (0) / 554 / 8,635 / 47 |
| Supports kept apart | test | 900 | 0 (0) / 287 / 6,774 / 272 |
| Several columns | test | 240 | 0 (0) / 44 / 979 / 114 |
| Tight columns | test | 200 | 0 (0) / 61 / 1,520 / 69 |
| Lists with a fingerprint line | test | 1,200 | 391 (0) / 963 / 16,147 / 78 |
| Red team with a fingerprint line | test | 838 | 475 (0) / 586 / 8,677 / 60 |
| Numbered lists with a fingerprint line | test | 800 | 539 (0) / 639 / 10,370 / 58 |
| Traps around the fingerprint line | test | 800 | 516 (0) / 576 / 8,141 / 130 |
| Labels before the fingerprint | test | 400 | 94 (0) / 121 / 1,616 / 254 |
| Real photos, development half | tuning | 65 | 25 (0) / 53 / 771 / 2 |
| Photos sent to the public test, before 2026-11-23 | tuning | 75 | 7 (0) / 57 / 1,016 / 6 |
| Simulated tuning pages | tuning | 1,978 | 0 (0) / 861 / 18,076 / 419 |
| Tuning lists with a fingerprint line | tuning | 779 | 291 (0) / 607 / 9,877 / 42 |
| Tuning labels | tuning | 200 | 34 (0) / 44 / 592 / 143 |

## Wrong verified seeds

| Set | Verified | Wrong verified | Upper bound (95 %) |
|---|---:|---:|---|
| Test benches | 2,019 | 0 | 0.15 % (rule of three) |
| Tuning benches | 357 | 0 | — |

The fingerprint check bounds a chance match of a wrong seed at 1e-6 per
photo, whatever the calibration of the networks (`fingerprint.py`).

## Results by writing and support

The 115 test photos with real backgrounds (real test half, cards with
simulated writing, hard device photos), exact seeds / photos and words read
correctly:

| Writing | Photos | Exact | Words |
|---|---:|---:|---:|
| Capitals | 54 | 35 | 567 |
| Lowercase | 37 | 15 | 237 |
| Neat cursive | 20 | 5 | 130 |
| Fast cursive | 2 | 1 | 19 |
| Other | 2 | 2 | 48 |

| Support | Photos | Exact | Words |
|---|---:|---:|---:|
| Official SeedSigner card | 11 | 11 | 132 |
| Card of another maker | 73 | 23 | 479 |
| Free list | 31 | 24 | 390 |

## Changes of 2026-11-22

Exact / verified, all benches unless stated:

| Change | Effect |
|---|---|
| Labels and fingerprint rows left out of a numbered list | labels before the fingerprint 92 to 93 verified; a real photo with a cursive label verified; no loss |
| Ink of the rows, empty numbered slots, rows across the columns | +220 exact, +208 verified over 11,383 pages, 3 isolated losses; photos sent to the public test, 77 photos: 54 to 57 exact, 5 to 7 verified |
| Number check, word crops read once per photo | +1 to +2 verified on the fingerprint and trap benches; photos sent to the public test, 82 photos: 61 to 62 exact, 8 to 9 verified |

## Trust gauge

Announced share of fully right seeds: level 3 at least 90 %, level 2 at least
40 %, chosen on simulated tuning pages. The measured share is at or above the
announced one on every bench but one: on the red team with a fingerprint
line, level 2 reaches 33 % (22 of 66), a difference that is not significant.

## Speed (Raspberry Pi Zero)

Application running, SeedSigner's C library, 16 real photos of the reference
set (13 taken by the device, 3 phone or web photos):

| Photos | Time |
|---|---|
| 12-word cards (4) | 8.9 to 13.8 s |
| 12-word lists verified by their fingerprint (6) | 22.0 to 25.1 s |
| All 16 real photos | 8.9 to 77.5 s, median 20.6 s |
| A list in three columns | 77.5 s |

## Word strips

Most probable BIP39 word, quantized networks run by the C library
(`haller.evaluate`):

| Strips | Capitals expert | Lowercase expert |
|---|---|---|
| Simulated evaluation strips (regenerated by `recipes/haller-1/evalsim/regenerate.sh`, 9,156) | 96.32 % | 96.13 % |

## Test set

Manifest `test-manifest.json`: 115 files (28 photos taken by the device, 10
phone photos, 7 web photos, 70 photos of blank cards with simulated
writing), SHA-256 of each file and composition, no image, no seed. SHA-256
of the manifest: `8091a21f79b37b3160c8ca9d8681a9def4a5ddbecc0705573319785feb91c0a9`.
The simulated test benches are regenerated by their generators; the photos
of the manifest stay private while the set is in use.
