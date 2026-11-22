# Haller-1 evaluation report

## Setup

| | |
|---|---|
| Weights | `haller-1-capitals.npz`, `haller-1-lowercase.npz`, `haller-1-detector.npz`, `haller-1-fingerprint.npz` (digests in `weights.sha256`) |
| Engine | SeedSigner fork, branch `handwritten-seed` as published with this release: `helpers/handwriting/__init__.py` SHA-256 `963e192eff140c40ceeb8934ce1882158921d28cb051d2254c9d485223a5446c`, `detector.py` `3b00fa45a4ce78740eef2881396b56525f619225b81241f35063ee111b962495`, `fingerprint.py` `410234a733ee2fea818534898d5ce9db6c5d20b3a8bcbe6af25a4f9509dc3c00`, `decoder.py` `2c0aaa24ea0a2477e6b29ce2d37ef24f73debb441684e7199145070aa28fdaf7` |
| Inference | SeedSigner's C library (16-bit integers); benches on x86 (Intel Xeon 8375C, Python 3.12, numpy 2.5, Pillow 13.0), timings on a Raspberry Pi Zero |
| Date of the runs | 2026-11-22 |

The benches were run on the engine before the comments of `__init__.py`,
`detector.py` and `fingerprint.py` were edited for publication; the code is
the same (identical syntax trees).

## Protocol

- Test seeds: random, valid BIP39 seeds of 12 and 24 words, generated for the
  evaluation and never used for funds.
- Each page is read by `read_seed`, as on the device, upright as stored: the
  engine follows slants and perspective, it does not turn a page upside down
  or by a quarter turn.
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
  optimistic than announced, no loss of more than 1 % of exact seeds by
  writing or by support, no case of the regression bench broken
  (`tests/handwriting_regression` of the fork), and no slowdown measured on a
  Pi Zero.

## Results

Verified (wrong) / exact / words / not found:

| Bench | Class | Pages | Result |
|---|---|---:|---|
| Real photos, test half | test | 35 | 0 (0) / 30 / 485 / 1 |
| Blank cards of other makers photographed, writing simulated | test | 70 | 0 (0) / 34 / 716 / 11 |
| Hard real photos taken by the device | test | 10 | 4 (0) / 7 / 99 / 1 |
| Usual simulated photos | test | 1,560 | 0 (0) / 1,403 / 21,065 / 44 |
| Badly framed | test | 1,167 | 0 (0) / 641 / 11,699 / 172 |
| Red team | test | 845 | 0 (0) / 556 / 8,680 / 47 |
| Supports kept apart | test | 900 | 0 (0) / 277 / 6,446 / 258 |
| Several columns | test | 240 | 0 (0) / 45 / 1,044 / 113 |
| Tight columns | test | 200 | 0 (0) / 61 / 1,441 / 68 |
| Lists with a fingerprint line | test | 1,200 | 397 (0) / 970 / 16,417 / 57 |
| Red team with a fingerprint line | test | 838 | 478 (0) / 593 / 8,894 / 39 |
| Numbered lists with a fingerprint line | test | 800 | 564 (0) / 654 / 10,824 / 31 |
| Traps around the fingerprint line | test | 800 | 541 (0) / 586 / 8,419 / 100 |
| Labels before the fingerprint | test | 400 | 97 (0) / 127 / 1,700 / 248 |
| Abbreviated words (first 4 letters) | test | 400 | 0 (0) / 109 / 2,422 / 116 |
| Real photos, development half | tuning | 65 | 25 (0) / 53 / 771 / 2 |
| Photos sent to the public test, before 2026-11-23 | tuning | 81 | 10 (0) / 63 / 1,109 / 5 |
| Simulated tuning pages | tuning | 1,978 | 0 (0) / 793 / 16,794 / 397 |
| Tuning lists with a fingerprint line | tuning | 779 | 296 (0) / 611 / 10,001 / 32 |
| Tuning labels | tuning | 200 | 37 (0) / 50 / 678 / 134 |
| Tuning abbreviated words | tuning | 200 | 0 (0) / 63 / 1,459 / 44 |

Simulated pages drawn upside down count as read wrong: 359 of them (180
simulated tuning pages, 106 badly framed, 38 supports kept apart, 11 tight
columns, 11 abbreviated, 9 several columns, 4 tuning abbreviated) are out of
what the engine reads since it stopped turning pages. On the other pages of
these benches, removing that search lost no exact seed.

The 70 blank cards were photographed by one device, 13 of them stored lying
on their side: they are given to the engine upright, turned without loss
(`jpegtran -perfect -rotate 270`); the manifest gives the digests of the
files as stored.

## Wrong verified seeds

| Set | Verified | Wrong verified | Upper bound (95 %) |
|---|---:|---:|---|
| Test benches | 2,081 | 0 | 0.14 % (rule of three) |
| Tuning benches | 368 | 0 | — |

Bound of the fingerprint check, whatever the calibration of the networks
(`fingerprint.py`): a candidate seed matches the line when the probability of
its fingerprint given the line, among the 2^32 strings of 8 hexadecimal
digits, is at least 7e-4. These probabilities sum to 1, so at most 1,428
strings pass, and the fingerprint of a wrong seed, uniform over 2^32 values,
matches by chance with a probability of at most 1 / (7e-4 x 2^32) =
3.33e-7. At most 3 seeds are compared with a fingerprint line over the whole
reading of a photo, both searches included (`FingerprintBudget`): a wrong
seed is verified by chance with a probability of at most 9.98e-7 per photo.

## Results by writing and support

The 115 test photos with real backgrounds (real test half, cards with
simulated writing, hard device photos), exact seeds / photos and words read
correctly:

| Writing | Photos | Exact | Words |
|---|---:|---:|---:|
| Capitals | 54 | 41 | 703 |
| Lowercase | 37 | 21 | 359 |
| Neat cursive | 20 | 6 | 171 |
| Fast cursive | 2 | 1 | 19 |
| Other | 2 | 2 | 48 |

| Support | Photos | Exact | Words |
|---|---:|---:|---:|
| Official SeedSigner card | 11 | 11 | 132 |
| Card of another maker | 73 | 36 | 778 |
| Free list | 31 | 24 | 390 |

## Changes of 2026-11-22

Measured on the frozen benches against the engine before each change:

| Change | Effect |
|---|---|
| Labels and fingerprint rows left out of a numbered list | labels before the fingerprint 92 to 93 verified; a real photo with a cursive label verified; no loss |
| Ink of the rows, empty numbered slots, rows across the columns | +220 exact, +208 verified over 11,383 pages, 3 isolated losses; photos sent to the public test, 77 photos: 54 to 57 exact, 5 to 7 verified |
| Number check, word crops read once per photo | +1 to +2 verified on the fingerprint and trap benches; photos sent to the public test, 82 photos: 61 to 62 exact, 8 to 9 verified |
| Fingerprint at the end of a word's row, optional fingerprint line | +71 verified on the fingerprint, trap and label benches, +2 on the photos sent to the public test; no loss |
| Words abbreviated to their first 4 letters | abbreviated test pages: 0 to 108 exact; real photos read the same; +3 to +5 exact on four simulated benches |
| At most 3 seeds compared with the fingerprint per photo, both searches included | the bound above; same readings on the frozen benches |
| No orientation search (the page is read as given) | the 359 upside-down simulated pages above; elsewhere no loss, +12 exact; real photos read the same |
| Order of two bands of words offered without line numbers; low trust when the order stays ambiguous | same readings on every bench; 98 readings whose order was ambiguous move from medium to low trust |
| A fingerprint line that matches no seed read is taken out and the list kept, unverified, at low trust | +6 to +15 exact on the fingerprint, trap and label benches, +1 exact on the photos sent to the public test; test photos read the same; no verified seed gained or lost |

## Trust gauge

Announced share of fully right seeds: level 3 at least 90 %, level 2 at least
40 %, chosen on simulated tuning pages. The measured share is at or above the
announced one on every bench but two, neither difference significant (upper
Wilson bound above the announced share): on the red team with a fingerprint
line, level 2 reaches 31 % (19 of 62); on the traps around the fingerprint
line, level 3 reaches 3 of 4. A reading whose order stays ambiguous (two
orders give a plausible seed, neither the line numbers nor the fingerprint
chose one) is given level 1.

## Speed (Raspberry Pi Zero)

Raspberry Pi Zero W (1 GHz ARMv6, Wi-Fi off), SeedSigner's C library,
`read_seed` alone with the application not running (the running application
adds about 5 %), 17 real photos of the reference set (13 taken by the
device, 4 phone or web photos):

| Photos | Time |
|---|---|
| 12-word cards (4) | 5.9 to 8.6 s |
| 12-word lists verified by their fingerprint (7) | 13.6 to 25.2 s |
| Other 12-word lists (5) | 6.8 to 15.6 s |
| A list in three columns | 54.2 s |
| All 17 real photos | 5.9 to 54.2 s, median 13.6 s |

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
