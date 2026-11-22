# Haller-1

First release of Haller, the handwritten BIP39 word reader of a proof of
concept: loading a seed from a photo of its handwritten words in a SeedSigner
fork ([`seedsigner-haller`, branch `handwritten-seed`](https://github.com/haller-project/seedsigner-haller/tree/handwritten-seed)).
Later versions will be Haller-2, Haller-3; Haller-N.M marks a change without
retraining.

**Status: proof of concept.** The engine is complete and measured; the
SeedSigner user experience is not final and the feature is not proposed
upstream. A seed is announced as verified only when the master key
fingerprint written with its words is read and matches them; every other
reading has to be compared word by word with the paper.

## Weights

Four networks, run offline in 16-bit integers by the C library of the fork.
The files are byte for byte the ones of the fork; check them with
`sha256sum -c SHA256SUMS`.

| File | Role | Parameters | SHA-256 |
|---|---|---|---|
| `haller-1-capitals.npz` | word reader, capitals expert; reads every photo | 163,557 | `2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14` |
| `haller-1-lowercase.npz` | word reader, lowercase and cursive expert; second reading, alone and blended | 163,557 | `5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509` |
| `haller-1-detector.npz` | U-Net word detector: words, line numbers, cards, number-word links, on every photo | 223,531 | `b33f6d02f1af87a4aff1cecc73368cc2230c2fa68d0eda977a112df437d501cc` |
| `haller-1-fingerprint.npz` | reader of the fingerprint line (8 hexadecimal digits) | 163,557 | `b92cca552dc0a718d309a98b3e5f7ed6f20ee0be32f92aec8df1804d809de454` |

Each file has a recipe in `recipes/`: data, float model, quantization.
Quantization, export and reference vectors are reproduced byte for byte from
the published float models; GPU training gives models of the same level, not
the same bits.

## Engine measured

`handwritten-seed` branch of the fork as published with this release: detector
on every photo, fingerprint check (at most 3 seeds compared with the
fingerprint per photo, a chance match of a wrong seed at most 9.98e-7 per
photo), four-level trust gauge, and the changes of 2026-11-22:

- **labels and fingerprint rows**: one or two rows at the top or bottom of a
  numbered list read as a label (`fingerprint`, `xfp`, `fp`…) or as a
  fingerprint are left out of the words;
- **ink of the rows**: a line extended along the ink of its row when the
  detector covers part of a word; empty numbered slots that hold strokes
  taken as written when the count of lines needs them; a title or
  fingerprint row across the columns of a list left out of the words;
- **number check**: line numbers read with probability 0.9 or more and linked
  to their line that contradict the order of the words make the list
  incoherent; it is read again as one list, else every word is uncertain;
  word crops are read once per photo across the searches;
- **fingerprint at the end of a row**: a fingerprint written after the first
  or last word of a column, on its row, is cut off that word and checked; a
  single line outside the words is checked as an optional fingerprint;
- **abbreviated words**: words written with their first 4 letters only (as
  on metal plates) are read, a list mostly abbreviated is read again as such;
- **orientation**: the page is read as given (EXIF orientation only), slants
  and perspective followed; no search for a page upside down or turned;
- **order of two bands**: two bands of words without line numbers are also
  read band by band; when the order stays ambiguous, the trust is low;
- **fingerprint hypothesis**: a line taken for a fingerprint that matches no
  seed read is taken out and the list is kept, unverified, at low trust.

Each change was kept only with no wrong seed verified on any bench, no loss
of more than 1 % of exact seeds by writing or by support, no case of the
regression bench of the fork broken (regression set to be published), and no slowdown measured on a Pi Zero.

## Results

Seeds verified by their fingerprint (wrong ones) / exact seeds / words read
correctly / pages not found. Test benches were never used to choose a
threshold or a rule.

| Photos | Result |
|---|---|
| Real photos, test half (35) | 0 (0) / 30 / 485 / 1 |
| Blank cards of other makers photographed, writing simulated (70) | 0 (0) / 34 / 716 / 11 |
| Hard real photos taken by the device (10) | 4 (0) / 7 / 99 / 1 |
| Usual simulated photos (1,560) | 0 (0) / 1,403 / 21,065 / 44 |
| Badly framed (1,167) | 0 (0) / 641 / 11,699 / 172 |
| Red team (845) | 0 (0) / 556 / 8,680 / 47 |
| Supports kept apart (900) | 0 (0) / 277 / 6,446 / 258 |
| Several columns (240) | 0 (0) / 45 / 1,044 / 113 |
| Tight columns (200) | 0 (0) / 61 / 1,441 / 68 |
| Lists with a fingerprint line (1,200) | 397 (0) / 970 / 16,417 / 57 |
| Red team with a fingerprint line (838) | 478 (0) / 593 / 8,894 / 39 |
| Numbered lists with a fingerprint line (800) | 564 (0) / 654 / 10,824 / 31 |
| Traps around the fingerprint line (800) | 541 (0) / 586 / 8,419 / 100 |
| Labels before the fingerprint (400) | 97 (0) / 127 / 1,700 / 248 |
| Abbreviated words (400) | 0 (0) / 109 / 2,422 / 116 |

Upside-down simulated pages (359 over the benches) count as read wrong: the
engine no longer turns a page.

**0 wrong seeds among 2,449 verified**: 2,081 on the test benches above (95 %
upper bound 0.14 %) and 368 on the tuning benches (real photos of the
development half and photos sent to the public test: 35 verified, 115 exact
of 146).

Exact seeds by writing on the 115 test photos (real test half, cards with
simulated writing, hard device photos): capitals 41 of 54, lowercase 21 of
37, neat cursive 6 of 20, fast cursive 1 of 2.

Word strips, most probable BIP39 word, quantized networks run by the C
library: 96.32 % (capitals expert) and 96.13 % (lowercase expert) of the
9,156 simulated evaluation strips regenerated by
`recipes/haller-1/evalsim/regenerate.sh`.

**Raspberry Pi Zero**, reading alone (the running application adds about
5 %), 17 real photos (13 taken by the device): 5.9 to 54.2 s, median 13.6 s;
12-word cards 5.9 to 8.6 s, 12-word lists verified by their fingerprint 13.6
to 24.5 s, a list in three columns 54.2 s.

## Test set

The real test photos stay private while they are in use; the manifest
`test-manifest.json` gives the SHA-256 of each of the 115 files and their
composition, without image or seed. SHA-256 of the manifest:
`8091a21f79b37b3160c8ca9d8681a9def4a5ddbecc0705573319785feb91c0a9`.
Other models can be evaluated on the same set on request (issue of this
repository).

## Known limitations

- **Cursive and lowercase** are read less well than capitals (see above);
  fast cursive worst.
- **Orientation**: the page must be upright in the photo; a page upside down
  or turned a quarter turn is not read (no wrong reading: the page is
  not found).
- **Two bands of words without numbers**: when both orders give a plausible
  seed and no fingerprint decides, the trust is low and every word is to be
  compared; write the line numbers or the fingerprint.
- **Abbreviated words**: lists written with 4-letter abbreviations are read
  less well than full words (109 of 400 test pages exact); short words in
  spaced capitals are often not found.
- **Fake cards**: part of a plain sheet can be taken for a seed card; its
  words are then read as another document and every word is uncertain. A
  rule on the shape of the card kernel was measured and left off: it freed
  the words of real cards cut by the frame of the photo.
- Without a matching fingerprint, the checksum and the networks can agree on
  a wrong seed (truncated words, look-alike words, column order); the review
  against the paper is required.
- Layouts in three columns or more, several documents in one photo, blurry
  photos and very small writing often fail; only English BIP39 words are read.
- Real test photos are few and come from few writers; most measurements are
  on simulated pages.

## Datasets

Published with this release under the MIT License, with their README and
`SHA256SUMS`:

| Archive | Content | Used by Haller-1 |
|---|---|---|
| `haller-1-detector-inputs.tar` | 1,117 of the 1,213 input files of the detector recipe (working versions of the photos below, annotations, masks, card variants), as listed in `recipes/haller-1-detector/data/inputs.sha256` | detector recipe |
| `haller-blank-supports.tar` | 144 photos of blank writing supports (phone and SeedSigner camera), masks, annotations | detector |
| `haller-blank-cards.tar` | 52 photos of blank seed cards of several makers and 237 variants on other backgrounds, annotations; the card designs, logos and trademarks of their makers are not covered by the MIT License | detector |
| `haller-gptimage2-pilot.tar` | 608 photos of handwritten test seeds generated by gpt-image-2, labels, 5,442 word strips | no |
| `haller-gptimage2-v2.tar.part-*` | 1,444 photos of handwritten test seeds generated by gpt-image-2, labels, 16,334 word strips, word boxes and exclusion masks of 1,305 pages | no |

Eighteen photos of blank supports that show identifiable printed matter
(an advertisement, a book cover, a product label), and the twenty-one card
variants composed on them, are withheld (96 input files of the detector
recipe): the shards that drew them cannot be regenerated exactly, the
detector is still rebuilt byte for byte from its float model. The drawings,
logos and trademarks of the third-party cards shown in the photos remain the
property of their holders and are not covered by the MIT License. The
generated photos are training data for later versions; they are generated
images, as their READMEs say, and their metadata was removed (pixels
unchanged). Each archive is under 2 GiB; rebuild the v2 archive with
`cat haller-gptimage2-v2.tar.part-* > haller-gptimage2-v2.tar`; the digests
of the archives and parts are in `DATASETS.sha256`.

Every seed in these datasets is a random test seed that never held funds;
none is a seed of a test bench.

## Name

Haller is named after Neil M. Haller of Bellcore, author of RFC 1760, "The
S/KEY One-Time Password System" (1995). S/KEY, designed with Phil Karn and
John Walden on an idea by Leslie Lamport, already wrote a secret as words from
a 2,048-word dictionary, 11 bits per word, with spare bits as a checksum: the
same layout BIP39 would use eighteen years later.

## License

Code, weights and datasets: MIT. Third-party data keep their own licenses,
listed in `NOTICE.md`, which accompanies the weights; the files of
`recipes/` derived from GNHK stay under CC BY 4.0.
