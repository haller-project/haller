# Haller

Haller reads handwritten BIP39 words. This repository is its training
pipeline: from public data only, it produces the quantized weights read by
`seedsigner.helpers.handwriting` in a SeedSigner fork, to load a seed from a
photo of handwritten words
([`handwritten-seed` branch](https://github.com/haller-project/seedsigner-haller/tree/handwritten-seed), see
`docs/handwritten_seeds.md` there). Photos taken by users are never used.

**Haller-1** is the first version (tag `haller-1`); the following ones will be
Haller-2, Haller-3, and so on.

## Status: proof of concept

The reading engine and the Haller-1 weights are complete and measured (see
[Evaluation](#evaluation)). The SeedSigner integration is a proof of concept:
its user experience is not final and it is not proposed upstream.

## Haller-1 model card

**Architecture.** Two networks of the same architecture, called experts: a
small fully convolutional network (`haller/model.py`, about 164,000
parameters with the channels 16/32/64) trained with CTC on 37 classes: blank,
a-z (case-insensitive), 0-9. It reads a whole word, without letter
segmentation; on the device its output is scored against the 2,048 BIP39
words, then the BIP39 checksum chooses the seed. Weights are quantized to
16-bit integers.

| Weights | Expert | Role in SeedSigner | SHA-256 |
|---|---|---|---|
| `haller-1-capitals.npz` | capitals | reads every photo | `2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14` |
| `haller-1-lowercase.npz` | lowercase and cursive | reads again the photos whose seed is not valid, alone and blended with the capitals expert | `5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509` |
| `haller-1-detector.npz` | word detector | finds the words, their line numbers and the cards on every photo | `b33f6d02f1af87a4aff1cecc73368cc2230c2fa68d0eda977a112df437d501cc` |
| `haller-1-fingerprint.npz` | fingerprint | reads the master key fingerprint written under the words; a seed is verified only when it matches | `b92cca552dc0a718d309a98b3e5f7ed6f20ee0be32f92aec8df1804d809de454` |

The fingerprint expert has the architecture of the word experts, fine-tuned
from the capitals expert on fingerprint lines
([recipe](recipes/haller-1-fingerprint/README.md)); the detector is described
in [Word detector](#word-detector) and [`MODEL_CARD.md`](MODEL_CARD.md). Each
file has an exact recipe in `recipes/` (data, float model, quantization).

**Intended use.** Reading BIP39 seeds written by hand on an official
SeedSigner card or a free list of 12 to 24 words, on an offline device, with a
mandatory review of every word by the user against the paper.

**Training data.** Synthetic words (handwriting fonts and EMNIST letters,
"photographed"), real words from GNHK, and, for the lowercase expert, cursive
fonts and BIP39 words generated in the hands of GNHK writers by Emuru; see
[Data](#data) and [Data sources and licenses](#data-sources-and-licenses).

**Results** of the engine of the SeedSigner fork as of 2026-11-22 on full
photos (seeds verified by their fingerprint (wrong ones) / exact seeds /
words read correctly; full report in
[`releases/haller-1/EVALUATION.md`](releases/haller-1/EVALUATION.md)):

| Photos | Result |
|---|---|
| Real photos, test half (35) | 0 (0) / 30 / 485 |
| Hard real photos taken by the device (10) | 4 (0) / 7 / 99 |
| Blank cards of other makers photographed, writing simulated (70) | 0 (0) / 21 / 417 |
| Usual simulated photos (1,560) | 0 (0) / 1,402 / 21,048 |
| Red team (845) | 0 (0) / 554 / 8,635 |
| Lists with a fingerprint line (1,200) | 391 (0) / 963 / 16,147 |
| Red team with a fingerprint line (838) | 475 (0) / 586 / 8,677 |

0 wrong seeds among 2,376 verified over all benches. On a Raspberry Pi Zero,
application running: 12-word cards 8.9 to 13.8 s, 12-word lists verified by
their fingerprint 22.0 to 25.1 s, 16 real photos 8.9 to 77.5 s (median
20.6 s).

**Limits.** A seed is announced as verified only when the fingerprint written
with its words is read and matches them; any other reading can be wrong even
with a valid checksum and has to be compared with the paper. Capitals are
read best; lowercase and cursive less well (115 test photos: 35 of 54 seeds
exact in capitals, 15 of 37 in lowercase, 5 of 20 in neat cursive). Part of
a plain sheet can be taken for a seed card; a fingerprint written to the
right of the last word is read as part of it. Blurry photos, very small
writing, lists cut by the frame and layouts in three columns or more often
fail. Only English BIP39 words are read. The real test photos are few; the
measurements rely mostly on simulated photos.

**Name.** Haller is named after Neil M. Haller of Bellcore, author of
RFC 1760, "The S/KEY One-Time Password System" (1995). S/KEY, designed with
Phil Karn and John Walden on an idea by Leslie Lamport, already wrote a secret
as words from a 2,048-word dictionary, 11 bits per word, with spare bits as a
checksum: the same layout BIP39 would use eighteen years later.

**License.** MIT (`LICENSE`), code and weights. The training data keep their
own licenses, listed in [Data sources and licenses](#data-sources-and-licenses)
and [`NOTICE.md`](NOTICE.md), which accompanies the weights.

The full model card, with the word detector and the evaluation of the
release, is [`MODEL_CARD.md`](MODEL_CARD.md); releases are made as described
in [`RELEASING.md`](RELEASING.md) and listed in [`CHANGELOG.md`](CHANGELOG.md).

## Data

Every sample is an ink map 32 pixels high and at most 256 wide, produced by
SeedSigner's own normalization (`seedsigner.helpers.handwriting.preprocess`,
imported by this repository). Training data and device input therefore go
through the same code; a change to `preprocess.py` requires regenerating all data.

- **Synthetic words** (`synth.py`, `generate.py`). A word (60 % English BIP39,
  40 % other words from `data/other_words.txt`, so that the model learns
  letters rather than a closed set of word shapes), in upper, lower or
  capitalized case, sometimes preceded by a line number, is written in one
  hand: a handwriting font (whole word or letter by letter with jitter) or
  EMNIST letters with word-level size, slant, stroke width and spacing. The
  writing is warped, placed on a line with optional neighboring lines, ruled or
  printed paper lines, then "photographed" (paper and ink colors, uneven
  lighting, shadows, scale, focus or motion blur, noise, JPEG) and normalized.
  Samples that become illegible or where normalization picks a neighboring line
  are discarded.
  *Why 60 %.* Since the device only outputs BIP39 words, training on BIP39
  words only could make the network learn the 2,048 word shapes as a prior
  (Wan et al. 2020, "Vocabulary Reliance in Scene Text Recognition") on top of
  the dictionary decoding, and become overconfident between close BIP39 words.
  This was tested with matched shards (`generate --matched --bip39-share S`:
  same hands and photographs sample by sample, only the words differ), 8 x
  50,000 words at 100, 80 and 60 % BIP39, plus filtered GNHK and Emuru, the
  capitals expert's architecture and schedule, two training seeds each. Engine replayed with
  one model on 1,605 photos (sure and exact / exact / false sure, sum of both
  seeds; capitals 1,301 photos, cursive 301):

  | BIP39 share | All photos | Capitals | Cursive | Letters, BIP39 / unseen words |
  |---|---|---|---|---|
  | 100 % | 2199 / 2399 / 6 | 2116 / 2226 / 6 | 83 / 171 / 0 | 50.2 / 44.1 % |
  | 80 % | 2180 / 2373 / 4 | 2107 / 2225 / 4 | 72 / 147 / 0 | 49.4 / 44.5 % |
  | 60 % | 2191 / 2397 / 3 | 2112 / 2232 / 1 | 79 / 163 / 2 | 49.8 / 44.1 % |

  The differences are within the training-seed noise (the same recipe varies
  by up to 16 sure and exact seeds and 7 points on GNHK test between seeds).
  Letter-by-letter reading of words never seen in training (English words
  outside BIP39 and `other_words.txt`) is the same at every share, and almost
  never turns into a BIP39 word (0.1 to 0.3 %); word-level calibration (ECE)
  and confusions between BIP39 words at edit distance 1-2 do not change with
  the share. Most false sure seeds come from page detection (a truncated word
  whose end is another BIP39 word, e.g. "pen" for "spend", or a wrong word
  count), at every share. No share is better: 60 % is kept.
- **Real words** from GNHK (`gnhk.py`): each annotated word is straightened and
  cropped with random margins, so that the crop includes part of its
  surroundings as on a SeedSigner line, then normalized.
- **Real names** from scanned forms (`names.py`, `filter.py`): several
  candidate crops per image; a trained model keeps the candidate it reads as
  the label, and one-edit readings are confirmed by a vision model. This set
  was not used for Haller-1.

### Datasets of the release

Published with the Haller-1 release under the MIT License, each with a
README and `SHA256SUMS`: `haller-1-detector-inputs` (1,168 of the 1,213
input files of the detector recipe; 45 withheld, listed in its README),
`haller-blank-supports` (155 photos of blank
writing supports, full resolution, masks and annotations),
`haller-blank-cards` (blank printed seed cards of several makers and their
variants; the printed designs remain the property of their authors), and
two sets of photos of handwritten test seeds generated by gpt-image-2
(`haller-gptimage2-pilot`, `haller-gptimage2-v2`, with word strips; not used
by Haller-1). Every seed in them is a random test seed that never held funds.

## Setup

Linux x86-64, Python 3.10 or later, `make` and a C compiler (the C library of
SeedSigner is built on first use), an NVIDIA GPU for training. The training
and quantization of Haller-1 used the versions pinned in `requirements-lock.txt`;
its data were generated with the versions given in
[`recipes/haller-1`](recipes/haller-1/README.md),
[`recipes/haller-1-detector`](recipes/haller-1-detector/README.md) and
[`recipes/haller-1-fingerprint`](recipes/haller-1-fingerprint/README.md),
which are not the same for every model.

```bash
# SeedSigner with the handwriting reader: its preprocessing and C library are
# used here. The handwritten-seed branch ships the four Haller-1 files of the
# weights table above (same SHA-256).
git clone -b handwritten-seed https://github.com/haller-project/seedsigner-haller ../seedsigner
export PYTHONPATH=$PWD/../seedsigner/src

python -m venv .venv && . .venv/bin/activate
pip install -c requirements-lock.txt -e '.[train,names]'
```

Inputs to download (not redistributed here); the commands below expect them in
the repository root:

- **Fonts** (Google Fonts, pinned to a commit of the google/fonts repository,
  SHA-256 checked, with their licenses in `LICENSES.json`):

  ```bash
  python -m haller.fonts fonts                                            # 287 fonts
  python -m haller.fonts fonts_cursive --list data/fonts_cursive.sha256   # 115 cursive fonts
  ```

- **EMNIST** *byclass* training images as `EMNIST.npz`: `x` (N, 28, 28) uint8,
  white ink on black, upright; `y` (N,) class index in the order `0-9 A-Z a-z`:

  ```bash
  curl -LO https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
  # SHA-256 fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e
  unzip gzip.zip && python -m haller.emnist EMNIST.npz gzip
  ```

- **GNHK**: page `.jpg` files and their `.json` annotations, in `GNHK/train`
  and `GNHK/test`. The official page (https://goodnotes.com/gnhk) no longer
  offers the download (November 2026); the Hugging Face dataset
  `aryaman27/goodnotes-handwritten-dataset`, revision `9042977`, holds the same
  files (`train_data/train`, `test_data/test`). Check them with
  `(cd GNHK && sha256sum -c ../data/gnhk.sha256)`.
- optionally the names dataset as Parquet files.

Run all commands from the repository root (default paths are relative to it).
Each generation process takes about 1 GB of memory and 20 minutes per shard of
50,000 words on a laptop CPU core: the loops below start
one process per shard; with less memory, start them in smaller groups.

### What can be reproduced

[`recipes/haller-1`](recipes/haller-1/README.md) reproduces the published
weights: the code versions that made their data, the float models, the data
that cannot be regenerated and the content digests of every file.

- **Data**: deterministic for given inputs, seeds and library versions. The
  shards of Haller-1 were made by the `synth.py` of 17/11/2026 (kept in
  `recipes/haller-1/haller1`), which differs from the current one (blur
  kernels up to 9x9, crops not clamped to the image, another draw of the
  scale): with it and the pinned versions, s1 to s24 are regenerated
  identically. The cursive shards, the cursive validation and the GNHK crops
  of the lowercase expert are regenerated identically by the current code.
  The commands below, with the current `synth.py`, give shards of the same
  kind with other samples.
- **Quantization is deterministic**: the published weights are obtained
  exactly (with `--ranges` on any processor) from the float models
  `recipes/haller-1/data/haller-1-capitals-float.npz` and
  `haller-1-lowercase-float.npz`, and those of the detector and of the
  fingerprint expert from theirs
  ([`recipes/haller-1-detector`](recipes/haller-1-detector/README.md),
  [`recipes/haller-1-fingerprint`](recipes/haller-1-fingerprint/README.md)).
- **Training** of Haller-1 is not bit-reproducible: it used
  `torch.backends.cudnn.benchmark` and the CUDA CTC loss, so two runs with the
  same seed differ. A retrained model reaches the same level, not the same
  weights. `haller.train --deterministic` makes new runs bit-identical on the
  same hardware and software.
- **The epoch kept** is the last one: `haller.train` writes the last epoch
  by default (`--select best` keeps the epoch with the best simulated score
  instead). Both Haller-1 word experts were trained with the best-epoch
  selection, which chose their last epoch; the fingerprint expert is epoch 1
  of a longer schedule (`--stop-after`). `--eval` is still required: every
  epoch is scored on the simulated strips of `EVAL.npz`. They are drawn with
  EMNIST test letters, so they are not stored in the repository:
  `recipes/haller-1/evalsim/regenerate.sh` downloads the official EMNIST
  archive and writes them to
  `recipes/haller-1/data/haller-1-eval-simulated.npz`, identical to those of
  Haller-1 (content digest checked). This file has no real strips (its
  `reel` field is always false) and replaces `EVAL.npz` in every command;
  any file of strips with `x`, `w` and `y` can be given instead, with or
  without `reel`. The 288 strips of real photos (a test set) are not
  published.
- **Generated handwriting** (Emuru) and the vision-model checks are not
  bit-reproducible (GPU sampling, served model): the strips used by the
  lowercase expert are published in `recipes/haller-1/data`.

## Reproducing Haller-1: capitals expert

```bash
# 1. Synthetic shards: 16 x 50,000 words, seed 1000 + i (one process per shard)
for i in $(seq 1 16); do
  python -m haller.generate s$i.npz 50000 $((1000 + i)) --fonts fonts \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait

# 2. GNHK training split: 4 random crops per word, seed 1
python -m haller.gnhk gnhk_train.npz GNHK/train 4 1

# 3. Training: 8 epochs, last epoch kept (for Haller-1 also the best on the
#    simulated evaluation strips, written by recipes/haller-1/evalsim/regenerate.sh)
python -m haller.train capitals.npz s{1..16}.npz gnhk_train.npz \
    --eval recipes/haller-1/data/haller-1-eval-simulated.npz --epochs 8

# 4. Quantization, calibrated on 3,000 words of shard 1
python -m haller.quantize capitals.npz s1.npz haller-1-capitals.npz --count 3000 --seed 1
```

The capitals expert was trained on GNHK crops made before `gnhk.py` applied the EXIF
orientation of the pages (about one page in six was cropped at the wrong
place); the current `gnhk.py` gives different, cleaner GNHK samples.

`train` also writes `capitals.json` (channels, epoch, scores). Other options:
`--channels C1 C2 C3`, `--hidden`, `--batch` (256), `--lr` (2e-3), `--seed`,
`--val VAL.npz` (second validation set, scored at every epoch),
`--select last|best` (weights written to the output: the last epoch, by
default, or the best one on the simulated EVAL score, with `--val` on the mean
of both scores), `--stop-after N` (stop after epoch N, the learning-rate
schedule still spanning `--epochs`), `--init WEIGHTS.npz` (start from a float
model written by `train`), `--fast` (shards cached and fed to the GPU by a
thread), `--keep-epochs`, `--eval-processes`
(processes of the evaluation by the C library, 1 by default), `--deterministic`
(bit-identical runs: deterministic GPU kernels, CTC loss on the CPU).
Quantization is deterministic: the same inputs give the same file on a given
processor; `--ranges WEIGHTS.npz` reuses the activation ranges of existing
weights and gives the same file on every machine.

**Version note.** The epoch is chosen on the quantized model, as computed by
SeedSigner's C library. The first Haller-1 weights were quantized at the
commit that introduced the name, with the calibration of the quantization in
numpy; its PyTorch version gives one unit of difference on 10 to 16 biases of
a layer (calibrated ranges differ by about 1e-6). The weights now shipped in
SeedSigner were regenerated by the current `haller.quantize` from the same
float models (same command, same inputs), so that step 4 above reproduces
them exactly from the published float models (see [What can be reproduced](#what-can-be-reproduced)): same share of strips read correctly on `EVAL.npz`
(capitals expert 96.32 % simulated, 98.26 % real; lowercase expert 96.13 % and
98.96 %), same five best words on the 62 strips of the reference vectors, and
on SeedSigner's 1,605 test photos the same results as before within one or two
seeds.

## Word detector

`haller-1-detector.npz` finds the words, the line numbers and the seed cards on
a photo reduced to 288 x 384 pixels (11 output maps, see `haller/detector.py`);
SeedSigner finds the words of every photo with it. `haller/detector.py` holds the network, its quantization
and the integer reference in numpy, `haller/train_detector.py` the training,
and `haller/detector_export.py` writes the weights read by the C library and
their reference vectors:

```bash
python -m haller.train_detector detector.npz PAGES_1.npz ... GNHK_PAGES.npz --val PAGES_VAL.npz --epochs 10 --fast
python -m haller.detector detector.npz PAGES_VAL.npz detector_q.npz --count 400   # --per-channel, --ranges R.npz
python -m haller.detector_export detector_q.npz OUT_DIR --name haller-1-detector
```

From the quantized model of Haller-1, `haller.detector_export` writes
`haller-1-detector.npz` and `haller-1-detector-vectors.json` byte for byte as
shipped in SeedSigner (quantized per channel, `--per-channel`). The training
pages of the shipped detector were written by a generator and a compositor
(simulated handwriting on photos of real blank supports and cards), checked
page by page by a variety audit and kept frozen in the recipe, and by
`haller.detector_gnhk` (GNHK pages with their word boxes); `haller.detector_pages` writes simulated
pages of the earlier 8-map detector (9 target channels, still read by the
training). The exact recipe of the detector of Haller-1 is
[`recipes/haller-1-detector/README.md`](recipes/haller-1-detector/README.md).

## Inference: SeedSigner's C library

The quantized network and the dictionary decoding are computed by
SeedSigner's C library (`src/seedsigner/helpers/handwriting/native/handwriting.c`),
the code that runs on the device; Haller has no inference code of its own.
`haller.native` builds it in the SeedSigner checkout found on `PYTHONPATH`
(with SeedSigner's Makefile; `make` and a C compiler are needed) and loads it
with `ctypes`. It is used by:

- `haller.evaluate`: share of strips read correctly by quantized weights
  (`python -m haller.evaluate WEIGHTS.npz STRIPS.npz [--by reel] [--processes N]`);
- `haller.train`: after each epoch, the model is quantized (calibration on
  3,000 words of the first shard) and evaluated by the C library, as the
  device will compute it; with `--select best`, the epoch is chosen on these
  scores;
- `haller.quantize` and `haller.vectors`: reference vectors.

The float model is computed by PyTorch only (`haller.model.folded_forward`,
with the exported weights), for the calibration of the quantization and the
filters.

### Reference vectors

`haller.quantize` writes, next to `OUT.npz`, `OUT-vectors.json` when given
`--strips` sources; `haller.vectors` writes them for existing weights:

```bash
python -m haller.vectors haller-1-capitals.npz \
    --strips capitals=recipes/haller-1/data/haller-1-eval-simulated.npz \
    --strips mixed=s1.npz --strips cursive=val.npz --float capitals.npz
```

For each source, 20 strips of spread widths (strips of real photos are never
used), plus an empty strip and a strip of noise; for each strip, the output
of the C library (SHA-256 of the logits and of the 2,048 word scores, 5 best
words and their scores). The file is shipped in SeedSigner next to the
weights, where `tests/test_handwriting_vectors.py` checks that any build of
the library gives the same bits. They must be written for every new set of
weights (Haller-2...) and copied into SeedSigner with them.

With `--float` (always, from `haller.quantize`), the strips are also read by
the float model and the CTC decoding of `haller.dictionary` (PyTorch, exact,
no beam), and the quantized C model must agree with it on the strips that
the float model reads clearly (best word with a log-likelihood of at least
-5, ahead of the second by at least 1): same best word, and a score that
differs by at most 2.5. Measured on 452 strips per Haller-1 expert: 613 such
strips, no different best word, largest score difference 1.94. Over all the
strips the best words differ on 5 to 6 %, always where the float model
hesitates or reads nothing (often words that are not BIP39 words). The check
fails the command otherwise; `tests/test_vectors.py` runs it again on the
vectors shipped in SeedSigner.

Names (optional, not in Haller-1):

```bash
python -m haller.names names_0.npz PARQUET [PARQUET ...] --part 0/8   # parts 0/8 to 7/8
python -m haller.filter read capitals.npz names names_*.npz            # -> names_exact.npz, names_near.npz
python -m haller.filter verify names_near.npz names_verified.npz --url http://127.0.0.1:8097/v1/vision
```

`verify` needs a local vision-model service at `--url` (threshold `--threshold`, 0.9).

Generated handwriting (optional): the 2,048 BIP39 words, in lowercase and in
capitals, written by the Emuru model in the hands of GNHK writers (one line of
a GNHK page as the style), then placed on a line and photographed like the
synthetic words. About a third of the GNHK styles give no usable words; a probe
batch skips the rest of such a style early. `filter read` keeps the generated
words that a trained model reads as their label (dictionary decoding, as on
the device, or letter by letter).

```bash
pip install -c requirements-lock.txt -c requirements-emuru-lock.txt -e '.[train,emuru]'   # on a GPU
python -m haller.emuru emuru.npz GNHK/train emuru_raw 7 --bip39 data/bip39_english.txt \
    --styles 500 --per-style 60 --batch 12
python -m haller.filter read capitals.npz emuru emuru.npz             # -> emuru_exact.npz
```

Generation is about 3 words per second on an RTX 3060; `emuru_raw` keeps the
generated words per style, so an interrupted run resumes where it stopped (each
file holds a key of the seed, the word list and the counts: a run with other
arguments stops on it). The code of the Emuru model is part of haller
(`haller/emuru_model.py`, MIT, copied from the Hugging Face repository at the
commit used for Haller-1), and every Hugging Face repository is read at a fixed
commit (`emuru_model.REVISIONS`): no code is downloaded and run.

## Reproducing Haller-1: lowercase expert

SeedSigner reads again with `haller-1-lowercase.npz` the photos whose seed
is not valid with the capitals expert. Same architecture, trained on more
lowercase and cursive writing:

```bash
# 1. Cursive shards: 8 x 50,000 words, 85 % lowercase, whole words in the 115
#    joined-script fonts listed in data/cursive_fonts.txt, seed 3000 + i
for i in $(seq 1 8); do
  python -m haller.generate c$i.npz 50000 $((3000 + i)) --cursive --fonts fonts_cursive \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait
# validation words of the same fonts, BIP39 labels only (1,274 of 3,000)
python -m haller.generate val_raw.npz 3000 3999 --cursive --fonts fonts_cursive \
    --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt
python -c "
import numpy as np
d = np.load('val_raw.npz'); bip39 = set(open('data/bip39_english.txt').read().split())
k = np.array([str(y) in bip39 for y in d['y']])
np.savez('val.npz', x=d['x'][k], w=d['w'][k], y=d['y'][k])"

# 2. GNHK with the EXIF orientation; samples that the capitals expert reads with a
#    character error rate above 0.8, or almost blank, are dropped
python -m haller.gnhk gnhk_train.npz GNHK/train 4 1
python -m haller.filter gnhk capitals.npz gnhk_train.npz gnhk_f.npz

# 3. Emuru, lowercase BIP39 words in the hands of the GNHK pages of
#    data/gnhk_cursive_pages.txt; raw words checked by the vision model
#    (threshold 0.9), then composed three times (seeds 31 to 33);
#    the strips of Haller-1 are recipes/haller-1/data/emuru_c3{1,2,3}.npz
python -m haller.emuru emuru_c.npz GNHK/train emuru_raw_c 21 --bip39 data/bip39_english.txt \
    --styles 200 --per-style 80 --batch 12 --lower --pages data/gnhk_cursive_pages.txt --generate-only

# 4. Training: 10 epochs, s1-s24 (seeds 1001 to 1024), the cursive data, and the
#    generated words of the "Generated handwriting" section (Haller-1:
#    recipes/haller-1/data/emuru_train.npz)
python -m haller.train lowercase.npz s{1..24}.npz c{1..8}.npz gnhk_f.npz gnhk_f.npz \
    emuru_exact.npz emuru_exact.npz emuru_c31.npz emuru_c32.npz emuru_c33.npz \
    --eval recipes/haller-1/data/haller-1-eval-simulated.npz --val val.npz --keep-epochs --epochs 10 --select best
python -m haller.quantize lowercase.npz s1.npz haller-1-lowercase.npz --count 3000 --seed 1
```

`data/cursive_fonts.txt` holds the fonts whose rendering of "minimum" has at
most 3 connected components, the evaluation fonts excluded (see
[Evaluation](#evaluation)). `data/gnhk_cursive_pages.txt` holds the GNHK pages
whose style line is handwritten (according to the vision model) and has at
most 0.9 connected components per letter. The raw Emuru words were checked
one by one by the vision model (the statement "The handwritten word in this
image is WORD.", as `haller.filter verify`) and composed by `haller.emuru.compose`
with seeds 31 to 33; [`recipes/haller-1`](recipes/haller-1/README.md) gives
the details and the data. The epoch selected was 10.

## Hardware

- Synthetic and real data: CPU, one process per shard.
- Generated handwriting (Emuru) and training: a single NVIDIA RTX 3060, about
  5 GB of VRAM; 8 epochs of the capitals expert take about 30 minutes (`train.py` falls
  back to the CPU without CUDA, much more slowly).
- Evaluation after each epoch (quantization and C library, on the CPU): about
  85 s for the 9,444 strips of the evaluation set with `--eval-processes 2`.
  The quantized Haller-1 capitals expert reads 96.32 % of its simulated strips
  and 98.26 % of its real strips (float model: 96.37 % and 98.26 %).

## Evaluation

`EVAL.npz` holds word strips cut from full photos of handwritten seeds by the
page reader of the time (`recipes/haller-1/evalsim/cut.py`), with their BIP39 word: 9,156 strips from
simulated photos and 288 from real photos (field `reel`, boolean). The metric is
the share of strips whose most probable BIP39 word is correct; since Haller-1
it is computed on the quantized model by SeedSigner's C library
(`haller.evaluate`; the table below was measured on the float models). The epochs of these models were selected
on simulated strips only (`--select best`); real strips are a test set. The simulated strips are regenerated from
the official EMNIST archive by `recipes/haller-1/evalsim/regenerate.sh`
(`recipes/haller-1/data/haller-1-eval-simulated.npz`); the real ones are not
published.

| Model | Channels | Data | Simulated | Real | Speed |
|---|---|---|---|---|---|
| Haller-1 capitals | 16/32/64 | synthetic + GNHK | 96.37 % | 98.26 % | 1x |
| Wide | 24/48/96 | synthetic + GNHK | 96.54 % | 98.61 % | 2x slower |
| B | 16/32/64 | synthetic + GNHK + Emuru | 96.43 % | 98.61 % | 1x |
| B2 | 16/32/64 | B + 8 synthetic shards + names | 96.41 % | 98.96 % | 1x |

Synthetic: 16 shards of 50,000 words; GNHK: train split; Emuru: 12,744
filtered words, listed twice; names: 151,000 filtered samples. The capitals expert 8
epochs, the others 10 (best epoch 8 for the wide model, 10 for B and B2).

### End to end

Development measurements of the word experts, with the engine of the time
(page rules before the word detector, no fingerprint check); the results of
the release are in [`releases/haller-1/EVALUATION.md`](releases/haller-1/EVALUATION.md).
The SeedSigner engine (page detection, word model, dictionary decoding,
checksum and confidence) on full photos of handwritten seeds. A seed is
*exact* when the 12 to 24 words read are the written ones, *sure and exact*
when the engine also reports it as valid, *false sure* when it reports a wrong
seed as valid. The 835 photos are 800 simulated photos (official cards, free
lists on blank or ruled paper, one or two columns, varied hands, lighting and
framing) and 35 real photos; the 460 others are simulated photos generated
independently, with other random seeds.

| Model | 835 photos: sure and exact / false sure / exact | 460 independent photos |
|---|---|---|
| Haller-1 capitals | 680 / 0 / 711 | 385 / 0 / 401 |
| B | 689 / 2 / 712 | 382 / 0 / 401 |
| B2 | 682 / 2 / 713 | 388 / 0 / 402 |
| Haller-1 lowercase alone | 1056 / 1 / 1106 (both sets) | |
| Capitals, then lowercase | 691 / 0 / 717 | 390 / 0 / 403 |
| Capitals, then lowercase or blend (Haller-1) | 692 / 0 / 718 | 391 / 0 / 403 |

The models have the same architecture and speed. The capitals expert alone
never reports a wrong seed as valid on these photos; the lowercase expert alone
reads lowercase and cursive much better but capitals slightly worse. Haller-1
in SeedSigner reads with the capitals expert, and reads again the photos whose
seed is not valid, with the lowercase expert (its seed needs a probability of
0.85 to be valid) and with a blend of both: for each word, the weighted sum of
the log-probabilities of the 2,048 words given by the capitals expert (0.65)
and the lowercase expert (0.35), which needs no more network passes. The most
probable reading is kept. Without false sure seeds, the blend weight can be
anywhere between 0.55 and 0.8; below, the blend reports wrong seeds as valid,
and so does the sum of both (product of experts). A style classifier choosing
the expert per page (capitals, lowercase, cursive) did not do better.

Lowercase and cursive, same engine: 301 simulated pages written in 13 fonts
never used in training (Playwrite FR Trad, GB J, DE SAS, IT Moderna, US
Modern, ES, NO, PL, VN, CA; League Script, Meddon, Fleur De Leah), and the GNHK
test split (BIP39 words written in lowercase, 2,980 crops):

| Engine | Cursive pages: sure and exact / false sure / exact | GNHK test, words |
|---|---|---|
| Haller-1 capitals | 50 / 0 / 86 | 64.6 % |
| Wide (24/48/96) | 68 / 1 / 108 | 64.3 % |
| B | 56 / 2 / 104 | 67.1 % |
| Haller-1 lowercase alone | 112 / 1 / 144 | 69.4 % |
| Capitals, then lowercase | 84 / 0 / 144 | |
| Blend of capitals (0.3 to 0.5) and lowercase, words only | | 71.2 % |
| Capitals, then lowercase or blend (Haller-1) | 95 / 0 / 147 | |

### Calibration

The engine uses the word scores of the decoder, divided by a temperature, as
log-likelihoods: the probability of a seed given the photo and the BIP39
checksum, and the probability of each word given the checksum (the percentage
SeedSigner shows next to a proposed word). The temperature and the
thresholds were chosen on 640 simulated pages kept apart from every test set:
400 pages of EMNIST test capitals and 240 pages in two lowercase fonts,
Playwrite DE Grund and Playwrite NZ Basic, not used by the test sets.
Playwrite NZ Basic is also one of the 287 training fonts; measured on the
calibration pages, this changes neither the reading of its pages (78.8 % of
the words, against 85.7 % for DE Grund) nor the choice of the temperature
(see [`recipes/haller-1`](recipes/haller-1/README.md#calibration-fonts)). The lowest
temperature with no wrong seed reported as valid is 2.2 (at 1.3 and 1.8, two
wrong seeds pass); a word below 0.95 given the checksum is then uncertain.

The networks are over-confident on single words: the probability of the best
word, before the temperature, is too high (expected calibration error 0.019
on the calibration pages, 0.025 on the test photos). The temperature that fits
the best word depends on the writing: 2.6 for capitals, about 1.7 for cards
and cursive; 2.2 is a compromise, and the likelihood of the true word among
the 30 candidates is best at 1.3. On the 1,605 test photos of SeedSigner
(1,301 in capitals, 301 in neat cursive), the percentage shown for a word has
a calibration error of 0.005 (0.003 for capitals, 0.020 for cursive): words
shown at 15 % are right 8 % of the time, at 45 % 52 %, at 75 % 82 %, at 86 %
89 %. The probability of the seed has a calibration error of 0.016 (0.070 with
the earlier products of word probabilities at temperature 1.2, which gave
15 to 35 % to seeds right 70 to 75 % of the time).

## Data sources and licenses

Datasets are downloaded by the user; no dataset is redistributed in this
repository. `recipes/haller-1/data` holds files derived from GNHK (words
generated in the style of GNHK writers, a selection mask of GNHK samples),
under the terms and attributions of [`NOTICE.md`](NOTICE.md). Files derived
from EMNIST are never stored: the simulated evaluation strips are regenerated
from the official archive.

- **Fonts**: 287 handwriting fonts from Google Fonts, plus 25 Playwrite fonts
  for the cursive shards, under the SIL Open Font License 1.1 or Apache
  License 2.0 (`data/fonts.sha256`, `data/fonts_cursive.sha256`). The per-font
  license list is written as `fonts/LICENSES.json` by `haller.fonts`.
- **EMNIST**: Cohen, Afshar, Tapson, van Schaik, "EMNIST: an extension of MNIST
  to handwritten letters", 2017; derived from NIST Special Database 19. No
  license is stated; the authors ask to be cited and NIST to be acknowledged
  as the source.
- **GNHK**: GoodNotes Handwriting in the Wild (GoodNotes), Lee et al.,
  "GNHK: A Dataset for English Handwriting in the Wild", ICDAR 2021, CC BY 4.0.
  Word images are cropped and normalized by `gnhk.py`.
- **Names**: Kaggle "Handwriting Recognition" dataset, CC0 (Hugging Face mirror
  `verbrannter/handwriting_recognition_from_kaggle`).
- **Emuru**: Pippi et al., "Zero-Shot Styled Text Image Generation, but Make
  It Autoregressive", CVPR 2025; model `blowing-up-groundhogs/emuru`, MIT.
- **BIP39 English wordlist** (`data/bip39_english.txt`): from the
  `bitcoin/bips` repository (MIT-licensed reference implementation).
- `data/other_words.txt`: the BIP39 lists of the other Latin-script languages,
  accents removed.

## License

MIT, see `LICENSE`.

## Citation

```bibtex
@software{haller1,
  title   = {Haller-1: a handwritten BIP39 word recognizer},
  author  = {{Pizza Bitcoiner}},
  year    = {2026},
  version = {haller-1},
  url     = {https://github.com/haller-project/haller}
}
```
