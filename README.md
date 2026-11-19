# Haller

Haller reads handwritten BIP39 words. This repository is its training
pipeline: from public data only, it produces the quantized weights read by
`seedsigner.helpers.handwriting` in a SeedSigner fork, to load a seed from a
photo of handwritten words
([`handwritten-seed` branch](https://github.com/pizza-bitcoiner/seedsigner/tree/handwritten-seed), see
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
| `haller-1-capitals.npz` | capitals | reads every photo | `1faede0402e7e3b48e922f8a82a3d72fcbb7e7354b4909298e31616431e7206a` |
| `haller-1-lowercase.npz` | lowercase and cursive | reads again the photos whose seed is not valid, alone and blended with the capitals expert | `55c2aba311f7e31e4fdbe6e82d17fc820165064c23c1792e1059fa6bf4e3863f` |

**Intended use.** Reading BIP39 seeds written by hand on an official
SeedSigner card or a free list of 12 to 24 words, on an offline device, with a
mandatory review of every word by the user against the paper.

**Training data.** Synthetic words (handwriting fonts and EMNIST letters,
"photographed"), real words from GNHK, and, for the lowercase expert, cursive
fonts and BIP39 words generated in the hands of GNHK writers by Emuru; see
[Data](#data) and [Data sources and licenses](#data-sources-and-licenses).

**Results** (SeedSigner engine on full photos, see [Evaluation](#evaluation)
for the protocol; seeds reported valid and exact / exact / wrong but reported
valid; words read correctly):

| Writing | Photos | Valid and exact | Exact | Wrong valid | Words |
|---|---|---|---|---|---|
| Capitals (simulated and real) | 1,295 | 1,083 | 1,121 | 0 | 87 % |
| Neat cursive (simulated, unseen fonts) | 301 | 95 | 147 | 0 | 70 % |
| Real photos, lowercase and cursive | 4 | 0 | 2 | 0 | 41 of 48 |

On a Raspberry Pi Zero with SeedSigner's C library: about 3.5 seconds to load
both experts, 5 seconds for a 12-word card, 9 to 12 seconds for a 12-word list
in capitals, 11 to 16 seconds in lowercase or cursive.

**Limits.** A reading can be wrong even with a valid checksum. Capitals are
read best; lowercase and cursive less well, a fast, sloppy cursive worst
(8 to 9 of 12 words on real photos), with the uncertain words marked. Blurry
photos, very small writing, lists cut by the frame of the photo and unusual
layouts fail. Only English BIP39 words are read. The real test photos are
few; the measurements rely mostly on simulated photos.

**Name.** Haller is named after Neil M. Haller of Bellcore, author of
RFC 1760, "The S/KEY One-Time Password System" (1995). S/KEY, designed with
Phil Karn and John Walden on an idea by Leslie Lamport, already wrote a secret
as words from a 2,048-word dictionary, 11 bits per word, with spare bits as a
checksum: the same layout BIP39 would use eighteen years later.

**License.** MIT (`LICENSE`), code and weights. The training data keep their
own licenses, listed in [Data sources and licenses](#data-sources-and-licenses).

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

## Setup

```bash
# a SeedSigner checkout that contains src/seedsigner/helpers/handwriting
export PYTHONPATH=/path/to/seedsigner/src
pip install -e '.[train,names]'
```

Inputs to download (not redistributed here):

- a directory of handwriting fonts (`.ttf`/`.otf`);
- EMNIST *byclass* training images as `EMNIST.npz`: `x` (N, 28, 28) uint8,
  white ink on black, upright; `y` (N,) class index in the order
  `0-9 A-Z a-z`, made by `python -m haller.emnist EMNIST.npz IDX_DIR` from
  the byclass IDX files of the official release;
- GNHK, split directory with the page `.jpg` files and their `.json` annotations;
- optionally the names dataset as Parquet files.

Run all commands from the repository root (default paths are relative to it).

## Reproducing Haller-1: capitals expert

```bash
# 1. Synthetic shards: 16 x 50,000 words, seed 1000 + i (one process per shard)
for i in $(seq 1 16); do
  python -m haller.generate s$i.npz 50000 $((1000 + i)) --fonts FONTS_DIR \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait

# 2. GNHK training split: 4 random crops per word, seed 1
python -m haller.gnhk gnhk_train.npz GNHK/train 4 1

# 3. Training: 8 epochs, best epoch on the simulated evaluation strips
python -m haller.train capitals.npz s{1..16}.npz gnhk_train.npz --eval EVAL.npz --epochs 8

# 4. Quantization, calibrated on 3,000 words of shard 1
python -m haller.quantize capitals.npz s1.npz haller-1-capitals.npz --count 3000 --seed 1
```

The capitals expert was trained on GNHK crops made before `gnhk.py` applied the EXIF
orientation of the pages (about one page in six was cropped at the wrong
place); the current `gnhk.py` gives different, cleaner GNHK samples.

`train` also writes `capitals.json` (channels, epoch, scores). Other options:
`--channels C1 C2 C3`, `--hidden`, `--batch` (256), `--lr` (2e-3), `--seed`,
`--val VAL.npz` (second validation set: the best epoch maximizes the mean of
the simulated EVAL score and of this one), `--keep-epochs`, `--eval-processes`
(processes of the evaluation by the C library, 1 by default).
Quantization is deterministic: the same inputs give the same file.

**Version note.** The Haller-1 weights were produced by the commands above at
tag `haller-1`, where the calibration of the quantization ran in numpy and the
epoch was chosen by the float model. Since then (see [Inference: SeedSigner's
C library](#inference-seedsigners-c-library)) both use other code: re-running
the commands gives files that differ from Haller-1 by one unit on a few biases
(10 to 28 of the 128 of a layer; calibrated ranges differ by about 1e-6), and
the epoch is chosen on the quantized model. The shipped Haller-1 weights are
unchanged.

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
  3,000 words of the first shard) and evaluated by the C library; the best
  epoch is chosen on these scores, as the device will compute them;
- `haller.quantize` and `haller.vectors`: reference vectors.

The float model is computed by PyTorch only (`haller.model.folded_forward`,
with the exported weights), for the calibration of the quantization and the
filters.

### Reference vectors

`haller.quantize` writes, next to `OUT.npz`, `OUT-vectors.json` when given
`--strips` sources; `haller.vectors` writes them for existing weights:

```bash
python -m haller.vectors haller-1-capitals.npz --strips capitals=EVAL.npz \
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
pip install torchvision 'transformers<5' diffusers einops   # with torch, on a GPU
python -m haller.emuru emuru.npz GNHK/train emuru_raw 7 --bip39 data/bip39_english.txt \
    --styles 500 --per-style 60 --batch 12
python -m haller.filter read capitals.npz emuru emuru.npz             # -> emuru_exact.npz
```

Generation is about 3 words per second on an RTX 3060; `emuru_raw` keeps the
generated words per style, so an interrupted run resumes where it stopped.

## Reproducing Haller-1: lowercase expert

SeedSigner reads again with `haller-1-lowercase.npz` the photos whose seed
is not valid with the capitals expert. Same architecture, trained on more
lowercase and cursive writing:

```bash
# 1. Cursive shards: 8 x 50,000 words, 85 % lowercase, whole words in the 115
#    joined-script fonts listed in data/cursive_fonts.txt, seed 3000 + i
for i in $(seq 1 8); do
  python -m haller.generate c$i.npz 50000 $((3000 + i)) --cursive --fonts CURSIVE_FONTS_DIR \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait
# validation words of the same fonts (BIP39 labels only are scored)
python -m haller.generate val.npz 3000 3999 --cursive --fonts CURSIVE_FONTS_DIR \
    --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt

# 2. GNHK with the EXIF orientation; samples that the capitals expert reads with a
#    character error rate above 0.8, or almost blank, are dropped (gnhk_f.npz)
python -m haller.gnhk gnhk_train.npz GNHK/train 4 1

# 3. Emuru, lowercase BIP39 words in the hands of the GNHK pages of
#    data/gnhk_cursive_pages.txt; raw words checked by the vision model
#    (threshold 0.9), then composed three times (seeds 31 to 33)
python -m haller.emuru emuru_c.npz GNHK/train emuru_raw_c 21 --bip39 data/bip39_english.txt \
    --styles 200 --per-style 80 --batch 12 --lower --pages data/gnhk_cursive_pages.txt --generate-only

# 4. Training: 10 epochs, all the data of B2's shards plus the above
python -m haller.train lowercase.npz s{1..24}.npz c{1..8}.npz gnhk_f.npz gnhk_f.npz \
    emuru_exact.npz emuru_exact.npz emuru_c31.npz emuru_c32.npz emuru_c33.npz \
    --eval EVAL.npz --val val.npz --keep-epochs --epochs 10
python -m haller.quantize lowercase.npz s1.npz haller-1-lowercase.npz --count 3000 --seed 1
```

`data/cursive_fonts.txt` holds the fonts whose rendering of "minimum" has at
most 3 connected components, the evaluation fonts excluded (see
[Evaluation](#evaluation)). `data/gnhk_cursive_pages.txt` holds the GNHK pages
whose style line is handwritten (according to the vision model) and has at
most 0.9 connected components per letter. The GNHK error filter, the vision
check of the raw Emuru words and their composition were run with helper
scripts that are not yet part of this repository. The epoch selected was 10.

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

`EVAL.npz` holds word strips cut by SeedSigner's page detection (`page.py`)
from full photos of handwritten seeds, with their BIP39 word: 9,156 strips from
simulated photos and 288 from real photos (field `reel`, boolean). The metric is
the share of strips whose most probable BIP39 word is correct; since Haller-1
it is computed on the quantized model by SeedSigner's C library
(`haller.evaluate`; the table below was measured on the float models). The best epoch is selected on simulated strips
only; real strips are a test set. The evaluation set is not published.

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

<!-- EVAL_V3: results on a new set of real photos (24 test seeds, varied paper,
pens, writers and lighting, taken with the SeedSigner camera) go here. -->

## Data sources and licenses

Datasets are downloaded by the user; none is redistributed in this repository.

- **Fonts**: 287 handwriting fonts from Google Fonts, plus 27 Playwrite fonts
  for the cursive shards, under the SIL Open Font License 1.1 or Apache
  License 2.0. The per-font license list is expected as
  `fonts/LICENSES.json`, alongside the font files.
- **EMNIST**: Cohen, Afshar, Tapson, van Schaik, "EMNIST: an extension of MNIST
  to handwritten letters", 2017; derived from NIST Special Database 19 (public
  domain, NIST terms of use).
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
  url     = {https://github.com/pizza-bitcoiner/haller}
}
```
