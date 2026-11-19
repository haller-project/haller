# seedsigner-ocr

Training pipeline of the handwritten-word recognizer used by a SeedSigner fork
to load a seed from a photo of handwritten BIP39 words
([`handwritten-seed` branch](https://github.com/pizza-bitcoiner/seedsigner/tree/handwritten-seed), see
`docs/handwritten_seeds.md` there). It produces
`src/seedsigner/resources/handwriting/word_model.npz`, the quantized weights
read by `seedsigner.helpers.handwriting`, from public data only. Photos taken
by users are never used.

## Status: proof of concept

The reading engine and these weights are complete and measured (see
[End to end](#end-to-end)). The SeedSigner integration is a proof of concept:
its user experience is not final and it is not proposed upstream. The shipped
weights have the SHA-256
`1faede0402e7e3b48e922f8a82a3d72fcbb7e7354b4909298e31616431e7206a` (model A,
`word_model.npz`) and
`55c2aba311f7e31e4fdbe6e82d17fc820165064c23c1792e1059fa6bf4e3863f` (model C2,
`word_model_lowercase.npz`).

The model is a small fully convolutional network (`seedsigner_ocr/model.py`,
about 164,000 parameters with the default channels 16/32/64) trained with CTC
on 37 classes: blank, a-z (case-insensitive), 0-9. On the device its output is
scored against the 2,048 BIP39 words; there is no letter segmentation.

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
- **Real words** from GNHK (`gnhk.py`): each annotated word is straightened and
  cropped with random margins, so that the crop includes part of its
  surroundings as on a SeedSigner line, then normalized.
- **Real names** from scanned forms (`names.py`, `filter.py`): several
  candidate crops per image; a trained model keeps the candidate it reads as
  the label, and one-edit readings are confirmed by a vision model. This set
  was not used for the shipped weights.

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
  `0-9 A-Z a-z`, made by `python -m seedsigner_ocr.emnist EMNIST.npz IDX_DIR` from
  the byclass IDX files of the official release;
- GNHK, split directory with the page `.jpg` files and their `.json` annotations;
- optionally the names dataset as Parquet files.

Run all commands from the repository root (default paths are relative to it).

## Reproducing the shipped weights

```bash
# 1. Synthetic shards: 16 x 50,000 words, seed 1000 + i (one process per shard)
for i in $(seq 1 16); do
  python -m seedsigner_ocr.generate s$i.npz 50000 $((1000 + i)) --fonts FONTS_DIR \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait

# 2. GNHK training split: 4 random crops per word, seed 1
python -m seedsigner_ocr.gnhk gnhk_train.npz GNHK/train 4 1

# 3. Training: 8 epochs, best epoch on the simulated evaluation strips
python -m seedsigner_ocr.train model_a.npz s{1..16}.npz gnhk_train.npz --eval EVAL.npz --epochs 8

# 4. Quantization, calibrated on 3,000 words of shard 1
python -m seedsigner_ocr.quantize model_a.npz s1.npz word_model.npz --count 3000 --seed 1
```

Model A was trained on GNHK crops made before `gnhk.py` applied the EXIF
orientation of the pages (about one page in six was cropped at the wrong
place); the current `gnhk.py` gives different, cleaner GNHK samples.

`train` also writes `model_a.json` (channels, epoch, scores). Other options:
`--channels C1 C2 C3`, `--hidden`, `--batch` (256), `--lr` (2e-3), `--seed`,
`--val VAL.npz` (second validation set: the best epoch maximizes the mean of
the simulated EVAL score and of this one), `--keep-epochs`.
Quantization is deterministic: the same inputs give the same file.

Names (optional, not in the shipped weights):

```bash
python -m seedsigner_ocr.names names_0.npz PARQUET [PARQUET ...] --part 0/8   # parts 0/8 to 7/8
python -m seedsigner_ocr.filter read model_a.npz names names_*.npz            # -> names_exact.npz, names_near.npz
python -m seedsigner_ocr.filter verify names_near.npz names_verified.npz --url http://127.0.0.1:8097/v1/vision
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
python -m seedsigner_ocr.emuru emuru.npz GNHK/train emuru_raw 7 --bip39 data/bip39_english.txt \
    --styles 500 --per-style 60 --batch 12
python -m seedsigner_ocr.filter read model_a.npz emuru emuru.npz             # -> emuru_exact.npz
```

Generation is about 3 words per second on an RTX 3060; `emuru_raw` keeps the
generated words per style, so an interrupted run resumes where it stopped.

## Reproducing the second (lowercase) weights

SeedSigner reads again with `word_model_lowercase.npz` the photos whose seed
is not valid with the first weights. Same architecture, trained on more
lowercase and cursive writing:

```bash
# 1. Cursive shards: 8 x 50,000 words, 85 % lowercase, whole words in the 115
#    joined-script fonts listed in data/cursive_fonts.txt, seed 3000 + i
for i in $(seq 1 8); do
  python -m seedsigner_ocr.generate c$i.npz 50000 $((3000 + i)) --cursive --fonts CURSIVE_FONTS_DIR \
      --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait
# validation words of the same fonts (BIP39 labels only are scored)
python -m seedsigner_ocr.generate val.npz 3000 3999 --cursive --fonts CURSIVE_FONTS_DIR \
    --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt

# 2. GNHK with the EXIF orientation; samples that model A reads with a
#    character error rate above 0.8, or almost blank, are dropped (gnhk_f.npz)
python -m seedsigner_ocr.gnhk gnhk_train.npz GNHK/train 4 1

# 3. Emuru, lowercase BIP39 words in the hands of the GNHK pages of
#    data/gnhk_cursive_pages.txt; raw words checked by the vision model
#    (threshold 0.9), then composed three times (seeds 31 to 33)
python -m seedsigner_ocr.emuru emuru_c.npz GNHK/train emuru_raw_c 21 --bip39 data/bip39_english.txt \
    --styles 200 --per-style 80 --batch 12 --lower --pages data/gnhk_cursive_pages.txt --generate-only

# 4. Training: 10 epochs, all the data of B2's shards plus the above
python -m seedsigner_ocr.train model_c2.npz s{1..24}.npz c{1..8}.npz gnhk_f.npz gnhk_f.npz \
    emuru_exact.npz emuru_exact.npz emuru_c31.npz emuru_c32.npz emuru_c33.npz \
    --eval EVAL.npz --val val.npz --keep-epochs --epochs 10
python -m seedsigner_ocr.quantize model_c2.npz s1.npz word_model_lowercase.npz --count 3000 --seed 1
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
  5 GB of VRAM; 8 epochs of model A take about 30 minutes (`train.py` falls
  back to the CPU without CUDA, much more slowly).

## Evaluation

`EVAL.npz` holds word strips cut by SeedSigner's page detection (`page.py`)
from full photos of handwritten seeds, with their BIP39 word: 9,156 strips from
simulated photos and 288 from real photos (field `reel`, boolean). The metric is
the share of strips whose most probable BIP39 word (CTC dictionary scoring,
`dictionary.py`) is correct. The best epoch is selected on simulated strips
only; real strips are a test set. The evaluation set is not published.

| Model | Channels | Data | Simulated | Real | Speed |
|---|---|---|---|---|---|
| A (shipped) | 16/32/64 | synthetic + GNHK | 96.37 % | 98.26 % | 1x |
| Wide | 24/48/96 | synthetic + GNHK | 96.54 % | 98.61 % | 2x slower |
| B | 16/32/64 | synthetic + GNHK + Emuru | 96.43 % | 98.61 % | 1x |
| B2 | 16/32/64 | B + 8 synthetic shards + names | 96.41 % | 98.96 % | 1x |

Synthetic: 16 shards of 50,000 words; GNHK: train split; Emuru: 12,744
filtered words, listed twice; names: 151,000 filtered samples. Model A 8
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
| A (shipped) | 680 / 0 / 711 | 385 / 0 / 401 |
| B | 689 / 2 / 712 | 382 / 0 / 401 |
| B2 | 682 / 2 / 713 | 388 / 0 / 402 |
| C2 alone | 1056 / 1 / 1106 (both sets) | |
| A, then C2 (shipped) | 691 / 0 / 717 | 390 / 0 / 403 |

The models have the same architecture and speed. A alone never reports a
wrong seed as valid on these photos; C2 alone reads lowercase and cursive much
better but capitals slightly worse. The shipped engine reads with A, and reads
again with C2 the photos whose seed is not valid (a C2 seed needs a
probability of 0.85 to be valid).

Lowercase and cursive, same engine: 301 simulated pages written in 13 fonts
never used in training (Playwrite FR Trad, GB J, DE SAS, IT Moderna, US
Modern, ES, NO, PL, VN, CA; League Script, Meddon, Fleur De Leah), and the GNHK
test split (BIP39 words written in lowercase, 2,980 crops):

| Engine | Cursive pages: sure and exact / false sure / exact | GNHK test, words |
|---|---|---|
| A | 50 / 0 / 86 | 64.6 % |
| Wide (24/48/96) | 68 / 1 / 108 | 64.3 % |
| B | 56 / 2 / 104 | 67.1 % |
| C2 alone | 112 / 1 / 144 | 69.4 % |
| A, then C2 (shipped) | 84 / 0 / 144 | |

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
