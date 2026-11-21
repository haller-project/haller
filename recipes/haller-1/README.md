# Haller-1: exact recipe of the published weights

This directory reproduces `haller-1-capitals.npz` and `haller-1-lowercase.npz`,
the weights shipped in SeedSigner, from public inputs. It holds the code
versions that made their data, the data that cannot be regenerated
bit for bit, the float models and the checksums of every intermediate file.
The commands of the main README use the current code; they give data and
weights of the same kind, not these ones.

| Step | Reproduced | How |
|---|---|---|
| Synthetic shards s1-s24 (capitals) | exactly | `haller1.generate` (code of the time, below), pinned versions |
| Cursive shards c1-c8 and cursive validation | exactly | current `haller.generate`, pinned versions |
| GNHK crops (both experts) | exactly | `haller1.gnhk` (capitals), current `haller.gnhk` (lowercase) |
| GNHK error filter (lowercase) | exactly | `haller.filter gnhk` with the published selection mask |
| Emuru words (lowercase) | published as data | generated on a GPU and checked by a vision model |
| Simulated evaluation strips | exactly | `evalsim/regenerate.sh`: official EMNIST archive, `haller.pages`, page reader of the time |
| Training | same level, not the same bits | float models published; see [Training](#training) |
| Quantization and reference vectors | exactly | current `haller.quantize`, from the published float models (see [Quantization](#quantization)) |

## Files

| File | Content |
|---|---|
| `haller1/` | the package as it was when the data and the float models were made (`synth.py`, `generate.py`, `gnhk.py`: 17/11/2026; `model.py`, `dictionary.py`, `train_capitals.py`: training of the capitals expert; `train_lowercase.py`: training of the lowercase expert). Files kept byte for byte; their docstrings use the earlier package name `seedsigner_ocr`. |
| `data/haller-1-capitals-float.npz`, `data/haller-1-lowercase-float.npz` | float models (batch normalization folded), inputs of the quantization |
| `data/train-capitals.log`, `data/train-lowercase.log` | their training logs (score per epoch) |
| `evalsim/` | regeneration of `data/haller-1-eval-simulated.npz`, the 9,156 simulated strips of `EVAL.npz` (derived from EMNIST, not stored): `regenerate.sh`, the page reader of the time (`cut.py`, `preprocess.py`), the page order (`pages.txt`) and the SHA-256 of the 800 photos (`photos.sha256`) |
| `data/gnhk_train_f.mask.npy` | GNHK samples kept by the error filter (packed booleans) |
| `data/emuru_train.npz` | 12,744 generated words, capitals and lowercase |
| `data/emuru_c31.npz`, `emuru_c32.npz`, `emuru_c33.npz` | generated lowercase words, three compositions |
| `SHA256SUMS.content` | content digest of every input, intermediate and output file |
| `digest.py` | computes and checks these digests |

The SHA-256 of an `.npz` file changes at every write (numpy stores the time
of writing in the archive). `digest.py` hashes the arrays only, so that a
regenerated file can be compared with the original:

```bash
python recipes/haller-1/digest.py --check recipes/haller-1/SHA256SUMS.content   # run where the files are
```

## Environment

Data (shards, GNHK crops): Python 3.12, numpy 1.26.4, Pillow 12.2.0 (wheels
with FreeType 2.13.2 and raqm 0.10.3), embit 0.8.0. Other versions change a few
samples: with Python 3.14, numpy 2.5.5 and Pillow 13.0.0, sample 357 of the
first 400 of s1 differs.

```bash
uv venv -p 3.12 venv-data
uv pip install -p venv-data/bin/python numpy==1.26.4 pillow==12.2.0 embit==0.8.0
```

Training and quantization: `requirements-lock.txt` (PyTorch 2.15.0 with CUDA
13.2). The capitals expert was trained with PyTorch 2.11.0 (CUDA 13.0, cuDNN
9.19), the lowercase expert with PyTorch 2.15.0 (CUDA 13.2, cuDNN 9.26), on an
NVIDIA RTX 3060.

Inputs, as in the main README: the 287 and 115 fonts (`haller.fonts`, checked
against `data/fonts.sha256` and `data/fonts_cursive.sha256`), EMNIST
(`haller.emnist`), the GNHK train split (checked against `data/gnhk.sha256`),
and the SeedSigner checkout given in the main README on `PYTHONPATH`. Its
`preprocess.py` gives the same samples as the version of 17/11/2026 (the later
changes only return `None` where the earlier code raised `ValueError`, and both
skip the sample).

In the commands below, `FONTS` and `FONTS_CURSIVE` are the two font
directories, `GNHK` the GNHK directory, `SS` the `src` directory of the
SeedSigner checkout, and `R=recipes/haller-1`.

## Capitals expert

```bash
# 1. Shards s1-s16, code of 17/11/2026 (run from recipes/haller-1)
cd $R
for i in $(seq 1 16); do
  PYTHONPATH=.:$SS venv-data/bin/python -m haller1.generate s$i.npz 50000 $((1000 + i)) --fonts FONTS \
      --emnist EMNIST.npz --bip39 ../../data/bip39_english.txt --other ../../data/other_words.txt &
done; wait

# 2. GNHK crops without the EXIF orientation
PYTHONPATH=.:$SS venv-data/bin/python -m haller1.gnhk gnhk_capitals.npz GNHK/train 4 1

# 3. Training (GPU required by this script): 8 epochs, the shards in this order
PYTHONPATH=.:$SS python -m haller1.train_capitals capitals.npz \
    s1.npz s10.npz s11.npz s12.npz s13.npz s14.npz s15.npz s16.npz \
    s2.npz s3.npz s4.npz s5.npz s6.npz s7.npz s8.npz s9.npz gnhk_capitals.npz \
    --eval data/haller-1-eval-simulated.npz --epochs 8 --bip39 ../../data/bip39_english.txt
cd ../..

# 4. Quantization (current code): gives haller-1-capitals.npz exactly
python -m haller.quantize $R/data/haller-1-capitals-float.npz $R/s1.npz haller-1-capitals.npz --count 3000 --seed 1 \
    --ranges $SS/seedsigner/resources/handwriting/haller-1-capitals.npz
```

The original run selected epoch 8 (`data/train-capitals.log`: 96.37 % of the
simulated strips, 98.26 % of the real ones, measured on the float model).

## Lowercase expert

```bash
cd $R
# 1. Shards s17-s24: same command as s1-s16
for i in $(seq 17 24); do
  PYTHONPATH=.:$SS venv-data/bin/python -m haller1.generate s$i.npz 50000 $((1000 + i)) --fonts FONTS \
      --emnist EMNIST.npz --bip39 ../../data/bip39_english.txt --other ../../data/other_words.txt &
done; wait
cd ../..

# 2. Cursive shards and cursive validation: current code
for i in $(seq 1 8); do
  PYTHONPATH=.:$SS $R/venv-data/bin/python -m haller.generate $R/c$i.npz 50000 $((3000 + i)) --cursive \
      --fonts FONTS_CURSIVE --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt &
done; wait
PYTHONPATH=.:$SS $R/venv-data/bin/python -m haller.generate $R/val_raw.npz 3000 3999 --cursive \
    --fonts FONTS_CURSIVE --emnist EMNIST.npz --bip39 data/bip39_english.txt --other data/other_words.txt
python -c "
import numpy as np
d = np.load('$R/val_raw.npz'); bip39 = set(open('data/bip39_english.txt').read().split())
k = np.array([str(y) in bip39 for y in d['y']])
np.savez('$R/val.npz', x=d['x'][k], w=d['w'][k], y=d['y'][k])"     # 1,274 BIP39 words

# 3. GNHK crops with the EXIF orientation, then the error filter of the capitals expert
PYTHONPATH=.:$SS $R/venv-data/bin/python -m haller.gnhk $R/gnhk_train.npz GNHK/train 4 1
python -m haller.filter gnhk $R/data/haller-1-capitals-float.npz $R/gnhk_train.npz $R/gnhk_f.npz \
    --mask $R/data/gnhk_train_f.mask.npy

# 4. Training (GPU required by this script): 10 epochs, the files in this order
cd $R
PYTHONPATH=.:$SS python -m haller1.train_lowercase lowercase.npz \
    s1.npz s10.npz s11.npz s12.npz s13.npz s14.npz s15.npz s16.npz s17.npz s18.npz s19.npz \
    s2.npz s20.npz s21.npz s22.npz s23.npz s24.npz s3.npz s4.npz s5.npz s6.npz s7.npz s8.npz s9.npz \
    c1.npz c2.npz c3.npz c4.npz c5.npz c6.npz c7.npz c8.npz gnhk_f.npz gnhk_f.npz \
    data/emuru_train.npz data/emuru_train.npz data/emuru_c31.npz data/emuru_c32.npz data/emuru_c33.npz \
    --eval data/haller-1-eval-simulated.npz --val val.npz --keep-epochs --epochs 10 \
    --bip39 ../../data/bip39_english.txt
cd ../..

# 5. Quantization (current code): gives haller-1-lowercase.npz exactly
python -m haller.quantize $R/data/haller-1-lowercase-float.npz $R/s1.npz haller-1-lowercase.npz --count 3000 --seed 1 \
    --ranges $SS/seedsigner/resources/handwriting/haller-1-lowercase.npz
```

The original run selected epoch 10 on the mean of the simulated EVAL score
and of the cursive validation (`data/train-lowercase.log`).

**GNHK filter.** The capitals expert reads every crop without dictionary;
crops with a character error rate above 0.8 or almost blank are dropped
(86,911 of 100,343 kept). The original selection was computed on a GPU; on a
CPU, 4 crops at the threshold come out differently, so `--mask` applies the
published selection (the command prints how many crops its own reading would
change).

**Emuru words (published as data).** BIP39 words written by the Emuru model
(`blowing-up-groundhogs/emuru`, MIT) in the hands of GNHK writers, one line of
a GNHK train page as the style, then placed on a line and photographed like
the synthetic words (`haller.emuru`):

- `emuru_train.npz`: the 2,048 words in lowercase and capitals, 285 styles of
  the train split, up to 60 words per style (the run was stopped after the
  285th style), composed four times with different seeds (15,662 strips),
  then filtered by the capitals expert
  (`haller.filter read`: read exactly, or decoded as its label with the BIP39
  dictionary: 12,621) and by the vision model for the strips read one edit away
  (`haller.filter verify`, threshold 0.9: 123 of 204), 12,744 strips;
- `emuru_c31-33.npz`: lowercase words only (`--lower`), styles of the 98 pages
  of `data/gnhk_cursive_pages.txt`, seed 21, `--styles 200 --per-style 80
  --generate-only`: 5,172 raw words, of which the vision model confirmed 4,623
  (threshold 0.9, each raw word on a blank line), composed with seeds 31, 32
  and 33 (3,278, 3,309 and 3,302 strips).

The vision model was Qwen3-VL-4B-Instruct (official GGUF files
`Qwen/Qwen3-VL-4B-Instruct-GGUF`, weights Q4_K_M, projector F16) served by
llama.cpp; the score of a word is the probability of the answer "1" to the
statement "The handwritten word in this image is WORD." Generation runs on a
GPU with sampling, and the vision scores depend on the server and its
batching: neither is bit-reproducible, so the selected strips are published
here (images generated from GNHK styles, CC BY 4.0 for the styles, MIT for the
model).

## Quantization

The quantization calibrates six activation ranges (99.99th percentile of each
layer's float32 activations over 3,000 words of s1), then rounds in float64.
The calibrated ranges depend slightly on the processor: on the machine that
made the weights (CPU or GPU) the command without `--ranges` gives the shipped
files exactly; on another x86-64 processor the ranges differed by at most
1.4e-6 and 10 to 20 biases by one unit. `--ranges` takes the six ranges stored
in the shipped file (field `amax`) instead and prints their difference with
the calibrated ones: the result is then the same on every machine. The
reference vectors are written again byte for byte by `haller.vectors` from
the shipped weights, the evaluation strips of `data/`, s1 and `val.npz`:

```bash
python -m haller.vectors haller-1-capitals.npz --strips capitals=$R/data/haller-1-eval-simulated.npz \
    --strips mixed=$R/s1.npz --strips cursive=$R/val.npz --float $R/data/haller-1-capitals-float.npz
```

## Evaluation strips

`EVAL.npz` holds 9,156 strips of simulated photos and 288 strips of real
photos of one person's handwriting. Only the simulated strips choose the
epoch; the real ones are a test set and stay private.
`data/haller-1-eval-simulated.npz` is the simulated part (fields `x`, `w`,
`y`, `page`, `layout` and `reel`, always false): with it, both training
scripts and `haller.train` select the epoch exactly as with `EVAL.npz`. The
simulated photos are seed lists and SeedSigner cards written with EMNIST test
letters (never seen in training).

The strips are derived from EMNIST, so the repository holds the way to make
them instead of the file. Run before the commands above that read it:

```bash
uv venv -p 3.12 venv-evalsim && uv pip install -p venv-evalsim/bin/python numpy==2.5.5 pillow==12.2.0
PYTHON=venv-evalsim/bin/python recipes/haller-1/evalsim/regenerate.sh evalsim-work 4   # work directory, processes
```

Pillow 13 changed its bilinear resize: with it, 6 of the 9,156 strips differ
by one gray level, so the script requires Pillow < 13 (versions 10.4 to 12.2
give the same strips; the photos are the same with Pillow 13 too). The
photos need numpy 2.x, and the script checks it: with numpy 1.26.4, 797 of
the 800 photos differ. Checked identical with Python 3.12.3, numpy 2.5.5 and
Pillow 12.2.0.

The script downloads the EMNIST archive from NIST (562 MB, SHA-256 checked),
converts the byclass test split with `haller.emnist`, writes 800 photos with
`haller.pages` (seeds 11 to 18, 100 photos each; every JPEG file is checked
against `evalsim/photos.sha256`), cuts them into word strips with
`evalsim/cut.py`, and checks the content digest of the result against
`SHA256SUMS.content`. `cut.py` is the page reader that cut the original
strips, reduced to the code that cutting uses and translated to English;
`evalsim/preprocess.py` is SeedSigner's word normalization as it was then.
Photos whose number of strips differs from their number of words are
skipped, as in the original run (654 of the 800 photos are kept), and
`evalsim/pages.txt` restores the original order of the pages.

## Checksums

`SHA256SUMS.content` lists the content digest (`digest.py`) of every file of
the two recipes: shards, GNHK crops and filtered crops, cursive validation,
Emuru data, evaluation strips, float models; then the SHA-256 of the
quantized weights. All the regenerated files of the table at the top were
compared with the originals of Haller-1: identical arrays.

## Training

The original runs used `torch.backends.cudnn.benchmark` and the CUDA CTC loss,
whose backward pass is not deterministic: two runs with the same data and seed
give different weights. The float models are therefore published, and the
quantization gives the shipped weights exactly from them.

The current `haller.train` has a `--deterministic` option (deterministic GPU
kernels, CTC loss on the CPU): two runs with the same data, seed, hardware and
software give the same weights.

**Retraining check (capitals expert).** The exact data of Haller-1 (s1-s16
and `gnhk_capitals.npz`, in the order above), the current `haller.train` with
`--deterministic`, the regenerated evaluation strips, 8 epochs. The losses per
epoch follow the original run to within 0.002 (1.5768 against 1.5784 at epoch
1, 0.6668 against 0.6683 at epoch 8), and epoch 8 is selected again. Two runs
give the same weights after their first epoch (same SHA-256). Quantized and
compared with the published weights:

| Capitals expert | EVAL simulated | EVAL real | GNHK test words | 1,605 photos: valid and exact / exact / wrong valid |
|---|---|---|---|---|
| Haller-1 (published) | 96.32 % | 98.26 % | 64.63 % | 1,197 / 1,288 / 0 |
| Retrained | 96.36 % | 97.92 % | 65.60 % | 1,201 / 1,290 / 1 |

(SeedSigner engine of the `handwritten-seed` commit given in the main README,
lowercase expert unchanged; the 288 real strips are one strip apart.) The
level is the same; the retrained weights report one simulated list in
capitals as valid with a wrong word, which the published weights read
correctly. The absence of wrong valid seeds on the test photos is measured on
the published weights, not guaranteed by the recipe: weights retrained from it
must be measured again before replacing them.

**Retraining check (lowercase expert).** The exact data (the files of the
lowercase recipe, in its order), the current `haller.train` with
`--deterministic`, `--val val.npz`, 10 epochs: epoch 10 is selected again,
the losses follow the original run to within 0.01. Quantized; the
1,605-photo column is the whole engine (published capitals expert, the
lowercase expert replaced):

| Lowercase expert | EVAL simulated | EVAL real | Cursive validation | GNHK test words | 1,605 photos | Neat cursive (301) |
|---|---|---|---|---|---|---|
| Haller-1 (published) | 96.13 % | 98.96 % | 80.14 % | 69.40 % | 1,197 / 1,288 / 0 | 106 / 153 / 0 |
| Retrained | 96.22 % | 98.26 % | 81.48 % | 69.19 % | 1,188 / 1,292 / 0 | 98 / 155 / 0 |

Same level, no wrong valid seed; the published weights report 9 more seeds
as valid and exact, and stay the shipped ones.

## Calibration fonts

The temperature of SeedSigner's word scores was chosen on 640 simulated pages:
400 in EMNIST test capitals and 240 in two lowercase fonts, Playwrite DE
Grund and Playwrite NZ Basic. Playwrite NZ Basic is also one of the 287 fonts
of the shards s1-s24 (about 0.2 % of the font-written words). Measured with
the shipped engine on the calibration pages: the NZ Basic pages are read less
well than the DE Grund pages (78.8 % of the words against 85.7 %, temperature
2.2) and give no wrong seed reported as valid at any temperature from 1.3 to
3.0; without them, the lowest temperature with no wrong valid seed is
unchanged (2.0 with the word threshold 0.95, 2.4 with 0.9), and the shipped
temperature 2.2 gives none. The weights and the temperature are kept.
