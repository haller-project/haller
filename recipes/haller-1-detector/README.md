# Haller-1 detector: exact recipe of the published weights

This directory reproduces `haller-1-detector.npz` and
`haller-1-detector-vectors.json`, the word detector shipped in SeedSigner
(`src/seedsigner/resources/handwriting/`, SHA-256 `8ddddf19…` and
`cc8564a0…`; 11 output channels, see `haller.detector`). It gives the code
that produced its 137 training and validation shards, frozen at the version
of the time (`gen/`), the exact training command, the float model it
produced, the activation ranges of its quantization and the export.

These shards were made on 20-21 November 2026, before the variety audit of
the synthetic data that the later generators of this project pass; they were
never audited. This recipe describes what was done, not a recommended way to
make training data.

The previous detector (8 output channels, SHA-256 `bcda59a2…`) and its
recipe (40 shards of `haller.detector_pages`) are in the history of this
directory, up to the commit that replaced them.

| Step | Reproduced | How |
|---|---|---|
| Simulated shards fv0, f0-f119 (`gen_univ.py`) | not exactly: about 3 % of the pages differ (environment of the run) | frozen generator, 287 pinned fonts, EMNIST train, card templates |
| Composed shards s0-s9, k0-k4, kb0 (`composer.py`) | exactly on Intel x86-64 (AVX-512), checked on s1, s5, k0, kb0 | frozen compositor, photos of blank supports and of official cards (manifest published, images not yet) |
| GNHK windows (train and test) | exactly | `haller.detector_gnhk` |
| Training | not the same bits | float model and log published (`data/`) |
| Quantization | exactly, with the published ranges | `haller.detector --ranges` |
| Export (weights and reference vectors for SeedSigner) | exactly, byte for byte | `haller.detector_export` |

## Files

| File | Content |
|---|---|
| `data/haller-1-detector-float.npz` | float model (batch normalization folded) after epoch 10, input of the quantization (SHA-256 `70400ced…`) |
| `data/train-detector.log` | its training log: losses and validation scores per epoch, then the ranges of a plain calibration on fv0 (not used, see [Quantization](#quantization)) |
| `data/haller-1-detector-ranges.npz` | `amax`: the 12 activation ranges of the shipped quantization, in the order of `haller.detector.LAYERS` |
| `data/quantize-photo.log` | ranges printed by that quantization (rounded) |
| `data/inputs.sha256` | SHA-256 of every input file of the compositor (images, masks, annotations, order files), under their public names |
| `data/order/{supports,cartes,variantes}.txt` | order in which the compositor reads the annotations (copy into each `annot` directory as `ordre.txt`) |
| `data/card_templates.json` | the 25 printed card templates drawn by `gen_univ.py`: vendor, license, source URL and page, SHA-256 of the rendered PNG |
| `gen/` | frozen generator, compositor, quantization and training scripts (see [Frozen code](#frozen-code)) |
| `SHA256SUMS.content` | content digests of the 137 shards, the GNHK windows, the validation file u40, the float and quantized models; SHA-256 of the exported files (comment lines at the end) |

`../haller-1/digest.py` computes and checks the content digests (arrays only,
independent of the zip container of the `.npz` files):

```bash
python recipes/haller-1/digest.py --check recipes/haller-1-detector/SHA256SUMS.content   # run where the files are
```

The `meta` array of the composed shards names the photo each page was made
from. The digests of `SHA256SUMS.content` are those of the shards with the
public names of the photos (`sv_*`, `cb_*`, `cv_*`), as the frozen code
writes them from the public inputs; the files used for the training carried
internal codes in `meta` and are otherwise the same arrays.

## Frozen code

`gen/` holds the files of the generator as they produced the shards, in their
original layout (the scripts find each other by relative paths). The
comments and identifiers are in French, as written. Changes made for the
publication, none of which changes the output (checked, see
[Verification](#verification)):

- paths: fonts from `HALLER_FONTS`, held-out test fonts from
  `HALLER_TEST_FONTS` (only for the test mode, not used here), haller package
  from `HALLER_DET` or this repository; local paths and machine names removed
  from the comments and the shell scripts;
- the module `train` (model loading of an older prototype, unused for
  generation) is always replaced by an empty module, as it already was on
  machines without PyTorch;
- `composer.py`: public names of the two papers whose printed lines are
  ignored (`sv_0130`, `sv_0152`); evaluation cards excluded by their
  `"jeu": "eval"` field alone (every card of the former list of internal
  evaluation codes has it); supports read in the order of `ANNOT/ordre.txt`
  (the order of the internal codes, under which the shards were drawn),
  since the public names do not sort in that order;
- `distributions_reelles.json`: a name removed from a comment string and from the label of one source of real photos (statistics read: `global` only, unchanged).

Main files:

| File | Role |
|---|---|
| `train/detecteur/gen_univ.py` | simulated pages (lists, sheets, printed cards from the templates, verso, altered words), 12 target channels |
| `train/detecteur/gen_det.py`, `train/pi/banc2/gen_sim.py`, `train/cartes.py`, `train/officiel.py`, `train/seed_sim.py`, … | its dependencies: handwriting from fonts and EMNIST, photo simulation, official SeedSigner cards (`officiel/`) |
| `train/supports/composer.py`, `lumiere.py`, `train/detecteur/supports/poser_ecriture.py` | compositor: simulated handwriting on photos of real blank supports and cards, lighting, occluders |
| `train/detecteur/distributions/distributions_reelles.json` | aggregated statistics of 76 private real photos (word height, orientation, colour temperature…), no image |
| `train/supports/cartes.py`, `traits.py`, `train/detecteur/modeles/annoter.py` | tools that made the compositor inputs (card variants, printed-line tracking, template boxes); run before the publication on internal names, not run by this recipe |
| `train/detecteur/gnhk_det.py` | earlier GNHK window extraction; `haller.detector_gnhk` gives the same content |
| `train/detecteur/nuit/quantifier.py`, `train_nuit.py` | the quantization run of the night (`--photo`), kept for the record |
| `train/detecteur/lots_univ5.sh`, `train/supports/lots_compo_u5.sh`, `train/detecteur/entrainer_univ5.sh` | the commands that made the shards and the training |

## Environment

The shards were made on three machines, in four Python environments, all
x86-64 Linux:

| Shards | Python | numpy | Pillow | scipy | opencv |
|---|---|---|---|---|---|
| fv0, f0-f75 | 3.12 | 1.26.4 | 12.2.0 | — | — |
| f76-f119 | 3.12.3 | 2.5.5 | 13.0.0 | (1.18.1) | (5.1.0.95) |
| s0-s4 | 3.13.1 | 2.5.5 | 13.0.0 | 1.18.1 | 5.1.0.95 (headless) |
| s5-s9, k0-k4, kb0 | 3.12.3 | 2.5.5 | 13.0.0 | 1.18.1 | 5.1.0.95 (headless) |

Training, quantization and export: `requirements-lock.txt` (Python 3.14.0,
PyTorch 2.15.0 with CUDA 13.2, numpy 2.5.5), training on an NVIDIA RTX 3060;
the export also needs the SeedSigner checkout given in the main README on
`PYTHONPATH`.

## Inputs

- Fonts: `python -m haller.fonts FONTS` (287 fonts, `data/fonts.sha256`), in
  `HALLER_FONTS`. Same 287 files on every machine of the run.
- EMNIST: `python -m haller.emnist EMNIST.npz IDX_DIR --split train`, then
  its digits and capitals in `gen/train/data/emnist_train.npz`:

  ```bash
  python -c "import numpy as np; d = np.load('EMNIST.npz'); k = d['y'] < 36
  np.savez('recipes/haller-1-detector/gen/train/data/emnist_train.npz', x=d['x'][k], y=d['y'][k])"
  ```

  533,993 images, identical in content and order to the file of the run
  (checked).
- Card templates: `gen/data/modeles_cartes/cases.json` (boxes of the 25
  templates) is here; the PNG files are not, because 18 of the 25 templates
  have no stated license that allows redistribution. `data/card_templates.json`
  gives, for each one, the source URL, the page and the SHA-256 of the PNG
  that was used (rendered from that page, in `gen/data/modeles_cartes/png/`).
- Photos of blank supports (162: papers, notebooks, cards, taken for this
  project, no handwriting) and of printed official cards (52 photos
  and 258 variants on paper backgrounds): **to be published; their manifest
  is published** (`data/inputs.sha256`, 1,213 files). The images used are the
  working versions (long side 2048 px) and the variants exactly as listed;
  the full-resolution exports that may be published elsewhere are not the
  same pixels. Layout expected by the compositor, `IN`:
  `supports/{annot,travail}`, `cartes/{annot,travail}`,
  `variantes/{annot,base}`, each `annot` holding the annotations, masks and
  `ordre.txt` (`data/order/`).
- GNHK: the `train` and `test` splits, checked against `data/gnhk.sha256`.
- `u40.npz` (validation only, 1,000 pages of an earlier configuration of the
  generator, seed 9040): not regenerated by this code; its digest is given.
  Validation draws no random numbers and does not choose the epoch, so it
  does not change the weights.

## Commands

From the root of the repository, `R=recipes/haller-1-detector`,
`G=$R/gen/train`, data written in `$R`:

```bash
# 1. Simulated shards: fv0 (seed 59999) and f0-f119 (seed 50000 + i), 1,000 pages each
OUT=$PWD/$R HALLER_FONTS=FONTS PYTHON=python3 bash $G/detecteur/lots_univ5.sh 8 fv:0:0 f:0:119

# 2. Composed shards: s0-s9 (70000 + i), k0-k4 (71000 + i), kb0 (72000)
OUT=$PWD/$R IN=IN HALLER_FONTS=FONTS PYTHON=python3 bash $G/supports/lots_compo_u5.sh 8 s:0:9 k:0:4 kb:0:0

# 3. GNHK windows: 6 per train page (seed 11), 2 per test page (seed 12)
python -m haller.detector_gnhk $R/gnhk_train.npz GNHK/train 6 11
python -m haller.detector_gnhk $R/gnhk_test.npz GNHK/test 2 12

# 4. Training (GPU, --fast loop): 10 epochs, the shards in this order, then gnhk_train 4 times
python -m haller.train_detector $R/detector-float.npz \
    $(for i in $(seq 0 119); do echo $R/f$i.npz; done) \
    $(for i in $(seq 0 9); do echo $R/s$i.npz; done) $(for i in $(seq 0 4); do echo $R/k$i.npz; done) $R/kb0.npz \
    $R/gnhk_train.npz $R/gnhk_train.npz $R/gnhk_train.npz $R/gnhk_train.npz \
    --arch base --epochs 10 --seed 1 --shards-in-memory 1 --fast \
    --val $R/fv0.npz $R/u40.npz $R/gnhk_test.npz $R/f0.npz

# 5. Quantization of the published float model with the published ranges
python -m haller.detector $R/data/haller-1-detector-float.npz $R/fv0.npz $R/haller-1-detector-q.npz \
    --ranges $R/data/haller-1-detector-ranges.npz

# 6. Weights and reference vectors for SeedSigner
mkdir -p $R/out
PYTHONPATH=$SS python -m haller.detector_export $R/haller-1-detector-q.npz $R/out
```

Step 4 is `entrainer_univ5.sh` with `TRAIN_OPTS=--fast GNHK_FOIS=4`; it
writes the model after each epoch and the last one under the given name,
with no selection. The order of the shards on the command line matters: the
training draws the shard order and the batches from its seed over the
positions of the files.

Step 5 prints the ranges of a plain calibration on the first 400 pages of
fv0, then the published ranges and the largest difference; it writes
`haller-1-detector-q.npz` (content digest `daf2a107…`). Step 6 prints the
largest difference between the C library and the integer reference in numpy
(0.0 for both vector images) and writes `haller-1-detector.npz` (SHA-256
`8ddddf19…`) and `haller-1-detector-vectors.json` (`cc8564a0…`).

## Quantization

The shipped quantization was not the plain calibration of `haller.detector`:
it calibrated on the first 400 pages of fv0 passed through the photo
augmentation of `train_nuit.py` (blur, gamma, contrast, lighting slope,
noise), with the frozen script:

```bash
cd $G/detecteur && python nuit/quantifier.py $R/data/haller-1-detector-float.npz $R/fv0.npz Q.npz --photo
```

The noise of that augmentation comes from PyTorch's default generator, which
the script never seeds: each run starts from another seed and gives other
ranges (three runs, three files, ranges within 0.02 of the shipped ones).
The ranges of the shipped run are therefore published, and step 5 gives the
shipped quantized model from them, byte for byte, on the machine of the run.
The quantization from given ranges is computed in float64 apart from the
ranges themselves.

## Training

`haller.train_detector --fast` trains on CUDA with bfloat16 autocast and
`cudnn.benchmark`; it has no deterministic mode and a second run is not
expected to give the same weights. The float model is therefore published,
and steps 5 and 6 give the shipped files exactly from it. `train-detector.log`
is the log of the run (output of step 4 and of the plain quantization of the
wrapper script), without two warnings of the Python launcher that held local
paths and a final marker line. For each epoch: mean losses, elapsed time,
precision and recall of words and numbers on fv0, u40, GNHK test and f0.
223,531 parameters, 4,369 s for 10 epochs.

## Verification

Checked on 21 November 2026 (content digests of `SHA256SUMS.content`):

- **Quantization and export.** From the published float model and ranges,
  step 5 gives the quantized model of the run byte for byte (same file,
  content `daf2a107…`), and step 6 gives `haller-1-detector.npz`
  (`8ddddf19…`) and `haller-1-detector-vectors.json` (`cc8564a0…`) byte for
  byte, on the x86-64 machine of the run.
- **Changes for the publication.** Original and published code, on 3 pages
  of each kind of shard (f, s, k, kb), give the same `x` and `y`, and the same
  `meta` once the internal codes are replaced by the public names.
- **Composed shards: identical.** s1 (Python 3.13.1), s5, k0 and kb0
  (Python 3.12.3), regenerated in full on an Intel Xeon Platinum 8375C (Ice
  Lake, the processor type of the original run of s5-s9, k and kb; s0-s4
  were made on an Intel Core i7-11800H): same content digests as the shards
  of the training. On an AMD EPYC 7R32 with the same versions, the same
  shards differ (k0: 86 pages by at most 5 gray levels; s and kb: the random
  draws part after a few hundred pages), so the composed shards are
  reproduced on Intel processors with AVX-512 only. s0 was made by a version
  of the compositor older by one commit, not kept; it was not regenerated.
- **Simulated shards: not identical.** f0 (Python 3.12, numpy 1.26.4,
  Pillow 12.2.0) and f76 (Python 3.12.3, numpy 2.5.5, Pillow 13.0.0), on the
  same Intel machine: 37 and 26 pages out of 1,000 differ (`meta` identical,
  so the same draws; targets different on 28 and 26 pages); on AMD the same
  pages differ, within 6. The original generator tree of the run, rerun on
  the same Intel machine, differs on the same page in its first 25: the
  difference does not come from the frozen code but from the environment of
  the run, which was not identified (same Python, numpy and Pillow versions,
  same FreeType 2.14.3 and raqm 0.10.3, same fonts, EMNIST file and card
  templates, same SHA-256). Pillow 10.2.0, which the base system of the
  machine of f76-f119 also had, cannot run the generator (`rounded_rectangle`
  rejects a box), so those shards were made with numpy 2.5.5 and Pillow
  13.0.0. The 120 simulated shards are therefore given by digest; a
  regeneration gives shards of the same distribution, about 97 % identical
  page for page.
- **GNHK windows**: same content as the windows of the previous detector
  recipe (`haller.detector_gnhk`).
- **EMNIST**: digits and capitals of `haller.emnist` identical in content and
  order to the file of the run.
