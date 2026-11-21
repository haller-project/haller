# Haller-1 detector: exact recipe of the published weights

This directory reproduces `haller-1-detector.npz` and
`haller-1-detector-vectors.json`, the word detector shipped in SeedSigner
(`src/seedsigner/resources/handwriting/`, SHA-256 `b33f6d02…` and
`7c995ffa…`; 11 output channels, see `haller.detector`). It gives the code
that produced its 138 training and validation shards, frozen at the version
of the time (`gen/`), the variety audit of every shard, the exact training
command, the float model it produced, the per-channel activation ranges of
its quantization and the export.

The earlier detectors and their recipes (11 output channels, SHA-256
`8ddddf19…`, trained on shards made before the variety audit; 8 output
channels, `bcda59a2…`) are in the history of this directory, up to the
commits that replaced them.

| Step | Reproduced | How |
|---|---|---|
| Simulated shards fv0, f0-f120 (`gen_univ.py`) | exactly, checked on f5 and f100 (filter included) | frozen generator, 287 pinned fonts, EMNIST train, card templates |
| Composed shards s0-s9, k0-k4, kb0 (`composer.py`) | exactly, checked on s5 and k0 | frozen compositor, photos of blank supports and of official cards (manifest published, images not yet) |
| Variety audit and filter of every shard | the audit of each shard is published (`data/audits.tsv`) | `variete.audit`, `variete.filtrer` |
| GNHK windows (train and test) | exactly | `haller.detector_gnhk` |
| Training | not the same bits | float model and log published (`data/`) |
| Quantization (per channel) | exactly, with the published ranges, on any processor | `haller.detector --per-channel --ranges` |
| Export (weights and reference vectors for SeedSigner) | exactly, byte for byte | `haller.detector_export` |

## Files

| File | Content |
|---|---|
| `data/haller-1-detector-float.npz` | float model (batch normalization folded) after epoch 10, input of the quantization |
| `data/train-detector.log` | its training log: losses and validation scores per epoch, then the ranges of a plain per-layer calibration on fv0 (not used, see [Quantization](#quantization)) |
| `data/haller-1-detector-ranges.npz` | the activation range of every output channel of the 12 layers (one array per layer, named after it), as used by the shipped quantization |
| `data/quantize.log` | output of step 6 below |
| `data/audits.tsv` | for each shard: signature of the generator configuration, seed, pages kept, pages removed by the filter and the factors that put them out of range, audit status |
| `data/inputs.sha256` | SHA-256 of every input file of the compositor (images, masks, annotations, order files), under their public names |
| `data/order/{supports,cartes,variantes}.txt` | order in which the compositor reads the annotations (copy into each `annot` directory as `ordre.txt`) |
| `data/card_templates.json` | the 25 printed card templates drawn by `gen_univ.py`: vendor, license, source URL and page, SHA-256 of the rendered PNG |
| `gen/` | frozen generator, compositor, variety audit, quantization and training scripts (see [Frozen code](#frozen-code)) |
| `SHA256SUMS.content` | content digests of the 138 shards, the GNHK windows, the validation file u40, the float and quantized models; SHA-256 of the exported files (comment lines at the end) |

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
comments and identifiers are in French, as written. The generator and the
compositor are those of the configurations validated on 21 November 2026 after
the variety audit: `e9dc866d13842135` (`gen_univ.py`, `gen_det.py`,
`gen_sim.py` with the variables of `lots_univ6.sh`) and `092dcb1e39fc103d`
(the same files with `composer.py`, `lumiere.py` and `poser_ecriture.py`,
with `UNIV_REEL=1 UNIV_POLICE_RARE=1` set by the compositor). Changes made
for the publication, none of which changes the output (checked, see
[Verification](#verification)):

- paths: fonts from `HALLER_FONTS`, held-out test fonts from
  `HALLER_TEST_FONTS` (only for the test mode, not used here); local paths
  and machine names removed from the comments and the shell scripts;
- the module `train` (model loading of an older prototype, unused for
  generation) is always replaced by an empty module, as it already was on
  machines without PyTorch;
- `composer.py`: public names of the two papers whose printed lines are
  ignored (`sv_0130`, `sv_0152`); evaluation cards excluded by their
  `"jeu": "eval"` field alone (every card of the former list of internal
  evaluation codes has it); supports read in the order of `ANNOT/ordre.txt`
  (the order of the internal codes, under which the shards were drawn),
  since the public names do not sort in that order;
- names of people and internal task numbers removed from comments, from the
  text fields of `variete/registre.json` (the plausible ranges and shares,
  which the audit reads, are unchanged) and from `variete/validations.json`
  (only the two configurations of this recipe are kept).

The signature of a configuration is the SHA-256 of the source files of the
generator and of its variables (`variete/journal.py`). Since the publication
changes comments in those files, shards written by the frozen code carry the
signatures `0868322f2bad8f77` and `760169ce364856c8` instead of `e9dc866d13842135` and `092dcb1e39fc103d`;
`variete/validations.json` lists them as the same configurations, so that the
variety guard (`variete/garde.py`) admits them.

Main files:

| File | Role |
|---|---|
| `train/detecteur/gen_univ.py` | simulated pages (lists, sheets, printed cards from the templates, verso, altered words, numbers on a dark disc), 12 target channels |
| `train/detecteur/gen_det.py`, `train/pi/banc2/gen_sim.py`, `train/cartes.py`, `train/officiel.py`, `train/seed_sim.py`, … | its dependencies: handwriting from fonts and EMNIST, photo simulation, official SeedSigner cards (`officiel/`) |
| `train/supports/composer.py`, `lumiere.py`, `train/detecteur/supports/poser_ecriture.py` | compositor: simulated handwriting on photos of real blank supports and cards, lighting, occluders |
| `train/variete/` | variety journal written with every page (`journal.py`, `resume.py`), audit of a shard against the plausible ranges of `registre.json` (`audit.py`), filter of the pages out of range (`filtrer.py`, `filtrer_lots.sh`), guard of the training (`garde.py`, `validations.json`) |
| `train/detecteur/distributions/distributions_reelles.json` | aggregated statistics of 76 private real photos (word height, orientation, colour temperature…), no image |
| `train/supports/cartes.py`, `traits.py`, `train/detecteur/modeles/annoter.py` | tools that made the compositor inputs (card variants, printed-line tracking, template boxes); run before the publication on internal names, not run by this recipe |
| `train/detecteur/gnhk_det.py` | earlier GNHK window extraction; `haller.detector_gnhk` gives the same content |
| `train/detecteur/nuit/quantifier.py`, `train_nuit.py` | the quantization script of the run (`--canal`, per channel), kept for the record: `haller.detector --per-channel` gives the same file |
| `train/detecteur/lots_univ6.sh`, `train/supports/lots_compo_u6.sh`, `train/detecteur/entrainer_univ6.sh` | the commands that made the shards and the training |

## Environment

All the shards were made with the same Python environment (Python 3.12.3,
numpy 2.5.5, Pillow 13.0.0, scipy 1.18.1, opencv 5.1.0.95 headless), on two
x86-64 Linux machines with AMD EPYC processors: f0-f87 on an AMD EPYC of the
c6a family, f88-f120, s0-s9, k0-k4 and kb0 on an AMD EPYC 7R32 (fv0 on one of
the two). Pillow 11 or later is required (Pillow 10.2 cannot draw the rounded
boxes of `gen_univ.py`).

Training: `requirements-lock.txt` (Python 3.14.0, PyTorch 2.15.0 with CUDA
13.2, numpy 2.5.5), on a rented NVIDIA RTX 3090. Quantization and export:
the same versions; the export also needs the SeedSigner checkout given in
the main README on `PYTHONPATH`.

## Inputs

- Fonts: `python -m haller.fonts FONTS` (287 fonts, `data/fonts.sha256`), in
  `HALLER_FONTS`. Same 287 files on every machine of the run.
- EMNIST: `python -m haller.emnist EMNIST.npz IDX_DIR --split train`, then
  its digits and capitals in `gen/train/data/emnist_train.npz`:

  ```bash
  python -c "import numpy as np; d = np.load('EMNIST.npz'); k = d['y'] < 36
  np.savez('recipes/haller-1-detector/gen/train/data/emnist_train.npz', x=d['x'][k], y=d['y'][k])"
  ```

  533,993 images, identical in content and order to the file of the run.
- Card templates: `gen/data/modeles_cartes/cases.json` (boxes of the 25
  templates) is here; the PNG files are not, because 18 of the 25 templates
  have no stated license that allows redistribution. `data/card_templates.json`
  gives, for each one, the source URL, the page and the SHA-256 of the PNG
  that was used (rendered from that page, in `gen/data/modeles_cartes/png/`);
  the 25 files of the run have these digests (checked).
- Photos of blank supports (162: papers, notebooks, cards, taken for this
  project, no handwriting) and of printed official cards (52 photos
  and 258 variants on paper backgrounds): **to be published; their manifest
  is published** (`data/inputs.sha256`, 1,213 files; the same files as for the
  previous detector, checked). The images used are the working versions (long
  side 2048 px) and the variants exactly as listed; the full-resolution
  exports that may be published elsewhere are not the same pixels. Layout
  expected by the compositor, `IN`: `supports/{annot,travail}`,
  `cartes/{annot,travail}`, `variantes/{annot,base}`, each `annot` holding the
  annotations, masks and `ordre.txt` (`data/order/`).
- GNHK: the `train` and `test` splits, checked against `data/gnhk.sha256`.
- `u40.npz` (validation only, 1,000 pages of an earlier configuration of the
  generator, seed 9040): not regenerated by this code; its digest is given.
  Validation draws no random numbers and does not choose the epoch, so it
  does not change the weights.

## Commands

From the root of the repository, `R=recipes/haller-1-detector`,
`G=$R/gen/train`, data written in `$R`. Run each generator in a shell without
`UNIV_*` variables: the scripts set the ones of their configuration, and any
other one changes the configuration.

```bash
# 1. Simulated shards: fv0 (seed 59999) and f0-f120 (seed 50000 + i), 1,000 pages each, journal and audit
OUT=$PWD/$R HALLER_FONTS=FONTS PYTHON=python3 bash $G/detecteur/lots_univ6.sh 8 fv:0:0 f:0:120

# 2. Composed shards: s0-s9 (70000 + i), k0-k4 (71000 + i), kb0 (72000), journal and audit
OUT=$PWD/$R IN=IN HALLER_FONTS=FONTS PYTHON=python3 bash $G/supports/lots_compo_u6.sh 8 s:0:9 k:0:4 kb:0:0

# 3. Pages out of the plausible ranges removed from the shards whose audit fails (191 pages of 138,000);
#    the raw shard is kept as <shard>.brut.npz, the audit is made again
bash $G/variete/filtrer_lots.sh python3 $PWD/$R

# 4. GNHK windows: 6 per train page (seed 11), 2 per test page (seed 12)
python -m haller.detector_gnhk $R/gnhk_train.npz GNHK/train 6 11
python -m haller.detector_gnhk $R/gnhk_test.npz GNHK/test 2 12

# 5. Training (GPU, --fast loop): 10 epochs, seed 2, the shards in this order, then gnhk_train 4 times;
#    the variety guard refuses any shard that is not audited or not of a validated configuration
LOTS=$PWD/$R GNHK=$PWD/$R U40=$PWD/$R/u40.npz RES=$PWD/$R/res HALLER=$PWD PYTHON=python GRAINE=2 \
    bash $G/detecteur/entrainer_univ6.sh detector-float 10 \
    $(for i in $(seq 0 120); do echo f$i; done) $(for i in $(seq 0 9); do echo s$i; done) k0 k1 k2 k3 k4 kb0

# 6. Quantization of the published float model with the published per-channel ranges
python -m haller.detector $R/data/haller-1-detector-float.npz $R/fv0.npz $R/haller-1-detector-q.npz \
    --per-channel --ranges $R/data/haller-1-detector-ranges.npz

# 7. Weights and reference vectors for SeedSigner
mkdir -p $R/out
PYTHONPATH=$SS python -m haller.detector_export $R/haller-1-detector-q.npz $R/out
```

Step 5 runs, from the root of the repository:

```bash
python -m haller.train_detector res/detector-float.npz f0.npz … f120.npz s0.npz … s9.npz k0.npz … k4.npz kb0.npz \
    gnhk_train.npz gnhk_train.npz gnhk_train.npz gnhk_train.npz \
    --arch base --epochs 10 --seed 2 --shards-in-memory 1 --fast --val fv0.npz u40.npz gnhk_test.npz f0.npz
```

It writes the model after each epoch and the last one under the given name,
with no selection, then a plain per-layer quantization that is not used. The
order of the shards on the command line matters: the training draws the
shard order and the batches from its seed over the positions of the files.
The seed was chosen among four trainings of the same recipe (seeds 1 to 4),
as the one that read the real photos of the development set best once
integrated in SeedSigner; the test photos were read once, after the choice.

Step 6 prints the largest channel of the ranges of a calibration on the
first 400 pages of fv0, then the largest difference with the published ranges,
and writes `haller-1-detector-q.npz`. Step 7 prints the largest difference
between the C library and the integer reference in numpy (0.0 for both vector
images) and writes `haller-1-detector.npz` (SHA-256 `b33f6d02…`) and
`haller-1-detector-vectors.json` (`7c995ffa…`).

## Quantization

The shipped detector is quantized per channel: each output channel of each
layer has its own activation range (99.99th percentile over the first 400
pages of fv0, at least 1e-3), and the next layer takes the scale of each of
its input channels into its weights. The integer format and the C library
are those of the per-layer quantization. Among four quantizations of the
float model (per layer or per channel, plain pages or pages passed through
the photo augmentation of `train_nuit.py`), it was chosen by a rule fixed
before the measurements: the best F1 score of the word boxes on the
development photos, within 0.01, then the best recall of the line numbers,
then the simplest.

The calibration has no random draw, but its result depends on the processor:
PyTorch and oneDNN choose their CPU kernels by instruction set, and the
summation order changes the ranges in the sixth decimal place, enough to
change some rounded weights. The run was calibrated on an Intel Core
i7-3770K (AVX, no AVX2). The same kernels are selected on a later processor
with

```bash
ATEN_CPU_CAPABILITY=default ONEDNN_MAX_CPU_ISA=AVX python -m haller.detector … --per-channel --save-ranges R.npz
```

which gives the shipped file byte for byte on an AMD EPYC 7R32 (checked);
without these variables, the ranges differ by at most 3.3e-6 and 18 of the
41 arrays of the quantized file differ. The published ranges
(`data/haller-1-detector-ranges.npz`, written that way) give the shipped file
byte for byte on any processor (step 6).

## Training

`haller.train_detector --fast` trains on CUDA with bfloat16 autocast and
`cudnn.benchmark`; it has no deterministic mode and a second run is not
expected to give the same weights. The float model is therefore published,
and steps 6 and 7 give the shipped files exactly from it.
`train-detector.log` is the log of the run (output of step 5 and of the plain
quantization of the wrapper script) without its final marker line. For each
epoch: mean losses, elapsed time, precision and recall of words and numbers on
fv0, u40, GNHK test and f0. 223,531 parameters, 6,796 s for 10 epochs.

## Verification

Checked on 21 November 2026 (content digests of `SHA256SUMS.content`):

- **Shards.** The frozen code of this directory, run with the commands of
  steps 1 to 3 (`env -i`, no `UNIV_*` variable) on an AMD EPYC 7R32 with the
  environment of the run, gives the same content digests as the shards of the
  training for f5 (made on the other machine of the run, AMD EPYC c6a: 2 pages
  removed by the filter), f100 (1 page removed), s5 and k0 (composed from the
  public inputs). Their variety journals are the same line for line (the
  composed ones up to the public names of the photos), their audits pass,
  and the variety guard admits them under the signatures of the published
  code. The other shards were not regenerated.
- **Compositor inputs.** The public inputs made from the working files of the
  run have the 1,213 digests of `data/inputs.sha256`, unchanged since the
  previous detector; the 25 card templates of the run have the digests of
  `data/card_templates.json`.
- **Quantization and export.** From the published float model and per-channel
  ranges, step 6 gives the quantized model of the run byte for byte (content
  `6bad6b32…`), and step 7 gives `haller-1-detector.npz` (`b33f6d02…`) and
  `haller-1-detector-vectors.json` (`7c995ffa…`) byte for byte, on an AMD EPYC
  7R32 (the run quantized on an Intel Core i7-3770K). The calibration itself
  gives the run's ranges on the AMD processor with the variables of
  [Quantization](#quantization) only. `nuit/quantifier.py --canal`, the
  script of the run, and `haller.detector --per-channel` give the same file.
- **Training**: not rerun. The float model is the one of the run, epoch 10 of
  seed 2 (log in `data/train-detector.log`).
- **GNHK windows and u40**: same content as for the previous detector
  (digests of `SHA256SUMS.content`).
- **EMNIST**: digits and capitals of `haller.emnist`, the same file as for the
  previous detector.
