# Haller-1 detector: exact recipe of the published weights

This directory reproduces `haller-1-detector.npz` and
`haller-1-detector-vectors.json`, the word detector shipped in SeedSigner
(`src/seedsigner/resources/handwriting/`), from public inputs: 40 shards of
simulated photos, two files of GNHK windows, the training, the quantization
and the export. The training is not bit-reproducible, so the float model it
produced is published here; the quantization and the export give the shipped
files byte for byte from it.

| Step | Reproduced | How |
|---|---|---|
| Simulated shards d1-d40 | exactly | `haller.detector_pages`, pinned fonts, EMNIST train, pinned versions |
| GNHK windows (train and test) | exactly | `haller.detector_gnhk` |
| Training | not the same bits | float model and log published (`data/`); see [Training](#training) |
| Quantization | exactly | `haller.detector`, from the published float model and d40 |
| Export (weights and reference vectors for SeedSigner) | exactly, byte for byte | `haller.detector_export` |

## Files

| File | Content |
|---|---|
| `data/haller-1-detector-float.npz` | float model (batch normalization folded) after epoch 14, input of the quantization |
| `data/train-detector.log` | its training log: losses and validation scores per epoch, calibrated ranges |
| `SHA256SUMS.content` | content digest of the shards, the GNHK windows, the float and quantized models; SHA-256 of the exported files (comment lines at the end, for `sha256sum -c` once uncommented) |

`../haller-1/digest.py` computes and checks the content digests (arrays only,
independent of the zip container of the `.npz` files; see
[`recipes/haller-1`](../haller-1/README.md)):

```bash
python recipes/haller-1/digest.py --check recipes/haller-1-detector/SHA256SUMS.content   # run where the files are
```

## Environment

Shards and GNHK windows: Python 3.12.3, numpy 1.26.4, Pillow 12.2.0 (wheel
with FreeType 2.14.3 and raqm 0.10.3), on x86-64 (Linux). With Python 3.14.0,
numpy 2.5.5 and Pillow 13.0.0 (FreeType 2.14.3, raqm 0.11.0), the target maps
and the metadata are the same but the images are not: in the first 100 photos
of d1, 51 differ by one gray level on a few pixels (6e-6 of the pixels).

```bash
uv venv -p 3.12 venv-data
uv pip install -p venv-data/bin/python numpy==1.26.4 pillow==12.2.0
```

Training, quantization and export: `requirements-lock.txt` (Python 3.14.0,
PyTorch 2.15.0 with CUDA 13.2, numpy 2.5.5), training on an NVIDIA RTX 3060;
the export also needs the SeedSigner checkout given in the main README on
`PYTHONPATH` (it builds and runs SeedSigner's C library).

Inputs:

- `FONTS`: the 287 fonts of `python -m haller.fonts FONTS` (google/fonts
  commit `7af0d90b`, checked against `data/fonts.sha256`). The cursive style
  draws from the fonts of `data/cursive_fonts.txt` that are in this directory
  (90 of them), the other styles from the other 197. Pass this directory
  alone: adding the 115 cursive fonts of `data/fonts_cursive.sha256` changes
  the font lists and therefore the shards. The directory used for
  haller-1-detector held the same 287 files (same SHA-256) and a license
  file, which is not a font and is ignored.
- `EMNIST.npz`: `python -m haller.emnist EMNIST.npz IDX_DIR --split train`
  from the official EMNIST archive (byclass train split). The generator keeps
  the digits and capitals (classes 0-35): 533,993 images, identical in content
  and order to those used for haller-1-detector.
- `GNHK`: the GNHK directory with its `train` and `test` splits, checked
  against `data/gnhk.sha256`.

In the commands below, `SS` is the `src` directory of the SeedSigner
checkout and `R=recipes/haller-1-detector`; the commands run from the root of
the repository and write the data files in `$R`.

## Commands

```bash
# 1. Simulated shards d1-d40: 1,000 photos each, seed 5000 + i (one process per shard)
for i in $(seq 1 40); do
  venv-data/bin/python -m haller.detector_pages train $R/d$i.npz 1000 $((5000 + i)) \
      --fonts FONTS --emnist EMNIST.npz --bip39 data/bip39_english.txt
done

# 2. GNHK windows: 6 per train page (seed 11), 2 per test page (seed 12)
venv-data/bin/python -m haller.detector_gnhk $R/gnhk_train.npz GNHK/train 6 11   # 3,089 windows
venv-data/bin/python -m haller.detector_gnhk $R/gnhk_test.npz GNHK/test 2 12     # 344 windows

# 3. Training (GPU): 14 epochs; d1 to d39 in numeric order, then gnhk_train twice
python -m haller.train_detector $R/detector-float.npz $(for i in $(seq 1 39); do echo $R/d$i.npz; done) \
    $R/gnhk_train.npz $R/gnhk_train.npz --val $R/d40.npz $R/gnhk_test.npz --epochs 14

# 4. Quantization of the published float model, calibrated on the first 400 photos of d40
python -m haller.detector $R/data/haller-1-detector-float.npz $R/d40.npz $R/haller-1-detector-q.npz --count 400

# 5. Weights and reference vectors for SeedSigner
mkdir -p $R/out
PYTHONPATH=$SS python -m haller.detector_export $R/haller-1-detector-q.npz $R/out
```

Step 3 writes the model after each epoch (`detector-float_eN.npz`) and the
last one under the given name; haller-1-detector is the model after epoch 14,
with no selection. The order of the shards on the command line matters: the
training draws the shard order and the batches from its seed (1) over the
positions of the files.

Step 4 prints the calibrated activation ranges, the same as at the end of
`data/train-detector.log`. Step 5 prints the largest difference between the
maps of the C library and of the integer reference in numpy (0.0 for both
vector images), and writes `haller-1-detector.npz` (SHA-256 `bcda59a2…`) and
`haller-1-detector-vectors.json`, identical byte for byte to the files of
SeedSigner.

## Verification

All the regenerated files of the table at the top were compared with the
originals of haller-1-detector:

- the 40 shards, regenerated with the commands above in the environment
  above: same content digests as the originals (`SHA256SUMS.content`);
- the two GNHK files: same content digests;
- quantization (step 4) of the published float model with the original d40:
  same content digest as the original quantized model, and in fact the same
  file (SHA-256 `0f149788…`); export (step 5): `haller-1-detector.npz` and
  `haller-1-detector-vectors.json` identical byte for byte to SeedSigner's.

The quantization was run with PyTorch on the CPU of the machine that made the
original quantized model (x86-64). The calibration takes the 99.99th
percentile of float32 activations; on another processor the ranges may differ
in their last bits, as observed for the word recognizers (see
[`recipes/haller-1`](../haller-1/README.md#quantization)), which would change
a few integer biases.

## Training

`haller.train_detector` (version of this repository, unchanged since the run)
trains on CUDA with bfloat16 autocast and the default cuDNN algorithms, and
has no deterministic mode: a second run on the same data is not expected to
give the same weights. The detector was not retrained for this recipe. The
float model is therefore published (`data/haller-1-detector-float.npz`), and
steps 4 and 5 give the shipped weights exactly from it; a retrained model
must be measured again before it replaces them.

`data/train-detector.log` is the log of the original run (output of step 3
and of step 4), without two warnings of the Python launcher that held local
paths and a final marker line written by the wrapper script. For each epoch:
mean losses (word and number kernels, card, corners, box height), elapsed
time, and the precision and recall of words and numbers on the first 600
photos of d40 and the first 600 windows of `gnhk_test.npz`.
