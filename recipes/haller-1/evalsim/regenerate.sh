#!/usr/bin/env bash
# Regenerates recipes/haller-1/data/haller-1-eval-simulated.npz, the 9,156
# simulated evaluation strips of Haller-1, from the official EMNIST archive.
# The file is derived from EMNIST and is therefore not stored in the repository.
#
#   recipes/haller-1/evalsim/regenerate.sh [WORK_DIR] [PROCESSES]
#
# Run from the repository root with numpy 2.x and Pillow < 13 installed. WORK_DIR
# (default: evalsim-work) receives the EMNIST archive (562 MB, downloaded from
# NIST unless already present), the 800 photos (about 400 MB) and the strips
# before they are copied into recipes/haller-1/data/.
#
# Steps, each checked against digests stored in this directory:
#   1. EMNIST archive from NIST (SHA-256 of gzip.zip);
#   2. byclass test split converted by haller.emnist;
#   3. 800 photos written by haller.pages, seeds 11 to 18, 100 photos each
#      (photos.sha256: every JPEG file byte for byte);
#   4. strips cut by cut.py, the page reader of the time, in the order of
#      pages.txt (content digest of SHA256SUMS.content).
set -euo pipefail

WORK=${1:-evalsim-work}
PROCS=${2:-4}
R=recipes/haller-1
E=$R/evalsim
URL=https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
ZIP_SHA256=fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e
PY=${PYTHON:-python3}

test -f haller/pages.py || { echo "run from the repository root" >&2; exit 1; }
# Versions. The photos need numpy 2.x: numpy 1.26 draws other random values
# and 797 of the 800 photos differ. Pillow 13 changed Image.resize (bilinear):
# 6 strips differ by one gray level; the strips need Pillow < 13 (tested:
# 10.4, 11.3, 12.0, 12.1, 12.2). Checked identical with Python 3.12.3,
# numpy 2.5.5 and Pillow 12.2.0:
#   uv venv -p 3.12 venv-evalsim && uv pip install -p venv-evalsim/bin/python numpy==2.5.5 pillow==12.2.0
#   PYTHON=venv-evalsim/bin/python recipes/haller-1/evalsim/regenerate.sh
$PY -c 'import PIL, sys; sys.exit(int(PIL.__version__.split(".")[0]) >= 13)' || {
    echo "Pillow < 13 is required (see the comment above)" >&2; exit 1; }
$PY -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) < 2)' || {
    echo "numpy 2.x is required (see the comment above)" >&2; exit 1; }
mkdir -p "$WORK"

# 1. EMNIST archive
if [ ! -f "$WORK/gzip.zip" ]; then
    curl -L --fail -o "$WORK/gzip.zip.part" "$URL"
    mv "$WORK/gzip.zip.part" "$WORK/gzip.zip"
fi
echo "$ZIP_SHA256  $WORK/gzip.zip" | sha256sum -c -

# 2. Test split
unzip -o -q "$WORK/gzip.zip" 'gzip/emnist-byclass-test-*' -d "$WORK"
PYTHONPATH=. $PY -m haller.emnist "$WORK/EMNIST_TEST.npz" "$WORK/gzip" --split test

# 3. Photos
for seed in 11 12 13 14 15 16 17 18; do
    PYTHONPATH=. $PY -m haller.pages "$WORK/photos/p$seed" 100 $seed \
        --emnist "$WORK/EMNIST_TEST.npz" --bip39 data/bip39_english.txt &
    while [ "$(jobs -rp | wc -l)" -ge "$PROCS" ]; do wait -n; done
done
wait
(cd "$WORK/photos" && sha256sum --quiet -c "$OLDPWD/$E/photos.sha256")
PYTHONPATH=. $PY - "$WORK/photos" <<'EOF'
import json, os, sys
root = sys.argv[1]
truth = {}
for d in sorted(os.listdir(root)):
    if os.path.isdir(os.path.join(root, d)):
        for name, info in json.load(open(os.path.join(root, d, "truth.json"))).items():
            truth[f"{d}/{name}"] = info
json.dump(truth, open(os.path.join(root, "truth.json"), "w"), indent=1)
print(len(truth), "photos")
EOF

# 4. Strips
PYTHONPATH=. $PY $E/cut.py "$WORK/photos" "$WORK/photos/truth.json" $E/pages.txt \
    $R/data/haller-1-eval-simulated.npz --processes "$PROCS"
expected=$(awk '$2 == "data/haller-1-eval-simulated.npz" {print $1}' $R/SHA256SUMS.content)
actual=$($PY $R/digest.py $R/data/haller-1-eval-simulated.npz | awk '{print $1}')
if [ -n "$expected" ] && [ "$expected" = "$actual" ]; then
    echo "haller-1-eval-simulated.npz: identical to Haller-1 ($actual)"
else
    echo "haller-1-eval-simulated.npz: content digest differs from SHA256SUMS.content" >&2
    exit 1
fi
