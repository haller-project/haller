"""
Reading of fingerprint lines by quantized weights, as SeedSigner computes it (C library).

    PYTHONPATH=<SeedSigner src> python recipes/haller-1-fingerprint/eval_fingerprint.py STRIPS.npz WEIGHTS.npz [...]

STRIPS.npz is a shard of haller.generate made with --hex-share 1.0 (every label is a fingerprint line: 8 hex
digits, possibly in two groups, after an optional prefix). On the first 2,000 lines, for the two word experts of
Haller-1 together (as in SeedSigner without the fingerprint expert), then for each WEIGHTS.npz alone and together
with the two word experts, it prints:

- the share of lines whose true fingerprint has a probability of at least 1e-4 among the 2^32 strings (the
  threshold at which SeedSigner accepts a fingerprint as matching the words);
- the share read exactly (most probable digit at each of the 8 positions).
"""
import sys

import numpy as np
from seedsigner.helpers.handwriting import fingerprint as F
from seedsigner.helpers.handwriting.word_model import MODEL_PATH, WordModel

LINES = 2000
THRESHOLD = 1e-4


def readings(model, x, w, n):
    return [F.FingerprintReading(model.log_probs(x[i, :, :w[i]])) for i in range(n)]


def score(readings_list, truth, name):
    match = exact = 0
    for i, t in enumerate(truth):
        check = F.FingerprintCheck([r[i] for r in readings_list])
        match += check.probabilities([t])[0] >= THRESHOLD
        exact += check.text == t
    n = len(truth)
    print(f"{name:45s} match {match / n:.4f}  exact {exact / n:.4f}", flush=True)


def main():
    d = np.load(sys.argv[1])
    x, w, y = d["x"], d["w"], d["y"]
    n = min(len(y), LINES)
    truth = [str(t)[-8:] for t in y[:n]]
    words = [readings(WordModel(MODEL_PATH), x, w, n),
             readings(WordModel(MODEL_PATH.replace("capitals", "lowercase")), x, w, n)]
    score(words, truth, "Haller-1 word experts (capitals + lowercase)")
    for path in sys.argv[2:]:
        own = readings(WordModel(path), x, w, n)
        name = path.split("/")[-1]
        score([own], truth, name)
        score(words + [own], truth, name + " + word experts")


if __name__ == "__main__":
    main()
