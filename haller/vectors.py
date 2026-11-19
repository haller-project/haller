"""
Reference vectors of quantized weights, shipped with them in SeedSigner
(resources/handwriting/<weights>-vectors.json) and checked by its tests
(tests/test_handwriting_vectors.py): any build of the C library (another
compiler, the ARMv6 SMLAD path or the generic C path) must give the same bits.

    python -m haller.vectors WEIGHTS.npz --strips capitals=EVAL.npz --strips cursive=VAL.npz \
        [--count 20] [--seed 1] [--float MODEL.npz]
        -> WEIGHTS-vectors.json

haller.quantize writes them with the weights. For each source, COUNT strips of
spread widths are drawn (strips with a true `reel` field, from photos, are
never used), plus an empty strip and a strip of noise of the maximum width.
For each strip, the file holds the ink map (zlib, base64) and the output of
the C library: SHA-256 of the logits (float32, little-endian), SHA-256 of the
scores of the 2048 BIP39 words after the log-softmax, and the TOP best words
with their scores.

With --float, the strips are also read by the float model before
quantization (PyTorch) and the agreement is checked (`check_float`).
"""
import argparse
import base64
import hashlib
import json
import os
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from embit.bip39 import WORDLIST

from . import native
from .dictionary import ctc_word_scores, dictionary_tables
from .model import folded_forward

TOP = 5

# Agreement of the quantized model with the float model (check_float), on the
# strips that the float model reads clearly: its best word has a
# log-likelihood of at least FLOAT_MIN_SCORE and is ahead of the second one by
# at least FLOAT_MARGIN. There, the best word must be the same and its score
# differ by at most SCORE_TOLERANCE. Measured on 452 strips of the three
# sources for each Haller-1 expert: 613 such strips, no different best word,
# largest score difference 1.94. The best words differ on 5 to 6 % of all the
# strips, always where the float model hesitates or reads nothing (often
# words that are not BIP39 words).
FLOAT_MIN_SCORE = -5.0
FLOAT_MARGIN = 1.0
SCORE_TOLERANCE = 2.5



def pick(d, count: int, rng: np.random.Generator) -> list[int]:
    """COUNT strips of spread widths among the strips that are not from photos."""
    usable = np.nonzero(~d["reel"].astype(bool))[0] if "reel" in d.files else np.arange(len(d["w"]))
    usable = usable[d["w"][usable] > 0]
    sample = rng.choice(usable, min(len(usable), 20 * count), replace=False)
    sample = sample[np.argsort(d["w"][sample], kind="stable")]
    return [int(sample[k]) for k in np.linspace(0, len(sample) - 1, count).round().astype(int)]



def strips(sources: dict, count: int, seed: int) -> list[dict]:
    """Strips of each source (name -> .npz path), then the two edge cases."""
    rng = np.random.default_rng(seed)
    out = []
    for style, path in sources.items():
        d = np.load(path)
        for i in pick(d, count, rng):
            out.append(dict(style=style, label=str(d["y"][i]), ink=d["x"][i, :, :int(d["w"][i])].copy()))
    out.append(dict(style="empty", label="", ink=np.zeros((32, 40), np.uint8)))
    out.append(dict(style="noise", label="", ink=rng.integers(0, 256, (32, 256), dtype=np.uint8)))
    return out



def c_outputs(model, decoder, ink: np.ndarray) -> dict:
    """Output of the C library for one ink map."""
    z = model.logits(ink)
    logits_sha256 = hashlib.sha256(z.astype("<f4").tobytes()).hexdigest()
    model.lib.hw_log_softmax(z.ctypes.data, z.shape[1])
    scores = decoder.scores(z)
    best = np.argsort(-scores, kind="stable")[:TOP]
    return dict(logits_sha256=logits_sha256, scores_sha256=hashlib.sha256(scores.astype("<f4").tobytes()).hexdigest(),
                words=[WORDLIST[k] for k in best], scores=[float(scores[k]) for k in best])



def float_scores(float_weights: dict, inks: list) -> list[np.ndarray]:
    """Scores of the BIP39 words by the float model and the exact CTC decoding (PyTorch)."""
    p = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in float_weights.items()}
    tables = dictionary_tables(WORDLIST, "cpu")
    out = []
    with torch.no_grad():
        for ink in inks:
            width = ink.shape[1]
            x = torch.zeros(1, 1, 32, (width + 16 + 3) // 4 * 4)
            x[0, 0, :, :width] = torch.tensor(ink, dtype=torch.float32) / 255
            lp = F.log_softmax(folded_forward(p, x), 1)
            out.append(ctc_word_scores(lp, torch.tensor([max(1, width // 4)]), tables)[0].numpy())
    return out



def check_float(entries: list[dict], float_weights: dict) -> dict:
    """
    Agreement of the C outputs (entries of the vectors file) with the float
    model; raises AssertionError beyond the tolerances above.
    """
    checked, deltas = 0, [0.0]
    for e, f in zip(entries, float_scores(float_weights, [decode_ink(e) for e in entries])):
        order = np.argsort(-f)
        best = int(order[0])
        if f[best] < FLOAT_MIN_SCORE or f[best] - f[order[1]] < FLOAT_MARGIN:
            continue
        assert e["words"][0] == WORDLIST[best], f"strip {e['label']!r}: {e['words'][0]} (C), {WORDLIST[best]} (float)"
        checked += 1
        deltas.append(abs(e["scores"][0] - float(f[best])))
    assert max(deltas) <= SCORE_TOLERANCE, f"score difference {max(deltas):.3f}"
    return dict(strips_checked=checked, max_score_difference=round(max(deltas), 4))



def encode_ink(ink: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(ink, np.uint8).tobytes(), 9)).decode()



def decode_ink(entry: dict) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(entry["ink"])), np.uint8).reshape(32, entry["width"])



def write(weights_path: str, sources: dict, count: int = 20, seed: int = 1, float_weights: dict = None) -> str:
    """Writes WEIGHTS-vectors.json next to the weights; returns its path."""
    model, decoder = native.word_model(weights_path), native.Decoder()
    entries = []
    for s in strips(sources, count, seed):
        entry = dict(style=s["style"], label=s["label"], width=int(s["ink"].shape[1]), ink=encode_ink(s["ink"]))
        entry.update(c_outputs(model, decoder, s["ink"]))
        entries.append(entry)
    with open(weights_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    vectors = dict(weights=os.path.basename(weights_path), weights_sha256=digest, beam=decoder.beam, top=TOP)
    if float_weights is not None:
        vectors["float_agreement"] = check_float(entries, float_weights)
    vectors["strips"] = entries
    out = weights_path[:-len(".npz")] + "-vectors.json"
    with open(out, "w") as f:
        json.dump(vectors, f, indent=0)
        f.write("\n")
    return out



def parse_sources(items: list[str]) -> dict:
    return dict(item.split("=", 1) for item in items)



def main():
    a = argparse.ArgumentParser()
    a.add_argument("weights")
    a.add_argument("--strips", action="append", required=True, help="STYLE=STRIPS.npz (x, w, y)")
    a.add_argument("--count", type=int, default=20)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--float", help="float weights before quantization (haller.train)")
    a = a.parse_args()
    float_weights = dict(np.load(a.float)) if a.float else None
    print(write(a.weights, parse_sources(a.strips), a.count, a.seed, float_weights))



if __name__ == "__main__":
    main()
