"""
Writes a shard of synthetic word samples.

    python -m seedsigner_ocr.generate OUT.npz COUNT SEED --fonts DIR --emnist EMNIST.npz \
        --bip39 data/bip39_english.txt --other data/other_words.txt

Output: x (N, 32, 256) uint8 ink maps padded with zeros, w (N,) widths,
y (N,) labels.
"""
import argparse
import time

import numpy as np

from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH
from .synth import EmnistLetters, Synthesizer, Vocabulary, list_fonts



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("count", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--fonts", required=True)
    p.add_argument("--emnist", required=True)
    p.add_argument("--bip39", required=True)
    p.add_argument("--other", required=True)
    p.add_argument("--cursive", action="store_true",
                   help="lowercase writing (85 %%) in the fonts only, rendered as whole words (cursive joins kept)")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    options = dict(font_share=1.0, lower_share=0.85, per_letter_share=0.0) if a.cursive else {}
    synth = Synthesizer(Vocabulary.load(a.bip39, a.other), list_fonts(a.fonts), EmnistLetters(a.emnist), **options)
    x = np.zeros((a.count, INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    widths, labels = np.zeros(a.count, np.int16), []
    n, t0 = 0, time.time()
    while n < a.count:
        try:
            s = synth.sample(rng)
        except (OSError, ValueError):  # a font that cannot render a character
            continue
        if s is None:
            continue
        x[n, :, :s[0].shape[1]] = s[0]
        widths[n] = s[0].shape[1]
        labels.append(s[1])
        n += 1
        if n % 5000 == 0:
            print(f"{n}/{a.count} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(a.out, x=x, w=widths, y=np.array(labels))



if __name__ == "__main__":
    main()
