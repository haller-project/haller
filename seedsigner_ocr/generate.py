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
    p.add_argument("--bip39-share", type=float, default=0.6, help="share of English BIP39 words (default 0.6)")
    p.add_argument("--matched", action="store_true",
                   help="one random stream per sample, words drawn from a separate stream: shards made with the "
                        "same seed and different --bip39-share differ only in their words")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    options = dict(font_share=1.0, lower_share=0.85, per_letter_share=0.0) if a.cursive else {}
    vocabulary = Vocabulary.load(a.bip39, a.other)
    vocabulary.bip39_share = a.bip39_share
    synth = Synthesizer(vocabulary, list_fonts(a.fonts), EmnistLetters(a.emnist), **options)
    x = np.zeros((a.count, INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    widths, labels = np.zeros(a.count, np.int16), []
    n, i, t0 = 0, 0, time.time()
    while n < a.count:
        if a.matched:
            rng, word_rng = np.random.default_rng([a.seed, i]), np.random.default_rng([a.seed, i, 1])
            i += 1
        else:
            word_rng = None
        try:
            s = synth.sample(rng, word_rng)
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
