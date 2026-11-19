"""
Benchmark of quantized weights on word strips, computed by SeedSigner's C
library (haller.native): share of strips whose most probable BIP39 word is
the label.

    python -m haller.evaluate WEIGHTS.npz STRIPS.npz [STRIPS.npz ...] [--processes 4] [--by FIELD]

STRIPS.npz: x (N, 32, 256) uint8 ink maps, w (N,) widths, y (N,) labels, as
written by haller.generate; strips whose label is not a BIP39 word are not
counted. --by FIELD gives the share for each value of a field of the file
(for instance `groupe` or `reel` of the evaluation sets).
"""
import argparse
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
from embit.bip39 import WORDLIST

from . import native

_engine = {}



def _start(weights):
    _engine["model"], _engine["decoder"] = native.word_model(weights), native.Decoder()



def _read(job):
    x, w = job
    model, decoder = _engine["model"], _engine["decoder"]
    return [int(np.argmax(decoder.scores(model.log_probs(ink[:, :max(int(width), 1)])))) for ink, width in zip(x, w)]



def best_words(weights, x: np.ndarray, w: np.ndarray, processes: int = 1, chunk: int = 256) -> np.ndarray:
    """
    Index in the BIP39 list of the most probable word of each strip.
    weights: path or dictionary of arrays (haller.native.word_model).
    """
    jobs = [(x[i:i + chunk], w[i:i + chunk]) for i in range(0, len(w), chunk)]
    if processes <= 1:
        _start(weights)
        return np.array([k for job in jobs for k in _read(job)], np.int64)
    # spawn: the caller may hold a CUDA context, which does not survive a fork
    with get_context("spawn").Pool(processes, _start, (weights,)) as pool:
        return np.array([k for part in pool.map(_read, jobs) for k in part], np.int64)



def accuracy(weights, x, w, labels, processes: int = 1, selection=None) -> float:
    """Share of the selected strips (default: all) read as their label, BIP39 labels only."""
    index = {word: i for i, word in enumerate(WORDLIST)}
    selection = np.arange(len(labels)) if selection is None else np.asarray(selection)
    selection = np.array([i for i in selection if str(labels[i]) in index], np.int64)
    if not len(selection):
        return 0.0
    best = best_words(weights, x[selection], w[selection], processes)
    return float(np.mean([b == index[str(labels[i])] for b, i in zip(best, selection)]))



def main():
    a = argparse.ArgumentParser()
    a.add_argument("weights")
    a.add_argument("strips", nargs="+")
    a.add_argument("--processes", type=int, default=1)
    a.add_argument("--by")
    a = a.parse_args()
    for path in a.strips:
        d = np.load(path)
        labels = [str(y) for y in d["y"]]
        groups = defaultdict(list)
        for i in range(len(labels)):
            groups[str(d[a.by][i]) if a.by else "all"].append(i)
        for group, selection in sorted(groups.items()):
            share = accuracy(a.weights, d["x"], d["w"], labels, a.processes, selection)
            print(f"{path} {group}: {100 * share:.2f} % of {len(selection)} strips")



if __name__ == "__main__":
    main()
