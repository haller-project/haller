"""
Quantized inference by SeedSigner's C library
(src/seedsigner/helpers/handwriting/native/handwriting.c), the code that runs
on the device: the benchmarks, the choice of the training epoch and the
reference vectors all use it, so that what is measured here is what SeedSigner
computes.

SeedSigner's sources must be importable (PYTHONPATH=SEEDSIGNER/src). The
library is built there, by SeedSigner's own Makefile, when it is missing or
older than its sources.
"""
import io
import os
import subprocess

import numpy as np
from seedsigner.helpers.handwriting import native
from seedsigner.helpers.handwriting.decoder import Decoder
from seedsigner.helpers.handwriting.word_model import WordModel

__all__ = ["Decoder", "load", "word_model"]



def load():
    """The C library, built first if needed (make -C .../native)."""
    subprocess.run(["make", "-s", "-C", os.path.dirname(native.LIBRARY_PATH)], check=True)
    return native.load()



def word_model(weights) -> WordModel:
    """
    WordModel of quantized weights: path of a .npz file, or dictionary of
    arrays as returned by haller.quantize.quantize.
    """
    load()
    if isinstance(weights, dict):
        buffer = io.BytesIO()
        np.savez(buffer, **weights)
        buffer.seek(0)
        weights = buffer
    return WordModel(weights)
