"""
The reference vectors shipped in SeedSigner (resources/handwriting/*-vectors.json)
match the C library, and the quantized model agrees with the float model it
comes from (haller.vectors.check_float).

    HALLER_FLOAT="haller-1-capitals=capitals.npz,haller-1-lowercase=lowercase.npz" \
        PYTHONPATH=.:SEEDSIGNER/src pytest tests

The float weights are not in this repository: without HALLER_FLOAT only the C
outputs are checked.
"""
import glob
import json
import os

import numpy as np
import pytest
from seedsigner.helpers.handwriting import native as seedsigner_native

from haller import native, vectors

RESOURCES = os.path.join(os.path.dirname(seedsigner_native.__file__), "..", "..", "..", "resources", "handwriting")
FILES = sorted(glob.glob(os.path.join(RESOURCES, "*-vectors.json")))
FLOAT = dict(item.split("=", 1) for item in os.environ.get("HALLER_FLOAT", "").split(",") if item)



@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_vectors(path):
    with open(path) as f:
        v = json.load(f)
    weights = os.path.join(RESOURCES, v["weights"])
    model, decoder = native.word_model(weights), native.Decoder()
    for entry in v["strips"]:
        out = vectors.c_outputs(model, decoder, vectors.decode_ink(entry))
        assert {k: entry[k] for k in out} == out, entry["label"]
    name = v["weights"][:-len(".npz")]
    if name in FLOAT:
        print(vectors.check_float(v["strips"], dict(np.load(FLOAT[name]))))


def test_files_found():
    assert len(FILES) >= 2
