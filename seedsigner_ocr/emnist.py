"""
EMNIST letters for the synthetic data (Cohen et al., 2017; NIST Special
Database 19).

Converts the "byclass" split of the official EMNIST release (IDX files, plain
or gzipped, from the "gzip" archive published by NIST) into the file expected
by seedsigner_ocr.generate --emnist:

    python -m seedsigner_ocr.emnist EMNIST.npz IDX_DIR [--split train]

Output: x (N, 28, 28) uint8, white ink on black, upright (the IDX images are
stored transposed); y (N,) int16 class index in the order 0-9 A-Z a-z.
"""
import argparse
import gzip
import os

import numpy as np



def read_idx(path: str) -> np.ndarray:
    """Array stored in an IDX file (unsigned bytes), gzipped or not."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\0\0" or data[2] != 8:
        raise ValueError(f"{path}: not an unsigned byte IDX file")
    dims = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(dims))
    return np.frombuffer(data, np.uint8, offset=4 + 4 * dims).reshape(shape)



def find(idx_dir: str, name: str) -> str:
    for candidate in (name, name + ".gz"):
        path = os.path.join(idx_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(os.path.join(idx_dir, name))



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("idx_dir")
    p.add_argument("--split", choices=("train", "test"), default="train")
    a = p.parse_args()

    images = read_idx(find(a.idx_dir, f"emnist-byclass-{a.split}-images-idx3-ubyte"))
    labels = read_idx(find(a.idx_dir, f"emnist-byclass-{a.split}-labels-idx1-ubyte"))
    np.savez_compressed(a.out, x=np.ascontiguousarray(images.transpose(0, 2, 1)), y=labels.astype(np.int16))
    print(len(labels), "images")



if __name__ == "__main__":
    main()
