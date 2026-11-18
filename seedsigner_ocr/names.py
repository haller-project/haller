"""
Handwritten capital-letter names from scanned forms (Kaggle "Handwriting
Recognition", CC0; Hugging Face mirror verbrannter/handwriting_recognition_from_kaggle).

Many images start with a printed field label ("NOM :", "PRENOM :") on the same
line as the handwritten name. Each image therefore yields several candidate
crops, starting at the beginning of each of the first groups of inked columns;
the filtering step (seedsigner_ocr.filter) keeps the candidate whose reading by an
earlier model matches the label, and drops the image if none does.

    python -m seedsigner_ocr.names OUT.npz PARQUET [PARQUET ...] --limit N --seed S --part I/N

--part I/N converts only every N-th row group, starting at the I-th (0-based),
so that N processes can share the work.
"""
import argparse
import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH, normalize_word
from .synth import photograph

MAX_CANDIDATES = 4



def candidate_starts(scan: Image.Image) -> list[int]:
    """
    Left edges of the first groups of inked columns of a clean scan, groups
    being separated by gaps wider than a quarter of the image height.
    """
    columns = np.nonzero((np.asarray(scan) < 128).sum(0))[0]
    if len(columns) == 0:
        return []
    gaps = np.nonzero(np.diff(columns) > max(4, scan.height // 4))[0]
    return [max(0, c - 2) for c in [columns[0], *columns[gaps + 1]][:MAX_CANDIDATES]]



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("parquet", nargs="+")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--part", default="0/1")
    a = p.parse_args()
    part, parts = map(int, a.part.split("/"))
    rng = np.random.default_rng([a.seed, part])

    samples, labels, groups = [], [], []
    image_id, group_index = 0, 0
    for path in a.parquet:
        file = pq.ParquetFile(path)
        for g in range(file.num_row_groups):
            group_index += 1
            if (group_index - 1) % parts != part:
                continue
            for row in file.read_row_group(g).to_pylist():
                text = (row["text"] or "").strip()
                if not (text.isalpha() and text.isascii()) or len(text) < 3 or text in ("UNREADABLE", "EMPTY"):
                    continue
                scan = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
                # scans are clean: add the camera conditions SeedSigner sees
                photo, scale = photograph(scan, max(16, scan.height // 3), rng)
                for start in candidate_starts(scan):
                    x = normalize_word(photo.crop((int(start * scale), 0, photo.width, photo.height)))
                    if x is not None:
                        samples.append(x)
                        labels.append(text.lower())
                        groups.append(image_id * parts + part)
                image_id += 1
                if a.limit and image_id >= a.limit:
                    break
            if a.limit and image_id >= a.limit:
                break
    x = np.zeros((len(samples), INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    for i, s in enumerate(samples):
        x[i, :, :s.shape[1]] = s
    np.savez_compressed(a.out, x=x, w=np.array([s.shape[1] for s in samples], np.int16), y=np.array(labels),
                        group=np.array(groups, np.int32))
    print(image_id, "images,", len(samples), "candidates")



if __name__ == "__main__":
    main()
