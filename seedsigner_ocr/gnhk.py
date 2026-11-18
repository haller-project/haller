"""
Real handwritten words from GNHK (GoodNotes Handwriting in the Wild, CC BY 4.0;
Lee et al., ICDAR 2021).

Each annotated word (4-point polygon on a photographed page) is straightened
and cropped with random margins, so that, like a SeedSigner line, the crop
holds the word plus some of its surroundings; then normalized. Several random
crops are taken per word.

    python -m seedsigner_ocr.gnhk OUT.npz GNHK_SPLIT_DIR VARIANTS SEED
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH, locate_word



def words(split_dir: str):
    """(page image path, polygon as 4 (x, y) points, label) of the words made
    of letters and digits only (special tokens such as %math% are skipped)."""
    for name in sorted(os.listdir(split_dir)):
        if not name.endswith(".json"):
            continue
        image = os.path.join(split_dir, name[:-5] + ".jpg")
        for item in json.load(open(os.path.join(split_dir, name))):
            text = item["text"]
            if not text.isalnum() or not text.isascii():
                continue
            p = item["polygon"]
            yield image, [(p[f"x{i}"], p[f"y{i}"]) for i in range(4)], text.lower()



def straighten(page: Image.Image, quad, rng: np.random.Generator, min_letters: int = 3) -> Image.Image | None:
    """The word's quadrilateral mapped to a rectangle, extended by random margins
    (in units of the word height) on every side."""
    q = np.array(quad, np.float64)
    # order: top-left, top-right, bottom-right, bottom-left
    center = q.mean(0)
    angles = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
    q = q[np.argsort(angles)]
    q = np.roll(q, -int(np.argmin(q.sum(1))), axis=0)
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if width < 4 or height < 4 or (width < height and min_letters > 2):
        return None  # tiny, or written vertically / rotated on the page
    ux, uy = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1), (q[3] - q[0]) / max(np.linalg.norm(q[3] - q[0]), 1)
    left, right = rng.uniform(0.1, 0.6) * height, rng.uniform(0.1, 0.6) * height
    top, bottom = rng.uniform(0.1, 0.8) * height, rng.uniform(0.0, 0.6) * height
    corners = [q[0] - left * ux - top * uy, q[1] + right * ux - top * uy,
               q[2] + right * ux + bottom * uy, q[3] - left * ux + bottom * uy]
    out_w, out_h = int(width + left + right), int(height + top + bottom)
    # PIL QUAD: upper-left, lower-left, lower-right, upper-right
    data = [*corners[0], *corners[3], *corners[2], *corners[1]]
    return page.transform((out_w, out_h), Image.QUAD, data, Image.BILINEAR)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("split_dir")
    p.add_argument("variants", type=int)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    samples, labels = [], []
    page_path, page = None, None
    for image, quad, label in words(a.split_dir):
        if image != page_path:
            page_path, page = image, ImageOps.exif_transpose(Image.open(image)).convert("L")
        for _ in range(a.variants):
            crop = straighten(page, quad, rng, len(label))
            if crop is None:
                continue
            if crop.height > 96:  # pages are large photos; keep processing cheap
                crop = crop.resize((int(crop.width * 96 / crop.height), 96), Image.BILINEAR)
            found = locate_word(crop)
            if found is None:
                continue
            samples.append(found[0])
            labels.append(label)
    x = np.zeros((len(samples), INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    for i, s in enumerate(samples):
        x[i, :, :s.shape[1]] = s
    np.savez_compressed(a.out, x=x, w=np.array([s.shape[1] for s in samples], np.int16), y=np.array(labels))
    print(len(samples), "samples")



if __name__ == "__main__":
    main()
