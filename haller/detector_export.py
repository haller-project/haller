"""
Detector weights and reference vectors for SeedSigner
(resources/handwriting/haller-1-detector.npz and
haller-1-detector-vectors.json), checked by its tests
(tests/test_handwriting_vectors.py).

    python -m haller.detector_export QUANTIZED.npz OUT_DIR [--name haller-1-detector]

The weights file holds the arrays read by the C library (layout of
haller.detector.quantize, without the calibration and layer description).
The vectors are images drawn from a fixed seed (a page of strokes, and a
card with printed lines and strokes, on a dark background): for each one,
the image (zlib, base64) and the SHA-256 of the output maps of the C library
(float32, little-endian). The maps are also compared with the integer
reference of haller.detector.integer_forward.
"""
import argparse
import base64
import ctypes
import hashlib
import json
import os
import zlib

import numpy as np
from PIL import Image, ImageDraw

from . import native as haller_native
from .detector import integer_forward

# Largest difference allowed between the maps of the C library (float32
# arithmetic) and of the numpy reference (float64 rounded to float32)
REFERENCE_TOLERANCE = 1e-3



def test_images(seed: int = 1) -> list[tuple[str, np.ndarray]]:
    """(label, image 384 x 288 uint8): a light page with lines of strokes, and a card with printed lines and
    strokes, on a dark background."""
    rng = np.random.default_rng(seed)
    page = Image.new("L", (288, 384), 60)
    draw = ImageDraw.Draw(page)
    draw.polygon([(40, 50), (250, 45), (255, 340), (35, 345)], fill=210)
    for row in range(12):
        y = 70 + 22 * row
        x = 55 + int(rng.integers(0, 10))
        for _ in range(int(rng.integers(3, 8))):
            dx, dy = int(rng.integers(4, 12)), int(rng.integers(-6, 7))
            draw.line([(x, y), (x + dx, y + dy)], fill=int(rng.integers(20, 70)), width=2)
            x += dx + int(rng.integers(1, 5))
    card = Image.new("L", (288, 384), 90)
    draw = ImageDraw.Draw(card)
    draw.polygon([(60, 90), (230, 80), (240, 320), (65, 330)], fill=230)
    for row in range(12):
        y = 120 + 16 * row
        draw.line([(90, y), (215, y - 3)], fill=150, width=1)
        x = 100
        for _ in range(int(rng.integers(2, 6))):
            dx, dy = int(rng.integers(3, 9)), int(rng.integers(-5, 2))
            draw.line([(x, y - 3), (x + dx, y - 3 + dy)], fill=int(rng.integers(10, 60)), width=2)
            x += dx + int(rng.integers(1, 4))
    return [("page", np.asarray(page)), ("card", np.asarray(card))]



def c_maps(lib, weights: dict, image: np.ndarray) -> np.ndarray:
    from seedsigner.helpers.handwriting import native
    net = native.detector({k: np.ascontiguousarray(v) for k, v in weights.items()})
    work = np.zeros(-(-lib.hw_detector_work_size(ctypes.byref(net)) // 4), np.int32)
    maps = np.empty((native.DETECTOR_OUTPUTS, 192, 144), np.float32)
    image = np.ascontiguousarray(image, np.uint8)
    lib.hw_detector_maps(ctypes.byref(net), image.ctypes.data, work.ctypes.data, maps.ctypes.data)
    return maps



def main():
    a = argparse.ArgumentParser()
    a.add_argument("quantized")
    a.add_argument("out_dir")
    a.add_argument("--name", default="haller-1-detector")
    a = a.parse_args()
    d = np.load(a.quantized)
    q = {k: d[k] for k in d.files}
    weights = {k: v for k, v in q.items() if k not in ("amax", "layers")}
    path = os.path.join(a.out_dir, a.name + ".npz")
    np.savez(path, **weights)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    lib = haller_native.load()
    vectors = {"weights": a.name + ".npz", "weights_sha256": sha, "images": []}
    for label, image in test_images():
        maps = c_maps(lib, weights, image)
        reference = integer_forward(q, image)
        gap = float(np.abs(maps - reference).max())
        if gap > REFERENCE_TOLERANCE:
            raise ValueError(f"image {label}: C maps differ from the integer reference by {gap}")
        vectors["images"].append({
            "label": label,
            "image": base64.b64encode(zlib.compress(image.tobytes(), 9)).decode(),
            "maps_sha256": hashlib.sha256(maps.astype("<f4").tobytes()).hexdigest(),
            "maps_max": [float(m.max()) for m in maps],
        })
        print(label, "reference gap", gap)
    with open(os.path.join(a.out_dir, a.name + "-vectors.json"), "w") as f:
        json.dump(vectors, f, indent=1)
    print(path, sha)



if __name__ == "__main__":
    main()
