"""
Real samples for the word detector from GNHK (GoodNotes Handwriting in the
Wild, CC BY 4.0; Lee et al., ICDAR 2021).

Random portrait windows of the GNHK pages, rotated and in slight perspective,
with the annotated words given by their polygons. Same format as the shards
of haller.detector_pages (x 384 x 288, y 192 x 144 x 9): channel 0 the words,
channel 1 the numbers ("12", "3.", "4)"), channel 8 (ignored area) the
illegible tokens (%...%) and blank labels; no card. A window is kept when it
holds at least two scored words or numbers.

    python -m haller.detector_gnhk OUT.npz GNHK_SPLIT_DIR WINDOWS_PER_PAGE SEED

GNHK_SPLIT_DIR is the train or test directory of GNHK (pages checked against
data/gnhk.sha256). The pages are read without their EXIF orientation.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from .detector_pages import NET_SIZE, OUT_SIZE, apply, core, dlt

NUMBER = re.compile(r"^\d{1,2}[.)]?$")



def window(page, words, rng):
    """Rotated portrait window of the page: (image 384 x 288, [(text, quad)]
    in the coordinates of the outputs, at half resolution)."""
    heights = [np.linalg.norm(np.array(q[3]) - np.array(q[0])) for _, q in words]
    word_height = float(np.median(heights))
    target = rng.uniform(4.5, 14)  # height of the words in the 288 x 384 input
    s = word_height * NET_SIZE[0] / target  # width of the window in the page
    w, h = s, s * NET_SIZE[1] / NET_SIZE[0]
    q = np.array([q for _, q in words], float).reshape(-1, 2)
    cx = rng.uniform(q[:, 0].min(), q[:, 0].max())
    cy = rng.uniform(q[:, 1].min(), q[:, 1].max())
    a = math.radians(rng.normal(0, 8) + (180 if rng.random() < 0.05 else 0))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    corners = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]) @ rot.T + (cx, cy)
    corners += rng.uniform(-0.04, 0.04, (4, 2)) * w
    red = max(1, int(s / (2 * NET_SIZE[0])))  # reduce large pages before the transform
    small = page.reduce(red) if red > 1 else page
    hm = dlt(corners / red, [(0, 0), (NET_SIZE[0], 0), (NET_SIZE[0], NET_SIZE[1]), (0, NET_SIZE[1])])
    inv = np.linalg.inv(hm)
    inv /= inv[2, 2]
    background = int(np.median(np.asarray(small.reduce(8))))
    x = small.transform(NET_SIZE, Image.PERSPECTIVE, tuple(inv.ravel()[:8]), Image.BILINEAR,
                        fillcolor=int(rng.uniform(15, background)))
    a = np.asarray(x, np.float32) * rng.uniform(0.6, 1.1) + rng.normal(0, rng.uniform(1, 4), (NET_SIZE[1], NET_SIZE[0]))
    x = np.clip(a, 0, 255).astype(np.uint8)
    objects = [(t, apply(hm, np.array(q) / red) / 2) for t, q in words]
    return x, objects



def targets(objects):
    """Target maps (see haller.detector_pages.targets) and the number of scored objects."""
    w, h = OUT_SIZE
    ims = [Image.new("L", (w, h), 0) for _ in range(9)]
    dr = [ImageDraw.Draw(i) for i in ims]
    n = 0
    for t, q in objects:
        if q[:, 0].max() < 0 or q[:, 0].min() > w or q[:, 1].max() < 0 or q[:, 1].min() > h:
            continue
        if t.startswith("%") or not t.strip():
            dr[8].polygon([tuple(p) for p in q], fill=255)
            continue
        k, height = core(q)
        pts = [tuple(p) for p in k]
        channel = 1 if NUMBER.match(t) else 0
        dr[channel].polygon(pts, fill=255)
        dr[channel].point(tuple(np.mean(k, 0)), fill=255)
        dr[7].polygon(pts, fill=int(min(255, height * 8)))
        n += 1
    return np.stack([np.asarray(i) for i in ims], -1), n



def page_words(path):
    """(text, quad) of the annotations of a GNHK page, each quad ordered
    top-left, top-right, bottom-right, bottom-left."""
    words = []
    for o in json.load(open(path)):
        p = o["polygon"]
        q = [(p[f"x{i}"], p[f"y{i}"]) for i in range(4)]
        # sort by angle around the center, then start from the top-left corner
        c = np.mean(q, 0)
        q = sorted(q, key=lambda v: math.atan2(v[1] - c[1], v[0] - c[0]))
        k = int(np.argmin([v[0] + v[1] for v in q]))
        words.append((o["text"], q[k:] + q[:k]))
    return words



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("split_dir")
    p.add_argument("per_page", type=int, help="windows drawn per page")
    p.add_argument("seed", type=int)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(a.split_dir, "*.json"))):
        words = page_words(f)
        if len(words) < 3:
            continue
        page = Image.open(f[:-5] + ".jpg").convert("L")
        for _ in range(a.per_page):
            x, objects = window(page, words, rng)
            y, n = targets(objects)
            if n >= 2:
                xs.append(x)
                ys.append(y)
    np.savez_compressed(a.out, x=np.array(xs), y=np.array(ys), meta=json.dumps([dict(style="gnhk")] * len(xs)))
    print(len(xs), "windows")



if __name__ == "__main__":
    main()
