"""
Simulated photos of handwritten seed pages.

A page is either a free list (plain or ruled paper; one or two columns; 12,
15, 18, 21 or 24 words; with or without numbers "1.", "1)" or "1") or one of
SeedSigner's printed cards for 12 and 24 words. The writing is careful (someone
copying a seed): EMNIST digits and capital letters, with size, spacing and
slant nearly constant over the page, stroke width changed by min/max filters,
mostly dark ink. The page is then photographed: placed on a table (or filling
the frame), rotated, in perspective, with an illumination gradient, blur,
noise and JPEG compression, as a portrait image of 1944 x 2592 pixels whose
orientation is already upright.

Haller-1 uses these photos, written with EMNIST test letters, as its simulated
evaluation pages (recipes/haller-1/evalsim); the word detector uses the same
drawing functions for its training pages.

    python -m haller.pages OUT_DIR COUNT SEED --emnist EMNIST_TEST.npz \
        --bip39 data/bip39_english.txt

Output: OUT_DIR/simSS_NNNN.jpg and OUT_DIR/truth.json
    {name: {layout, group, words, rotation, count, columns, numbers, paper}}
EMNIST_TEST.npz is written by `python -m haller.emnist EMNIST_TEST.npz gzip
--split test`. A given seed, EMNIST file and word list always give the same
JPEG files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PAGE = (1400, 1980)  # A5 sheet at about 190 dpi
FRAME = (1944, 2592)  # portrait photo
CARDS = Path(__file__).resolve().parent.parent / "data" / "cards"

BILINEAR = Image.Resampling.BILINEAR



def load_emnist(path: str):
    """EMNIST digits and capitals of a file written by haller.emnist: images
    (N, 28, 28), white ink on black, upright, and the indices of each class
    0-35 (0-9 digits, 10-35 capitals)."""
    with np.load(path) as d:
        x, y = d["x"], d["y"]
    keep = y < 36
    x, y = x[keep], y[keep]
    return x, {k: np.nonzero(y == k)[0] for k in range(36)}



def mnemonic(entropy: bytes, wordlist: list) -> list:
    """BIP39 words of an entropy: entropy and the first bits of its SHA-256,
    cut into 11-bit indices."""
    n = len(entropy) * 8
    bits = int.from_bytes(entropy, "big") << (n // 32)
    bits |= hashlib.sha256(entropy).digest()[0] >> (8 - n // 32)
    count = (n + n // 32) // 11
    return [wordlist[(bits >> (11 * (count - 1 - i))) & 2047] for i in range(count)]



def homography(src, dst):
    """Coefficients of Image.transform(PERSPECTIVE) mapping the output points
    dst onto the input points src."""
    a, b = [], []
    for (u, v), (x, y) in zip(dst, src):
        a.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); b.append(x)
        a.append([0, 0, 0, u, v, 1, -y * u, -y * v]); b.append(y)
    return np.linalg.solve(np.array(a, float), np.array(b, float))



def card_template(size: int) -> dict:
    """Front of SeedSigner's printed card for 12 or 24 words (1004 x 1401
    pixels at 400 dpi): image, writing lines (x0, x1, baseline y) in word
    order, and line spacing."""
    img = Image.open(CARDS / f"card{size}.png").convert("L")
    ys = [367 + 76.6 * r for r in range(12)]
    if size == 12:
        lines = [(180 if r < 9 else 221, 812, int(ys[r])) for r in range(12)]
    else:
        lines = [(153 if r < 9 else 188, 474, int(ys[r])) for r in range(12)] + \
                [(621, 912, int(ys[r])) for r in range(12)]
    return dict(img=img, lines=lines, spacing=76.6)



def glyph(images, by_class, cls, height, rng, weight):
    """One EMNIST character as dark ink on white (255), scaled to the height,
    stroke thickened (weight > 0) or thinned (weight < 0), slightly rotated."""
    i = rng.choice(by_class[cls])
    g = Image.fromarray(255 - images[i]).crop(Image.fromarray(images[i]).getbbox())
    hh = max(5, int(height * rng.uniform(0.95, 1.04)))
    g = g.resize((max(3, int(g.width * hh / g.height)), hh), BILINEAR)
    if weight > 0:  # dark ink on a light background: minimum filter
        g = g.filter(ImageFilter.MinFilter(2 * weight + 1))
    elif weight < 0:
        g = g.filter(ImageFilter.MaxFilter(3))
    return np.asarray(g.rotate(rng.uniform(-3, 3), expand=True, fillcolor=255))



def compose(text, height, images, by_class, rng, weight, slant, spacing):
    """Image of a text in capitals and digits (background 255) and the row of
    its baseline. "." and ")" are drawn; " " is a blank."""
    elems = []
    for ch in text:
        if ch == " ":
            elems.append(None)
        elif ch == ".":
            r = max(2, int(height * rng.uniform(0.06, 0.1)))
            im = Image.fromarray(np.full((2 * r + 2, 2 * r + 2), 255, np.uint8))
            ImageDraw.Draw(im).ellipse([1, 1, 2 * r, 2 * r], fill=0)
            elems.append(np.asarray(im))
        elif ch == ")":
            im = Image.new("L", (max(6, int(height * 0.3)), int(height)), 255)
            ImageDraw.Draw(im).arc([-int(height * 0.3), 0, int(height * 0.25), int(height) - 1], -70, 70, fill=0,
                                   width=max(2, int(height * 0.07)))
            elems.append(np.asarray(im))
        else:
            cls = int(ch) if ch.isdigit() else 10 + ord(ch.upper()) - 65
            elems.append(glyph(images, by_class, cls, height, rng, weight))
    widths = [e.shape[1] if e is not None else int(height * 0.6) for e in elems]
    gaps = [int(height * max(0.0, rng.normal(spacing, 0.035))) for _ in elems]
    w = sum(widths) + sum(gaps) + int(height)
    base = int(height * 1.2)
    h = base + int(height * 0.4)
    out = np.full((h, w), 255, np.uint8)
    x = int(height * 0.3)
    for k, (e, ew) in enumerate(zip(elems, widths)):
        if k:
            x += gaps[k]
        if e is not None:
            y = base - e.shape[0] + int(rng.integers(-1, 2))
            y = max(0, min(h - e.shape[0], y))
            cw = min(e.shape[1], w - x)
            out[y:y + e.shape[0], x:x + cw] = np.minimum(out[y:y + e.shape[0], x:x + cw], e[:, :cw])
        x += ew
    im = Image.fromarray(out)
    if slant:  # shear: the top of the letters moves right for a positive slant
        s = np.tan(np.radians(slant))
        margin = int(abs(s) * h) + 2
        im2 = Image.new("L", (w + 2 * margin, h), 255)
        im2.paste(im, (margin, 0))
        im = im2.transform(im2.size, Image.Transform.AFFINE, (1, s, -s * base, 0, 1, 0), BILINEAR, fillcolor=255)
    return np.asarray(im), base



def ink(page, m, x, y, level):
    """Lays the text m (255 = paper) on the float page at (x, y) with ink of
    the given gray level."""
    m = np.asarray(m, np.float32) / 255
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    zone = page[y0:y1, x0:x1]
    page[y0:y1, x0:x1] = np.minimum(zone, zone * sub + level * (1 - sub))



def list_page(words, images, by_class, rng, opts):
    """Free list on an A5 sheet: opts columns (1 or 2), numbers (bool), paper
    ("plain" or "ruled"). Words run down the first column, then the second."""
    w, h = PAGE
    background = rng.uniform(205, 245)
    page = np.full((h, w), background, np.float32)
    n = len(words)
    ncol = opts["columns"]
    per_col = -(-n // ncol)
    height = rng.uniform(50, 80) if ncol == 1 else rng.uniform(45, 65)
    pitch = height * rng.uniform(1.45, 2.1)
    ytop = rng.uniform(0.06, 0.18) * h
    if per_col * pitch + ytop > 0.95 * h:  # page too small: smaller writing
        f = 0.95 * h / (per_col * pitch + ytop)
        height, pitch = height * f, pitch * f
    if opts["paper"] == "ruled":
        y0 = ytop + height * 1.2
        im = Image.fromarray(page.astype(np.uint8))
        d = ImageDraw.Draw(im)
        gray = int(rng.uniform(140, 200))
        for yy in np.arange(y0 % pitch, h, pitch):
            d.line([(0, yy), (w, yy)], fill=gray, width=int(rng.integers(1, 4)))
        if rng.random() < 0.5:  # margin line
            xm = int(rng.uniform(0.08, 0.14) * w)
            d.line([(xm, 0), (xm, h)], fill=gray, width=2)
        page = np.asarray(im, np.float32)
    # careful writing: mostly dark ink, regular stroke, constant slant
    level = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 100)
    weight = int(rng.choice([-1, 0, 0, 0, 1, 1]))
    slant = rng.normal(0, 5)
    spacing = rng.uniform(0.1, 0.25)
    fmt = rng.choice(["{}.", "{})", "{}"])
    xg = rng.uniform(0.07, 0.15) * w
    col_width = (w - xg) / ncol
    for k, word in enumerate(words):
        col, row = divmod(k, per_col)
        text = word.upper()
        if opts["numbers"]:
            text = fmt.format(k + 1) + " " + text
        m, base = compose(text, height, images, by_class, rng, weight, slant + rng.normal(0, 1), spacing)
        if m.shape[1] > 0.95 * col_width:
            f = 0.95 * col_width / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), max(5, int(m.shape[0] * f)))))
            base = int(base * f)
        yb = ytop + height * 1.2 + row * pitch + rng.normal(0, height * 0.03)
        if opts["paper"] == "ruled":
            yb -= rng.uniform(0, 0.12) * height
        x = int(xg + col * col_width + rng.normal(0, height * 0.06))
        im = Image.fromarray(m).rotate(rng.normal(0, 0.8), expand=False, fillcolor=255)
        ink(page, np.asarray(im), x, int(yb - base), level)
    return Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))



def card_page(words, images, by_class, rng):
    """SeedSigner card filled in, inside the black border of the physical card."""
    tpl = card_template(12 if len(words) == 12 else 24)
    img = tpl["img"].copy()
    level = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 100)
    weight = int(rng.choice([-1, 0, 0, 0, 1, 1]))
    slant = rng.normal(0, 5)
    a = np.asarray(img, np.float32).copy()
    height0 = tpl["spacing"] * rng.uniform(0.45, 0.85)  # writing size of the card
    spacing = rng.uniform(0.1, 0.25)
    for word, (x0, x1, yl) in zip(words, tpl["lines"]):
        height = height0 * rng.uniform(0.96, 1.04)
        m, base = compose(word.upper(), height, images, by_class, rng, weight, slant, spacing)
        if m.shape[1] > x1 - x0 + 10:
            f = (x1 - x0 + 10) / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), m.shape[0])))
        x = x0 + int(rng.uniform(-0.25, 0.1) * height)
        ink(a, m, x, int(yl - rng.uniform(0, 0.15) * height - base), level)
    w, h = img.size
    border = int(0.07 * w)
    card = np.full((h + 2 * border, w + 2 * border), rng.uniform(20, 45), np.float32)
    card[border:border + h, border:border + w] = a
    return Image.fromarray(np.clip(card, 0, 255).astype(np.uint8))



def photograph(page, rng, fill=False):
    """Photo of the page lying on a table (fill: the sheet overflows the frame)."""
    w, h = FRAME
    hp = h * rng.uniform(1.0, 1.25) if fill else h * rng.uniform(0.7, 0.95)
    wp = hp * page.width / page.height
    ang = np.radians(rng.normal(0, 4))
    cx, cy = w / 2 + rng.normal(0, 0.04 * w), h / 2 + rng.normal(0, 0.03 * h)
    corners = np.array([(-wp / 2, -hp / 2), (wp / 2, -hp / 2), (wp / 2, hp / 2), (-wp / 2, hp / 2)])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    persp = rng.uniform(-0.035, 0.035, (4, 2)) * hp
    dst = [tuple(p) for p in corners @ rot.T + (cx, cy) + persp]
    coef = homography([(0, 0), (page.width, 0), (page.width, page.height), (0, page.height)], dst)
    table = rng.uniform(15, 110)
    out = page.transform((w, h), Image.Transform.PERSPECTIVE, coef, BILINEAR, fillcolor=int(table))
    a = np.asarray(out, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-0.5, 0.5, 2)
    light = 1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    a = a * light * rng.uniform(0.55, 1.0)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    a = np.asarray(out, np.float32) + rng.normal(0, rng.uniform(2, 5), (h, w))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))



# (group, weight, options); None: drawn at random for each page
GROUPS = [
    ("list", 3, dict(layout="list", count=[12], columns=1, numbers=False, paper="plain")),
    ("numbered_list", 3, dict(layout="list", count=[12], columns=1, numbers=True, paper="plain")),
    ("ruled_list", 3, dict(layout="list", count=[12], columns=1, numbers=None, paper="ruled")),
    ("two_columns", 2, dict(layout="list", count=[24, 12, 18], columns=2, numbers=None, paper=None)),
    ("lengths", 2, dict(layout="list", count=[15, 18, 21, 24], columns=1, numbers=None, paper=None)),
    ("card12", 2, dict(layout="card", count=[12])),
    ("card24", 1, dict(layout="card", count=[24])),
]



def simulate(out_dir, count, seed, images, by_class, wordlist):
    """Writes count photos and truth.json in out_dir; returns the truth."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    weights = np.array([g[1] for g in GROUPS], float)
    truth = {}
    for i in range(count):
        group, _, o = GROUPS[rng.choice(len(GROUPS), p=weights / weights.sum())]
        n = int(rng.choice(o["count"]))
        words = list(mnemonic(rng.bytes(n * 4 // 3), wordlist))
        if o["layout"] == "card":
            page = card_page(words, images, by_class, rng)
            img = photograph(page, rng)
            info = dict(columns=1 if n == 12 else 2, numbers=True, paper="card")
        else:
            numbers = bool(rng.random() < 0.5) if o["numbers"] is None else o["numbers"]
            paper = str(rng.choice(["plain", "ruled"])) if o["paper"] is None else o["paper"]
            opts = dict(columns=o["columns"], numbers=numbers, paper=paper)
            page = list_page(words, images, by_class, rng, opts)
            img = photograph(page, rng, fill=rng.random() < 0.25)
            info = opts
        name = f"sim{seed:02d}_{i:04d}.jpg"
        img.save(os.path.join(out_dir, name), quality=int(rng.integers(70, 95)))
        truth[name] = dict(layout=o["layout"], group=group, words=words, rotation=0, count=n, **info)
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    return truth



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("count", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--emnist", required=True, help="EMNIST file of haller.emnist (the test split for evaluation)")
    p.add_argument("--bip39", required=True)
    a = p.parse_args()
    images, by_class = load_emnist(a.emnist)
    wordlist = open(a.bip39).read().split()
    simulate(a.out_dir, a.count, a.seed, images, by_class, wordlist)



if __name__ == "__main__":
    main()
