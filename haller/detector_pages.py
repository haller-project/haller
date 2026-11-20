"""
Annotated simulated photos for the word detector (haller.detector).

The pages are laid out and drawn as in haller.pages (EMNIST capitals) or with
handwriting fonts, but the number and the word of each line are composed
separately, and the exact position of every element is recorded: ink box of
each word and number (a quadrilateral in the photo), corners of the light area
of each printed SeedSigner card, row and column of the word.

Framing defects (`--defect` in bench mode): normal, crop_top, crop_bottom,
crop_side, close, far, perspective, rotation (up to 35 degrees), upside_down
(180 degrees), sheet (several cards of a printed sheet), shadow (cast shadows
and objects lying on the page), lighting (very uneven).

Two modes:

    python -m haller.detector_pages train OUT.npz COUNT SEED --fonts FONTS \
        --emnist EMNIST.npz --bip39 data/bip39_english.txt [--test]

writes a training shard: x (COUNT, 384, 288) uint8 network inputs, y (COUNT,
192, 144, 9) uint8 target maps (see targets()), meta (JSON list of the style,
defect, layout and word count of each photo);

    python -m haller.detector_pages bench OUT_DIR COUNT SEED --fonts FONTS \
        --emnist EMNIST.npz --bip39 data/bip39_english.txt [--defect D] [--style S] [--layout L] [--test]

writes full photos (1944 x 2592 JPEG) and OUT_DIR/truth.json with the boxes.

FONTS is the directory of the 287 fonts of haller.fonts (data/fonts.sha256);
the cursive ones are those of data/cursive_fonts.txt present in it. EMNIST.npz
is written by haller.emnist (train split). With --test, the photos use the
EMNIST test split (pass it as --emnist) and the nine fonts of
data/detector_test_fonts.sha256, which are not among the training fonts
(download them with `python -m haller.fonts DIR --list
data/detector_test_fonts.sha256`). A given seed, input files and software
versions always give the same shard.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .pages import FRAME, PAGE, card_template, compose, ink, load_emnist, mnemonic

DATA = Path(__file__).resolve().parent.parent / "data"

NET_SIZE = (288, 384)  # detector input (width, height)
OUT_SIZE = (144, 192)  # detector outputs (half resolution)
DEFECTS = ["normal", "crop_top", "crop_bottom", "crop_side", "close", "far", "perspective", "rotation",
           "upside_down", "sheet", "shadow", "lighting"]
DEFECT_WEIGHTS = [26, 9, 7, 7, 7, 7, 8, 9, 4, 6, 5, 5]
STYLES = ["capitals", "lowercase", "cursive"]
STYLE_WEIGHTS = [5, 2, 3]

# Labels written in the meta field of the training shards. They are those of
# the shards of haller-1-detector, so that a regenerated shard has the same
# content digest as the original one.
META_LABELS = {
    "normal": "normal", "crop_top": "coupe_haut", "crop_bottom": "coupe_bas", "crop_side": "coupe_cote",
    "close": "proche", "far": "loin", "perspective": "perspective", "rotation": "rotation",
    "upside_down": "retournee", "sheet": "planche", "shadow": "ombre", "lighting": "eclairage",
    "capitals": "capitales", "lowercase": "minuscules", "cursive": "cursive", "card": "carte", "list": "liste",
}

# Corners of the light area of the cards (top-left, top-right, bottom-right,
# bottom-left) in the 1004 x 1401 template of haller.pages.card_template
CARD_CORNERS = {
    12: [(52.5, 82.96), (937.5, 77.05), (937.5, 1337.5), (52.5, 1337.5)],
    24: [(33.22, 57.67), (1002.5, 49.89), (1002.5, 1369.85), (32.04, 1373.89)],
}



def font_lists(fonts_dir: str, test: bool) -> dict:
    """Font paths for the "cursive" and "lowercase" styles (capitals written
    with a font use the "lowercase" list)."""
    if test:
        names = [os.path.basename(line.split()[1]) for line in open(DATA / "detector_test_fonts.sha256")]
        paths = [os.path.join(fonts_dir, n) for n in names]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"missing test fonts: {missing}")
        return {"cursive": paths, "lowercase": paths}
    cursive = set(open(DATA / "cursive_fonts.txt").read().split())
    names = sorted(f for f in os.listdir(fonts_dir) if f.lower().endswith((".ttf", ".otf")))
    cur = [os.path.join(fonts_dir, f) for f in names if f in cursive]
    other = [os.path.join(fonts_dir, f) for f in names if f not in cursive]
    return {"cursive": cur or other, "lowercase": other}



_x_heights = {}


def x_height(path: str) -> float:
    """Height of the letter x of a font, as a fraction of the font size."""
    if path not in _x_heights:
        try:
            bb = ImageFont.truetype(path, 100).getbbox("x", anchor="ls")
            _x_heights[path] = max(0.2, -bb[1] / 100)
        except Exception:
            _x_heights[path] = 0.5
    return _x_heights[path]



class Hand:
    """Writing of one page: style, font, slant, stroke weight, ink level."""

    def __init__(self, style, rng, images, by_class, fonts):
        self.style, self.images, self.by_class = style, images, by_class
        self.weight = int(rng.choice([-1, 0, 0, 0, 1, 1]))
        self.slant = rng.normal(0, 5)
        self.spacing = rng.uniform(0.1, 0.25)
        self.level = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 110)
        self.font = None
        if style != "capitals" or rng.random() < 0.3:
            pool = fonts[style if style != "capitals" else "lowercase"]
            self.font = pool[rng.integers(len(pool))]
            self.x = rng.uniform(0.5, 0.65)  # x-height as a fraction of the writing height
            self.stroke = rng.uniform(0.05, 0.09)
            self.case = "upper" if style == "capitals" else ("title" if rng.random() < 0.15 else "lower")

    def compose(self, text, height, rng, slant=None):
        """Image of the text (background 255) and the row of its baseline."""
        slant = self.slant if slant is None else slant
        if self.font is None:
            return compose(text, height, self.images, self.by_class, rng, self.weight, slant, self.spacing)
        if self.case == "upper":
            text = text.upper()
            size = max(8, int(height / 0.72))
        else:
            text = text.lower() if self.case == "lower" else text.capitalize()
            size = max(8, int(self.x * height / x_height(self.font)))
        f = ImageFont.truetype(self.font, size)
        w = int(size * (0.9 * len(text) + 3))
        h = int(size * 2.4)
        base = int(size * 1.5)
        im = Image.new("L", (w, h), 255)
        ImageDraw.Draw(im).text((int(size * 0.4), base), text, font=f, fill=0, anchor="ls")
        bb = Image.eval(im, lambda v: 255 - v).getbbox()
        if bb:
            im = im.crop((0, 0, min(w, bb[2] + int(size * 0.3)), h))
        stroke = int(round(self.stroke * height * (self.x if self.case != "upper" else 0.7) / 2)) + max(0, self.weight)
        if stroke > 0:
            im = im.filter(ImageFilter.MinFilter(2 * stroke + 1))
        s = math.tan(math.radians(slant))
        if abs(s) > 1e-3:  # shear: the top of the letters moves right for a positive slant
            margin = int(abs(s) * h) + 2
            im2 = Image.new("L", (im.width + 2 * margin, h), 255)
            im2.paste(im, (margin, 0))
            im = im2.transform(im2.size, Image.AFFINE, (1, s, -s * base, 0, 1, 0), Image.BILINEAR, fillcolor=255)
        return np.asarray(im), base



def place(page, m, x, y, level, objects, info):
    """Inks the text m (255 = paper) at (x, y) and records its ink box in objects."""
    ink(page, m, x, y, level)
    ys, xs = np.nonzero(np.asarray(m) < 128)
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
    objects.append(dict(info, quad=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))



def list_page(words, hand, rng, opts):
    """Free list on an A5 sheet (opts: columns, numbers, paper "plain" or
    "ruled"); returns the page, its objects and no card."""
    w, h = PAGE
    background = rng.uniform(205, 245)
    page = np.full((h, w), background, np.float32)
    n, ncol = len(words), opts["columns"]
    per_col = -(-n // ncol)
    height = rng.uniform(50, 80) if ncol == 1 else rng.uniform(45, 65)
    pitch = height * rng.uniform(1.45, 2.1)
    ytop = rng.uniform(0.06, 0.18) * h
    if per_col * pitch + ytop > 0.95 * h:  # page too small: smaller writing
        f = 0.95 * h / (per_col * pitch + ytop)
        height, pitch = height * f, pitch * f
    if opts["paper"] == "ruled":
        im = Image.fromarray(page.astype(np.uint8))
        d = ImageDraw.Draw(im)
        gray = int(rng.uniform(140, 200))
        for yy in np.arange((ytop + height * 1.2) % pitch, h, pitch):
            d.line([(0, yy), (w, yy)], fill=gray, width=int(rng.integers(1, 4)))
        if rng.random() < 0.5:  # margin line
            xm = int(rng.uniform(0.08, 0.14) * w)
            d.line([(xm, 0), (xm, h)], fill=gray, width=2)
        page = np.asarray(im, np.float32).copy()
    fmt = rng.choice(["{}.", "{})", "{}"])
    xg = rng.uniform(0.07, 0.15) * w
    col_width = (w - xg) / ncol
    objects = []
    for k, word in enumerate(words):
        col, row = divmod(k, per_col)
        slant = hand.slant + rng.normal(0, 1)
        yb = ytop + height * 1.2 + row * pitch + rng.normal(0, height * 0.03)
        if opts["paper"] == "ruled":
            yb -= rng.uniform(0, 0.12) * height
        x = int(xg + col * col_width + rng.normal(0, height * 0.06))
        if opts["numbers"]:
            mn, bn = hand.compose(fmt.format(k + 1), height, rng, slant)
            place(page, mn, x, int(yb - bn), hand.level, objects, dict(kind="number", k=k, row=row, col=col))
            x += mn.shape[1] + int(height * rng.uniform(0.25, 0.7))
        m, base = hand.compose(word.upper() if hand.font is None else word, height, rng, slant)
        if m.shape[1] > 0.95 * col_width:
            f = 0.95 * col_width / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), max(5, int(m.shape[0] * f)))))
            base = int(base * f)
        m = np.asarray(Image.fromarray(m).rotate(rng.normal(0, 0.8), expand=False, fillcolor=255))
        place(page, m, x, int(yb - base), hand.level, objects, dict(kind="word", k=k, row=row, col=col, word=word))
    return Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)), objects, None



_printed_numbers = {}


def printed_numbers(n):
    """Ink boxes (x0, y0, x1, y1) of the numbers printed on the blank card
    template, in word order (None where no ink is found)."""
    if n not in _printed_numbers:
        tpl = card_template(n)
        a = np.asarray(tpl["img"])
        boxes = []
        for x0, x1, y in tpl["lines"]:
            z = a[y - 50:y + 6, max(0, x0 - 140):x0 - 4] < 128
            ys, xs = np.nonzero(z)
            boxes.append((max(0, x0 - 140) + xs.min(), y - 50 + ys.min(), max(0, x0 - 140) + xs.max() + 1,
                          y - 50 + ys.max() + 1) if len(xs) else None)
        _printed_numbers[n] = boxes
    return _printed_numbers[n]



def card_face(words, hand, rng, n):
    """Front of a card (template with the black border of the physical card),
    written with words (None: blank); objects and light-area corners in its
    coordinates."""
    tpl = card_template(n)
    a = np.asarray(tpl["img"], np.float32).copy()
    objects = []
    height0 = tpl["spacing"] * rng.uniform(0.45, 0.85)
    for k, (x0, x1, yl) in enumerate(tpl["lines"]):
        if words is None:
            break
        height = height0 * rng.uniform(0.96, 1.04)
        m, base = hand.compose(words[k].upper() if hand.font is None else words[k], height, rng)
        if m.shape[1] > x1 - x0 + 10:
            f = (x1 - x0 + 10) / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), m.shape[0])))
        x = x0 + int(rng.uniform(-0.25, 0.1) * height)
        place(a, m, x, int(yl - rng.uniform(0, 0.15) * height - base), hand.level, objects,
              dict(kind="word", k=k, row=k % 12, col=k // 12, word=words[k]))
    for k, b in enumerate(printed_numbers(n)):
        if b:
            objects.append(dict(kind="number", k=k, row=k % 12, col=k // 12, printed=True,
                                quad=[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]))
    w, h = tpl["img"].size
    border = int(0.07 * w)
    card = np.full((h + 2 * border, w + 2 * border), rng.uniform(20, 45), np.float32)
    card[border:border + h, border:border + w] = a

    def shift(q):
        return [(x + border, y + border) for x, y in q]

    for o in objects:
        o["quad"] = shift(o["quad"])
    return card, objects, shift(CARD_CORNERS[n])



def card_page(words, hand, rng, wordlist, sheet=False):
    """Written card alone, or (sheet) among blank or written cards of a printed
    sheet; returns the image, its objects and its cards."""
    n = len(words)
    card, objects, corners = card_face(words, hand, rng, n)
    cards = [dict(corners=corners, target=True, n=n)]
    if not sheet:
        return Image.fromarray(np.clip(card, 0, 255).astype(np.uint8)), objects, cards
    # printed sheet: grid of cards on a sheet, the target card at a random position
    h, w = card.shape
    nx, ny = int(rng.integers(2, 4)), int(rng.integers(1, 3))
    ix, iy = int(rng.integers(nx)), int(rng.integers(ny))
    gap = int(rng.uniform(0, 0.08) * w)
    margin = int(rng.uniform(0.03, 0.1) * w)
    paper = np.full((ny * h + (ny - 1) * gap + 2 * margin, nx * w + (nx - 1) * gap + 2 * margin),
                    rng.uniform(215, 245), np.float32)
    every, cards = [], []
    for j in range(ny):
        for i in range(nx):
            ox, oy = margin + i * (w + gap), margin + j * (h + gap)
            if (i, j) == (ix, iy):
                c, ob, co, target = card, objects, corners, True
            else:
                others = list(mnemonic(rng.bytes(n * 4 // 3), wordlist)) if rng.random() < 0.25 else None
                c, ob, co = card_face(others, hand, rng, n)
                target = False
            paper[oy:oy + h, ox:ox + w] = c
            for o in ob:
                every.append(dict(o, quad=[(x + ox, y + oy) for x, y in o["quad"]], other=not target))
            cards.append(dict(corners=[(x + ox, y + oy) for x, y in co], target=target, n=n))
    return Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)), every, cards



def dlt(src, dst):
    """3 x 3 homography mapping the 4 points src onto dst."""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(a, float))
    hm = vt[-1].reshape(3, 3)
    return hm / hm[2, 2]



def apply(hm, pts):
    """Points mapped by the homography hm."""
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ hm.T
    return p[:, :2] / p[:, 2:3]



def framing(page, defect, rng, card):
    """Homography page -> photo (coordinates of the full 1944 x 2592 photo) for the defect."""
    w, h = FRAME
    fill = (not card) and rng.random() < 0.25  # the sheet overflows the frame
    if card:
        hp = h * rng.uniform(0.42, 0.62) if page.height < 2000 else h * rng.uniform(0.75, 0.95)
    else:
        hp = h * (rng.uniform(1.0, 1.25) if fill else rng.uniform(0.7, 0.95))
    ang = rng.normal(0, 4)
    persp = 0.035
    cx, cy = w / 2 + rng.normal(0, 0.04 * w), h / 2 + rng.normal(0, 0.03 * h)
    if defect == "close":
        hp *= rng.uniform(1.3, 1.9)
        cx += rng.normal(0, 0.1 * w)
        cy += rng.normal(0, 0.1 * h)
    elif defect == "far":
        hp *= rng.uniform(0.35, 0.55)
        cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.3, 0.7)
    elif defect == "perspective":
        persp = rng.uniform(0.08, 0.16)
    elif defect == "rotation":
        ang = rng.choice([-1, 1]) * rng.uniform(8, 35)
    elif defect == "upside_down":
        ang = 180 + rng.normal(0, 6)
    wp = hp * page.width / page.height
    if defect in ("rotation", "upside_down"):  # the rotated page must fit in the frame
        a = math.radians(ang)
        ext_w = abs(wp * math.cos(a)) + abs(hp * math.sin(a))
        ext_h = abs(wp * math.sin(a)) + abs(hp * math.cos(a))
        f = min(1.0, 0.95 * w / ext_w, 0.95 * h / ext_h) if not fill else 1.0
        wp, hp = wp * f, hp * f
    cut = rng.uniform(0.08, 0.35)
    if defect == "crop_top":
        cy = hp / 2 - cut * hp
    elif defect == "crop_bottom":
        cy = h - hp / 2 + cut * hp
    elif defect == "crop_side":
        c = rng.uniform(0.06, 0.25)
        cx = (wp / 2 - c * wp) if rng.random() < 0.5 else (w - wp / 2 + c * wp)
    a = math.radians(ang)
    corners = np.array([(-wp / 2, -hp / 2), (wp / 2, -hp / 2), (wp / 2, hp / 2), (-wp / 2, hp / 2)])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = corners @ rot.T + (cx, cy) + rng.uniform(-persp, persp, (4, 2)) * hp
    return dlt([(0, 0), (page.width, 0), (page.width, page.height), (0, page.height)], dst)



def photograph(page, hm, defect, rng, scale):
    """Photo (size FRAME * scale) of the page through the homography hm (page
    -> full photo); returns the photo and the mask of the objects lying on it
    (None without objects)."""
    w, h = int(round(FRAME[0] * scale)), int(round(FRAME[1] * scale))
    s = np.diag([scale, scale, 1.0]) @ hm
    inv = np.linalg.inv(s)
    inv = inv / inv[2, 2]
    table = rng.uniform(15, 110)
    if defect == "sheet" or rng.random() < 0.15:
        table = rng.uniform(110, 200)  # light table
    out = page.transform((w, h), Image.PERSPECTIVE, tuple(inv.ravel()[:8]), Image.BILINEAR, fillcolor=int(table))
    a = np.asarray(out, np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w
    yy /= h
    g = 0.5 if defect != "lighting" else rng.uniform(0.9, 1.6)
    gx, gy = rng.uniform(-g, g, 2)
    light = 1 + gx * (xx - 0.5) + gy * (yy - 0.5)
    if defect == "lighting":
        cx, cy, r = rng.uniform(0, 1, 2).tolist() + [rng.uniform(0.2, 0.6)]
        light = light * (1 - rng.uniform(0.3, 0.6) * np.clip(((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2, 0, 1))
        if rng.random() < 0.5:  # specular highlight
            hx, hy = rng.uniform(0.2, 0.8, 2)
            a += rng.uniform(40, 120) * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * rng.uniform(0.03, 0.1) ** 2))
    a = a * np.clip(light, 0.15, 2.0) * rng.uniform(0.55, 1.0)
    mask = None
    if defect == "shadow":
        a, mask = shadows_and_objects(a, rng, w, h)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6) * scale))
    a = np.asarray(out, np.float32) + rng.normal(0, rng.uniform(2, 5) * min(1.0, math.sqrt(scale * 2)), (h, w))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), mask



def shadows_and_objects(a, rng, w, h):
    """Cast shadows (blurred darkening) and opaque objects (pen, finger at the
    edge of the frame); returns the image and the mask of the objects."""
    mask = Image.new("L", (w, h), 0)
    for _ in range(int(rng.integers(1, 3))):
        sh = Image.new("L", (w, h), 0)
        pts = [(rng.uniform(-0.2, 1.2) * w, rng.uniform(-0.2, 1.2) * h) for _ in range(int(rng.integers(3, 6)))]
        ImageDraw.Draw(sh).polygon(pts, fill=255)
        sh = np.asarray(sh.filter(ImageFilter.GaussianBlur(rng.uniform(0.01, 0.05) * w)), np.float32) / 255
        a = a * (1 - rng.uniform(0.3, 0.65) * sh)
    if rng.random() < 0.7:
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d, dm = ImageDraw.Draw(im), ImageDraw.Draw(mask)
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:  # pen: long strip
                cx, cy, length, width, t = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 0.7) * w, \
                    rng.uniform(0.02, 0.04) * w, rng.uniform(0, math.pi)
                dx, dy = math.cos(t) * length / 2, math.sin(t) * length / 2
                nx, ny = -math.sin(t) * width / 2, math.cos(t) * width / 2
                pts = [(cx - dx + nx, cy - dy + ny), (cx + dx + nx, cy + dy + ny), (cx + dx - nx, cy + dy - ny),
                       (cx - dx - nx, cy - dy - ny)]
            else:  # finger or object at the edge of the frame
                edge = int(rng.integers(4))
                r = rng.uniform(0.05, 0.12) * w
                t = rng.uniform(0.1, 0.9)
                cx, cy = [(t * w, 0), (w, t * h), (t * w, h), (0, t * h)][edge]
                pts = [(cx + r * math.cos(u), cy + 1.6 * r * math.sin(u)) for u in np.linspace(0, 2 * math.pi, 16)]
            gray = int(rng.uniform(30, 200))
            d.polygon(pts, fill=gray)
            dm.polygon(pts, fill=255)
        a = np.asarray(im, np.float32)
    return a, np.asarray(mask) > 127



def generate(rng, images, by_class, fonts, wordlist, defect=None, style=None, scale=1.0, layout=None):
    """One annotated photo: (image, truth). The quadrilaterals are in the
    coordinates of the full photo. layout: card12, card24, list1 or list2
    (None: drawn at random)."""
    defect = defect or DEFECTS[rng.choice(len(DEFECTS), p=np.array(DEFECT_WEIGHTS) / sum(DEFECT_WEIGHTS))]
    style = style or STYLES[rng.choice(3, p=np.array(STYLE_WEIGHTS) / sum(STYLE_WEIGHTS))]
    hand = Hand(style, rng, images, by_class, fonts)
    if layout is None:
        layout = "card" if defect == "sheet" else rng.choice(["card12", "card24", "list1", "list2"],
                                                             p=[0.27, 0.13, 0.38, 0.22])
        if layout == "card":
            layout = "card12" if rng.random() < 0.65 else "card24"
    if layout.startswith("card"):
        n = 12 if layout == "card12" else 24
        words = list(mnemonic(rng.bytes(n * 4 // 3), wordlist))
        page, objects, cards = card_page(words, hand, rng, wordlist, sheet=defect == "sheet")
        info = dict(layout="card", columns=1 if n == 12 else 2, numbers=True, paper="card")
    else:
        ncol = 1 if layout == "list1" else 2
        n = int(rng.choice([12, 15, 18, 21, 24])) if ncol == 1 else int(rng.choice([24, 12, 18]))
        if ncol == 1 and rng.random() < 0.5:
            n = 12
        words = list(mnemonic(rng.bytes(n * 4 // 3), wordlist))
        opts = dict(columns=ncol, numbers=bool(rng.random() < 0.5), paper=str(rng.choice(["plain", "ruled"])))
        page, objects, cards = list_page(words, hand, rng, opts)
        info = dict(layout="list", **opts)
    hm = framing(page, defect, rng, layout.startswith("card") and defect != "sheet")
    img, mask = photograph(page, hm, defect, rng, scale)
    for o in objects:
        o["quad"] = [tuple(map(float, p)) for p in apply(hm, o["quad"])]
    for c in cards or []:
        c["corners"] = [tuple(map(float, p)) for p in apply(hm, c["corners"])]
    if mask is not None:  # objects lying on the page: words hidden by more than 25 % are marked
        for o in objects:
            q = np.array(o["quad"]) * scale
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, mask.shape[1]), min(y1, mask.shape[0])
            if x1 > x0 and y1 > y0 and mask[y0:y1, x0:x1].mean() > 0.25:
                o["hidden"] = True
    truth = dict(info, words=words, count=n, rotation=0, style=style, defect=defect, objects=objects, cards=cards,
                 font=os.path.basename(hand.font) if hand.font else "emnist")
    return img, truth



# ---------- training targets ----------

def visible(q, w, h):
    """True if the quadrilateral q overlaps the w x h image."""
    q = np.asarray(q)
    return not (q[:, 0].max() < 0 or q[:, 0].min() > w or q[:, 1].max() < 0 or q[:, 1].min() > h)



def core(q, f_h=0.4, f_w=0.25):
    """Reduced quadrilateral of a box: central band (f_h of the height), ends
    moved in by f_w of the height; and the height of the box."""
    q = np.asarray(q, float)
    a, b, c, d = q
    h = (np.linalg.norm(d - a) + np.linalg.norm(c - b)) / 2
    w = (np.linalg.norm(b - a) + np.linalg.norm(c - d)) / 2
    ux = ((b - a) + (c - d)) / 2
    ux /= max(1e-6, np.linalg.norm(ux))
    uy = ((d - a) + (c - b)) / 2
    uy /= max(1e-6, np.linalg.norm(uy))
    center = q.mean(0)
    hw = max(w / 2 - f_w * h, min(w / 2, 0.15 * h))
    hh = f_h * h / 2
    return [center - ux * hw - uy * hh, center + ux * hw - uy * hh, center + ux * hw + uy * hh,
            center - ux * hw + uy * hh], h



def targets(truth, scale_out):
    """Target maps at the output resolution (OUT_SIZE), uint8: 0 word core,
    1 number core, 2 light area of a card, 3-6 card corners (top-left,
    top-right, bottom-right, bottom-left; Gaussians), 7 box height (output
    pixels x 8), 8 ignored area (255: not scored)."""
    w, h = OUT_SIZE
    ims = [Image.new("L", (w, h), 0) for _ in range(9)]
    dr = [ImageDraw.Draw(i) for i in ims]
    for o in truth["objects"]:
        q = np.array(o["quad"]) * scale_out
        if not visible(q, w, h):
            continue
        k, height = core(q)
        pts = [tuple(p) for p in k]
        channel = 0 if o["kind"] == "word" else 1
        if o.get("hidden"):
            dr[8].polygon([tuple(p) for p in q], fill=255)
            continue
        dr[channel].polygon(pts, fill=255)
        c = np.mean(k, 0)
        dr[channel].point(tuple(c), fill=255)
        dr[7].polygon(pts, fill=int(min(255, height * 8)))
        dr[7].point(tuple(c), fill=int(min(255, height * 8)))
    for c in truth["cards"] or []:
        q = np.array(c["corners"]) * scale_out
        dr[2].polygon([tuple(p) for p in q], fill=255)
    y = np.stack([np.asarray(i) for i in ims], -1).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    for c in truth["cards"] or []:
        q = np.array(c["corners"]) * scale_out
        s = max(1.0, 0.012 * np.linalg.norm(q[2] - q[0]))
        for j in range(4):
            g = np.exp(-((xx - q[j, 0]) ** 2 + (yy - q[j, 1]) ** 2) / (2 * s * s))
            y[..., 3 + j] = np.maximum(y[..., 3 + j], (g * 255).astype(np.uint8))
    return y



def shard(out, count, seed, images, by_class, fonts, wordlist):
    """Training shard: photos drawn at half the input size of the full frame
    scale, then reduced by 2 to NET_SIZE."""
    rng = np.random.default_rng(seed)
    scale = 2 * NET_SIZE[0] / FRAME[0]
    x = np.zeros((count, NET_SIZE[1], NET_SIZE[0]), np.uint8)
    y = np.zeros((count, OUT_SIZE[1], OUT_SIZE[0], 9), np.uint8)
    meta = []
    for i in range(count):
        img, truth = generate(rng, images, by_class, fonts, wordlist, scale=scale)
        x[i] = np.asarray(img.reduce(2))
        y[i] = targets(truth, OUT_SIZE[0] / FRAME[0])
        meta.append(dict(style=META_LABELS[truth["style"]], defaut=META_LABELS[truth["defect"]],
                         type=META_LABELS[truth["layout"]], n=truth["count"]))
    np.savez_compressed(out, x=x, y=y, meta=json.dumps(meta))



def bench(out_dir, count, seed, images, by_class, fonts, wordlist, defect=None, style=None, layout=None):
    """Full-size JPEG photos and truth.json."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    truth = {}
    for i in range(count):
        img, t = generate(rng, images, by_class, fonts, wordlist, defect=defect, style=style, layout=layout)
        name = f"det{seed:03d}_{i:04d}.jpg"
        img.save(os.path.join(out_dir, name), quality=int(rng.integers(70, 95)))
        truth[name] = t
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth, f, ensure_ascii=False)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("train", "bench"))
    p.add_argument("out", help="shard .npz (train) or directory (bench)")
    p.add_argument("count", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--fonts", required=True, help="directory of the fonts of haller.fonts")
    p.add_argument("--emnist", required=True, help="EMNIST file of haller.emnist (test split with --test)")
    p.add_argument("--bip39", required=True)
    p.add_argument("--test", action="store_true", help="fonts of data/detector_test_fonts.sha256")
    p.add_argument("--defect", choices=DEFECTS, help="bench: framing defect of every photo")
    p.add_argument("--style", choices=STYLES, help="bench: writing style of every photo")
    p.add_argument("--layout", choices=("card12", "card24", "list1", "list2"), help="bench: page layout")
    a = p.parse_args()
    images, by_class = load_emnist(a.emnist)
    fonts = font_lists(a.fonts, a.test)
    wordlist = open(a.bip39).read().split()
    if a.mode == "train":
        shard(a.out, a.count, a.seed, images, by_class, fonts, wordlist)
    else:
        bench(a.out, a.count, a.seed, images, by_class, fonts, wordlist, a.defect, a.style, a.layout)



if __name__ == "__main__":
    main()
