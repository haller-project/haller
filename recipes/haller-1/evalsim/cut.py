"""
    Cutting of simulated seed photos into word strips (haller-1 simulated eval set).

    Each photo shows one BIP39 seed, written either on an official SeedSigner
    card (12 or 24 words) or as a free list on paper. The photo is classified
    (card or list), straightened, and cut into one strip per written line, in
    word order. Each strip then goes through preprocess.normalize_word and is
    padded into a 32 x 256 array.

    This is the code path of the reader that cut haller-1-eval-simulated.npz,
    reduced to what that cutting executes, with the same arithmetic: results
    are bit-identical to the published file. Thresholds that the original code
    computed through NumPy 1.x scalar promotion rules are written with explicit
    casts, so that NumPy 2 gives the same values.

    Usage:
        python cut.py PHOTO_DIR TRUTH_JSON PAGES_TXT OUT.npz [--processes N]

    TRUTH_JSON maps photo paths (relative to PHOTO_DIR) to their description,
    with the word list under "words" (or "mots"). PAGES_TXT lists the photos to
    cut, one name per line (file name without ".jpg"), in output order.
    Photos whose strip count differs from their word count are skipped.

    Dependencies: numpy (1.26 and 2.x give the same result), Pillow < 13.
    Pillow 13 changed Image.resize: the BILINEAR resize in
    preprocess.normalize_word then differs by 1 grey level on a few pixels of
    6 strips, and the content digest no longer matches.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess  # noqa: E402

CARD_DIR = Path(__file__).resolve().parents[3] / "data" / "cards"

STRIP_HEIGHT, STRIP_WIDTH = 32, 256
SEED_LENGTHS = (12, 15, 18, 21, 24)
LAYOUT_NAMES = {"card12": "card12", "card24": "card24", "list": "list"}

# Ink threshold: pixel < t x local mean, t = 1 - ADAPT (1 - contrast), clamped
ADAPT, THRESHOLD_MIN, THRESHOLD_MAX = 0.4, 0.72, 0.9
# Card location
CORNERS_TARGET = 384           # side of the reduced image used to find the card (pixels)
QUAD_THRESHOLDS = (0.85, 0.92, 1.0)
RULE_SCALE = 0.35              # scale of the card model when counting printed rules
CARD24_MIN = 0.5               # 24-word model tried only if the 12-word model finds this share
LIST_MAX_RULES = 2             # list if at most this many of 12 printed rules are found
CARD_BAND = (0.95, 0.3)        # strip above and below the printed rule, in line spacings
# Free list
LIST_REDUCTION = 4
LIST_MARGIN_H, LIST_MARGIN_V = 0.6, 0.35
LIST_MERGE, LIST_SPLIT = 0.25, 1.6
LINE_HYPOTHESES = (("peaks", 0.25), ("profile", None), ("peaks", 0.15), ("peaks", 0.4))
PAGE_ANGLE_MAX, PAGE_ANGLE_MIN = 12, 0.3


# ---------- image helpers ----------
def box_mean(a, r):
    """Mean over a (2r+1) x (2r+1) window, through Pillow's box blur on the
    image rounded to uint8."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.BoxBlur(r)), np.float32)


def otsu(v):
    """Otsu threshold of the values v (clipped to 0-255)."""
    hist = np.bincount(np.clip(v, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    tot = hist.sum(); cum = hist.cumsum(); mcum = (hist * np.arange(256)).cumsum()
    w0 = cum / tot; m0 = mcum / np.maximum(cum, 1); m1 = (mcum[-1] - mcum) / np.maximum(tot - cum, 1)
    return int(np.argmax(w0 * (1 - w0) * (m0 - m1) ** 2))


def remove_long_runs(m, frac=0.25):
    """Removes horizontal runs of ink longer than frac x the mask width (at
    least 40 pixels): printed rules, ruled lines, table edges."""
    lmin = max(40, int(frac * m.shape[1]))
    starts = m.copy()
    starts[:, 1:] &= ~m[:, :-1]
    ids = np.cumsum(starts.ravel()).reshape(m.shape) * m
    length = np.bincount(ids.ravel())
    length[0] = 0
    return m & ~(length >= lmin)[ids]


def binarize(a):
    """Ink = pixel < t x local mean, t adapted to the ink contrast of the
    strip; pixels isolated vertically or horizontally are dropped."""
    v = a[::2, ::2].ravel()
    k2, km = int(0.02 * (len(v) - 1)), (len(v) - 1) // 2
    q = np.partition(v, (k2, km))
    c = float(q[k2]) / max(float(q[km]), 1.0)
    t = min(max(1 - ADAPT * (1 - c), THRESHOLD_MIN), THRESHOLD_MAX)
    m = a < t * box_mean(a, 15)
    v = np.zeros_like(m); v[1:-1] = m[1:-1] & (m[:-2] | m[2:])
    h = np.zeros_like(m); h[:, 1:-1] = m[:, 1:-1] & (m[:, :-2] | m[:, 2:])
    return v & h


def erase_printed_lines(a):
    """Rows darker than their surroundings over at least 85 % of the width
    (printed or ruled lines) are replaced by the local background."""
    background = box_mean(a, 15)
    d = np.percentile(background - a, 15, axis=1)
    rows = d > 6
    rows = rows | np.roll(rows, 1) | np.roll(rows, -1)
    if rows.any():
        a = a.copy()
        a[rows] = np.maximum(a[rows], background[rows] * 1.02)
    return a


def clean_lines(m):
    """Removes long horizontal runs, then nearly full rows and columns."""
    m = remove_long_runs(m)
    rows = m.mean(1) > 0.5
    m[rows] = False
    cols = m.mean(0) > 0.6
    m[:, cols] = False
    return m


def segment(m):
    """Main text band of a line mask and its fragments: ([[x0, x1], ...], (y0, y1))."""
    prof = m.sum(1)
    active = prof > max(2, 0.08 * prof.max())
    if not active.any():
        return [], (0, 0)
    blocks, d = [], None
    for y, v in enumerate(np.append(active, False)):
        if v and d is None:
            d = y
        elif not v and d is not None:
            if blocks and d - blocks[-1][1] <= 2:
                blocks[-1][1] = y
            else:
                blocks.append([d, y])
            d = None
    y0, y1 = max(blocks, key=lambda b: prof[b[0]:b[1]].sum())
    while y0 > 0 and prof[y0 - 1] > 0:
        y0 -= 1
    while y1 < len(prof) and prof[y1] > 0:
        y1 += 1
    h = y1 - y0
    col = m[y0:y1].sum(0)
    inked = col > 0
    segs, x = [], 0
    while x < len(inked):
        if inked[x]:
            d = x
            while x < len(inked) and inked[x]:
                x += 1
            segs.append([d, x])
        x += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 2:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    merged = [s for s in merged if m[y0:y1, s[0]:s[1]].sum() > 0.01 * h * h]
    # Blocks that are too wide (touching letters) are split at profile minima
    width = np.median([s[1] - s[0] for s in merged]) if merged else h
    ref = max(width, 0.55 * h)
    out = []
    for d, f in merged:
        n = int(round((f - d) / ref))
        if (f - d) > 1.55 * ref and n >= 2:
            bounds = [d]
            for k in range(1, n):
                c = d + (f - d) * k // n
                window = slice(max(d + 2, c - int(0.25 * ref)), min(f - 2, c + int(0.25 * ref)))
                cs = col[window]
                bounds.append(window.start + int(cs.argmin()) if len(cs) else c)
            bounds.append(f)
            out += [[bounds[k], bounds[k + 1]] for k in range(n)]
        else:
            out.append([d, f])
    return out, (y0, y1)


def homography(src, dst):
    """Pillow PERSPECTIVE coefficients mapping output points dst to input points src."""
    A, B = [], []
    for (u, v), (x, y) in zip(dst, src):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); B.append(y)
    return np.linalg.solve(np.array(A, float), np.array(B, float))


# ---------- card location ----------
def fit_line(t, v):
    """Least-squares fit v = a t + b, outliers removed iteratively."""
    t = np.asarray(t, np.float64)
    v = np.asarray(v, np.float64)
    m = np.ones(len(t), bool)
    for _ in range(4):
        tm, vm = t[m], v[m]
        if len(tm) < 2:
            raise ValueError("fit_line: fewer than 2 points")
        mt, mv = tm.mean(), vm.mean()
        dt = tm - mt
        den = float(dt @ dt)
        a = float(dt @ (vm - mv)) / den if den > 0 else 0.0
        b = mv - a * mt
        r = np.abs(v - (a * t + b))
        m = r < max(2.0, 2.5 * np.median(r[m]))
    return a, b


def _propagate_runs(reached, m, axis):
    """Extends `reached` to every contiguous run of m (along the axis) that it touches."""
    if axis == 0:
        return _propagate_runs(reached.T, m.T, 1).T
    starts = m.copy()
    starts[:, 1:] &= ~m[:, :-1]
    ids = np.cumsum(starts.ravel()).reshape(m.shape) * m
    touched = np.bincount(ids.ravel(), weights=reached.ravel(), minlength=int(ids.max()) + 1) > 0
    touched[0] = False
    return touched[ids]


def inner_component(m):
    """Light area that is not 4-connected to the image border (inside of the
    card or sheet); the whole area if less than 2 % of the image remains."""
    reached = np.zeros_like(m)
    reached[0], reached[-1], reached[:, 0], reached[:, -1] = m[0], m[-1], m[:, 0], m[:, -1]
    n = -1
    while reached.sum() != n:
        n = reached.sum()
        reached = _propagate_runs(reached, m, 1)
        reached = _propagate_runs(reached, m, 0)
    inner = m & ~reached
    return inner if inner.sum() > 0.02 * m.size else m


def quad_from_mask(m, r):
    """Corners of the area m (lines fitted on its 4 edges), in pixels of the
    image reduced by r, scaled back by r; None if the area is too small."""
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    cols = np.arange(int(x0 + 0.15 * (x1 - x0)), int(x1 - 0.15 * (x1 - x0)))
    rows = np.arange(int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0)))
    if len(cols) < 5 or len(rows) < 5:
        return None
    mc, mr = m[:, cols], m[rows]
    kc, kr = mc.any(0), mr.any(1)
    top = np.argmax(mc, 0)[kc]
    bottom = (m.shape[0] - 1 - np.argmax(mc[::-1], 0))[kc]
    left = np.argmax(mr, 1)[kr]
    right = (m.shape[1] - 1 - np.argmax(mr[:, ::-1], 1))[kr]
    cols, rows = cols[kc], rows[kr]
    at, bt = fit_line(cols, top); ab, bb = fit_line(cols, bottom)
    al, bl = fit_line(rows, left); ar, br = fit_line(rows, right)

    def intersect(a1, b1, a2, b2):
        y = (a1 * b2 + b1) / (1 - a1 * a2)
        return (a2 * y + b2, y)

    pts = [intersect(at, bt, al, bl), intersect(at, bt, ar, br), intersect(ab, bb, ar, br), intersect(ab, bb, al, bl)]
    return [((x + 0.5) * r, (y + 0.5) * r) for x, y in pts]


def plausible(q, size, ratio):
    """Large enough quadrilateral whose width / height is close to ratio."""
    if q is None:
        return False
    p = np.array(q)
    side = lambda i, j: np.hypot(*(p[i] - p[j]))
    width = (side(0, 1) + side(3, 2)) / 2
    height = (side(0, 3) + side(1, 2)) / 2
    area = 0.5 * abs(sum(p[i, 0] * p[(i + 1) % 4, 1] - p[(i + 1) % 4, 0] * p[i, 1] for i in range(4)))
    if area < 0.08 * size[0] * size[1] or height <= 0:
        return False
    return ratio is None or abs(width / height / ratio - 1) < 0.15


def light_corners(img, ratio=None):
    """Corners of the light area: first on the image corrected for lighting,
    then with a global Otsu threshold; the first plausible one is returned."""
    r = max(1, -(-max(img.size) // CORNERS_TARGET))
    small = np.asarray(img.reduce(r), np.float32)
    background = box_mean(small, max(6, min(small.shape) // 10))
    norm = np.clip(small / np.maximum(background, 1) * 128, 0, 255)
    masks = [inner_component(norm > 0.92 * 128), inner_component(small > otsu(small))]
    for m in masks:
        q = quad_from_mask(m, r)
        if plausible(q, img.size, ratio):
            return q
    return quad_from_mask(masks[-1], r)


_CARDS = {}


def card_models():
    """Blank card models: image, printed rules (x0, x1, y), line spacing, and
    reference corners found by light_corners on the card laid on a dark background."""
    if not _CARDS:
        for name in ("card12", "card24"):
            img = Image.open(CARD_DIR / f"{name}.png").convert("L")
            ys = [367 + 76.6 * r for r in range(12)]
            if name == "card12":
                rules = [(180 if r < 9 else 221, 812, int(ys[r])) for r in range(12)]
            else:
                rules = ([(153 if r < 9 else 188, 474, int(ys[r])) for r in range(12)]
                         + [(621, 912, int(ys[r])) for r in range(12)])
            w, h = img.size
            canvas = Image.new("L", (w + 400, h + 400), 60)
            canvas.paste(img, (200, 200))
            ref = [(x - 200, y - 200) for x, y in light_corners(canvas)]
            _CARDS[name] = dict(img=img, rules=rules, spacing=76.6, ref=ref)
    return _CARDS


def card_ratio():
    pr = np.array(card_models()["card12"]["ref"])
    return (np.hypot(*(pr[0] - pr[1])) + np.hypot(*(pr[3] - pr[2]))) / (np.hypot(*(pr[0] - pr[3])) + np.hypot(*(pr[1] - pr[2])))


def candidate_quads(photo):
    """Plausible quadrilaterals for the light inside of a card: the one of
    light_corners, then other thresholds on the lighting-corrected image,
    alone or combined with the global Otsu threshold. Duplicates (corners
    within 2 % of the photo size of an earlier candidate) are skipped."""
    ratio = card_ratio()
    tol = 0.02 * max(photo.size)
    seen = []

    def new(q):
        if not plausible(q, photo.size, ratio):
            return False
        if any(max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(q, u)) < tol for u in seen):
            return False
        seen.append(q)
        return True

    try:
        q = light_corners(photo, ratio=ratio)
        if new(q):
            yield q
    except Exception:
        pass
    r = max(1, -(-max(photo.size) // CORNERS_TARGET))
    small = np.asarray(photo.reduce(r), np.float32)
    background = box_mean(small, max(6, min(small.shape) // 10))
    norm = np.clip(small / np.maximum(background, 1) * 128, 0, 255)
    light = small > otsu(small)
    for k in QUAD_THRESHOLDS:
        for m in (norm > k * 128, (norm > k * 128) & light):
            try:
                q = quad_from_mask(inner_component(m), r)
            except Exception:
                continue
            if new(q):
                yield q


def darkness(photo, q, ref, w, h, f, cache):
    """Local darkening (local mean - image) of the photo straightened onto a
    w x h model at scale f; cached per (quad, reference corners, scale)."""
    key = (tuple(map(tuple, q)), tuple(map(tuple, ref)), f)
    if key not in cache:
        coef = homography(q, [(x * f, y * f) for x, y in ref])
        a = np.asarray(photo.transform((int(w * f), int(h * f)), Image.PERSPECTIVE, coef, Image.BILINEAR), np.float32)
        cache[key] = box_mean(a, max(4, int(round(15 * f)))) - a
    return cache[key]


def p30_max(zone):
    """Maximum over the rows of the 30th percentile of each row (np.partition)."""
    if zone.size == 0:
        return 0.0
    k = int(0.3 * (zone.shape[1] - 1))
    return float(np.partition(zone, k, axis=1)[:, k].max())


def count_rules(photo, q, name, cache):
    """Printed rules found at the expected positions of the card model `name`
    after straightening: darkening > 6 on the right half of the rule, none
    beyond its end, and none in the gap between the two columns of a 24-word card."""
    card = card_models()[name]
    w, h = card["img"].size
    f = RULE_SCALE
    try:
        dark = darkness(photo, q, card["ref"], w, h, f, cache)
    except Exception:
        return 0
    e = card["spacing"] * f
    rules = [(x0 * f, x1 * f, yl * f) for x0, x1, yl in card["rules"]]
    W = w * f
    gap = ((max(l[1] for l in rules if l[0] < W / 2), min(l[0] for l in rules if l[0] >= W / 2))
           if any(l[0] >= W / 2 for l in rules) else None)
    n = 0
    for x0, x1, yl in rules:
        ya, yb = max(0, int(yl - 0.45 * e)), int(yl + 0.45 * e)
        found = p30_max(dark[ya:yb, int(x0 + 0.5 * (x1 - x0)):int(x1 - 0.05 * (x1 - x0))]) > 6
        if found:
            found = not p30_max(dark[ya:yb, int(x1 + 0.04 * W):int(min(W - 0.03 * W, x1 + 0.12 * W))]) > 6
        if found and gap:
            found = not p30_max(dark[ya:yb, int(gap[0] + 0.2 * (gap[1] - gap[0])):int(gap[1] - 0.2 * (gap[1] - gap[0]))]) > 6
        n += found
    return n


def best_quad(photo):
    """(quad, share of printed rules found, card model). The 24-word model is
    chosen when it finds at least 75 % of its rules; the search stops at the
    first candidate with all rules found, otherwise the best one is kept."""
    cache = {}
    n12 = len(card_models()["card12"]["rules"])
    n24 = len(card_models()["card24"]["rules"])
    best = (None, 0.0, "card12")
    for q in candidate_quads(photo):
        f = {"card12": count_rules(photo, q, "card12", cache) / n12}
        f["card24"] = count_rules(photo, q, "card24", cache) / n24 if f["card12"] >= CARD24_MIN else 0.0
        name = "card24" if f["card24"] >= 0.75 else "card12"
        if f[name] > best[1]:
            best = (q, f[name], name)
        if f[name] >= 1.0:
            return best
    return best


def place_rule(dark, x0, x1, yl, e):
    """Actual row of a printed rule, searched within +-0.45 spacing of the
    model position (the darkest row by 30th percentile over the right half of
    the rule); the model position if no clear rule is found."""
    ya, yb = int(yl - 0.45 * e), int(yl + 0.45 * e)
    xa, xb = int(x0 + 0.5 * (x1 - x0)), int(x1 - 0.05 * (x1 - x0))
    zone = dark[max(0, ya):yb, xa:xb]
    if zone.size == 0:
        return yl
    v = np.percentile(zone, 30, axis=1)
    k = int(np.argmax(v))
    return max(0, ya) + k if v[k] > 6 else yl


def erase_rule(a, dark, background, xa, xb, yl, r=7):
    """Erases a printed rule column by column: the darkest row within +-r of
    yl and its two neighbours on each side are set to the local background."""
    ya, yb = max(0, yl - r), min(a.shape[0], yl + r + 1)
    xa, xb = max(0, xa), min(a.shape[1], xb)
    zone = dark[ya:yb, xa:xb]
    if zone.size == 0:
        return
    ok = zone.max(0) > 6
    xs = np.arange(xa, xb)[ok]
    ys = ya + np.argmax(zone, 0)[ok]
    for d in range(-2, 3):
        yy = np.clip(ys + d, 0, a.shape[0] - 1)
        a[yy, xs] = background[yy, xs]


def card_strips(photo, q, name):
    """Card straightened onto the model, printed rules erased, one strip per rule."""
    card = card_models()[name]
    w, h = card["img"].size
    page = photo.transform((w, h), Image.PERSPECTIVE, homography(q, card["ref"]), Image.BILINEAR)
    ec = card["spacing"]
    a = np.asarray(page, np.float32)
    dark = box_mean(a, 15) - a
    background = box_mean(a, 15)
    a = a.copy()
    rows = []
    for x0, x1, yl in card["rules"]:
        yl = place_rule(dark, x0, x1, yl, ec)
        rows.append(yl)
        erase_rule(a, dark, background, x0 - 4, x1 + 6, yl)
    page = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    top, bottom = CARD_BAND
    return [page.crop((x0 - 4, int(yl - top * ec), x1 + 6, int(yl + bottom * ec)))
            for (x0, x1, _), yl in zip(card["rules"], rows)]


# ---------- free list ----------
def ink_mask(photo, R):
    """Ink mask of the sheet on the photo reduced by R, or None if no paper is found."""
    a = np.asarray(photo.reduce(R), np.float32)
    background = box_mean(a, 12)
    t = otsu(background.ravel())
    low, high = background[background <= t], background[background > t]
    if len(low) and len(high) and low.mean() > 0.5 * high.mean():
        paper = np.ones_like(background, bool)  # no dark surroundings: the sheet fills the frame
    else:
        paper = background > t
    paper[:12], paper[-12:], paper[:, :12], paper[:, -12:] = False, False, False, False
    # 25 x 25 erosion through a box blur and a threshold
    paper = box_mean(paper.astype(np.float32) * 255, 12) > 254
    if not paper.any():
        return None
    # Ink contrast measured on the light paper only
    light = paper & (background > np.float32(0.7 * float(np.median(background[paper]))))
    inside = a[light] if light.any() else a[paper]
    c = float(np.percentile(inside, 1) / max(np.median(inside), 1))
    ink = (a < min(max(1 - ADAPT * (1 - c), THRESHOLD_MIN), THRESHOLD_MAX) * background) & paper
    paper_width = max(int(paper.any(0).sum()), 1)
    return remove_long_runs(ink, frac=0.3 * paper_width / ink.shape[1])


def page_angle(ink):
    """Page tilt in degrees: shear that maximises the sum of squares of the
    horizontal ink projection; 0.5 degree grid, then refined to 0.1 degree."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0
    xs = xs - xs.mean()
    H = ink.shape[0]

    def score(deg):
        y = np.round(ys - xs * np.tan(np.radians(deg))).astype(int)
        y = y[(y >= 0) & (y < H)]
        p = np.bincount(y, minlength=H).astype(np.float64)
        return float((p * p).sum())

    grid = np.arange(-PAGE_ANGLE_MAX, PAGE_ANGLE_MAX + 0.01, 0.5)
    best = max(grid, key=score)
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    return float(max(fine, key=score))


def straighten_page(photo, R):
    """(page, angle, ink mask of the page). Below PAGE_ANGLE_MIN the photo is
    kept as is; otherwise the photo and its ink mask are rotated."""
    ink = ink_mask(photo, R)
    if ink is None:
        return photo, 0.0, None
    ang = page_angle(ink)
    if abs(ang) < PAGE_ANGLE_MIN:
        return photo, ang, ink
    fill = int(np.median(np.asarray(photo.reduce(8))))
    page = photo.rotate(ang, resample=Image.BILINEAR, fillcolor=fill)
    ink = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).rotate(ang, resample=Image.NEAREST, fillcolor=0)) > 127
    return page, ang, ink


def rotation_matrix(ang, size):
    """Affine coefficients (a, b, c, d, e, f) of Image.rotate(ang) without
    expansion: pixel (x, y) of the rotated image comes from
    (a x + b y + c, d x + e y + f) of the original image."""
    w, h = size
    t = -math.radians(ang % 360.0)
    a, b, d, e = round(math.cos(t), 15), round(math.sin(t), 15), round(-math.sin(t), 15), round(math.cos(t), 15)
    cx, cy = w / 2.0, h / 2.0
    c = a * -cx + b * -cy + cx
    f = d * -cx + e * -cy + cy
    return a, b, c, d, e, f


class BoxCutter:
    """Cuts boxes found on the rotated page of a photo reduced by an integer
    factor f directly from the full-resolution photo: each strip is extracted
    by an affine transform limited to the strip (same rotation as the reduced
    page, coordinates multiplied by f)."""

    def __init__(self, photo, ang, f, reduced_size):
        self.photo, self.f = photo, f
        self.m = rotation_matrix(ang, reduced_size) if ang else None
        self.fill = int(np.median(np.asarray(photo.reduce(16)))) if ang else 255

    def __call__(self, box):
        f = self.f
        X0, Y0, X1, Y1 = (int(round(v * f)) for v in box)
        if self.m is None:
            return self.photo.crop((X0, Y0, X1, Y1))
        a, b, c, d, e, g = self.m
        coef = (a, b, a * X0 + b * Y0 + f * c, d, e, d * X0 + e * Y0 + f * g)
        return self.photo.transform((X1 - X0, Y1 - Y0), Image.AFFINE, coef, Image.BILINEAR, fillcolor=self.fill)


def lines_from_peaks(ink, peak_min):
    """Writing lines from the peaks of the horizontal projection: line pitch
    from the autocorrelation, peaks at least 0.55 pitch apart and at least
    peak_min x the median of the strongest; each line spans the minima around
    it. Returns [(y0, y1, x0, x1)] in mask coordinates."""
    prof = ink.sum(1).astype(np.float64)
    if prof.sum() == 0:
        return []
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    p = np.convolve(prof, k / k.sum(), mode="same")
    c = p - p.mean()
    ac = np.correlate(c, c, mode="full")[len(c) - 1:]
    lo, hi = 6, min(len(ac) - 1, 200)
    if hi <= lo:
        return []
    cand = [t for t in range(lo + 1, hi) if ac[t] >= ac[t - 1] and ac[t] >= ac[t + 1] and ac[t] > 0.1 * ac[0]]
    if not cand:
        return []
    P = cand[0]
    sig = max(1.0, P / 6)
    k = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    q = np.convolve(prof, k / k.sum(), mode="same")
    peaks = [y for y in range(1, len(q) - 1) if q[y] >= q[y - 1] and q[y] > q[y + 1]]
    peaks.sort(key=lambda y: -q[y])
    kept = []
    for y in peaks:
        if all(abs(y - g) >= 0.55 * P for g in kept):
            kept.append(y)
    if not kept:
        return []
    ref = np.median(sorted((q[y] for y in kept), reverse=True)[:max(3, len(kept) // 2)])
    kept = sorted(y for y in kept if q[y] >= peak_min * ref)
    out = []
    for i, yc in enumerate(kept):
        y0 = int(np.argmin(q[kept[i - 1]:yc]) + kept[i - 1]) if i else max(0, int(yc - 0.6 * P))
        y1 = int(np.argmin(q[yc:kept[i + 1]]) + yc) if i + 1 < len(kept) else min(len(q), int(yc + 0.6 * P))
        y0, y1 = max(y0, int(yc - 0.75 * P)), min(y1, int(yc + 0.75 * P))
        band = ink[y0:y1]
        colv = band.sum(0)
        cols = np.nonzero(colv > 0)[0]
        if len(cols) == 0:
            continue
        h = max(1, y1 - y0)
        groups = [[cols[0], cols[0] + 1]]
        for x in cols[1:]:
            if x - groups[-1][1] < 0.8 * h:
                groups[-1][1] = x + 1
            else:
                groups.append([x, x + 1])
        x0, x1 = max(groups, key=lambda g: colv[g[0]:g[1]].sum())
        rows = np.nonzero(band[:, x0:x1].sum(1) > 0)[0]
        if len(rows):
            y0, y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        out.append((y0, y1, int(x0), int(x1)))
    return out


def lines_from_profile(ink):
    """Writing lines from a threshold on the horizontal projection, with
    merging of close bands and splitting of tall ones: [(y0, y1, x0, x1)]."""
    prof = ink.sum(1)
    pos = prof[prof > 0]
    ref = np.percentile(pos, 90) if len(pos) else 0
    active = prof > max(2, 0.15 * ref)
    bands, d = [], None
    for y, v in enumerate(np.append(active, False)):
        if v and d is None:
            d = y
        elif not v and d is not None:
            bands.append([d, y])
            d = None
    if not bands:
        return []
    hmed = np.median([b1 - b0 for b0, b1 in bands])
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= LIST_MERGE * hmed:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    hmed = np.median([b1 - b0 for b0, b1 in merged])
    split, stack = [], list(merged)
    while stack:
        y0, y1 = stack.pop(0)
        if y1 - y0 > LIST_SPLIT * hmed:
            yc = y0 + int(0.3 * hmed) + int(np.argmin(prof[y0 + int(0.3 * hmed):y1 - int(0.3 * hmed)]))
            stack[:0] = [[y0, yc], [yc + 1, y1]]
        else:
            split.append([y0, y1])
    hmed = np.median([b1 - b0 for b0, b1 in split])
    out = []
    for y0, y1 in split:
        if y1 - y0 < 0.4 * hmed or ink[y0:y1].sum() <= 0.5 * hmed * hmed:
            continue
        h = y1 - y0
        colv = ink[y0:y1].sum(0)
        cols = np.nonzero(colv > 0)[0]
        groups = [[cols[0], cols[0] + 1]]
        for x in cols[1:]:
            if x - groups[-1][1] < h:
                groups[-1][1] = x + 1
            else:
                groups.append([x, x + 1])
        x0, x1 = max(groups, key=lambda g: colv[g[0]:g[1]].sum())
        out.append((y0, y1, int(x0), int(x1)))
    return out


def text_columns(ink):
    """Text columns [(xa, xb)]: an empty vertical corridor at least 1.5 line
    heights wide, with at least 25 % of the ink on each side, separates two
    columns. A column touching the frame edge is dropped if another remains."""
    lines = lines_from_profile(ink)
    if not lines:
        return []
    hmed = float(np.median([y1 - y0 for y0, y1, _, _ in lines]))
    colv = ink.sum(0).astype(float)
    xs = np.nonzero(colv > 0)[0]
    cols = [(int(xs[0]), int(xs[-1]) + 1)]
    tot = colv.sum()
    empty = colv <= 0.002 * colv.max()
    best = None
    x = int(xs[0])
    while x < xs[-1]:
        if empty[x]:
            d = x
            while x < xs[-1] and empty[x]:
                x += 1
            g, dr = colv[:d].sum(), colv[x:].sum()
            if x - d >= 1.5 * hmed and min(g, dr) >= 0.25 * tot and (best is None or x - d > best[1] - best[0]):
                best = (d, x)
        x += 1
    if best:
        cols = [(int(xs[0]), best[0]), (best[1], int(xs[-1]) + 1)]
    W = ink.shape[1]
    kept = [c for c in cols if c[0] > 0.015 * W + 12 and c[1] < W - 0.015 * W - 12]
    return kept or cols


def line_boxes(page, ink, method, R):
    """Boxes (page pixels) of the writing lines, per column."""
    out = []
    for xa, xb in text_columns(ink):
        sub = np.zeros_like(ink)
        sub[:, xa:xb] = ink[:, xa:xb]
        boxes = []
        for y0, y1, x0, x1 in method(sub):
            h = y1 - y0
            boxes.append((max(0, int((x0 - LIST_MARGIN_H * h) * R)), max(0, int((y0 - LIST_MARGIN_V * h) * R)),
                          min(page.width, int((x1 + LIST_MARGIN_H * h) * R)), min(page.height, int((y1 + LIST_MARGIN_V * h) * R))))
        if boxes:
            out.append(boxes)
    return out


def detect_lines(page, ink, R):
    """Line boxes per column, left to right, each column top to bottom. The
    hypotheses of LINE_HYPOTHESES are tried in order; the first whose line
    count is a BIP39 length is kept, otherwise the first one."""
    first = None
    for method, peak_min in LINE_HYPOTHESES:
        if method == "peaks":
            cols = line_boxes(page, ink, lambda m: lines_from_peaks(m, peak_min), R)
        else:
            cols = line_boxes(page, ink, lines_from_profile, R)
        if first is None:
            first = cols
        if sum(len(c) for c in cols) in SEED_LENGTHS:
            return cols
    return first


def word_starts(page, boxes):
    """Numbered list: per line, the widest gap among the first 4 (>= 0.25 h,
    >= 1.5 x the next 3, within 2.2 h of the start of the ink) separates the
    number from the word. The list counts as numbered if at least 2/3 of the
    lines have one and the word starts are aligned (< 0.6 h from the median).
    Returns per line the abscissa where to cut, or None."""
    cand = []
    for box in boxes:
        a = erase_printed_lines(np.asarray(page.crop(box), np.float32))
        m = clean_lines(binarize(a))
        segs, (y0, y1) = segment(m)
        h = max(y1 - y0, 1)
        gaps = [segs[j + 1][0] - segs[j][1] for j in range(len(segs) - 1)]
        best = None
        for j in range(min(4, len(gaps))):
            if segs[j][1] - segs[0][0] > 2.2 * h:
                break
            after = gaps[j + 1:j + 4] or [0]
            if gaps[j] >= 0.25 * h and gaps[j] >= 1.5 * max(after) and (best is None or gaps[j] > best[1]):
                best = (box[0] + segs[j + 1][0], gaps[j], h)
        cand.append(best)
    found = [c for c in cand if c]
    if len(found) < 2 * len(boxes) / 3:
        return [None] * len(boxes)
    xm = float(np.median([c[0] for c in found]))
    hm = float(np.median([c[2] for c in found]))
    if len([c for c in found if abs(c[0] - xm) < 0.6 * hm]) < 2 * len(boxes) / 3:
        return [None] * len(boxes)
    return [int((c[0] if c and abs(c[0] - xm) < 0.6 * hm else xm) - 0.15 * hm) for c in cand]


def list_strips(photo, half):
    """Strips of a free list. Rotation, lines and numbers are found on the
    half-resolution photo; strips are cut from the full-resolution photo."""
    R = max(1, int(round(LIST_REDUCTION * (half.width / photo.width))))
    page, ang, ink = straighten_page(half, R)
    if ink is None:
        raise ValueError("no paper found")
    cols = detect_lines(page, ink, R)
    n = sum(len(c) for c in cols)
    if n not in SEED_LENGTHS:
        raise ValueError(f"{n} lines found, need 12-24")
    boxes = []
    for c in cols:  # numbers removed column by column
        cuts = word_starts(page, c)
        boxes += [(x, b[1], b[2], b[3]) if x is not None and x > b[0] else b for b, x in zip(c, cuts)]
    f = int(round(photo.width / half.width))
    cutter = BoxCutter(photo, ang if abs(ang) >= PAGE_ANGLE_MIN else 0.0, f, half.size)
    return [cutter(b) for b in boxes]


# ---------- photo ----------
def open_photo(path, rotation=0, half=False):
    """Grayscale photo, JPEG decoded directly to luminance (at half resolution if half)."""
    im = Image.open(path)
    im.draft("L", (im.width // 2, im.height // 2) if half else im.size)
    return im.convert("L").rotate(rotation, expand=True)


def cut_photo(path, rotation=0):
    """(layout, strips) of one photo: layout in {"card12", "card24", "list"},
    strips as grayscale Pillow images in word order."""
    photo = open_photo(path, rotation)
    half = open_photo(path, rotation, half=True)
    q, share, name = best_quad(half)
    if share > LIST_MAX_RULES / 12:
        f = photo.width / half.width
        return name, card_strips(photo, [(x * f, y * f) for x, y in q], name)
    return "list", list_strips(photo, half)


def strip_array(strip):
    """Strip -> (32 x 256 uint8 array, width), via preprocess.normalize_word;
    width 0 if no ink is found."""
    out = np.zeros((STRIP_HEIGHT, STRIP_WIDTH), np.uint8)
    x = preprocess.normalize_word(strip)
    if x is None:
        return out, 0
    out[:, :x.shape[1]] = x
    return out, x.shape[1]


# ---------- command line ----------
def _job(args):
    page, path, words, rotation = args
    try:
        layout, strips = cut_photo(path, rotation)
    except Exception as e:
        return page, None, f"{type(e).__name__}: {e}"
    if len(strips) != len(words):
        return page, None, f"{len(strips)} strips for {len(words)} words"
    return page, (layout, [strip_array(s) for s in strips]), None


def _find_photos(photo_dir, truth):
    """page name -> (photo path, words, rotation), from the truth file."""
    by_name = {}
    for key, d in truth.items():
        name = os.path.basename(key)
        path = photo_dir / key
        if not path.exists():
            path = photo_dir / name
        words = d.get("words", d.get("mots"))
        by_name[os.path.splitext(name)[0]] = (path, words, d.get("rotation", 0))
    return by_name


def main():
    ap = argparse.ArgumentParser(description="Cut simulated seed photos into word strips.")
    ap.add_argument("photo_dir", type=Path)
    ap.add_argument("truth_json", type=Path)
    ap.add_argument("pages_txt", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--processes", type=int, default=1)
    args = ap.parse_args()
    by_name = _find_photos(args.photo_dir, json.load(open(args.truth_json)))
    pages = [line.strip() for line in open(args.pages_txt) if line.strip()]
    jobs = [(p, str(by_name[p][0]), by_name[p][1], by_name[p][2]) for p in pages]
    if args.processes > 1:
        from multiprocessing import Pool
        with Pool(args.processes) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]
    X, W, Y, PAGE, LAYOUT = [], [], [], [], []
    skipped = 0
    for (page, path, words, _), (_, res, err) in zip(jobs, results):
        if res is None:
            skipped += 1
            print(f"{page}: skipped ({err})", file=sys.stderr)
            continue
        layout, strips = res
        for (x, w), word in zip(strips, words):
            X.append(x); W.append(w); Y.append(word); PAGE.append(page); LAYOUT.append(LAYOUT_NAMES[layout])
    np.savez_compressed(args.out_npz, x=np.stack(X), w=np.array(W, np.int16), y=np.array(Y, "<U8"),
                        reel=np.zeros(len(Y), bool), page=np.array(PAGE), layout=np.array(LAYOUT))
    print(f"{len(Y)} strips from {len(pages) - skipped} photos ({skipped} skipped)")


if __name__ == "__main__":
    main()
