"""
Synthetic handwritten word images.

Each sample is one word written in a single, consistent hand, with the
irregularities of real handwriting, placed on a line of paper with its context
(neighboring lines, ruled or printed lines, a line number), photographed under
varied conditions, then normalized exactly as SeedSigner does before recognition.

Hands:
- a handwriting font (287 OFL/Apache fonts), rendered either as a whole word
  (keeps the joins of cursive fonts) or letter by letter with per-letter jitter;
- EMNIST letters (NIST handwriting), with word-level constants (size, slant,
  stroke width, spacing) and per-letter jitter.
Both are then warped by a smooth random displacement field.
"""
import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from seedsigner.helpers.handwriting.preprocess import locate_word

EMNIST_CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"



# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
@dataclass
class Vocabulary:
    """English BIP39 words, and other words so that the model learns letters
    rather than a closed set of word shapes (the BIP39 lists of the other
    Latin-script languages, accents removed)."""
    bip39: list
    other: list
    bip39_share: float = 0.6

    @classmethod
    def load(cls, bip39_path: str, other_path: str):
        bip39 = [w.strip() for w in open(bip39_path) if w.strip()]
        other = [w.strip() for w in open(other_path) if w.strip()]
        return cls(bip39, other)


    def word(self, rng: np.random.Generator) -> str:
        words = self.bip39 if rng.random() < self.bip39_share else self.other
        return words[rng.integers(len(words))]



def apply_case(word: str, rng: np.random.Generator, lower_share: float = 0.35) -> str:
    """Capitals, lowercase (lower_share) or capitalized (0.15)."""
    r = rng.random()
    if r < 0.85 - lower_share:
        return word.upper()
    if r < 0.85:
        return word.lower()
    return word.capitalize()



def line_prefix(rng: np.random.Generator) -> str:
    """Optional line number as people write it: "7.", "12)", "3 ", "#4"."""
    if rng.random() > 0.25:
        return ""
    n = int(rng.integers(1, 25))
    return rng.choice([f"{n}. ", f"{n}) ", f"{n} ", f"{n}.", f"{n}-"])



# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------
def warp(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """
    Smooth random displacement (elastic distortion): the same text never looks
    twice the same. `strength` is the displacement in pixels.
    """
    h, w = img.shape
    if strength <= 0 or h < 8 or w < 8:
        return img
    grid = (max(2, h // 12), max(2, w // 12))
    dx = np.asarray(Image.fromarray(rng.normal(0, 1, grid).astype(np.float32)).resize((w, h), Image.BICUBIC))
    dy = np.asarray(Image.fromarray(rng.normal(0, 1, grid).astype(np.float32)).resize((w, h), Image.BICUBIC))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.clip(xx + strength * dx, 0, w - 1)
    y = np.clip(yy + strength * dy, 0, h - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    a = img.astype(np.float32)
    out = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy)
           + a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)
    return out.astype(np.uint8)



def stroke(img: Image.Image, width: int) -> Image.Image:
    """Thicker (width > 0) or thinner (width < 0) strokes; ink is dark."""
    if width > 0:
        return img.filter(ImageFilter.MinFilter(2 * width + 1))
    if width < 0:
        return img.filter(ImageFilter.MaxFilter(3))
    return img



def stroke_for(size: float, rng: np.random.Generator) -> int:
    """Stroke width change in pixels, in proportion to the size of the writing."""
    return int(round(rng.choice([-0.5, 0, 0, 0, 0.5, 1]) * size / 40))



def shear(img: Image.Image, degrees: float) -> Image.Image:
    """Slant the writing (positive: leaning right)."""
    s = np.tan(np.radians(degrees))
    margin = int(abs(s) * img.height) + 2
    wide = Image.new("L", (img.width + 2 * margin, img.height), 255)
    wide.paste(img, (margin, 0))
    return wide.transform(wide.size, Image.AFFINE, (1, s, -s * img.height, 0, 1, 0), Image.BILINEAR, fillcolor=255)



@dataclass
class FontHand:
    path: str
    size: int            # pixels
    slant: float         # degrees
    stroke_width: int
    per_letter: bool     # letter-by-letter rendering with jitter
    jitter: float        # 0..1, amount of per-letter irregularity
    tracking: float      # extra spacing between letters, in em

    _cache = {}

    @classmethod
    def random(cls, fonts: list, size: float, rng: np.random.Generator, per_letter_share: float = 0.5):
        return cls(
            path=fonts[rng.integers(len(fonts))],
            size=int(size),
            slant=float(rng.normal(0, 6)),
            stroke_width=stroke_for(size, rng),
            per_letter=bool(rng.random() < per_letter_share),
            jitter=float(rng.uniform(0.2, 1.0)),
            tracking=float(rng.uniform(-0.02, 0.15)),
        )


    def font(self):
        key = (self.path, self.size)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = ImageFont.truetype(self.path, self.size)
        return self._cache[key]


    def render(self, text: str, rng: np.random.Generator) -> Image.Image:
        font = self.font()
        s = self.size
        img = Image.new("L", (int(s * (1.2 * len(text) + 2)), int(2.2 * s)), 255)
        draw = ImageDraw.Draw(img)
        if not self.per_letter:
            draw.text((s // 2, s // 3), text, font=font, fill=0)
        else:
            x = s / 2
            for ch in text:
                if ch == " ":
                    x += 0.35 * s * rng.uniform(0.8, 1.3)
                    continue
                j = self.jitter
                size = max(6, int(s * (1 + j * rng.normal(0, 0.06))))
                letter_font = ImageFont.truetype(self.path, size) if size != s else font
                glyph = Image.new("L", (3 * s, 3 * s), 255)
                ImageDraw.Draw(glyph).text((s, s), ch, font=letter_font, fill=0)
                glyph = glyph.rotate(j * rng.normal(0, 3), resample=Image.BILINEAR, fillcolor=255)
                dy = j * rng.normal(0, 0.05) * s
                img.paste(Image.new("L", glyph.size, 0), (int(x - s), int(s // 3 - s + dy)), Image.eval(glyph, lambda v: 255 - v))
                x += letter_font.getlength(ch) + s * (self.tracking + j * rng.normal(0, 0.04))
        img = stroke(shear(img, self.slant), self.stroke_width)
        return img



class EmnistLetters:
    """EMNIST byclass glyphs indexed by character."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.images = d["x"]
        labels = d["y"]
        self.index = {c: np.nonzero(labels == i)[0] for i, c in enumerate(EMNIST_CLASSES)}


    def glyph(self, ch: str, rng: np.random.Generator) -> np.ndarray | None:
        choices = self.index.get(ch)
        if choices is None or len(choices) == 0:
            return None
        return 255 - self.images[choices[rng.integers(len(choices))]]



@dataclass
class EmnistHand:
    letters: EmnistLetters
    size: int
    slant: float
    stroke_width: int
    spacing: float       # in units of the letter height
    jitter: float

    @classmethod
    def random(cls, letters: EmnistLetters, size: float, rng: np.random.Generator):
        return cls(letters, int(size), float(rng.normal(0, 6)), stroke_for(size, rng),
                   float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.2, 1.0)))


    def render(self, text: str, rng: np.random.Generator) -> Image.Image:
        s = self.size
        img = Image.new("L", (int(s * (1.3 * len(text) + 2)), int(2.2 * s)), 255)
        base = int(1.6 * s)
        x = s // 2
        for ch in text:
            if ch == " ":
                x += int(0.5 * s)
                continue
            g = self.letters.glyph(ch, rng)
            if g is None:  # punctuation of the line number
                g = self._punctuation(ch)
            box = Image.fromarray(255 - g).getbbox()
            if box is None:
                continue
            glyph = Image.fromarray(g).crop(box)
            low = ch.islower() and ch not in "bdfhklt"
            h = s * {".": 0.15, "-": 0.1, ")": 0.9, "#": 0.8}.get(ch, 0.62 if low else 1.0)
            h *= 1 + self.jitter * rng.normal(0, 0.06)
            if ch in "gjpqy":
                h *= 1.4
            glyph = glyph.resize((max(2, int(glyph.width * h / glyph.height)), max(2, int(h))), Image.BILINEAR)
            glyph = glyph.rotate(self.jitter * rng.normal(0, 3), resample=Image.BILINEAR, expand=True, fillcolor=255)
            top = base - glyph.height + int(self.jitter * rng.normal(0, 0.04) * s)
            if ch in "gjpqy":
                top += int(0.4 * s * 0.62)
            img.paste(Image.new("L", glyph.size, 0), (x, max(0, top)), Image.eval(glyph, lambda v: 255 - v))
            x += glyph.width + int(s * (self.spacing + self.jitter * rng.normal(0, 0.04)))
        return stroke(shear(img, self.slant), self.stroke_width)


    @staticmethod
    def _punctuation(ch: str) -> np.ndarray:
        g = Image.new("L", (28, 28), 255)
        d = ImageDraw.Draw(g)
        if ch == ".":
            d.ellipse([11, 20, 16, 25], fill=0)
        elif ch == ")":
            d.arc([2, 2, 18, 26], -70, 70, fill=0, width=3)
        elif ch == "-":
            d.line([6, 14, 20, 14], fill=0, width=3)
        elif ch == "#":
            d.text((6, 4), "#", fill=0)
        return np.asarray(g)



def crop_ink(img: Image.Image, pad: int) -> Image.Image:
    box = Image.eval(img, lambda v: 255 - v).getbbox()
    if box is None:
        return img
    # bounded by the image: Pillow fills the area outside it with black
    return img.crop((max(0, box[0] - pad), max(0, box[1] - pad),
                     min(img.width, box[2] + pad), min(img.height, box[3] + pad)))



# ---------------------------------------------------------------------------
# Line on paper, camera
# ---------------------------------------------------------------------------
def compose_line(target: Image.Image, neighbors: list, size: int, rng: np.random.Generator) -> tuple[Image.Image, tuple]:
    """
    The target word on its line, possibly with its neighbors above and below
    (same hand), cropped with random margins that may cut into the neighbors.
    Returns the line and the rows (top, bottom) of the target in it.
    """
    spacing = int(size * rng.uniform(1.4, 2.4))
    width = max([target.width] + [n.width for n in neighbors]) + 2 * size
    height = target.height + 2 * spacing
    page = Image.new("L", (width, height), 255)
    y_target = spacing
    page.paste(target, (int(rng.integers(0, size)), y_target))
    if neighbors:
        page.paste(neighbors[0], (int(rng.integers(0, size)), y_target - spacing))
        page.paste(neighbors[1], (int(rng.integers(0, size)), y_target + spacing))
    top = max(0, int(y_target - size * rng.uniform(0.1, 0.9)))
    bottom = min(height, int(y_target + target.height + size * rng.uniform(0.0, 0.7)))
    return page.crop((0, top, width, bottom)), (y_target - top, y_target - top + target.height)



def add_paper_lines(a: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Ruled paper or the printed line of a seed card, under and between words."""
    kind = rng.choice(["none", "none", "ruled", "printed"])
    if kind == "none":
        return a
    h, w = a.shape
    darkness = rng.uniform(0.55, 0.9) if kind == "ruled" else rng.uniform(0.3, 0.7)
    tilt = rng.normal(0, 0.01)
    gap = size * rng.uniform(1.4, 2.2)
    y = h * rng.uniform(0.55, 0.95)
    xs = np.arange(w)
    rows = [y] if kind == "printed" else [y - gap, y, y + gap]
    for r in rows:
        ys = np.round(r + tilt * (xs - w / 2)).astype(int)
        thick = int(rng.integers(1, 3))
        for t in range(thick):
            ok = (ys + t >= 0) & (ys + t < h)
            a[ys[ok] + t, xs[ok]] = np.minimum(a[ys[ok] + t, xs[ok]], 255 * darkness)
    return a



def photograph(line: Image.Image, size: int, rng: np.random.Generator) -> tuple[Image.Image, float]:
    """
    Paper and ink colors, uneven lighting, paper lines, scale, blur (from sharp
    to strongly out of focus, or motion), sensor noise and JPEG compression.
    Returns the image and its scale.
    """
    low = max(0.35, 22 / size)  # letters at least ~15 px high
    scale = rng.uniform(low, max(low, 1.3))
    line = line.resize((max(8, int(line.width * scale)), max(8, int(line.height * scale))), Image.BILINEAR)
    size = max(4, int(size * scale))
    ink_map = np.asarray(line, np.float32) / 255
    h, w = ink_map.shape

    paper = rng.uniform(100, 245)
    ink = paper * (rng.uniform(0.05, 0.35) if rng.random() < 0.8 else rng.uniform(0.35, 0.75))  # pen or faint pencil
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1 + rng.uniform(-0.25, 0.25) * (xx / w - 0.5) + rng.uniform(-0.2, 0.2) * (yy / h - 0.5)
    if rng.random() < 0.2:  # shadow edge
        edge = rng.uniform(0, w)
        light *= np.where(xx < edge, rng.uniform(0.6, 0.9), 1.0)
    a = (ink + (paper - ink) * ink_map) * light
    a = add_paper_lines(a, size, rng)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    # Blur in proportion to the size of the writing: from sharp to strongly out
    # of focus, but still legible to a human
    blur = rng.choice(["sharp", "mild", "strong", "motion"], p=[0.3, 0.45, 0.1, 0.15])
    if blur == "mild":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.01, 0.05) * size))
    elif blur == "strong":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.09) * size))
    elif blur == "motion":
        n = max(3, min(5, int(rng.uniform(0.05, 0.15) * size)) | 1)  # Pillow kernels: 3x3 or 5x5
        k = np.zeros((n, n), np.float32)
        k[n // 2] = 1
        kern = Image.fromarray((k * 255).astype(np.uint8)).rotate(rng.uniform(0, 180))
        kv = np.asarray(kern, np.float32).ravel()
        kv = kv / max(kv.sum(), 1)
        img = img.filter(ImageFilter.Kernel((n, n), list(kv), scale=1))

    a = np.asarray(img, np.float32) + rng.normal(0, rng.uniform(1, 8), (img.height, img.width))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(35, 95)))
        img = Image.open(buf).convert("L")
    return img, scale



# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
class Synthesizer:

    def __init__(self, vocabulary: Vocabulary, fonts: list, emnist: EmnistLetters, font_share: float = 0.65,
                 lower_share: float = 0.35, per_letter_share: float = 0.5):
        self.vocabulary = vocabulary
        self.fonts = fonts
        self.emnist = emnist
        self.font_share = font_share
        self.lower_share = lower_share
        self.per_letter_share = per_letter_share


    def hand(self, size: float, rng: np.random.Generator):
        if rng.random() < self.font_share:
            return FontHand.random(self.fonts, size, rng, self.per_letter_share)
        return EmnistHand.random(self.emnist, size, rng)


    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, str] | None:
        """(normalized ink map, label) or None. Labels are lowercase letters and
        digits only: case and punctuation are not predicted."""
        word = apply_case(self.vocabulary.word(rng), rng, self.lower_share)
        text = line_prefix(rng) + word
        size = rng.uniform(36, 90)
        hand = self.hand(size, rng)
        strength = rng.uniform(0, 0.06) * size

        def write(t):
            return Image.fromarray(warp(np.asarray(crop_ink(hand.render(t, rng), int(size // 4))), rng, strength))

        target = write(text)
        neighbors = []
        if rng.random() < 0.5:
            neighbors = [write(line_prefix(rng) + apply_case(self.vocabulary.word(rng), rng, self.lower_share))
                         for _ in range(2)]
        line, (t0, t1) = compose_line(target, neighbors, int(size), rng)
        photo, scale = photograph(line, int(size), rng)
        found = locate_word(photo)
        if found is None or not legible(found[0]):
            return None
        # the normalized word must be the target, not a neighboring line
        x, (_, top, _, bottom) = found
        t0, t1 = t0 * scale, t1 * scale
        if min(bottom, t1) - max(top, t0) < 0.7 * (t1 - t0) or bottom - top > 1.6 * (t1 - t0):
            return None
        label = "".join(c for c in text.lower() if c.isalnum())
        return x, label



def legible(x: np.ndarray) -> bool:
    """Rejects samples degraded beyond what a person could read: almost no
    strong ink left after normalization."""
    return np.percentile(x, 99) >= 120 and (x > 96).mean() >= 0.04



def list_fonts(directory: str) -> list:
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".ttf", ".otf")))
