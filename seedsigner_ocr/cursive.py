"""
Procedural cursive handwriting (no generative model).

A word is planned as a sequence of virtual targets taken from letter
skeletons (French school cursive or American Palmer/D'Nealian cursive, with
allographs, or detached lowercase "script"), joined by connecting strokes that
depend on the previous and the next letter. The pen trajectory is then produced
by the sigma-lognormal model of the Kinematic Theory of rapid human movements
(Plamondon): each segment between two targets is a stroke whose speed has a
lognormal time profile and whose direction turns linearly with the traveled
fraction (a circular arc); strokes overlap in time and their velocities add,
which rounds corners and loops the way a hand does. Variability comes from
noise on the targets and on the lognormal parameters, and from hand constants
(slant, proportions, loops, spacing, wavy baseline, tremor, speed). The ink
follows the speed and a simulated pressure, for a ballpoint, a felt pen or a
pencil.

References: R. Plamondon, "A kinematic theory of rapid human movements",
Biological Cybernetics 72, 1995; R. Plamondon, C. O'Reilly, J. Galbally,
A. Almaksour, E. Anquetil, "Recent developments in the study of rapid human
movements with the kinematic theory: Applications to handwriting and signature
synthesis", Pattern Recognition Letters 35, 2014.

    python -m seedsigner_ocr.cursive OUT.npz COUNT SEED --bip39 data/bip39_english.txt \
        --other data/other_words.txt

Output: same format as `generate.py`.
"""
import argparse
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH, locate_word
from .synth import Vocabulary, compose_line, crop_ink, legible, line_prefix, photograph, warp

STYLES = ("french", "american", "script")



# ---------------------------------------------------------------------------
# Sigma-lognormal kinematics
# ---------------------------------------------------------------------------
def erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz and Stegun 7.1.26 (absolute error < 1.5e-7)."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) \
        * np.exp(-x * x)
    return s * y



@dataclass
class Stroke:
    """One lognormal component: the pen travels the circular arc of chord
    `chord` turning by `bend` radians, with a lognormal speed profile."""
    chord: np.ndarray
    bend: float
    t0: float
    mu: float
    sigma: float


    def displacement(self, t: np.ndarray) -> np.ndarray:
        """Pen displacement due to this stroke at times t, shape (len(t), 2)."""
        dt = np.maximum(t - self.t0, 1e-9)
        u = 0.5 * (1 + erf((np.log(dt) - self.mu) / (self.sigma * np.sqrt(2))))  # traveled fraction
        u[t <= self.t0] = 0
        length = np.hypot(*self.chord)
        theta = np.arctan2(self.chord[1], self.chord[0])
        b = self.bend
        if abs(b) < 1e-3:
            return np.outer(u, self.chord)
        radius_scale = length / (2 * np.sin(b / 2))  # D / bend, with D the arc length
        ts = theta - b / 2
        phi = ts + b * u
        return np.stack([radius_scale * (np.sin(phi) - np.sin(ts)),
                         radius_scale * (np.cos(ts) - np.cos(phi))], axis=1)



def plan_strokes(targets: np.ndarray, bends: np.ndarray, hand, rng: np.random.Generator) -> list:
    """Lognormal strokes through a sequence of targets. Consecutive strokes
    overlap in time by a hand-dependent amount (smaller spacing: rounder)."""
    strokes, center = [], 0.0
    prev_t = None
    for i in range(1, len(targets)):
        chord = targets[i] - targets[i - 1]
        length = float(np.hypot(*chord))
        if length < 1e-4:
            continue
        b = bends[i]
        arc = length if abs(b) < 1e-3 else length * (b / 2) / np.sin(b / 2)
        t = hand.tempo * (0.25 + arc) ** 0.5 * np.exp(rng.normal(0, 0.08))
        sigma = hand.sigma * np.exp(rng.normal(0, 0.1))
        if prev_t is not None:
            center += hand.overlap * np.exp(rng.normal(0, 0.12)) * (prev_t + t) / 2
        mu = np.log(t)
        strokes.append(Stroke(chord, float(b), center - t, mu, sigma))
        prev_t = t
    return strokes



def trajectory(start: np.ndarray, strokes: list, dt: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """Pen positions (N, 2) and speeds (N,) from the sum of the strokes."""
    if not strokes:
        return start[None], np.zeros(1)
    t_begin = min(s.t0 for s in strokes)
    t_end = max(s.t0 + np.exp(s.mu + 3.2 * s.sigma) for s in strokes)
    t = np.arange(t_begin, t_end + dt, dt)
    pos = np.tile(start, (len(t), 1)).astype(np.float64)
    for s in strokes:
        pos += s.displacement(t)
    speed = np.r_[0, np.hypot(*np.diff(pos, axis=0).T)] / dt
    return pos, speed



# ---------------------------------------------------------------------------
# Letter skeletons
# ---------------------------------------------------------------------------
# Units: x-height 1, baseline y = 0, y upwards. A path is a list of targets
# (x, y, bend in degrees of the arc that arrives at the target, positive
# counterclockwise). path[0] is where the letter body starts: the connecting
# stroke from the previous letter arrives there (its bend is path[0][2], or a
# default when None). The last target is the exit. `lifts` are pen-up strokes
# written after the word (i and j dots, t and x crosses).

@dataclass
class Letter:
    path: list
    width: float
    high_exit: bool = False
    lifts: list = field(default_factory=list)
    start: tuple | None = None  # pen-down point when the letter is not joined, if not path[0]



def asc_loop(x: float, top: float, loop: float) -> list:
    """Ascender: upstroke right of the stem, over the top, down the stem at x.
    loop = 0 gives a retraced stick."""
    if loop < 0.06:
        return [(x, top, 5), (x, 0.6, 0)]
    return [(x + 0.55 * loop, top * 0.62, None), (x + 0.15 * loop, top, 85), (x - 0.3 * loop, top - 0.45 * loop - 0.2, 70),
            (x - 0.05, 0.6, 15)]



def desc_loop_left(x: float, depth: float, loop: float, exit_x: float) -> list:
    """Descender of g, j, y: down the stem, clockwise loop to the left, back up
    across the stem to the baseline."""
    return [(x, -depth + 0.35, 0), (x - 0.25 - 0.4 * loop, -depth, -110), (x - 0.35 - 0.6 * loop, -depth + 0.45, -60),
            (exit_x, 0.3, -35)]



def desc_loop_right(x: float, depth: float, loop: float, exit_x: float) -> list:
    """Descender of f and q: counterclockwise loop to the right, back up to the
    baseline."""
    return [(x, -depth + 0.35, 0), (x + 0.3 + 0.4 * loop, -depth + 0.15, 120), (x + 0.15, -0.05, 40),
            (exit_x, 0.3, 30)]



def bowl(x: float, top: float = 1.0) -> list:
    """Counterclockwise oval of a, d, g, q (and o), starting at its top right."""
    return [(x + 0.8, top * 0.95, -85), (x + 0.4, top, 45), (x + 0.02, 0.5 * top, 85), (x + 0.38, 0.0, 85),
            (x + 0.8, 0.6 * top, 80)]



def exit_low(x: float) -> tuple:
    return (x, 0.32, 65)



def cursive_letters(h) -> dict:
    """Joined lowercase letters for hand h (style, proportions, allographs)."""
    A, D, L = h.ascender, h.descender, h.loop
    french = h.style == "french"
    L_ = {}

    stem = 0.8
    L_["a"] = Letter(bowl(0) + [(stem, 1.0, 0), (stem, 0.12, 0), exit_low(1.1)], 1.1)
    L_["c"] = Letter([(0.55, 0.98, -70), (0.75, 0.8, -110), (0.42, 1.0, 60), (0.02, 0.5, 85), (0.4, 0.0, 85), (0.95, 0.32, 40)], 0.95)
    d_top = A * (0.75 if french else 0.95)
    d_stem = [(stem + 0.02, d_top, 0), (stem, 0.12, 0)] if h.alt["d"] == 0 else \
             [(stem + 0.45 * L, d_top * 0.8, 20), (stem + 0.1 * L, d_top, 85), (stem - 0.2 * L, d_top - 0.4, 70), (stem, 0.12, 10)]
    L_["d"] = Letter(bowl(0) + d_stem + [exit_low(1.1)], 1.1)
    L_["e"] = Letter([(0.55, 0.55, 10), (0.3, 1.0, 110), (0.04, 0.45, 75), (0.38, 0.0, 80), (0.9, 0.32, 45)], 0.9, start=(0.05, 0.4))
    f_top = A
    L_["f"] = Letter(asc_loop(0.32, f_top, L)[:-1] + [(0.3, 0.0, 0)] + desc_loop_right(0.3, D, L, 0.75), 0.75)
    L_["g"] = Letter(bowl(0) + [(stem, 1.0, 0)] + desc_loop_left(stem, D, L, 1.05), 1.05)
    L_["h"] = Letter(asc_loop(0.15, A, L) + [(0.15, 0.0, 0), (0.5, 0.98, -70), (0.75, 0.5, -60), (0.76, 0.12, 0),
                                              exit_low(1.08)], 1.08)
    L_["i"] = Letter([(0.25, 1.0, None), (0.22, 0.12, 0), exit_low(0.55)], 0.55, lifts=[[(0.3, 1.55), (0.33, 1.62)]])
    L_["j"] = Letter([(0.25, 1.0, None)] + desc_loop_left(0.22, D, L, 0.6), 0.6, lifts=[[(0.3, 1.55), (0.33, 1.62)]])
    k_body = [(0.15, 0.0, 0), (0.55, 0.95, -70), (0.6, 0.55, -110), (0.25, 0.48, -50), (0.72, 0.05, 30), exit_low(1.0)] \
        if h.alt["k"] == 0 else \
        [(0.15, 0.0, 0), (0.15, 0.45, 0), (0.62, 0.95, -20), (0.3, 0.48, 0), (0.72, 0.05, 15), exit_low(1.0)]
    L_["k"] = Letter(asc_loop(0.15, A, L) + k_body, 1.0)
    L_["l"] = Letter(asc_loop(0.2, A, L) + [(0.2, 0.12, 0), exit_low(0.55)], 0.55)
    L_["m"] = Letter([(0.12, 1.0, -45), (0.12, 0.0, 0), (0.42, 0.98, -70), (0.62, 0.5, -60), (0.62, 0.0, 0),
                      (0.92, 0.98, -70), (1.12, 0.5, -60), (1.12, 0.12, 0), exit_low(1.42)], 1.42)
    L_["n"] = Letter([(0.12, 1.0, -45), (0.12, 0.0, 0), (0.42, 0.98, -70), (0.62, 0.5, -60), (0.62, 0.12, 0),
                      exit_low(0.92)], 0.92)
    o_exit = [(0.45, 0.98, 75), (0.6, 0.85, 120), (0.95, 0.88, -30)] if h.alt["o"] == 0 else \
             [(0.5, 0.98, 75), (0.95, 0.88, -20)]
    L_["o"] = Letter([(0.62, 0.92, -75), (0.35, 1.0, 35), (0.02, 0.5, 85), (0.35, 0.0, 90), (0.7, 0.5, 90)] + o_exit,
                     0.95, high_exit=True)
    L_["p"] = Letter([(0.22, 1.05, None), (0.18, -D, 0), (0.2, 0.55, 0), (0.55, 1.0, -85), (0.82, 0.5, -80),
                      (0.48, 0.0, -80), (0.22, 0.15, -40), exit_low(0.95)], 0.95)
    L_["q"] = Letter(bowl(0) + [(stem, 1.0, 0)] + desc_loop_right(stem, D, L, 1.2), 1.2)
    if h.alt["r"] == 0:  # French "r a epaule": a small step at the top
        r_body = [(0.15, 1.05, None), (0.45, 0.92, -40), (0.55, 0.95, 60), (0.55, 0.12, -5), exit_low(0.85)]
    else:  # American: a peak, a small dip, a shoulder, down
        r_body = [(0.18, 1.08, None), (0.35, 0.85, 0), (0.62, 0.98, -60), (0.66, 0.12, -10), exit_low(0.95)]
    L_["r"] = Letter(r_body, r_body[-1][0], start=(0.0, 0.3))
    if h.alt["s"] == 0:
        s_body = [(0.3, 1.2, None), (0.6, 0.6, -20), (0.42, 0.0, -110), (0.12, 0.22, -60), (0.3, 0.3, -90),
                  exit_low(0.8)]
    else:  # pointed s closing on the stroke
        s_body = [(0.3, 1.2, None), (0.62, 0.55, -30), (0.42, 0.0, -110), (0.12, 0.22, -50), exit_low(0.8)]
    L_["s"] = Letter(s_body, 0.8, start=(0.05, 0.25))
    t_top = A * (0.75 if french else 0.85)
    L_["t"] = Letter([(0.35, t_top, None), (0.3, 0.12, 0), exit_low(0.7)], 0.7,
                     lifts=[[(0.0, 1.1), (0.68, 1.1)]])
    L_["u"] = Letter([(0.12, 1.0, None), (0.12, 0.25, 0), (0.38, 0.0, 80), (0.68, 0.5, 55), (0.7, 1.0, 0),
                      (0.68, 0.12, 0), exit_low(0.98)], 0.98)
    L_["v"] = Letter([(0.1, 1.0, -40), (0.38, 0.0, 30), (0.72, 1.0, 35), (0.62, 0.85, 120), (0.98, 0.88, -30)],
                     0.98, high_exit=True)
    L_["w"] = Letter([(0.1, 1.0, None), (0.28, 0.0, 30), (0.55, 0.85, 30), (0.78, 0.0, 30), (1.08, 1.0, 35),
                      (0.98, 0.85, 120), (1.32, 0.88, -30)], 1.32, high_exit=True)
    L_["x"] = Letter([(0.1, 0.85, -30), (0.35, 1.0, -60), (0.55, 0.12, 0), exit_low(0.85)], 0.85,
                     lifts=[[(0.8, 1.0), (0.22, 0.02)]])
    L_["y"] = Letter([(0.12, 1.0, None), (0.12, 0.25, 0), (0.38, 0.0, 80), (0.68, 0.5, 55), (0.7, 1.0, 0)]
                     + desc_loop_left(0.7, D, L, 1.0), 1.0)
    if h.alt["z"] == 0:  # looped z
        z_body = [(0.12, 0.92, -30), (0.42, 1.0, -60), (0.58, 0.72, -70), (0.28, 0.45, -40), (0.6, 0.15, -70),
                  (0.5, -D + 0.3, -50), (0.25, -D, -90), (0.12, -D + 0.45, -70), exit_low(0.9)]
    else:
        z_body = [(0.1, 1.0, -20), (0.7, 1.0, 0), (0.12, 0.0, 0), (0.6, 0.02, 0), exit_low(0.88)]
    L_["z"] = Letter(z_body, 0.88)
    b_top = A
    L_["b"] = Letter(asc_loop(0.15, b_top, L) + [(0.15, 0.25, 0), (0.45, 0.0, 70), (0.75, 0.6, 70), (0.5, 0.95, 70),
                                                 (0.62, 0.8, 120), (0.95, 0.88, -30)], 0.95, high_exit=True)
    return L_



def script_letters(h) -> dict:
    """Detached lowercase letters (print), each a list of pen-down strokes."""
    A, D = h.ascender, h.descender
    o = [(0.75, 0.85), (0.4, 1.0, 50), (0.02, 0.5, 85), (0.4, 0.0, 85), (0.78, 0.5, 85), (0.75, 0.85, 40)]
    S = {
        "a": [o[:-1] + [(0.78, 1.0, 0), (0.8, 0.0, 0)]],
        "b": [[(0.1, A), (0.1, 0.0, 0)], [(0.1, 0.55), (0.45, 1.0, -70), (0.8, 0.5, -90), (0.45, 0.0, -90), (0.1, 0.2, -60)]],
        "c": [[(0.75, 0.85), (0.4, 1.0, 50), (0.02, 0.5, 85), (0.4, 0.0, 85), (0.8, 0.15, 40)]],
        "d": [o[:-1] + [(0.8, A, 0), (0.8, 0.0, 0)]],
        "e": [[(0.05, 0.5), (0.8, 0.55, 0), (0.42, 1.0, 100), (0.02, 0.5, 80), (0.4, 0.0, 85), (0.8, 0.15, 40)]],
        "f": [[(0.75, A * 0.92), (0.45, A, 70), (0.3, A * 0.7, 50), (0.3, 0.0, 0)], [(0.0, 0.95), (0.65, 0.95)]],
        "g": [o[:-1] + [(0.78, 1.0, 0), (0.78, -D + 0.3, 0), (0.4, -D, -90), (0.05, -D + 0.35, -60)]],
        "h": [[(0.1, A), (0.1, 0.0, 0)], [(0.1, 0.55), (0.45, 1.0, -70), (0.75, 0.6, -60), (0.75, 0.0, 0)]],
        "i": [[(0.2, 1.0), (0.2, 0.0, 0)], [(0.2, 1.55), (0.22, 1.62)]],
        "j": [[(0.4, 1.0), (0.4, -D + 0.3, 0), (0.1, -D, -90), (-0.15, -D + 0.35, -50)], [(0.4, 1.55), (0.42, 1.62)]],
        "k": [[(0.1, A), (0.1, 0.0, 0)], [(0.7, 1.0), (0.12, 0.42, 0), (0.75, 0.0, 0)]],
        "l": [[(0.2, A), (0.2, 0.15, 0), (0.45, 0.0, 60)]],
        "m": [[(0.1, 1.0), (0.1, 0.0, 0)], [(0.1, 0.6), (0.4, 1.0, -70), (0.6, 0.6, -50), (0.6, 0.0, 0)],
              [(0.6, 0.6), (0.9, 1.0, -70), (1.1, 0.6, -50), (1.1, 0.0, 0)]],
        "n": [[(0.1, 1.0), (0.1, 0.0, 0)], [(0.1, 0.6), (0.42, 1.0, -70), (0.7, 0.6, -50), (0.7, 0.0, 0)]],
        "o": [o],
        "p": [[(0.1, 1.0), (0.1, -D, 0)], [(0.1, 0.55), (0.45, 1.0, -70), (0.8, 0.5, -90), (0.45, 0.0, -90), (0.1, 0.2, -60)]],
        "q": [o[:-1] + [(0.78, 1.0, 0), (0.78, -D, 0), (1.0, -D + 0.2, 60)]],
        "r": [[(0.1, 1.0), (0.1, 0.0, 0)], [(0.1, 0.55), (0.4, 1.0, -70), (0.7, 0.92, -30)]],
        "s": [[(0.7, 0.9), (0.4, 1.0, 60), (0.08, 0.75, 80), (0.4, 0.5, 60), (0.72, 0.25, -80), (0.4, 0.0, -80),
               (0.05, 0.1, -50)]],
        "t": [[(0.3, A * 0.8), (0.3, 0.15, 0), (0.6, 0.0, 70)], [(0.0, 0.95), (0.6, 0.95)]],
        "u": [[(0.1, 1.0), (0.1, 0.3, 0), (0.4, 0.0, 80), (0.75, 0.4, 50)], [(0.75, 1.0), (0.75, 0.0, 0)]],
        "v": [[(0.0, 1.0), (0.38, 0.0, 0), (0.76, 1.0, 0)]],
        "w": [[(0.0, 1.0), (0.25, 0.0, 0), (0.55, 0.9, 0), (0.85, 0.0, 0), (1.1, 1.0, 0)]],
        "x": [[(0.0, 1.0), (0.7, 0.0, 0)], [(0.7, 1.0), (0.0, 0.0, 0)]],
        "y": [[(0.0, 1.0), (0.4, 0.0, 0)], [(0.8, 1.0), (0.3, -D, 0), (0.0, -D + 0.1, -30)]],
        "z": [[(0.0, 1.0), (0.72, 1.0, 0), (0.0, 0.0, 0), (0.75, 0.0, 0)]],
    }
    widths = {"m": 1.25, "w": 1.15, "i": 0.4, "l": 0.5, "j": 0.5, "t": 0.65, "f": 0.75, "r": 0.72, "q": 1.0}
    return {c: (strokes, widths.get(c, 0.85)) for c, strokes in S.items()}



def digit_strokes() -> dict:
    """Digits and the punctuation of line numbers, 1.6 high, as pen-down strokes."""
    return {
        "0": ([[(0.5, 1.6), (0.05, 0.8, 90), (0.5, 0.0, 90), (0.95, 0.8, 90), (0.5, 1.6, 90)]], 1.0),
        "1": ([[(0.15, 1.2), (0.5, 1.6, -20), (0.5, 0.0, 0)]], 0.7),
        "2": ([[(0.1, 1.2), (0.5, 1.6, -90), (0.85, 1.15, -90), (0.05, 0.0, -10), (0.9, 0.0, 0)]], 1.0),
        "3": ([[(0.1, 1.4), (0.5, 1.6, -60), (0.8, 1.25, -90), (0.4, 0.85, -80), (0.85, 0.4, -100), (0.45, 0.0, -80),
                (0.05, 0.25, -50)]], 1.0),
        "4": ([[(0.65, 0.0), (0.65, 1.6, 0), (0.05, 0.5, 0), (0.95, 0.5, 0)]], 1.0),
        "5": ([[(0.85, 1.6), (0.2, 1.6, 0), (0.15, 0.9, 0), (0.85, 0.5, -120), (0.45, 0.0, -80), (0.05, 0.25, -40)]], 1.0),
        "6": ([[(0.8, 1.5), (0.15, 0.6, 60), (0.5, 0.0, 90), (0.85, 0.45, 90), (0.5, 0.85, 90), (0.15, 0.5, 70)]], 1.0),
        "7": ([[(0.05, 1.6), (0.9, 1.6, 0), (0.35, 0.0, 0)]], 1.0),
        "8": ([[(0.8, 1.4), (0.45, 1.6, 80), (0.15, 1.2, 90), (0.5, 0.85, 60), (0.85, 0.4, -90), (0.5, 0.0, -90),
                (0.15, 0.4, -90), (0.8, 1.4, -20)]], 1.0),
        "9": ([[(0.85, 1.3), (0.5, 1.6, 70), (0.15, 1.2, 90), (0.5, 0.85, 90), (0.85, 1.25, 80), (0.8, 0.0, -10)]], 1.0),
        ".": ([[(0.1, 0.02), (0.13, 0.08)]], 0.35),
        ")": ([[(0.1, 1.6), (0.35, 0.8, -60), (0.1, 0.0, -60)]], 0.5),
        "-": ([[(0.1, 0.75), (0.6, 0.75, 0)]], 0.7),
    }



# ---------------------------------------------------------------------------
# Hand
# ---------------------------------------------------------------------------
@dataclass
class CursiveHand:
    style: str             # french, american or script
    x_height: float        # pixels
    slant: float           # degrees, positive: leaning right
    ascender: float        # in x-heights
    descender: float
    loop: float            # loop width, 0 = retraced sticks
    width: float           # horizontal scale of the letters
    spacing: float         # extra space between letters
    tempo: float           # stroke duration scale (s)
    sigma: float           # lognormal spread
    overlap: float         # stroke spacing in time, in durations (smaller: rounder)
    noise: float           # target noise, in x-heights
    lift_share: float      # chance of a pen lift at a join
    lead_in: bool          # entry stroke before the first letter
    baseline_wave: float   # amplitude, in x-heights
    tremor: float          # in x-heights
    pen: str               # ballpoint, felt or pencil
    pen_width: float       # in x-heights
    alt: dict              # allograph index per letter

    @classmethod
    def random(cls, size: float, rng: np.random.Generator, style: str | None = None):
        style = style or str(rng.choice(STYLES, p=[0.4, 0.4, 0.2]))
        french = style == "french"
        pen = str(rng.choice(["ballpoint", "felt", "pencil"], p=[0.6, 0.25, 0.15]))
        return cls(
            style=style,
            x_height=size * rng.uniform(0.38, 0.5),
            slant=float(rng.normal(14 if style != "script" else 4, 7)),
            ascender=float(rng.uniform(2.1, 2.7) if french else rng.uniform(1.7, 2.2)),
            descender=float(rng.uniform(1.1, 1.6) if french else rng.uniform(0.8, 1.2)),
            loop=float(rng.choice([0.0, rng.uniform(0.25, 0.5)], p=[0.1, 0.9]) if french
                       else rng.choice([0.0, rng.uniform(0.15, 0.4)], p=[0.3, 0.7])),
            width=float(rng.uniform(0.8, 1.25)),
            spacing=float(rng.uniform(-0.08, 0.2)),
            tempo=float(rng.uniform(0.09, 0.14)),
            sigma=float(rng.uniform(0.2, 0.32)),
            overlap=float(rng.uniform(0.45, 0.75)),
            noise=float(rng.uniform(0.01, 0.05)),
            lift_share=float(rng.choice([0.0, rng.uniform(0.05, 0.3)], p=[0.6, 0.4])),
            lead_in=bool(rng.random() < (0.85 if french else 0.3)),
            baseline_wave=float(rng.uniform(0, 0.12)),
            tremor=float(rng.choice([0.0, rng.uniform(0.005, 0.02)], p=[0.7, 0.3])),
            pen=pen,
            pen_width=float({"ballpoint": rng.uniform(0.06, 0.1), "felt": rng.uniform(0.1, 0.17),
                             "pencil": rng.uniform(0.06, 0.1)}[pen]),
            alt={"d": int(rng.random() < (0.15 if french else 0.5)), "k": int(rng.random() < 0.4),
                 "o": int(rng.random() < 0.3), "r": int(rng.random() < (0.15 if french else 0.8)),
                 "s": int(rng.random() < 0.5), "z": int(rng.random() < (0.2 if french else 0.6))},
        )


    # -- planning ---------------------------------------------------------
    def plan(self, text: str, rng: np.random.Generator) -> list:
        """Pen-down pieces of the text: each a list of (x, y, bend in degrees)."""
        letters = cursive_letters(self) if self.style != "script" else None
        script = script_letters(self) if self.style == "script" else None
        digits = digit_strokes()
        pieces, delayed = [], []
        current, high, x, started = None, False, 0.0, False
        n = self.noise

        def jitter(p):
            return (p[0] + rng.normal(0, n), p[1] + rng.normal(0, n * 0.8)) + tuple(p[2:])

        for ch in text:
            if ch == " ":
                current = None
                x += 0.6 * self.width
                continue
            lower = ch.lower()
            if lower in digits or script is not None or lower not in (letters or {}):
                strokes, w = digits.get(lower) or (script or script_letters(self)).get(lower, ([], 0.6))
                current = None
                for s in strokes:
                    pieces.append([(x + p[0] * self.width, p[1]) + tuple(p[2:]) for p in map(jitter, s)])
                x += (w + 0.15 + self.spacing) * self.width
                continue
            letter = letters[lower]
            body = [(x + p[0] * self.width, p[1], p[2]) for p in map(jitter, letter.path)]
            first_bend = body[0][2]
            if current is not None and rng.random() < self.lift_share:
                current = None
            if current is None:
                if self.lead_in and not started:
                    if body[0][1] > 0.7 and letter.path[0][0] > 0.5:  # bowl: short rise to its top
                        current = [(body[0][0] - 0.4 * self.width, 0.5, 0)]
                    else:
                        left = x + (min(p[0] for p in letter.path) - 0.35) * self.width
                        current = [(left, 0.0 if body[0][1] > 0.3 else body[0][1] - 0.4, 0)]
                    high = False
                elif letter.start is not None:
                    current = [(x + letter.start[0] * self.width, letter.start[1], 0)]
                else:
                    current = [(body[0][0], body[0][1], 0)]
                    body = body[1:]
                pieces.append(current)
            if body and len(body) == len(letter.path):  # joined: the first segment is the connecting stroke
                if first_bend is None:
                    first_bend = (15 if high else 30) if body[0][1] > 0.4 else 15
                body[0] = body[0][:2] + (first_bend,)
            current.extend(body)
            delayed += [[(x + p[0] * self.width + rng.normal(0, n), p[1] + rng.normal(0, n)) for p in lift]
                        for lift in letter.lifts]
            high, started = letter.high_exit, True
            x += (letter.width + self.spacing) * self.width
        if high and current is not None and len(current) > 2:  # no high outstroke after the last letter
            current.pop()
        elif pieces and rng.random() < 0.3 and len(pieces[-1]) > 2:  # unfinished exit stroke
            last = pieces[-1][-1]
            pieces[-1][-1] = (last[0] - 0.12, last[1] - 0.1, last[2])
        return pieces + delayed


    # -- kinematics and ink -----------------------------------------------
    def strokes(self, text: str, rng: np.random.Generator) -> list:
        """Pen trajectories in x-height units: list of (positions, speeds)."""
        out = []
        for piece in self.plan(text, rng):
            if len(piece) == 2 and np.hypot(piece[1][0] - piece[0][0], piece[1][1] - piece[0][1]) < 0.15:
                d = np.array(piece[1][:2]) - np.array(piece[0][:2])  # a dot
                piece = [piece[0], (piece[0][0] + d[0] + rng.normal(0, 0.03), piece[0][1] + d[1] + rng.normal(0, 0.03))]
            pts = np.array([p[:2] for p in piece], np.float64)
            bends = np.radians([p[2] if len(p) > 2 and p[2] is not None else 0 for p in piece])
            bends = bends + np.radians(rng.normal(0, 6, len(bends)))
            out.append(trajectory(pts[0], plan_strokes(pts, bends, self, rng)))
        return out


    def render(self, text: str, rng: np.random.Generator) -> Image.Image:
        """The text on white paper, dark ink (same convention as the font hands)."""
        trajectories = self.strokes(text, rng)
        scale = 3  # supersampling
        u = self.x_height * scale
        shear = np.tan(np.radians(self.slant))
        wave_phase, wave_len = rng.uniform(0, 2 * np.pi), rng.uniform(4, 9)
        tremor_phase = rng.uniform(0, 2 * np.pi, 4)
        all_pts = []
        for pos, speed in trajectories:
            p = pos.copy()
            p[:, 1] += self.baseline_wave * np.sin(2 * np.pi * p[:, 0] / wave_len + wave_phase)
            if self.tremor:
                k = np.arange(len(p)) * 0.004
                p[:, 0] += self.tremor * np.sin(2 * np.pi * 9 * k + tremor_phase[0]) * np.sin(2 * np.pi * 1.3 * k + tremor_phase[1])
                p[:, 1] += self.tremor * np.sin(2 * np.pi * 11 * k + tremor_phase[2]) * np.sin(2 * np.pi * 1.7 * k + tremor_phase[3])
            p[:, 0] += shear * p[:, 1]
            all_pts.append(p)
        if not all_pts:
            return Image.new("L", (8, 8), 255)
        stacked = np.vstack(all_pts)
        lo, hi = stacked.min(0), stacked.max(0)
        pad = 0.6
        w = int((hi[0] - lo[0] + 2 * pad) * u) + 2
        h = int((hi[1] - lo[1] + 2 * pad) * u) + 2
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        base_width = self.pen_width * u
        for p, (_, speed) in zip(all_pts, trajectories):
            xy = np.stack([(p[:, 0] - lo[0] + pad) * u, (hi[1] + pad - p[:, 1]) * u], axis=1)
            v = speed / (np.percentile(speed, 90) + 1e-6)
            if self.pen == "felt":
                widths = base_width * (1.25 - 0.35 * np.clip(v, 0, 1.5))
            elif self.pen == "pencil":
                widths = base_width * (1.1 - 0.2 * np.clip(v, 0, 1.5))
            else:
                widths = base_width * (1.1 - 0.15 * np.clip(v, 0, 1.5))
            pressure = np.clip(1 - 0.25 * np.clip(v - 0.8, 0, 1), 0.5, 1) if self.pen != "felt" else np.ones(len(v))
            step = max(1, len(xy) // 400)
            for i in range(0, len(xy) - 1, step):
                j = min(i + step, len(xy) - 1)
                wd = max(1, int(round(widths[i])))
                level = int(255 * pressure[i])
                draw.line([tuple(xy[i]), tuple(xy[j])], fill=level, width=wd)
                r = wd / 2
                draw.ellipse([xy[j][0] - r, xy[j][1] - r, xy[j][0] + r, xy[j][1] + r], fill=level)
        ink = np.asarray(canvas.resize((max(1, w // scale), max(1, h // scale)), Image.BOX), np.float32) / 255
        if self.pen == "pencil":
            grain = rng.uniform(0.55, 1.0, ink.shape).astype(np.float32)
            ink = ink * grain * rng.uniform(0.55, 0.85)
        elif self.pen == "ballpoint":
            ink = ink * np.clip(rng.normal(0.95, 0.05, ink.shape), 0.75, 1).astype(np.float32)
        return Image.fromarray(np.clip(255 * (1 - ink), 0, 255).astype(np.uint8))



# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
class CursiveSynthesizer:
    """Same sampling as `synth.Synthesizer`, with a procedural cursive hand:
    lowercase (90 %) or capitalized words (written in lowercase: there are no
    capital skeletons; labels are case-insensitive)."""

    def __init__(self, vocabulary: Vocabulary, style: str | None = None):
        self.vocabulary = vocabulary
        self.style = style


    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, str] | None:
        word = self.vocabulary.word(rng)
        text = line_prefix(rng) + word
        size = rng.uniform(36, 90)
        hand = CursiveHand.random(size, rng, self.style)
        strength = rng.uniform(0, 0.04) * size

        def write(t):
            return Image.fromarray(warp(np.asarray(crop_ink(hand.render(t, rng), int(size // 4))), rng, strength))

        target = write(text)
        neighbors = []
        if rng.random() < 0.5:
            neighbors = [write(line_prefix(rng) + self.vocabulary.word(rng)) for _ in range(2)]
        line, (t0, t1) = compose_line(target, neighbors, int(size), rng)
        photo, scale = photograph(line, int(size), rng)
        found = locate_word(photo)
        if found is None or not legible(found[0]):
            return None
        x, (_, top, _, bottom) = found
        t0, t1 = t0 * scale, t1 * scale
        if min(bottom, t1) - max(top, t0) < 0.7 * (t1 - t0) or bottom - top > 1.6 * (t1 - t0):
            return None
        return x, "".join(c for c in text.lower() if c.isalnum())



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("count", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--bip39", required=True)
    p.add_argument("--other", required=True)
    p.add_argument("--style", choices=STYLES)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    synth = CursiveSynthesizer(Vocabulary.load(a.bip39, a.other), a.style)
    x = np.zeros((a.count, INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    widths, labels = np.zeros(a.count, np.int16), []
    n, t0 = 0, time.time()
    while n < a.count:
        s = synth.sample(rng)
        if s is None:
            continue
        x[n, :, :s[0].shape[1]] = s[0]
        widths[n] = s[0].shape[1]
        labels.append(s[1])
        n += 1
        if n % 5000 == 0:
            print(f"{n}/{a.count} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(a.out, x=x, w=widths, y=np.array(labels))



if __name__ == "__main__":
    main()
