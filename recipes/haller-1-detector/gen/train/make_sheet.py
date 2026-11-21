"""Génère des photos simulées d'une grille remplie à la main (24 mots x 4 lettres).

Lettres tirées du jeu de test EMNIST (jamais vu à l'entraînement), encre sombre
sur papier, grille 8 colonnes x 12 lignes, quatre repères noirs aux coins.
La « photo » 720x480 subit perspective, rotation, flou, éclairage inégal et bruit.
Sortie : sheets/sheet_<i>.png et sheets/sheet_<i>.json (vérité terrain).
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from train import charger

ICI = Path(__file__).parent
COLS, ROWS, CELL, MARGE, REP = 8, 12, 64, 60, 36


def feuille(x, y, rng):
    W = COLS * CELL + 2 * MARGE
    H = ROWS * CELL + 2 * MARGE
    img = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(img)
    # Repères aux quatre coins
    for cx, cy in [(0, 0), (W - REP, 0), (0, H - REP), (W - REP, H - REP)]:
        d.rectangle([cx + 6, cy + 6, cx + REP - 6, cy + REP - 6], fill=0)
    texte = []
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = MARGE + c * CELL, MARGE + r * CELL
            d.rectangle([x0, y0, x0 + CELL, y0 + CELL], outline=150, width=2)
            i = rng.integers(len(x))
            g = Image.fromarray(255 - x[i]).resize((48, 48), Image.BILINEAR)
            g = g.rotate(rng.uniform(-8, 8), fillcolor=255)
            ox, oy = rng.integers(-4, 5, 2)
            img.paste(Image.composite(Image.new("L", g.size, 30), img.crop((x0 + 8 + ox, y0 + 8 + oy, x0 + 56 + ox, y0 + 56 + oy)),
                                      Image.fromarray(255 - np.array(g))), (int(x0 + 8 + ox), int(y0 + 8 + oy)))
            texte.append(chr(65 + int(y[i])))
    return img, texte


def photo(img, rng):
    W, H = img.size
    # Perspective : coins de la feuille déplacés dans l'image 720x480
    sx, sy = 720, 480
    # La feuille est en portrait : on la tourne de 90° pour remplir le cadre paysage
    img = img.rotate(90, expand=True)
    W, H = img.size
    j = lambda: rng.uniform(-25, 25)
    dst = [(40 + j(), 20 + j()), (680 + j(), 20 + j()), (680 + j(), 460 + j()), (40 + j(), 460 + j())]
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    A, B = [], []
    for (u, v), (xs, ys) in zip(dst, src):
        A.append([u, v, 1, 0, 0, 0, -xs * u, -xs * v]); B.append(xs)
        A.append([0, 0, 0, u, v, 1, -ys * u, -ys * v]); B.append(ys)
    coef = np.linalg.solve(np.array(A), np.array(B))
    out = img.transform((sx, sy), Image.PERSPECTIVE, coef, Image.BILINEAR, fillcolor=90)
    a = np.asarray(out, np.float32)
    # Éclairage inégal, flou, bruit
    yy, xx = np.mgrid[0:sy, 0:sx]
    a = a * (0.6 + 0.4 * (xx / sx * rng.uniform(0.3, 1) + yy / sy * rng.uniform(0, 0.5)))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.4)))
    a = np.asarray(out, np.float32) + rng.normal(0, 6, (sy, sx))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), dst


def main():
    x, y = charger(False)
    x, y = x.numpy(), y.numpy()
    rng = np.random.default_rng(7)
    out = ICI / "sheets"
    out.mkdir(exist_ok=True)
    for k in range(5):
        img, texte = feuille(x, y, rng)
        ph, coins = photo(img, rng)
        ph.save(out / f"sheet_{k}.png")
        (out / f"sheet_{k}.json").write_text(json.dumps({"texte": "".join(texte)}))
    print("ok")


if __name__ == "__main__":
    main()
