"""Lit une photo de grille (make_sheet.py) : repères, redressement, cases, OCR.

Dépendances : numpy et Pillow seulement (présents dans SeedSigner OS).
Usage : python read_sheet.py <modele.npz> <photo.png> [<photo.png> ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from infer_np import predire

COLS, ROWS, CELL, MARGE, REP = 8, 12, 64, 60, 36
SW, SH = COLS * CELL + 2 * MARGE, ROWS * CELL + 2 * MARGE  # feuille portrait


def moyenne_locale(a, r):
    # Moyenne sur une fenêtre (2r+1)^2 par sommes cumulées
    p = np.pad(a, r + 1, mode="edge").cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]
    return s[: a.shape[0], : a.shape[1]] / (k * k)


def reperes(a):
    # Pixels sombres par rapport à leur voisinage ; dans chaque quart de l'image,
    # le pixel sombre le plus proche du coin, puis le centroïde de sa tache.
    h, w = a.shape
    sombre = a < 0.45 * moyenne_locale(a, 25)
    ys, xs = np.nonzero(sombre)
    out = []
    for cx, cy in [(0, 0), (w, 0), (w, h), (0, h)]:
        d = np.hypot(xs - cx, ys - cy)
        i = d.argmin()
        proche = np.hypot(xs - xs[i], ys - ys[i]) < 18
        out.append((xs[proche].mean(), ys[proche].mean()))
    return out


def homographie(src, dst):
    # Coefficients PIL (sortie -> entrée) pour Image.transform(PERSPECTIVE)
    A, B = [], []
    for (u, v), (x, y) in zip(dst, src):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); B.append(y)
    return np.linalg.solve(np.array(A, float), np.array(B, float))


def vers_emnist(c):
    # c : case en niveaux de gris, encre sombre. Sortie 28x28, encre claire centrée.
    c = c.astype(np.float32)
    fond = np.percentile(c, 90)
    # Contraste minimal imposé : une case vide (bruit seul) ne doit pas être étirée
    encre = np.clip((fond - c) / max(fond - c.min(), 50), 0, 1)
    m = encre > 0.35
    if m.sum() < 15:
        return None
    ys, xs = np.nonzero(m)
    e = encre[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    s = max(e.shape)
    carre = np.zeros((s, s), np.float32)
    oy, ox = (s - e.shape[0]) // 2, (s - e.shape[1]) // 2
    carre[oy : oy + e.shape[0], ox : ox + e.shape[1]] = e
    g = Image.fromarray((carre * 255).astype(np.uint8)).resize((24, 24), Image.BILINEAR)
    out = np.zeros((28, 28), np.uint8)
    out[2:26, 2:26] = np.asarray(g)
    return out


def lire(p, chemin):
    t0 = time.perf_counter()
    img = Image.open(chemin).convert("L")
    a = np.asarray(img, np.float32)
    rep = reperes(a)
    t1 = time.perf_counter()
    # Feuille vue tournée de 90° : redresser dans le repère paysage puis remettre en portrait
    cw, ch = SH, SW
    h2 = REP / 2
    dst = [(h2, h2), (cw - h2, h2), (cw - h2, ch - h2), (h2, ch - h2)]
    coef = homographie(rep, dst)
    feuille = img.transform((cw, ch), Image.PERSPECTIVE, coef, Image.BILINEAR).rotate(-90, expand=True)
    f = np.asarray(feuille)
    cases = []
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = MARGE + c * CELL, MARGE + r * CELL
            cases.append(vers_emnist(f[y0 + 9 : y0 + CELL - 9, x0 + 9 : x0 + CELL - 9]))
    t2 = time.perf_counter()
    x = np.stack([c if c is not None else np.zeros((28, 28), np.uint8) for c in cases])
    pred = np.concatenate([predire(p, x[i : i + 32]) for i in range(0, len(x), 32)])
    t3 = time.perf_counter()
    texte = "".join(chr(65 + int(k)) for k in pred)
    return texte, (t1 - t0, t2 - t1, t3 - t2)


def main():
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[1]).items()}
    tot_ok = tot = 0
    for ch in sys.argv[2:]:
        verite = json.loads(Path(ch).with_suffix(".json").read_text())["texte"]
        texte, (ta, tb, tc) = lire(p, ch)
        ok = sum(a == b for a, b in zip(texte, verite))
        tot_ok += ok
        tot += len(verite)
        print(f"{Path(ch).name}: {ok}/{len(verite)} lettres justes ; repères {ta:.2f}s, redressement+cases {tb:.2f}s, OCR {tc:.2f}s")
    print(f"total : {tot_ok}/{tot} = {tot_ok/tot*100:.2f} %")


if __name__ == "__main__":
    main()
