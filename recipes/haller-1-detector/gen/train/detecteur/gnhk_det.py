"""Échantillons réels du détecteur à partir de GNHK (CC BY 4.0) : fenêtres portrait des pages, mots annotés
par leur polygone. Même format que gen_det.py train (x 288x384, y 144x192x9) : canal 0 les mots, canal 1
les numéros (« 12 », « 3. », « 4) »), canal 8 (exclusion) les jetons illisibles (%...%) ; pas de carte.
GNHK_CHIFFRES_EXCLUS=1 : tout jeton qui contient un chiffre va dans l'exclusion (ni mot ni numéro). Dans GNHK, les
chiffres sont des dates, prix, heures, rarement des numéros de seed : sans cible de numéro fiable, ils apprenaient au
réseau qu'un nombre écrit est un mot ou du fond (essais comparables : GNHK -0,05 de rappel des numéros réels).

    python3 gnhk_det.py SORTIE.npz DOSSIER_GNHK FENETRES_PAR_PAGE GRAINE
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_det import NET_SIZE, OUT_SIZE, appliquer, dlt, noyau  # noqa: E402

NUMERO = re.compile(r"^\d{1,2}[.)]?$")
EXCLURE_CHIFFRES = bool(os.environ.get("GNHK_CHIFFRES_EXCLUS"))


def fenetre(page, mots, rng):
    """Fenêtre portrait tournée de la page ; (image 288x384, objets dans les coordonnées de sortie 1/2)."""
    hs = [np.linalg.norm(np.array(q[3]) - np.array(q[0])) for _, q in mots]
    h_mot = float(np.median(hs))
    cible = rng.uniform(4.5, 14)  # hauteur des mots dans l'entrée 288x384
    s = h_mot * NET_SIZE[0] / cible  # largeur de la fenêtre dans la page
    w, h = s, s * NET_SIZE[1] / NET_SIZE[0]
    q = np.array([q for _, q in mots], float).reshape(-1, 2)
    cx = rng.uniform(q[:, 0].min(), q[:, 0].max())
    cy = rng.uniform(q[:, 1].min(), q[:, 1].max())
    a = math.radians(rng.normal(0, 8) + (180 if rng.random() < 0.05 else 0))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    coins = np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]) @ rot.T + (cx, cy)
    coins += rng.uniform(-0.04, 0.04, (4, 2)) * w
    red = max(1, int(s / (2 * NET_SIZE[0])))
    petite = page.reduce(red) if red > 1 else page
    Hm = dlt(coins / red, [(0, 0), (NET_SIZE[0], 0), (NET_SIZE[0], NET_SIZE[1]), (0, NET_SIZE[1])])
    inv = np.linalg.inv(Hm)
    inv /= inv[2, 2]
    fond = int(np.median(np.asarray(petite.reduce(8))))
    x = petite.transform(NET_SIZE, Image.PERSPECTIVE, tuple(inv.ravel()[:8]), Image.BILINEAR,
                         fillcolor=int(rng.uniform(15, fond)))
    a = np.asarray(x, np.float32) * rng.uniform(0.6, 1.1) + rng.normal(0, rng.uniform(1, 4), (NET_SIZE[1], NET_SIZE[0]))
    x = np.clip(a, 0, 255).astype(np.uint8)
    objets = [(t, appliquer(Hm, np.array(q) / red) / 2) for t, q in mots]
    return x, objets


def cibles(objets):
    W, H = OUT_SIZE
    ims = [Image.new("L", (W, H), 0) for _ in range(9)]
    dr = [ImageDraw.Draw(i) for i in ims]
    n = 0
    for t, q in objets:
        if q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H:
            continue
        if t.startswith("%") or not t.strip() or (EXCLURE_CHIFFRES and re.search(r"\d", t)):
            dr[8].polygon([tuple(p) for p in q], fill=255)
            continue
        k, h = noyau(q)
        pts = [tuple(p) for p in k]
        canal = 1 if NUMERO.match(t) else 0
        dr[canal].polygon(pts, fill=255)
        dr[canal].point(tuple(np.mean(k, 0)), fill=255)
        dr[7].polygon(pts, fill=int(min(255, h * 8)))
        n += 1
    return np.stack([np.asarray(i) for i in ims], -1), n


def main():
    sortie, dossier, par_page, graine = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(graine)
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(dossier, "*.json"))):
        ann = json.load(open(f))
        mots = []
        for o in ann:
            p = o["polygon"]
            q = [(p[f"x{i}"], p[f"y{i}"]) for i in range(4)]
            # ordre HG, HD, BD, BG : tri par angle autour du centre, en partant du haut gauche
            c = np.mean(q, 0)
            q = sorted(q, key=lambda v: math.atan2(v[1] - c[1], v[0] - c[0]))
            k = int(np.argmin([v[0] + v[1] for v in q]))
            mots.append((o["text"], q[k:] + q[:k]))
        if len(mots) < 3:
            continue
        page = Image.open(f[:-5] + ".jpg").convert("L")
        for _ in range(par_page):
            x, objets = fenetre(page, mots, rng)
            y, n = cibles(objets)
            if n >= 2:
                X.append(x)
                Y.append(y)
    np.savez_compressed(sortie, x=np.array(X), y=np.array(Y), meta=json.dumps([dict(style="gnhk")] * len(X)))
    print(len(X))


if __name__ == "__main__":
    main()
