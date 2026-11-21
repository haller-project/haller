"""Seed écrite sur le modèle officiel SeedSigner (mnemonic_24word_wfingerprint.pdf).

Simulation : modèle rendu à 200 dpi (1700x2200), 24 mots BIP39 en capitales
EMNIST posés sur les lignes, feuille photographiée en entier sur une table sombre
par le capteur en pleine résolution (2592x1944), perspective, éclairage, flou, bruit.
Lecture : repérage de la feuille (zone claire, coins extrêmes), redressement dans
les coordonnées du modèle, puis lecture de chaque ligne (ligne.lire_ligne) et
choix du mot par le dictionnaire, puis checksum.
Usage : python officiel.py <modele.npz> [nb_seeds] [--demo]
"""
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

import ligne as L
from seed_sim import checksum_ok, mnemonique

MODELE = Image.open(__file__.rsplit("/", 1)[0] + "/../officiel/m24_200-1.png").convert("L")
PW, PH = MODELE.size  # 1700 x 2200
LIGNES_Y = [426 + 130 * r for r in range(12)]
COLONNES = [(364, 789), (994, 1419)]
CAPTEUR = (2592, 1944)


def poser_mot(img, mot, x0, x1, yb, ex, par_lettre, rng):
    haut = int(rng.integers(34, 48))  # 4,3 à 6 mm à 200 dpi
    lettres = []
    for ch in mot.upper():
        i = rng.choice(par_lettre[ord(ch) - 65])
        g = Image.fromarray(255 - ex[i]).crop(Image.fromarray(ex[i]).getbbox())
        s = haut / g.height * rng.uniform(0.9, 1.08)
        g = g.resize((max(4, int(g.width * s)), max(4, int(g.height * s))), Image.BILINEAR)
        lettres.append(np.asarray(g.rotate(rng.uniform(-6, 6), expand=True, fillcolor=255)))
    esp = [int(rng.choice([0, 3, 5, 8, 11, 15], p=[.08, .17, .25, .25, .15, .1])) for _ in lettres]
    larg = sum(l.shape[1] for l in lettres) + sum(esp[1:])
    f = min(1.0, (x1 - x0 - 10) / larg)
    x = x0 + int(rng.integers(2, 12))
    for k, l in enumerate(lettres):
        if f < 1:
            l = np.asarray(Image.fromarray(l).resize((max(3, int(l.shape[1] * f)), l.shape[0])))
        if k:
            x += int(esp[k] * f)
        y = yb - l.shape[0] - int(rng.integers(2, 9))
        zone = np.asarray(img.crop((x, y, x + l.shape[1], y + l.shape[0])))
        img.paste(Image.fromarray(np.minimum(zone, np.where(l < 200, rng.integers(25, 60), 255).astype(np.uint8))), (x, y))
        x += l.shape[1]


def photographier(feuille, rng):
    W, H = CAPTEUR
    k = W / 2592
    hp = rng.uniform(1650, 1850) * k
    wp = hp * PW / PH
    cx, cy = W / 2 + rng.uniform(-150, 150) * k, H / 2 + rng.uniform(-40, 40) * k
    j = lambda: rng.uniform(-70, 70) * k
    dst = [(cx - wp / 2 + j(), cy - hp / 2 + j()), (cx + wp / 2 + j(), cy - hp / 2 + j()),
           (cx + wp / 2 + j(), cy + hp / 2 + j()), (cx - wp / 2 + j(), cy + hp / 2 + j())]
    coef = homographie([(0, 0), (PW, 0), (PW, PH), (0, PH)], dst)
    fond = int(rng.integers(40, 90))
    out = feuille.transform((W, H), Image.PERSPECTIVE, coef, Image.BILINEAR, fillcolor=fond)
    a = np.asarray(out, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    a = a * (0.6 + 0.4 * (xx / W * rng.uniform(0.2, 1) + yy / H * rng.uniform(0, 0.6)))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.2) * k))
    a = np.asarray(out, np.float32) + rng.normal(0, 5, (H, W))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), dst


def homographie(src, dst):
    # Coefficients PIL (sortie -> entrée) : points dst (sortie) vers src (entrée)
    A, B = [], []
    for (u, v), (x, y) in zip(dst, src):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); B.append(y)
    return np.linalg.solve(np.array(A, float), np.array(B, float))


def coins_feuille(photo):
    # Image réduite, seuil d'Otsu, coins extrêmes de la zone claire
    r = 8
    small = np.asarray(photo.resize((photo.width // r, photo.height // r), Image.BILINEAR), np.float32)
    hist = np.bincount(small.astype(np.uint8).ravel(), minlength=256).astype(float)
    tot = hist.sum(); cum = hist.cumsum(); mcum = (hist * np.arange(256)).cumsum()
    w0 = cum / tot; m0 = mcum / np.maximum(cum, 1); m1 = (mcum[-1] - mcum) / np.maximum(tot - cum, 1)
    t = int(np.argmax(w0 * (1 - w0) * (m0 - m1) ** 2))
    ys, xs = np.nonzero(small > t)
    s, d = xs + ys, xs - ys
    pts = [(xs[s.argmin()], ys[s.argmin()]), (xs[d.argmax()], ys[d.argmax()]),
           (xs[s.argmax()], ys[s.argmax()]), (xs[d.argmin()], ys[d.argmin()])]
    return [((x + 0.5) * r, (y + 0.5) * r) for x, y in pts]


def lire_page(p, photo):
    t0 = time.perf_counter()
    coins = coins_feuille(photo)
    coef = homographie(coins, [(0, 0), (PW, 0), (PW, PH), (0, PH)])
    page = photo.transform((PW, PH), Image.PERSPECTIVE, coef, Image.BILINEAR)
    t1 = time.perf_counter()
    res = []
    for c, (x0, x1) in enumerate(COLONNES):
        for r, yb in enumerate(LIGNES_Y):
            band = page.crop((x0 - 8, yb - 100, x1 + 12, yb + 8))
            L.BANDE = (0, 0, band.width, band.height)
            G, n = L.lire_ligne(p, 0.65, band)
            res.append(L.mots_candidats(G, n))
    return res, page, (t1 - t0, time.perf_counter() - t1)


def main():
    from train import charger
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[1]).items()}
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
    demo = "--demo" in sys.argv
    ex, ey = charger(False)
    ex, ey = ex.numpy(), ey.numpy()
    par_lettre = {k: np.nonzero(ey == k)[0] for k in range(26)}
    rng = np.random.default_rng(31)
    st = dict(mots=0, top1=0, top3=0, seeds_ok=0, faux_valides=0, ta=0.0, tb=0.0)
    for f in range(nseeds):
        mots = mnemonique(rng.bytes(32))
        feuille = MODELE.copy()
        for k, mot in enumerate(mots):
            x0, x1 = COLONNES[k // 12]
            poser_mot(feuille, mot, x0, x1, LIGNES_Y[k % 12] - 2, ex, par_lettre, rng)
        photo, _ = photographier(feuille, rng)
        if demo and f == 0:
            feuille.save("officiel_demo_feuille.png"); photo.save("officiel_demo_photo.png")
        res, page, (ta, tb) = lire_page(p, photo)
        if demo and f == 0:
            page.save("officiel_demo_redressee.png")
        st["ta"] += ta; st["tb"] += tb
        for (c, s), mot in zip(res, mots):
            top = [c[i] for i in np.argsort(-s)[:3]]
            st["mots"] += 1; st["top1"] += top[0] == mot; st["top3"] += mot in top
        final, mode = L.corriger_seed(res)
        st["seeds_ok"] += final == mots
        st["faux_valides"] += final != mots and checksum_ok(final)
    n = st["mots"]
    print(f"{nseeds} seeds ({n} mots), modèle officiel 24 mots, photo {CAPTEUR[0]}x{CAPTEUR[1]}")
    print(f"mot juste : {st['top1']/n*100:.1f} % ; dans les 3 variantes : {st['top3']/n*100:.1f} %")
    print(f"seeds exactes après checksum : {st['seeds_ok']}/{nseeds} ; valides mais fausses : {st['faux_valides']}")
    print(f"temps x86 par page : repérage+redressement {st['ta']/nseeds:.2f} s, 24 lignes {st['tb']/nseeds:.2f} s")


if __name__ == "__main__":
    main()
