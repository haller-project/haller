"""Seeds écrites sur les supports officiels SeedSigner, en capitales ou en minuscules.

Supports (rendus des PDF de docs/seed_qr/printable_templates) :
- page24 : mnemonic_24word_wfingerprint.pdf (Letter), photo de la page entière ;
- carte12 : trading_card_21x21_w12words.pdf, recto (63 x 88 mm) ;
- carte24 : trading_card_25x25_w24words.pdf, recto, 2 colonnes.
Écriture : lettres EMNIST de test ; en minuscules, hauteur d'x 0,6, hampes
(b d f h k l t) et jambages (g j p q y) sous la ligne de base.
Lecture : coins de la zone claire (Otsu), redressement vers le modèle, lecture de
chaque ligne (ligne.lire_ligne, casse fusionnée pour le modèle 52 classes),
dictionnaire BIP39, checksum.
Usage : python cartes.py <support> <maj|min> <modele.npz> [nb_seeds] [--demo]
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import ligne as L
from officiel import homographie
from seed_sim import checksum_ok, mnemonique

OFF = Path(__file__).parent.parent / "officiel"
CAPTEUR = (2592, 1944)
BANDE_HAUT, BANDE_BAS = 0.95, 0.3  # bande lue autour du trait imprimé, en écarts de ligne
COINS_CIBLE = 384  # côté de l'image réduite pour le repérage de la carte (pixels)
HAUTEUR_CARTE = (1070, 1360)  # hauteur de la carte dans la photo (pixels)
X_HAUT = set("acemnorsuvwxz")
HAMPES = set("bdfhklt")
JAMBAGES = set("gjpqy")


def _support(nom):
    if nom == "page24":
        img = Image.open(OFF / "m24_200-1.png").convert("L")
        lignes = [(x0, x1, 426 + 130 * r) for x0, x1 in [(364, 789), (994, 1419)] for r in range(12)]
        return dict(img=img, lignes=lignes, ecart=130, hauteur_photo=(1650, 1850))
    src = Image.open(OFF / ("c12-1.png" if nom == "carte12" else "c24-1.png")).convert("L")
    img = src.crop((696, 400, 1700, 1801))  # recto découpé (400 dpi)
    ys = [367 + 76.6 * r for r in range(12)]
    if nom == "carte12":
        lignes = [(180 if r < 9 else 221, 812, int(ys[r])) for r in range(12)]
    else:
        lignes = [(153 if r < 9 else 188, 474, int(ys[r])) for r in range(12)] + [(621, 912, int(ys[r])) for r in range(12)]
    # La carte occupe 55 à 70 % de la hauteur de l'image
    return dict(img=img, lignes=lignes, ecart=76.6, hauteur_photo=HAUTEUR_CARTE)


def _droite(t, v):
    # Ajuste v = a*t + b en écartant itérativement les points aberrants ; moindres
    # carrés en forme fermée (même résultat que np.polyfit, bien plus rapide sur le Pi)
    t = np.asarray(t, np.float64)
    v = np.asarray(v, np.float64)
    m = np.ones(len(t), bool)
    for _ in range(4):
        tm, vm = t[m], v[m]
        if len(tm) < 2:
            raise ValueError("droite : moins de 2 points")
        mt, mv = tm.mean(), vm.mean()
        dt = tm - mt
        den = float(dt @ dt)
        a = float(dt @ (vm - mv)) / den if den > 0 else 0.0
        b = mv - a * mt
        r = np.abs(v - (a * t + b))
        m = r < max(2.0, 2.5 * np.median(r[m]))
    return a, b


def _propager_runs(atteint, m, axe):
    """Étend « atteint » à toute plage contiguë de m (le long de l'axe) qui le touche."""
    if axe == 0:
        return _propager_runs(atteint.T, m.T, 1).T
    debut = m.copy()
    debut[:, 1:] &= ~m[:, :-1]
    ids = np.cumsum(debut.ravel()).reshape(m.shape) * m
    touche = np.bincount(ids.ravel(), weights=atteint.ravel(), minlength=int(ids.max()) + 1) > 0
    touche[0] = False
    return touche[ids]


def composante_interieure(m):
    """Zone claire qui ne communique pas avec le bord de l'image (intérieur de la
    feuille ou de la carte, quelle que soit la couleur de la table). Les zones
    claires reliées au bord sont retirées par propagation le long des rangées et
    des colonnes, sans boucle par pixel. Si rien ne reste, la zone claire entière."""
    atteint = np.zeros_like(m)
    atteint[0], atteint[-1], atteint[:, 0], atteint[:, -1] = m[0], m[-1], m[:, 0], m[:, -1]
    n = -1
    while atteint.sum() != n:
        n = atteint.sum()
        atteint = _propager_runs(atteint, m, 1)
        atteint = _propager_runs(atteint, m, 0)
    interieur = m & ~atteint
    return interieur if interieur.sum() > 0.02 * m.size else m


def _otsu(v):
    hist = np.bincount(np.clip(v, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    tot = hist.sum(); cum = hist.cumsum(); mcum = (hist * np.arange(256)).cumsum()
    w0 = cum / tot; m0 = mcum / np.maximum(cum, 1); m1 = (mcum[-1] - mcum) / np.maximum(tot - cum, 1)
    return int(np.argmax(w0 * (1 - w0) * (m0 - m1) ** 2))


def _quad(m, r):
    # Droites ajustées sur les 4 bords de la zone m, coins = intersections
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
    haut = np.argmax(mc, 0)[kc]
    bas = (m.shape[0] - 1 - np.argmax(mc[::-1], 0))[kc]
    gauche = np.argmax(mr, 1)[kr]
    droite = (m.shape[1] - 1 - np.argmax(mr[:, ::-1], 1))[kr]
    cols, rows = cols[kc], rows[kr]
    ah, bh = _droite(cols, haut); ab, bb = _droite(cols, bas)
    ag, bg = _droite(rows, gauche); ad, bd = _droite(rows, droite)

    def inter(a1, b1, a2, b2):
        y = (a1 * b2 + b1) / (1 - a1 * a2)
        return (a2 * y + b2, y)

    pts = [inter(ah, bh, ag, bg), inter(ah, bh, ad, bd), inter(ab, bb, ad, bd), inter(ab, bb, ag, bg)]
    return [((x + 0.5) * r, (y + 0.5) * r) for x, y in pts]


def _plausible(q, taille, rapport):
    # Quadrilatère convexe, assez grand, aux proportions proches de celles attendues
    if q is None:
        return False
    p = np.array(q)
    cote = lambda i, j: np.hypot(*(p[i] - p[j]))
    larg = (cote(0, 1) + cote(3, 2)) / 2
    haut = (cote(0, 3) + cote(1, 2)) / 2
    aire = 0.5 * abs(sum(p[i, 0] * p[(i + 1) % 4, 1] - p[(i + 1) % 4, 0] * p[i, 1] for i in range(4)))
    if aire < 0.08 * taille[0] * taille[1] or haut <= 0:
        return False
    return rapport is None or abs(larg / haut / rapport - 1) < 0.15


def coins_clairs(img, rapport=None):
    """Coins de la zone claire (feuille, ou intérieur d'une carte) par droites
    ajustées sur ses 4 bords. Deux essais, le premier plausible est retenu :
    1. éclairage corrigé (image divisée par sa moyenne locale large : ombres et
       dégradés effacés), zone claire entourée d'un bord plus sombre ;
    2. seuil d'Otsu global (feuille claire sur une table sombre).
    rapport : largeur / hauteur attendue de la zone (None : non vérifié)."""
    r = max(1, -(-max(img.size) // COINS_CIBLE))  # réduction entière vers ~COINS_CIBLE px de côté
    small = np.asarray(img.reduce(r), np.float32)
    essais = []
    fond = L.moyenne_locale(small, max(6, min(small.shape) // 10))
    norm = np.clip(small / np.maximum(fond, 1) * 128, 0, 255)
    essais.append(composante_interieure(norm > 0.92 * 128))
    essais.append(composante_interieure(small > _otsu(small)))
    for m in essais:
        q = _quad(m, r)
        if _plausible(q, img.size, rapport):
            return q
    return _quad(essais[-1], r)


def composer_mot(mot, haut, ex, par52, rng, casse):
    """Image du mot (fond 255, encre sombre) et rangée de la ligne de base."""
    elems = []
    for ch in mot:
        if casse == "maj":
            cls, h, desc = ord(ch.upper()) - 65, 1.0, 0.0
        else:
            cls = 26 + ord(ch) - 97
            h, desc = (0.6, 0.0) if ch in X_HAUT else (1.0, 0.0) if ch in HAMPES else (1.0, 0.4) if ch in JAMBAGES else (0.85, 0.0)
        i = rng.choice(par52[cls])
        g = Image.fromarray(255 - ex[i]).crop(Image.fromarray(ex[i]).getbbox())
        hh = max(4, int(haut * h * rng.uniform(0.92, 1.06)))
        g = g.resize((max(3, int(g.width * hh / g.height)), hh), Image.BILINEAR)
        elems.append((np.asarray(g.rotate(rng.uniform(-5, 5), expand=True, fillcolor=255)), desc))
    esp = [int(haut * rng.choice([0.0, 0.08, 0.14, 0.2, 0.3], p=[.08, .25, .32, .23, .12])) for _ in elems]
    W = sum(e.shape[1] for e, _ in elems) + sum(esp[1:]) + 4
    base = int(haut * 1.15)
    H = base + int(haut * 0.55)
    out = np.full((H, W), 255, np.uint8)
    x = 2
    for k, (e, desc) in enumerate(elems):
        if k:
            x += esp[k]
        y = base + int(desc * haut) - e.shape[0] + int(rng.integers(-1, 2))
        y = max(0, min(H - e.shape[0], y))
        out[y : y + e.shape[0], x : x + e.shape[1]] = np.minimum(out[y : y + e.shape[0], x : x + e.shape[1]], e)
        x += e.shape[1]
    return out, base


def ecrire(sup, mots, ex, par52, rng, casse):
    img = sup["img"].copy()
    for mot, (x0, x1, yl) in zip(mots, sup["lignes"]):
        haut = sup["ecart"] * rng.uniform(0.42, 0.55)
        m, base = composer_mot(mot, haut, ex, par52, rng, casse)
        if m.shape[1] > x1 - x0 - 6:  # mot trop long pour la ligne : écriture resserrée
            m = np.asarray(Image.fromarray(m).resize((x1 - x0 - 6, m.shape[0]), Image.BILINEAR))
        encre = rng.integers(25, 60)
        m = np.where(m < 200, encre + (m.astype(int) * (255 - encre)) // 255, 255).astype(np.uint8)
        x = x0 + int(rng.integers(2, max(3, min(14, x1 - x0 - m.shape[1]))))
        y = yl - 3 - base
        zone = np.asarray(img.crop((x, y, x + m.shape[1], y + m.shape[0])))
        img.paste(Image.fromarray(np.minimum(zone, m)), (x, y))
    return img


def photographier(img, sup, rng):
    W, H = CAPTEUR
    hp = rng.uniform(*sup["hauteur_photo"])
    wp = hp * img.width / img.height
    cx, cy = W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-40, 40)
    j = lambda: rng.uniform(-0.04, 0.04) * hp
    dst = [(cx - wp / 2 + j(), cy - hp / 2 + j()), (cx + wp / 2 + j(), cy - hp / 2 + j()),
           (cx + wp / 2 + j(), cy + hp / 2 + j()), (cx - wp / 2 + j(), cy + hp / 2 + j())]
    coef = homographie([(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)], dst)
    out = img.transform((W, H), Image.PERSPECTIVE, coef, Image.BILINEAR, fillcolor=int(rng.integers(40, 90)))
    a = np.asarray(out, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    a = a * (0.6 + 0.4 * (xx / W * rng.uniform(0.2, 1) + yy / H * rng.uniform(0, 0.6)))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.2)))
    a = np.asarray(out, np.float32) + rng.normal(0, 5, (H, W))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def caler_ligne(sombre, x0, x1, yl, e):
    """Position réelle du trait imprimé d'une ligne, cherchée à ±0,45 écart de la
    position du gabarit (impression à une autre échelle, image redessinée).
    Le trait est la rangée sombre sur au moins 70 % de la moitié droite de la
    ligne, que l'écriture couvre rarement. Sans trait net, position du gabarit."""
    ya, yb = int(yl - 0.45 * e), int(yl + 0.45 * e)
    xa, xb = int(x0 + 0.5 * (x1 - x0)), int(x1 - 0.05 * (x1 - x0))
    zone = sombre[max(0, ya) : yb, xa:xb]
    if zone.size == 0:
        return yl
    v = np.percentile(zone, 30, axis=1)
    k = int(np.argmax(v))
    return max(0, ya) + k if v[k] > 6 else yl


def effacer_trait(a, sombre, fond, xa, xb, yl, r=7):
    """Efface le trait imprimé suivi colonne par colonne : dans chaque colonne, la
    rangée la plus sombre à ±r pixels de yl (trait éventuellement incliné), et ses
    voisines, sont remplacées par le fond local."""
    ya, yb = max(0, yl - r), min(a.shape[0], yl + r + 1)
    xa, xb = max(0, xa), min(a.shape[1], xb)
    zone = sombre[ya:yb, xa:xb]
    if zone.size == 0:
        return
    ok = zone.max(0) > 6
    xs = np.arange(xa, xb)[ok]
    ys = ya + np.argmax(zone, 0)[ok]
    for d in range(-2, 3):
        yy = np.clip(ys + d, 0, a.shape[0] - 1)
        a[yy, xs] = fond[yy, xs]


CHRONO = {}  # durée de chaque étape de lire(), pour les mesures


def _chrono(nom, t0):
    CHRONO[nom] = CHRONO.get(nom, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def lire(p, sup, photo, ref):
    t0 = time.perf_counter()
    tc = t0
    pr = np.array(ref)
    rapport = (np.hypot(*(pr[0] - pr[1])) + np.hypot(*(pr[3] - pr[2]))) / (np.hypot(*(pr[0] - pr[3])) + np.hypot(*(pr[1] - pr[2])))
    coef = homographie(coins_clairs(photo, rapport=rapport), ref)
    tc = _chrono("repérage de la carte", tc)
    w, h = sup["img"].size
    page = photo.transform((w, h), Image.PERSPECTIVE, coef, Image.BILINEAR)
    tc = _chrono("redressement", tc)
    t1 = time.perf_counter()
    res = []
    e = sup["ecart"]
    a = np.asarray(page, np.float32)
    sombre = L.moyenne_locale(a, 15) - a
    fond = L.moyenne_locale(a, 15)
    a = a.copy()
    for x0, x1, yl in sup["lignes"]:
        yl = caler_ligne(sombre, x0, x1, yl, e)
        effacer_trait(a, sombre, fond, x0 - 4, x1 + 6, yl)
    page = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    tc = _chrono("recalage + effacement traits", tc)
    for x0, x1, yl in sup["lignes"]:
        yl = caler_ligne(sombre, x0, x1, yl, e)
        band = page.crop((x0 - 4, int(yl - BANDE_HAUT * e), x1 + 6, int(yl + BANDE_BAS * e)))
        L.BANDE = (0, 0, band.width, band.height)
        G, n = L.lire_ligne(p, 0.65, band)
        tc = _chrono("lecture des lignes (OCR)", tc)
        res.append(L.mots_candidats(G, n))
        tc = _chrono("dictionnaire", tc)
    return res, page, (t1 - t0, time.perf_counter() - t1)


def main():
    from train import charger
    nom, casse = sys.argv[1], sys.argv[2]
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[3]).items()}
    nseeds = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 20
    demo = "--demo" in sys.argv
    global HAUTEUR_CARTE
    if "--pres" in sys.argv:  # carte photographiée de près : 85 à 95 % de la hauteur
        HAUTEUR_CARTE = (1650, 1850)
    sup = _support(nom)
    if nom != "page24":
        sup["hauteur_photo"] = HAUTEUR_CARTE
    nmots = len(sup["lignes"])
    # Coins de référence : même méthode appliquée au modèle vierge posé sur un fond sombre
    W, H = sup["img"].size
    fond = Image.new("L", (W + 400, H + 400), 60)
    fond.paste(sup["img"], (200, 200))
    ref = [(x - 200, y - 200) for x, y in coins_clairs(fond)]
    ex, ey = charger(False, "les2")
    ex, ey = ex.numpy(), ey.numpy()
    par52 = {k: np.nonzero(ey == k)[0] for k in range(52)}
    rng = np.random.default_rng(41)
    st = dict(mots=0, top1=0, top3=0, ok=0, faux=0, ta=0.0, tb=0.0)
    for f in range(nseeds):
        mots = mnemonique(rng.bytes(16 if nmots == 12 else 32))
        img = ecrire(sup, mots, ex, par52, rng, casse)
        photo = photographier(img, sup, rng)
        res, page, (ta, tb) = lire(p, sup, photo, ref)
        if demo and f == 0:
            photo.save(f"demo_{nom}_{casse}_photo.png"); page.save(f"demo_{nom}_{casse}_page.png")
        st["ta"] += ta; st["tb"] += tb
        for (c, s), mot in zip(res, mots):
            top = [c[i] for i in np.argsort(-s)[:3]]
            st["mots"] += 1; st["top1"] += top[0] == mot; st["top3"] += mot in top
        final, _ = L.corriger_seed(res)
        st["ok"] += final == mots
        st["faux"] += final != mots and checksum_ok(final)
    n = st["mots"]
    print(f"{nom:8s} {casse}  mot juste {st['top1']/n*100:5.1f} %  3 variantes {st['top3']/n*100:5.1f} %  "
          f"seeds exactes {st['ok']}/{nseeds}  valides mais fausses {st['faux']}  "
          f"(x86 : redressement {st['ta']/nseeds:.2f} s, lignes {st['tb']/nseeds:.2f} s)", flush=True)


if __name__ == "__main__":
    main()
