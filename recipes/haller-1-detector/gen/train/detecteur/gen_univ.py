"""Pages simulées annotées pour le détecteur universel (cartes de toutes sortes, listes sur tous supports).

Complète gen_det.py (dont il reprend l'écriture, le cadrage et la prise de vue) :
  - cartes procédurales : 12/18/20/24/25/33 cases, 1 à 3 colonnes, numérotation par colonne ou par
    rangée, numéros imprimés de styles variés (1. 01 #1 1) cerclé, en case, en exposant), cases en
    trait plein, pointillé, tirets, rectangles, fond gris, sans case ; titre et logo factices ; papier
    blanc, crème, coloré, plastique, métal gravé ; carte en partie remplie (12 mots sur 24 cases) ;
  - modèles de cartes du commerce (data/modeles_cartes, cases annotées dans cases.json) ;
  - la carte SeedSigner (gabarit de gen_sim) ;
  - listes sur supports variés : blanc, ligné, ligné avec marge, Séyès, petits carreaux, quadrillé,
    millimétré, pointillé (dot grid), carnet, Post-it, enveloppe, ticket, papier coloré ou recyclé,
    papier plié ou froissé, plaque de métal poinçonnée ; 1 à 3 colonnes, écrites par colonne ou par
    rangée, avec ou sans numéros ;
  - encres : bille, feutre, crayon ;
  - plusieurs documents dans le champ (autre carte vierge ou remplie, autre feuille).

Vérité (en plus de celle de gen_det) : format (seedsigner, carte, liste), support, encre, modele,
ordre (colonne, rangee), objets « numero » avec leur valeur, liens numéro -> mot, cartes avec leur
noyau ; `ambigu` si un autre document rempli d'une seed est dans le champ (aucune lecture sûre attendue).

Cibles (cibles_univ) : 0 mot, 1 numéro, 2 zone de carte, 3-6 coins, 7 hauteur, 8 noyau de carte
(zone réduite, sépare les cartes jointives), 9 lien numéro-mot (bande entre le numéro et son mot),
10 mot coupé ou caché (en partie hors du cadre, ou masqué par un doigt, un stylo, un reflet, une ombre dure),
11 masque d'exclusion (mot caché à 90 % ou plus).

    python3 gen_univ.py train SORTIE.npz NOMBRE GRAINE [--test]
    python3 gen_univ.py banc DOSSIER NOMBRE GRAINE [--test] [--format F] [--support S] [--style S] [--defaut D]
--test : écritures, polices, modèles de cartes et familles de supports réservés à l'évaluation.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import gen_det as D  # noqa: E402
from gen_det import G, mnemonique  # noqa: E402

DEFAUTS = D.DEFAUTS
POIDS_DEFAUTS = list(D.POIDS_DEFAUTS)
if os.environ.get("UNIV_ROT"):  # itération 2 : plus de cartes retournées et de rotations fortes
    POIDS_DEFAUTS[DEFAUTS.index("rotation")] = 14
    POIDS_DEFAUTS[DEFAUTS.index("retournee")] = 10
REEL = D.REEL   # UNIV_REEL=1 : distributions des photos réelles (distributions/README.md)
if REEL:   # réel : inclinaison 0,8° [0-5,9] ; quelques cas forts et retournés, en proportion
    POIDS_DEFAUTS[DEFAUTS.index("rotation")] = 4
    POIDS_DEFAUTS[DEFAUTS.index("retournee")] = 5
    POIDS_DEFAUTS[DEFAUTS.index("loin")] = 4
STYLES = D.STYLES
SEED_LENGTHS = (12, 15, 18, 21, 24)

# Supports de liste. Familles réservées à l'évaluation (jamais générées en entraînement) : pointillé,
# millimétré, enveloppe.
SUPPORTS_TRAIN = ["blanc", "ligne", "ligne_marge", "seyes", "petits_carreaux", "quadrille", "carnet", "postit",
                  "ticket", "colore", "recycle", "plie", "froisse", "metal"]
POIDS_SUPPORTS_TRAIN = [14, 10, 7, 9, 9, 5, 6, 5, 3, 4, 4, 5, 5, 2]
SUPPORTS_TEST = SUPPORTS_TRAIN + ["pointille", "millimetre", "enveloppe"]
POIDS_SUPPORTS_TEST = POIDS_SUPPORTS_TRAIN + [8, 5, 4]
# Essais comparables : supports lignés et à carreaux retirés de l'entraînement (UNIV_SANS_LIGNES=1)
LIGNES_CARREAUX = ("ligne", "ligne_marge", "seyes", "petits_carreaux", "quadrille", "carnet")
if os.environ.get("UNIV_SANS_LIGNES"):
    POIDS_SUPPORTS_TRAIN = [0 if s in LIGNES_CARREAUX else w for s, w in zip(SUPPORTS_TRAIN, POIDS_SUPPORTS_TRAIN)]
CATEGORIE_SUPPORT = {"blanc": "blanc", "colore": "blanc", "recycle": "blanc", "plie": "blanc", "froisse": "blanc",
                     "ligne": "ligné", "ligne_marge": "ligné", "carnet": "ligné", "ticket": "autre",
                     "seyes": "grands carreaux", "petits_carreaux": "petits carreaux", "quadrille": "petits carreaux",
                     "millimetre": "autre", "pointille": "pointillé", "postit": "autre", "enveloppe": "autre",
                     "metal": "autre", "carte": "carte"}
ENCRES = ["bille", "feutre", "crayon"]

FONTS_IMPRIME = [f for f in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf") if os.path.exists(f)]
MOTS_TITRE = ["RECOVERY", "SEED", "PHRASE", "BACKUP", "WALLET", "WORDS", "SECRET", "MNEMONIC", "KEEP", "SAFE", "CARD",
              "SHEET", "BITCOIN", "VAULT", "COLD", "STORAGE", "NO", "DATE", "NAME", "PASSPHRASE", "FINGERPRINT"]

MODELES = os.environ.get("MODELES_CARTES", os.path.join(ICI, "..", "..", "data", "modeles_cartes"))


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


COULEURS_LIGNES = [(80, 120, 220), (150, 180, 230), (120, 120, 120), (170, 170, 170), (100, 180, 100),
                   (240, 150, 60), (150, 110, 200), (60, 60, 60), (110, 160, 210)]


def gris_ligne(rng, clair=False):
    c = COULEURS_LIGNES[rng.integers(len(COULEURS_LIGNES))]
    g = lum(c)
    return float(np.clip(g + (rng.uniform(10, 50) if clair else rng.uniform(-20, 20)), 40, 225))


# ---------- encre ----------

class Encre:
    """Encre d'une page : niveau, épaisseur ajoutée, texture (crayon), sauts (bille)."""
    def __init__(self, rng, kind=None):
        self.kind = kind or str(rng.choice(ENCRES, p=[0.6, 0.25, 0.15]))
        if self.kind == "bille":
            self.niveau, self.epais = rng.uniform(15, 75), int(rng.choice([-1, 0, 0, 0, 1]))
        elif self.kind == "feutre":
            self.niveau, self.epais = rng.uniform(5, 40), int(rng.choice([1, 1, 2]))
        else:
            self.niveau, self.epais = rng.uniform(85, 150), int(rng.choice([-1, 0, 0]))

    def poser(self, page, m, x, y, rng):
        m = np.asarray(m, np.float32) / 255
        h, w = m.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
        cover = 1 - sub
        if self.kind == "crayon":
            cover = cover * np.clip(rng.normal(0.85, 0.25, sub.shape), 0.2, 1)
        elif self.kind == "bille" and rng.random() < 0.3:
            cover = cover * np.clip(rng.normal(1, 0.25, sub.shape), 0.4, 1)
        zone = page[y0:y1, x0:x1]
        page[y0:y1, x0:x1] = np.minimum(zone, zone * (1 - cover) + self.niveau * cover)


def poser(page, m, x, y, encre, rng, objets, info):
    encre.poser(page, m, x, y, rng)
    ys, xs = np.nonzero(np.asarray(m) < 128)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
    o = dict(info, quad=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    objets.append(o)
    return o


def ecrire_mot(main, texte, haut, rng, maxw=None):
    m, base = main.composer(texte.upper() if main.font is None else texte, haut, rng, main.pente + rng.normal(0, 1))
    if maxw and m.shape[1] > maxw:
        f = maxw / m.shape[1]
        m = np.asarray(Image.fromarray(m).resize((max(4, int(m.shape[1] * f)), max(5, int(m.shape[0] * f)))))
        base = int(base * f)
    return m, base


# ---------- altérations d'un mot (itération 2) ----------
# Mots pâles en partie, stylo qui saute, grand blanc dans le mot (la boîte cible couvre le mot entier), mot barré
# puis réécrit à droite ou au-dessus (seul le mot réécrit est un mot ; le mot barré reste du fond). Tirage propre au
# générateur : aucun jeu red team ni de test n'est relu.
ALTERATIONS = ["pale", "saut", "blanc", "rature_droite", "rature_dessus"]
POIDS_ALTERATIONS = [3, 2, 2, 2, 2]
P_ALTERE = float(os.environ.get("UNIV_ALTERE", 0))   # part des pages dont 1 à 3 mots sont altérés
P_VERSO = float(os.environ.get("UNIV_VERSO", 0))     # part des listes avec le texte du verso vu par transparence
P_MULTI = float(os.environ.get("UNIV_MULTI", 0))     # part des listes écrites sur 3 à 6 colonnes
_bip39 = []


def bip39():
    if not _bip39:
        _bip39.extend(open(os.path.join(ICI, "..", "bip39_english.txt")).read().split())
    return _bip39


def choisir_alterations(rng, n):
    """{position: altération} d'une page (vide pour la plupart des pages)."""
    if n <= 0 or rng.random() >= P_ALTERE:
        return {}
    ks = rng.choice(n, int(rng.integers(1, 4)), replace=False)
    p = np.array(POIDS_ALTERATIONS, float)
    return {int(k): str(rng.choice(ALTERATIONS, p=p / p.sum())) for k in ks}


def barrer(m, rng):
    """Mot barré : un ou deux traits, un gribouillis en zigzag ou une croix."""
    im = Image.fromarray(np.asarray(m, np.uint8).copy())
    d = ImageDraw.Draw(im)
    h, w = m.shape
    ep = max(2, int(h * rng.uniform(0.06, 0.13)))
    kind = rng.choice(["trait", "trait", "zigzag", "croix"])
    if kind == "trait":
        for _ in range(int(rng.integers(1, 3))):
            ya = h * rng.uniform(0.35, 0.7)
            d.line([(0, ya + rng.normal(0, h * 0.06)), (w, ya + rng.normal(0, h * 0.06))], fill=0, width=ep)
    elif kind == "zigzag":
        pts, x = [], 0.0
        while x < w:
            pts.append((x, h * rng.uniform(0.2, 0.8)))
            x += h * rng.uniform(0.15, 0.4)
        d.line(pts, fill=0, width=ep)
    else:
        d.line([(0, h * 0.2), (w, h * 0.85)], fill=0, width=ep)
        d.line([(0, h * 0.85), (w, h * 0.2)], fill=0, width=ep)
    return np.asarray(im)


def alterer(m, kind, haut, rng):
    """(image dessinée, image de la boîte) d'un mot pâle en partie, au stylo qui saute ou coupé d'un grand blanc."""
    m = np.asarray(m, np.float32)
    h, w = m.shape
    if kind == "blanc":
        ink = (255 - m).sum(0)
        lo, hi = int(0.3 * w), max(int(0.3 * w) + 1, int(0.7 * w))
        cut = lo + int(np.argmin(ink[lo:hi]))
        gap = np.full((h, int(haut * rng.uniform(1.0, 2.2))), 255, np.float32)
        m = np.concatenate([m[:, :cut], gap, m[:, cut:]], 1)
        return m, m
    box = m.copy()
    if kind == "pale":
        span = int(w * rng.uniform(0.25, 0.5))
        side = rng.choice(["debut", "fin", "fin", "deux"])
        g = rng.uniform(0.15, 0.4)
        zones = {"debut": [(0, span)], "fin": [(w - span, w)], "deux": [(0, span // 2), (w - span // 2, w)]}[side]
        for a, b in zones:
            m[:, a:b] = 255 - (255 - m[:, a:b]) * g
    else:  # saut : bandes sans encre, encre irrégulière
        for _ in range(int(rng.integers(2, 5))):
            x = int(rng.uniform(0, w))
            m[:, x:x + max(2, int(haut * rng.uniform(0.04, 0.12)))] = 255
        m = 255 - (255 - m) * np.clip(rng.normal(0.8, 0.2, (1, w)), 0.3, 1)
    return m, box


def poser_mot(a, main, mot, haut, x, yb, encre, rng, objets, info, maxw=None, alt=None, rot=0.0):
    """Écrit `mot` sur la ligne de base yb à partir de x (altéré selon `alt`) ; objet du mot lisible ou None."""
    m, base = ecrire_mot(main, mot, haut, rng, maxw=maxw)
    if rot:
        m = np.asarray(Image.fromarray(m).rotate(rot, expand=False, fillcolor=255))
    if alt in ("pale", "saut", "blanc"):
        m, box = alterer(m, alt, haut, rng)
        encre.poser(a, m, x, int(yb - base), rng)
        ys, xs = np.nonzero(box < 128)
        if len(xs) == 0:
            return None
        y = int(yb - base)
        o = dict(info, alteration=alt, quad=[(x + xs.min(), y + ys.min()), (x + xs.max() + 1, y + ys.min()),
                                              (x + xs.max() + 1, y + ys.max() + 1), (x + xs.min(), y + ys.max() + 1)])
        objets.append(o)
        return o
    if alt in ("rature_droite", "rature_dessus"):
        faux = mot
        while faux == mot:
            faux = bip39()[int(rng.integers(2048))]
        mf, bf = ecrire_mot(main, faux, haut * rng.uniform(0.85, 1.0), rng)
        mf = barrer(mf, rng)
        if alt == "rature_droite":
            gap = int(haut * rng.uniform(0.3, 0.9))
            if maxw and mf.shape[1] + gap + m.shape[1] > 1.3 * maxw:
                f = 1.3 * maxw / (mf.shape[1] + gap + m.shape[1])
                mf = np.asarray(Image.fromarray(mf).resize((max(4, int(mf.shape[1] * f)), max(5, int(mf.shape[0] * f)))))
                m = np.asarray(Image.fromarray(m).resize((max(4, int(m.shape[1] * f)), max(5, int(m.shape[0] * f)))))
                bf, base, gap = int(bf * f), int(base * f), int(gap * f)
            encre.poser(a, mf, x, int(yb - bf), rng)
            return poser(a, m, x + mf.shape[1] + gap, int(yb - base), encre, rng, objets,
                         dict(info, alteration=alt))
        encre.poser(a, mf, x, int(yb - bf), rng)
        s = rng.uniform(0.6, 0.85)
        m = np.asarray(Image.fromarray(m).resize((max(4, int(m.shape[1] * s)), max(5, int(m.shape[0] * s)))))
        y2 = yb - haut * rng.uniform(0.85, 1.15)
        x2 = x + int(rng.uniform(0, 0.3) * mf.shape[1])
        return poser(a, m, x2, int(y2 - base * s), encre, rng, objets, dict(info, alteration=alt))
    return poser(a, m, x, int(yb - base), encre, rng, objets, info)


def verso(a, main, rng):
    """Texte du verso vu par transparence : mots écrits au dos (miroir, parfois aussi retourné), pâles et flous."""
    H, W = a.shape
    dos = np.full((H, W), 255, np.float32)
    haut = rng.uniform(0.03, 0.06) * W
    ncol = int(rng.integers(1, 3))
    y = rng.uniform(0.05, 0.5) * H
    while y < H - haut:
        for c in range(ncol):
            if rng.random() < 0.85:
                m, base = ecrire_mot(main, bip39()[int(rng.integers(2048))], haut, rng)
                x = int(rng.uniform(0.05, 0.1) * W + c * W / ncol)
                Encre(rng, "bille").poser(dos, m, x, int(y - base), rng)
        y += haut * rng.uniform(1.5, 2.2)
    dos = dos[:, ::-1] if rng.random() < 0.6 else dos[::-1]
    im = Image.fromarray(dos.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 3.0)))
    cover = (255 - np.asarray(im, np.float32)) / 255 * rng.uniform(0.12, 0.35)
    return a * (1 - cover)


# ---------- texte imprimé ----------

def texte_imprime(txt, taille, font=None, gras=False):
    f = ImageFont.truetype(font or FONTS_IMPRIME[0], max(6, int(taille)))
    bb = f.getbbox(txt, anchor="ls")
    W, H = bb[2] - bb[0] + 4, int(taille * 1.6)
    im = Image.new("L", (W, H), 255)
    ImageDraw.Draw(im).text((2 - bb[0], int(taille * 1.2)), txt, font=f, fill=0, anchor="ls")
    if gras:
        im = im.filter(ImageFilter.MinFilter(3))
    return np.asarray(im), int(taille * 1.2)


def imprimer(page, m, x, y, niveau=30.0):
    G.encrer(page, m, x, y, niveau)
    ys, xs = np.nonzero(np.asarray(m) < 128)
    if len(xs) == 0:
        return None
    return [(x + xs.min(), y + ys.min()), (x + xs.max() + 1, y + ys.min()), (x + xs.max() + 1, y + ys.max() + 1),
            (x + xs.min(), y + ys.max() + 1)]


def trait(page, p0, p1, gris, ep=2, style="plein", periode=10):
    im = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    if style == "plein":
        d.line([p0, p1], fill=int(gris), width=max(1, int(ep)))
    else:
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(1, int(L / periode))
        on = 0.25 if style == "pointille" else 0.6
        for i in range(n):
            a, b = i / n, (i + on) / n
            q0 = (p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]))
            q1 = (p0[0] + b * (p1[0] - p0[0]), p0[1] + b * (p1[1] - p0[1]))
            if style == "pointille":
                r = max(1, ep * 0.7)
                d.ellipse([q0[0] - r, q0[1] - r, q0[0] + r, q0[1] + r], fill=int(gris))
            else:
                d.line([q0, q1], fill=int(gris), width=max(1, int(ep)))
    page[:] = np.asarray(im, np.float32)


def rect(page, box, gris, ep=2, fond=None, rayon=0):
    im = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    if rayon:
        d.rounded_rectangle(box, radius=rayon, outline=int(gris), width=max(1, int(ep)),
                            fill=None if fond is None else int(fond))
    else:
        d.rectangle(box, outline=int(gris), width=max(1, int(ep)), fill=None if fond is None else int(fond))
    page[:] = np.asarray(im, np.float32)


def logo(page, rng, cx, cy, r, gris):
    im = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    k = int(rng.integers(4))
    if k == 0:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=int(gris), width=max(2, int(r / 6)))
    elif k == 1:
        d.regular_polygon((cx, cy, r), int(rng.integers(3, 8)), rotation=float(rng.uniform(0, 60)), fill=int(gris))
    elif k == 2:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(gris))
        d.ellipse([cx - r / 2, cy - r / 2, cx + r / 2, cy + r / 2], fill=255)
    else:
        for i in range(3):
            d.rectangle([cx - r + i * r * 0.7, cy - r, cx - r + i * r * 0.7 + r * 0.45, cy + r], fill=int(gris))
    page[:] = np.asarray(im, np.float32)


# ---------- papier ----------

def papier(rng, W, H, kind):
    """Fond de page : (tableau float, ton du papier)."""
    if kind == "colore":
        ton = rng.uniform(150, 225)
    elif kind in ("recycle", "enveloppe"):
        ton = rng.uniform(170, 215)
    elif kind == "postit":
        ton = lum((250, 240, 120)) + rng.uniform(-25, 5)
    elif kind == "metal":
        ton = rng.uniform(120, 190)
    elif kind == "ticket":
        ton = rng.uniform(215, 245)
    else:
        ton = rng.uniform(200, 245)
    a = np.full((H, W), ton, np.float32)
    if kind in ("recycle", "enveloppe"):  # fibres et grain
        a += rng.normal(0, rng.uniform(3, 8), (H, W))
        for _ in range(int(rng.integers(50, 300))):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            a[int(y) % H, int(x) % W] -= rng.uniform(20, 80)
    if kind == "metal":  # brossé
        a += np.repeat(rng.normal(0, rng.uniform(4, 12), (H, 1)), W, 1)
        a += rng.normal(0, 4, (H, W))
    return a, ton


def plier_froisser(a, rng, kind):
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "plie":
        for _ in range(int(rng.integers(1, 3))):
            t = rng.uniform(0, math.pi)
            c = (rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H)
            dist = (xx - c[0]) * math.cos(t) + (yy - c[1]) * math.sin(t)
            a *= np.where(dist > 0, rng.uniform(0.82, 0.97), 1.0)
            a -= rng.uniform(15, 50) * np.exp(-dist ** 2 / (2 * rng.uniform(1.5, 4) ** 2))
    else:  # froissé : ombrage par un champ lisse
        cell = rng.uniform(60, 140)
        f = rng.normal(0, 1, (int(H / cell) + 2, int(W / cell) + 2))
        f = np.asarray(Image.fromarray(((f - f.min()) / (np.ptp(f) + 1e-6) * 255).astype(np.uint8))
                       .resize((W, H), Image.BICUBIC), np.float32) / 255
        gy, gx = np.gradient(f)
        a *= 1 + np.clip(rng.uniform(0.15, 0.4) * cell * (gx + gy), -0.3, 0.2)
    return a


# ---------- supports de liste ----------

def support_liste(rng, kind):
    """Page vierge d'un support : (tableau, dict(pas des lignes en px ou None, y de la 1re ligne, marge gauche,
    taille de page))."""
    mm = G.PAGE[0] / 148
    W, H = G.PAGE
    if kind == "postit":
        W = H = int(76 * mm * rng.uniform(0.9, 1.3))
    elif kind == "ticket":
        W, H = int(rng.uniform(58, 80) * mm), int(rng.uniform(150, 260) * mm)
    elif kind == "enveloppe":
        W, H = int(rng.uniform(160, 230) * mm), int(rng.uniform(110, 160) * mm)
    elif kind == "carnet":
        W, H = int(rng.uniform(90, 148) * mm), int(rng.uniform(140, 210) * mm)
    elif kind == "metal":
        W, H = int(rng.uniform(60, 100) * mm), int(rng.uniform(80, 180) * mm)
    a, ton = papier(rng, W, H, kind if kind in ("colore", "recycle", "postit", "enveloppe", "ticket", "metal") else "")
    info = dict(pas=None, y0=rng.uniform(0.06, 0.16) * H, marge=None, W=W, H=H, ton=ton)
    gl = gris_ligne(rng)
    ep = int(rng.choice([1, 1, 2, 2, 3]))
    if kind in ("ligne", "ligne_marge", "carnet"):
        pas = rng.uniform(7, 10) * mm if kind != "carnet" else rng.uniform(5.5, 8) * mm
        y0 = rng.uniform(0.04, 0.12) * H
        for y in np.arange(y0, H, pas):
            trait(a, (0, y), (W, y), gl, ep)
        info.update(pas=pas, y0=y0)
        if kind == "ligne_marge" or (kind == "carnet" and rng.random() < 0.4):
            xm = rng.uniform(0.08, 0.16) * W
            trait(a, (xm, 0), (xm, H), lum((220, 70, 70)) + rng.uniform(-20, 30), ep + 1)
            info["marge"] = xm
        if kind == "carnet":  # spirale ou trous
            cote = rng.random() < 0.5
            for t in np.arange(rng.uniform(5, 10) * mm, (H if cote else W), rng.uniform(6, 9) * mm):
                r = rng.uniform(1.5, 2.5) * mm
                cx, cy = (rng.uniform(4, 7) * mm, t) if cote else (t, rng.uniform(4, 7) * mm)
                rect(a, [cx - r, cy - r, cx + r, cy + r], rng.uniform(10, 60), 1, fond=rng.uniform(10, 80), rayon=int(r))
    elif kind == "seyes":
        pas = 8 * mm * rng.uniform(0.9, 1.1)
        y0 = rng.uniform(0.04, 0.1) * H
        gfin = gris_ligne(rng, clair=True)
        for y in np.arange(y0, H, pas / 4):
            major = abs(((y - y0) / pas) - round((y - y0) / pas)) < 1e-3
            trait(a, (0, y), (W, y), gl if major else gfin, ep if major else 1)
        xm = rng.uniform(0.12, 0.2) * W
        for x in np.arange(xm, W, pas):
            trait(a, (x, 0), (x, H), gl, 1)
        trait(a, (xm, 0), (xm, H), lum((220, 70, 70)) + rng.uniform(-20, 30), ep + 1)
        info.update(pas=pas, y0=y0, marge=xm)
    elif kind in ("petits_carreaux", "quadrille", "millimetre"):
        pas = {"petits_carreaux": 5, "quadrille": rng.uniform(4, 10), "millimetre": 1}[kind] * mm * rng.uniform(0.95, 1.05)
        if kind == "millimetre":
            gl = lum([(240, 150, 60), (100, 180, 100), (80, 120, 220)][rng.integers(3)])
        x0, y0 = rng.uniform(0, pas), rng.uniform(0, pas)
        for i, y in enumerate(np.arange(y0, H, pas)):
            fort = kind == "millimetre" and i % 5 == 0
            trait(a, (0, y), (W, y), gl - (30 if fort else 0) + (25 if kind == "millimetre" and not fort else 0),
                  2 if fort else 1)
        for i, x in enumerate(np.arange(x0, W, pas)):
            fort = kind == "millimetre" and i % 5 == 0
            trait(a, (x, 0), (x, H), gl - (30 if fort else 0) + (25 if kind == "millimetre" and not fort else 0),
                  2 if fort else 1)
        info.update(pas=pas * (1 if kind != "millimetre" else 5) * (2 if kind == "petits_carreaux" else 1), y0=y0)
        if kind == "quadrille" and pas < 6 * mm:
            info["pas"] = 2 * pas
    elif kind == "pointille":
        pas = rng.uniform(4, 6) * mm
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        r = rng.uniform(0.6, 1.4) * mm / 2
        for y in np.arange(rng.uniform(0, pas), H, pas):
            for x in np.arange(rng.uniform(0, pas), W, pas):
                d.ellipse([x - r, y - r, x + r, y + r], fill=int(gl))
        a = np.asarray(im, np.float32).copy()
        info.update(pas=2 * pas, y0=pas)
    elif kind == "ticket":  # texte imprimé du ticket en haut
        y = 5 * mm
        for _ in range(int(rng.integers(3, 8))):
            t = " ".join(str(rng.choice(MOTS_TITRE)) for _ in range(int(rng.integers(1, 4))))
            m, _ = texte_imprime(t, rng.uniform(2.2, 3.2) * mm, FONTS_IMPRIME[-1])
            imprimer(a, m, int(rng.uniform(2, 8) * mm), int(y), rng.uniform(80, 150))
            y += rng.uniform(4, 5) * mm
        info["y0"] = y + 5 * mm
    elif kind == "enveloppe":  # rabat en diagonale
        g = ton - rng.uniform(15, 35)
        trait(a, (0, 0), (W / 2, H * rng.uniform(0.35, 0.5)), g, 2)
        trait(a, (W, 0), (W / 2, H * rng.uniform(0.35, 0.5)), g, 2)
        info["y0"] = rng.uniform(0.15, 0.4) * H
    if kind in ("plie", "froisse"):
        a = plier_froisser(a, rng, kind)
    return a, info


# ---------- numéros manuscrits tracés (itération 3) ----------
# Chiffres tracés au trait avec des variantes de forme fréquentes sur les vraies listes : « 1 » à drapeau court ou
# long (continental), avec ou sans pied ; « 7 » barré ou non ; « 4 » ouvert ou fermé ; « 9 » à queue droite ou
# courbe (comme un g) ; « 2 » à boucle ; points de contrôle déformés au hasard. Coordonnées (x, y) dans une case de
# largeur ~0,6 et de hauteur 1, y vers le haut.
P_NUM_TRACE = float(os.environ.get("UNIV_NUM", 0))   # part des listes numérotées dont les numéros sont tracés
P_SERRE = float(os.environ.get("UNIV_SERRE", 0))     # part des listes à plusieurs colonnes écrites serrées


def _arc(cx, cy, rx, ry, a0, a1, n=10):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


def formes_chiffre(c, rng):
    """Traits (listes de points) du chiffre c, variante tirée au hasard."""
    v = rng.random()
    if c == "0":
        return [_arc(0.3, 0.5, 0.28, 0.5, 90, 450, 18)]
    if c == "1":
        traits = [[(0.32, 1.0), (0.32, 0.0)]]
        if v < 0.35:
            traits.append([(0.32, 1.0), (0.05, 0.62)])            # drapeau long (continental)
        elif v < 0.8:
            traits.append([(0.32, 1.0), (0.12, 0.82)])            # drapeau court
        if rng.random() < 0.25:
            traits.append([(0.12, 0.0), (0.52, 0.0)])             # pied
        return traits
    if c == "2":
        haut = _arc(0.3, 0.72, 0.27, 0.28, 160, -20, 10)
        bas = [haut[-1], (0.02, 0.0), (0.58, 0.02)]
        if v < 0.3:  # boucle en bas à gauche
            bas = [haut[-1], (0.08, 0.05)] + _arc(0.12, 0.08, 0.08, 0.08, 270, 90, 6) + [(0.58, 0.02)]
        return [haut + bas[1:]]
    if c == "3":
        return [_arc(0.28, 0.75, 0.26, 0.25, 150, -90, 10) + _arc(0.28, 0.25, 0.3, 0.25, 90, -150, 10)]
    if c == "4":
        if v < 0.5:  # fermé
            return [[(0.42, 0.0), (0.42, 1.0), (0.0, 0.32), (0.6, 0.32)]]
        return [[(0.12, 1.0), (0.05, 0.38), (0.6, 0.38)], [(0.45, 0.75), (0.45, 0.0)]]
    if c == "5":
        return [[(0.55, 1.0), (0.1, 1.0), (0.06, 0.58)] + _arc(0.28, 0.3, 0.3, 0.3, 120, -150, 12)]
    if c == "6":
        return [[(0.5, 1.0), (0.18, 0.6)] + _arc(0.3, 0.28, 0.27, 0.28, 180, 540, 16)]
    if c == "7":
        traits = [[(0.0, 0.95), (0.6, 1.0), (0.2, 0.0)]]
        if v < 0.45:
            traits.append([(0.15, 0.5), (0.55, 0.52)])            # barre
        return traits
    if c == "8":
        return [_arc(0.3, 0.75, 0.22, 0.25, -90, 270, 14), _arc(0.3, 0.25, 0.28, 0.25, 90, 450, 14)]
    if c == "9":
        boucle = _arc(0.3, 0.7, 0.26, 0.3, 0, 360, 14)
        if v < 0.45:  # queue courbe (comme un g)
            queue = [(0.56, 0.7), (0.52, 0.2)] + _arc(0.32, 0.15, 0.22, 0.17, 0, -160, 8)
        else:
            queue = [(0.56, 0.7), (0.46, 0.0)]
        return [boucle, queue]
    return []


def tracer_nombre(texte, haut, rng, epais=0, pente=0.0):
    """Image (255 papier) et ligne de base du nombre `texte` (chiffres, « . », « ) », « : », « - », « # ») tracé."""
    hc = haut * rng.uniform(0.85, 1.3)
    larg = rng.uniform(0.75, 1.1)
    trait = max(1, int(round(hc * rng.uniform(0.06, 0.11)))) + max(0, epais)
    pas = hc * rng.uniform(0.55, 0.85) * larg
    W, H = int(pas * (len(texte) + 1) + 2 * trait + 4), int(hc * 1.5 + 2 * trait)
    base = int(hc * 1.2 + trait)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    x = trait + 2
    jit = 0.06
    for ch in texte:
        if ch.isdigit():
            for t in formes_chiffre(ch, rng):
                pts = [(x + (px + rng.normal(0, jit) * 0.5) * hc * larg, base - (py + rng.normal(0, jit) * 0.5) * hc)
                       for px, py in t]
                d.line(pts, fill=0, width=trait, joint="curve")
            x += pas + rng.uniform(-0.05, 0.12) * hc
        elif ch in ".:":
            r = max(1, trait // 2 + 1)
            for yy in ([0.05] if ch == "." else [0.05, 0.5]):
                d.ellipse((x, base - yy * hc - r, x + 2 * r, base - yy * hc + r), fill=0)
            x += 0.35 * hc
        elif ch == ")":
            d.line(_arc(x, base - 0.5 * hc, 0.18 * hc, 0.55 * hc, -80, 80, 8), fill=0, width=trait)
            x += 0.35 * hc
        elif ch == "-":
            d.line([(x, base - 0.45 * hc), (x + 0.35 * hc, base - 0.45 * hc)], fill=0, width=trait)
            x += 0.5 * hc
        elif ch == "#":
            for k in (0.2, 0.45):
                d.line([(x + k * hc, base), (x + (k + 0.1) * hc, base - hc)], fill=0, width=trait)
            for k in (0.3, 0.7):
                d.line([(x, base - k * hc), (x + 0.65 * hc, base - k * hc)], fill=0, width=trait)
            x += 0.8 * hc
        else:
            x += 0.4 * hc
    bb = Image.eval(im, lambda v: 255 - v).getbbox()
    if bb:
        im = im.crop((0, 0, min(W, bb[2] + trait), H))
    sl = math.tan(math.radians(pente))
    if abs(sl) > 1e-3:
        marge = int(abs(sl) * H) + 2
        im2 = Image.new("L", (im.width + 2 * marge, H), 255)
        im2.paste(im, (marge, 0))
        im = im2.transform(im2.size, Image.AFFINE, (1, sl, -sl * base, 0, 1, 0), Image.BILINEAR, fillcolor=255)
    return np.asarray(im), base


def ecrire_liste(mots, main, encre, rng, kind, ordre=None, ncol=None, numeros=None):
    a, inf = support_liste(rng, kind)
    W, H = inf["W"], inf["H"]
    n = len(mots)
    if ncol is None:
        if rng.random() < P_MULTI:  # grille de 3 à 6 colonnes
            ncol = int(rng.choice([3, 4, 6], p=[0.5, 0.35, 0.15]))
            if ncol == 6 and n % 6:
                ncol = 4
        else:
            ncol = int(rng.choice([1, 2, 3], p=[0.55, 0.38, 0.07]))
        if W < 0.75 * H and n >= 18 and ncol == 1 and kind in ("postit", "ticket"):
            ncol = 2
    if os.environ.get("UNIV_NCOL"):            # bancs : nombre de colonnes imposé
        ncol = int(os.environ["UNIV_NCOL"])
    ncol = min(ncol, 6)
    if numeros is None:
        numeros = bool(rng.random() < (0.5 if REEL else 0.6)) if not os.environ.get("UNIV_NUMEROS") else True
    if ordre is None:
        p_rangee = (0.6 if numeros else 0.45) if ncol >= 3 else (0.45 if numeros else 0.2)
        ordre = "rangee" if ncol > 1 and rng.random() < p_rangee else "colonne"
    alts = choisir_alterations(rng, n)
    if rng.random() < P_VERSO:
        a = verso(a, main, rng)
    rows = -(-n // ncol)
    ytop = inf["y0"]
    pas = inf["pas"]
    avail = 0.96 * H - ytop
    if pas:
        mult = max(1, int(round(rng.uniform(1.0, 1.6) * 9 * (G.PAGE[0] / 148) / pas)))
        pas_ecr = pas * mult
        if rows * pas_ecr > avail:
            mult = max(1, int(avail / rows / pas))
            pas_ecr = pas * mult
        if rows * pas_ecr > avail:
            pas_ecr = avail / rows
        haut = min(pas_ecr * rng.uniform(0.5, 0.78), pas * mult * 0.85)
    else:
        haut = rng.uniform(45, 80) * W / G.PAGE[0] * (1 if ncol == 1 else 0.85)
        pas_ecr = haut * rng.uniform(1.45, 2.1)
        if rows * pas_ecr > avail:
            f = avail / (rows * pas_ecr)
            haut, pas_ecr = haut * f, pas_ecr * f
    if REEL and rng.random() < 0.75:
        # réel : mots = 0,056 du petit côté de l'image [0,036-0,096] (généré avant : 0,024) ; la liste occupe le plus
        # souvent la hauteur de la feuille : interligne = part F de la hauteur disponible, F tiré dans [0,75 ; 1]
        F = rng.uniform(0.75, 1.0)
        cible = F * avail / rows
        if pas:
            mult = max(1, int(round(cible / pas)))
            pas_ecr = pas * mult
            if rows * pas_ecr > avail:
                mult = max(1, int(avail / rows / pas))
                pas_ecr = pas * mult
            haut = min(pas_ecr * rng.uniform(0.6, 0.85), pas_ecr * 0.85)
        else:
            pas_ecr = cible
            haut = pas_ecr * rng.uniform(0.6, 0.85)
        haut = min(haut, 0.92 * (W - 0.1 * W) / (ncol * 4.8))   # le mot moyen tient dans sa colonne
    haut = max(haut, 14)
    xg = (inf["marge"] + rng.uniform(0.01, 0.04) * W) if inf["marge"] else rng.uniform(0.05, 0.14) * W
    largeur_col = (W - xg - 0.02 * W) / ncol
    fmt = str(rng.choice(["{}.", "{})", "{}", "{} -", "#{}", "{}:"], p=[0.35, 0.15, 0.25, 0.08, 0.07, 0.1]))
    if main.font is None:  # tracé EMNIST : chiffres, « . » et « ) » seulement
        fmt = fmt if fmt in ("{}.", "{})", "{}") else "{}."
    objets, liens = [], []
    trace = numeros and rng.random() < P_NUM_TRACE
    if trace:
        fmt = str(rng.choice(["{}", "{}.", "{})", "{}:", "{} -", "#{}"], p=[0.5, 0.25, 0.1, 0.05, 0.05, 0.05]))
    colle = numeros and rng.random() < (0.35 if trace else 0.1)    # numéro collé au mot
    serre = ncol > 1 and rng.random() < P_SERRE                     # colonne suivante contre le mot le plus long
    if serre:  # colonnes juste assez larges : la page doit les contenir (mot moyen ~6,5 hauteurs avec son numéro)
        haut = min(haut * rng.uniform(1.0, 1.25), 0.92 * (W - xg) / (ncol * 6.5))
    debut_col, fin_col = {}, {}
    places = [((k // ncol, k % ncol) if ordre == "rangee" else (k % rows, k // rows)) + (k,) for k in range(n)]
    if serre:
        places.sort(key=lambda t: (t[1], t[0]))
    for row, col, k in places:
        mot = mots[k]
        yb = ytop + (row + 1) * pas_ecr - rng.uniform(0.05, 0.2) * haut if pas else \
            ytop + haut * 1.2 + row * pas_ecr + rng.normal(0, haut * 0.03)
        if serre:
            if col not in debut_col:
                debut_col[col] = xg if col == 0 else fin_col.get(col - 1, xg) + haut * rng.uniform(0.05, 0.6)
            x = int(debut_col[col] + rng.normal(0, haut * 0.05))
        else:
            x = int(xg + col * largeur_col + rng.normal(0, haut * 0.06))
        on = None
        if numeros:
            if trace:
                mn, bn = tracer_nombre(fmt.format(k + 1), haut, rng, main.epais, main.pente)
            else:
                mn, bn = main.composer(fmt.format(k + 1), haut, rng, main.pente)
            on = poser(a, mn, x, int(yb - bn), encre, rng, objets, dict(classe="numero", k=k, valeur=k + 1,
                                                                         rang=row, col=col))
            x += mn.shape[1] + int(haut * (rng.uniform(-0.08, 0.12) if colle else rng.uniform(0.25, 0.7)))
        if serre:
            maxw_mot = max(0.98 * W - x, 2.5 * haut)
        else:
            maxw_mot = 0.95 * largeur_col - (x - xg - col * largeur_col)
        om = poser_mot(a, main, mot, haut, x, yb, encre, rng, objets, dict(classe="mot", k=k, rang=row, col=col, mot=mot),
                       maxw=maxw_mot, alt=alts.get(k), rot=rng.normal(0, 0.8))
        if om is not None:
            fin_col[col] = max(fin_col.get(col, 0), max(p[0] for p in om["quad"]))
        if on is not None and om is not None:
            liens.append((on, om))
    if kind == "metal":  # poinçonné : relief
        a = estamper(a, inf["ton"])
    info = dict(colonnes=ncol, numeros=numeros, ordre=ordre, papier=kind, numeros_traces=bool(trace),
                serre=bool(serre), colle=bool(colle))
    return a, objets, liens, info


def estamper(a, ton):
    """Encre transformée en creux poinçonnés (relief clair/sombre)."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    enc = np.asarray(im, np.float32) < ton - 40
    relief = np.asarray(Image.fromarray((enc * 255).astype(np.uint8)).filter(ImageFilter.EMBOSS), np.float32) - 128
    out = a.copy()
    out[enc] = ton - 45
    return np.clip(out + relief * 0.6, 0, 255)


# ---------- cartes procédurales ----------

def spec_carte(rng):
    slots = int(rng.choice([12, 18, 20, 24, 25, 33], p=[0.3, 0.08, 0.06, 0.4, 0.08, 0.08]))
    if slots == 12:
        cols = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.2]))
    elif slots in (18, 33):
        cols = 3
    elif slots == 25:
        cols = int(rng.choice([1, 5], p=[0.2, 0.8]))
        cols = 5 if cols == 5 else 1
    else:
        cols = int(rng.choice([2, 3, 4], p=[0.6, 0.3, 0.1]))
    ordre = "rangee" if cols > 1 and rng.random() < (0.55 if cols >= 3 else 0.25) else "colonne"
    return dict(slots=slots, cols=cols, ordre=ordre,
                case=str(rng.choice(["ligne", "pointille", "tirets", "rect", "arrondi", "fond", "rien", "souligne"],
                                    p=[0.25, 0.12, 0.08, 0.2, 0.1, 0.1, 0.05, 0.1])),
                num=str(rng.choice(["{}.", "{}", "{:02d}", "#{}", "{})", "cercle", "case", "exposant", "{}:"],
                                   p=[0.2, 0.18, 0.15, 0.08, 0.08, 0.08, 0.1, 0.08, 0.05])),
                paysage=bool(rng.random() < 0.4),
                # UNIV_PASTILLE=p (nuit du 20-21/11) : numéros en chiffres clairs sur pastille sombre (carré ou rond
                # plein, comme les cartes Trezor ou blockdyor) sur une part p des cartes ; tirage seulement si posée
                pastille=bool(os.environ.get("UNIV_PASTILLE") and rng.random() < float(os.environ["UNIV_PASTILLE"])),
                matiere=str(rng.choice(["papier", "creme", "colore", "plastique", "metal"], p=[0.5, 0.15, 0.12, 0.13, 0.1])))


def carte_proc(mots, main, encre, rng, spec=None):
    """Carte procédurale : (tableau, objets, liens, coins de la carte, spec)."""
    spec = spec or spec_carte(rng)
    slots, cols = spec["slots"], spec["cols"]
    alts = choisir_alterations(rng, len(mots) if mots else 0)
    rows = -(-slots // cols)
    if spec["paysage"] and cols >= 2:
        W = int(rng.uniform(1300, 1700))
        H = int(W / rng.uniform(1.3, 1.65))
    else:
        W = int(rng.uniform(900, 1300))
        H = int(W * rng.uniform(1.2, 1.55))
        if rows > 13:
            H = int(W * rng.uniform(1.5, 1.9))
    mat = spec["matiere"]
    a, ton = papier(rng, W, H, {"papier": "", "creme": "colore", "colore": "colore", "plastique": "",
                                "metal": "metal"}[mat])
    if mat == "creme":
        a += 225 - ton
        ton = 225
    imp = rng.uniform(10, 90) if mat != "metal" else ton - 60
    font = FONTS_IMPRIME[rng.integers(len(FONTS_IMPRIME))]
    if rng.random() < 0.7:
        rect(a, [6, 6, W - 7, H - 7], imp, int(rng.integers(2, 6)), rayon=int(rng.choice([0, 20, 40])))
    # en-tête
    top = int(rng.uniform(0.08, 0.2) * H)
    if rng.random() < 0.85:
        t = " ".join(str(rng.choice(MOTS_TITRE)) for _ in range(int(rng.integers(1, 4))))
        m, b = texte_imprime(t, rng.uniform(0.025, 0.05) * H, font, gras=rng.random() < 0.5)
        if m.shape[1] < 0.8 * W:
            imprimer(a, m, int(rng.uniform(0.08, 0.5) * (W - m.shape[1])), int(top * 0.25), imp)
        if rng.random() < 0.6:
            logo(a, rng, W * rng.choice([0.12, 0.88]), top * 0.5, top * 0.3, imp)
    bot = H - int(rng.uniform(0.04, 0.14) * H)
    if rng.random() < 0.4:  # pied de page : date, empreinte
        t = str(rng.choice(["DATE: ________", "FINGERPRINT ________", "PASSPHRASE? Y / N", "NAME ______"]))
        m, b = texte_imprime(t, rng.uniform(0.015, 0.025) * H, font)
        if m.shape[1] < 0.8 * W:
            imprimer(a, m, int(0.08 * W), bot + 4, imp)
    mx = rng.uniform(0.05, 0.1) * W
    cw = (W - 2 * mx) / cols
    rh = min((bot - top) / rows, cw * rng.uniform(0.25, 0.45))
    top += ((bot - top) - rows * rh) * rng.uniform(0.2, 0.5)
    num_w = 0 if spec["num"] in ("exposant",) else min(0.3 * cw, rh * rng.uniform(0.45, 0.8)
                                                         * (1.1 if "02" in spec["num"] else 0.9))
    objets, liens = [], []
    gl = imp if rng.random() < 0.5 else gris_ligne(rng)
    n_mots = len(mots) if mots is not None else 0
    hn = min(rh * rng.uniform(0.28, 0.45), max(num_w, 0.3 * rh) * 0.55)
    for s in range(slots):
        r, c = (s // cols, s % cols) if spec["ordre"] == "rangee" else (s % rows, s // rows)
        x0, y0 = mx + c * cw, top + r * rh
        xs0, xs1 = x0 + num_w + rh * 0.12, x0 + cw - rh * 0.25
        ybase = y0 + rh * 0.78
        case = spec["case"]
        if case in ("ligne", "pointille", "tirets"):
            trait(a, (xs0, ybase), (xs1, ybase), gl, int(rng.choice([1, 2, 3])) if s == 0 else 2,
                  {"ligne": "plein", "pointille": "pointille", "tirets": "tirets"}[case], periode=rh * 0.15)
        elif case in ("rect", "arrondi", "fond"):
            box = [xs0 - rh * 0.05, y0 + rh * 0.12, xs1, y0 + rh * 0.92]
            rect(a, box, gl, 2, fond=(ton - rng.uniform(15, 35)) if case == "fond" else None,
                 rayon=int(rh * 0.15) if case == "arrondi" else 0)
        elif case == "souligne":
            m, b = texte_imprime("_" * int((xs1 - xs0) / (rh * 0.3)), rh * 0.5, FONTS_IMPRIME[-1])
            imprimer(a, m[:, : int(xs1 - xs0)], int(xs0), int(ybase - b + 2), imp + 40)
        # numéro imprimé
        f = spec["num"]
        txt = str(s + 1) if f in ("cercle", "case", "exposant") else f.format(s + 1)
        taille = hn if f != "exposant" else hn * 0.6
        m, b = texte_imprime(txt, taille, font)
        nx = x0 + (num_w - m.shape[1]) / 2 if f != "exposant" else xs0 + rh * 0.02
        ny = ybase - b if f != "exposant" else y0 + rh * 0.14
        if f == "cercle":
            rr = taille * 0.75
            cxn, cyn = x0 + num_w / 2, ybase - taille * 0.4
            rect(a, [cxn - rr, cyn - rr, cxn + rr, cyn + rr], imp, 2, rayon=int(rr))
        elif f == "case":
            rr = taille * 0.75
            cxn, cyn = x0 + num_w / 2, ybase - taille * 0.4
            rect(a, [cxn - rr * 1.1, cyn - rr, cxn + rr * 1.1, cyn + rr], imp, 2, fond=ton - 25)
        if spec.get("pastille"):
            # pastille sombre pleine, chiffres couleur du papier ; la boîte du numéro est la pastille
            rr = max(m.shape) * 0.5 + taille * 0.25
            cxn, cyn = nx + m.shape[1] / 2, ny + b - taille * 0.4
            pb = [cxn - rr, cyn - rr, cxn + rr, cyn + rr]
            rect(a, pb, imp, 1, fond=imp, rayon=int(rr) if spec["case"] == "arrondi" else int(rr * 0.15))
            y_, x_ = int(ny), int(nx)
            zone = a[y_:y_ + m.shape[0], x_:x_ + m.shape[1]]
            mm = np.asarray(m)[: zone.shape[0], : zone.shape[1]] < 128
            zone[mm] = ton - 10
            q = [(pb[0], pb[1]), (pb[2], pb[1]), (pb[2], pb[3]), (pb[0], pb[3])]
        else:
            q = imprimer(a, m, int(nx), int(ny), imp)
        on = None
        if q:
            on = dict(classe="numero", k=s, valeur=s + 1, rang=r, col=c, imprime=True, quad=q)
            objets.append(on)
        # mot écrit
        if s < n_mots:
            haut = rh * (rng.uniform(0.5, 0.8) if REEL else rng.uniform(0.38, 0.6))
            maxw = xs1 - xs0 - (rh * 0.4 if f == "exposant" else 0)
            xw = int(xs0 + rng.uniform(0, 0.15) * rh + (rh * 0.35 if f == "exposant" else 0))
            om = poser_mot(a, main, mots[s], haut, xw, ybase - rng.uniform(0.04, 0.15) * rh, encre, rng, objets,
                           dict(classe="mot", k=s, rang=r, col=c, mot=mots[s]), maxw=maxw, alt=alts.get(s))
            if on is not None and om is not None:
                liens.append((on, om))
    if mat == "metal":
        a = estamper(a, ton)
    elif mat == "plastique":  # reflet
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        t = rng.uniform(0, math.pi)
        d = (xx / W - 0.5) * math.cos(t) + (yy / H - 0.5) * math.sin(t) - rng.uniform(-0.3, 0.3)
        a += rng.uniform(20, 60) * np.exp(-d ** 2 / (2 * rng.uniform(0.02, 0.08) ** 2))
    coins = [(0, 0), (W, 0), (W, H), (0, H)]
    return a, objets, liens, coins, spec


# ---------- modèles du commerce ----------

_modeles = {}


def modeles(test):
    """Modèles annotés (cases.json) : réservés à l'évaluation si leur champ `split` vaut test."""
    if test not in _modeles:
        f = os.path.join(MODELES, "cases.json")
        lst = []
        if os.path.exists(f) and not os.environ.get("UNIV_SANS_MODELES"):  # essais comparables : aucun modèle
            for m in json.load(open(f)):
                if (m.get("split") == "test") == test and os.path.exists(os.path.join(MODELES, m["png"])):
                    lst.append(m)
        _modeles[test] = lst
    return _modeles[test]


def carte_modele(mots, main, encre, rng, m):
    """Carte d'un modèle annoté : cases = liste de [x0, y0, x1, y1, numéro] (rectangle d'écriture, ligne de base en
    bas), numéros = boîtes [x0, y0, x1, y1, numéro] des numéros imprimés, carte = [x0, y0, x1, y1]."""
    im = Image.open(os.path.join(MODELES, m["png"])).convert("L")
    a = np.asarray(im, np.float32).copy()
    objets, liens = [], []
    alts = choisir_alterations(rng, len(mots) if mots else 0)
    nums = {int(b[4]): b for b in m.get("numeros", [])}
    for b in m.get("numeros", []):
        objets.append(dict(classe="numero", k=int(b[4]) - 1, valeur=int(b[4]), imprime=True,
                           quad=[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]))
    cases = sorted(m["cases"], key=lambda c: c[4])
    for c in cases:
        k = int(c[4]) - 1
        if mots is None or k >= len(mots):
            continue
        x0, y0, x1, y1 = c[:4]
        rh = y1 - y0
        haut = rh * (rng.uniform(0.65, 0.95) if REEL else rng.uniform(0.55, 0.85))
        om = poser_mot(a, main, mots[k], haut, int(x0 + rng.uniform(0.02, 0.1) * (x1 - x0)),
                       y1 - rng.uniform(0.04, 0.15) * rh, encre, rng, objets, dict(classe="mot", k=k, mot=mots[k]),
                       maxw=(x1 - x0) * 0.95, alt=alts.get(k))
        on = [o for o in objets if o["classe"] == "numero" and o["valeur"] == k + 1]
        if on and om is not None:
            liens.append((on[0], om))
    x0, y0, x1, y1 = m.get("carte", [0, 0, im.width, im.height])
    coins = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if rng.random() < 0.8:  # la carte seule, avec une marge de page ou de table
        mg = int(rng.uniform(0.0, 0.06) * max(x1 - x0, y1 - y0))
        cx0, cy0 = max(0, x0 - mg), max(0, y0 - mg)
        cx1, cy1 = min(a.shape[1], x1 + mg), min(a.shape[0], y1 + mg)
        a = a[cy0:cy1, cx0:cx1].copy()
        for o in objets:
            o["quad"] = [(x - cx0, y - cy0) for x, y in o["quad"]]
        coins = [(x - cx0, y - cy0) for x, y in coins]
    return a, objets, liens, coins


def carte_seedsigner(mots, main, encre, rng):
    n = 24 if mots is not None and len(mots) > 12 else (24 if mots is None and rng.random() < 0.4 else 12)
    carte, objets, coins = D.face_carte(mots, main, rng, n)
    for o in objets:
        if o["classe"] == "numero":
            o["valeur"] = o["k"] + 1
    liens = []
    by = {}
    for o in objets:
        by.setdefault(o["k"], {})[o["classe"]] = o
    for k, d in by.items():
        if "numero" in d and "mot" in d:
            liens.append((d["numero"], d["mot"]))
    return carte, objets, liens, coins


# ---------- occultations ----------

def occulter(a, rng, kinds=None):
    """Occultations réalistes sur la photo (tableau float H x W) : doigt au bord, stylo en travers, reflet, ombre
    dure. Renvoie (photo, masque 0-1 de l'encre rendue illisible)."""
    H, W = a.shape
    cache = np.zeros((H, W), np.float32)
    kinds = kinds or [str(rng.choice(["doigt", "stylo", "reflet", "ombre_dure"], p=[0.35, 0.25, 0.25, 0.15]))
                      for _ in range(int(rng.integers(1, 3)))]
    for k in kinds:
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        if k == "doigt":
            bord = int(rng.integers(4))
            r = rng.uniform(0.05, 0.12) * W
            t = rng.uniform(0.1, 0.9)
            cx, cy = [(t * W, 0), (W, t * H), (t * W, H), (0, t * H)][bord]
            depth = rng.uniform(0.6, 2.2) * r
            ang = [math.pi / 2, math.pi, -math.pi / 2, 0][bord] + rng.normal(0, 0.3)
            ex, ey = cx + math.cos(ang) * depth, cy + math.sin(ang) * depth
            d.line([(cx, cy), (ex, ey)], fill=255, width=int(2 * r))
            d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=255)
            m = np.asarray(im.filter(ImageFilter.GaussianBlur(r * 0.08)), np.float32) / 255
            peau = rng.uniform(110, 200) + 25 * np.asarray(im.filter(ImageFilter.GaussianBlur(r * 0.5)),
                                                           np.float32) / 255
            a = a * (1 - m) + peau * m
            cache = np.maximum(cache, (m > 0.5).astype(np.float32))
        elif k == "stylo":
            cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H
            L, e, t = rng.uniform(0.35, 0.8) * W, rng.uniform(0.015, 0.04) * W, rng.uniform(0, math.pi)
            p0 = (cx - math.cos(t) * L / 2, cy - math.sin(t) * L / 2)
            p1 = (cx + math.cos(t) * L / 2, cy + math.sin(t) * L / 2)
            d.line([p0, p1], fill=255, width=int(e))
            m = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
            g = rng.uniform(20, 220)
            hl = Image.new("L", (W, H), 0)
            ImageDraw.Draw(hl).line([p0, p1], fill=255, width=max(1, int(e * 0.2)))
            a = a * (1 - m) + g * m + 60 * np.asarray(hl, np.float32) / 255 * m
            cache = np.maximum(cache, (m > 0.5).astype(np.float32))
            if rng.random() < 0.6:  # ombre portée du stylo
                sh = np.roll(np.roll(m, int(e * 0.6), 0), int(e * 0.4), 1)
                a = a * (1 - 0.35 * sh * (1 - m))
        elif k == "reflet":
            cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
            rx, ry = rng.uniform(0.04, 0.18) * W, rng.uniform(0.02, 0.1) * H
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
            m = np.asarray(im.filter(ImageFilter.GaussianBlur(min(rx, ry) * 0.5)), np.float32) / 255
            a = a + m * rng.uniform(150, 260)
            cache = np.maximum(cache, (m > 0.6).astype(np.float32))
        else:  # ombre dure
            pts = [(rng.uniform(-0.2, 1.2) * W, rng.uniform(-0.2, 1.2) * H) for _ in range(int(rng.integers(3, 6)))]
            d.polygon(pts, fill=255)
            m = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), np.float32) / 255
            f = rng.uniform(0.12, 0.4)
            a = a * (1 - (1 - f) * m)
            if f < 0.2:
                cache = np.maximum(cache, (m > 0.5).astype(np.float32))
    return np.clip(a, 0, 255), cache


def part_hors_champ(q, W, H):
    """Part de la boîte (quadrilatère) hors du cadre W x H, par échantillonnage."""
    q = np.asarray(q, float)
    u, v = np.meshgrid(np.linspace(0.05, 0.95, 9), np.linspace(0.1, 0.9, 5))
    u, v = u.ravel()[:, None], v.ravel()[:, None]
    pts = (1 - u) * (1 - v) * q[0] + u * (1 - v) * q[1] + u * v * q[2] + (1 - u) * v * q[3]
    out = (pts[:, 0] < 0) | (pts[:, 0] > W) | (pts[:, 1] < 0) | (pts[:, 1] > H)
    return float(out.mean())


# ---------- scène ----------

def coller(scene, item, ox, oy, angle, rng):
    """Pose l'image `item` (tableau) tournée de `angle` degrés, coin haut gauche de son cadre tourné en (ox, oy) ;
    renvoie la transformation des points de l'item vers la scène."""
    h, w = item.shape
    im = Image.fromarray(np.clip(item, 0, 255).astype(np.uint8))
    mask = Image.new("L", (w, h), 255)
    rot = im.rotate(angle, expand=True, resample=Image.BILINEAR, fillcolor=0)
    mrot = mask.rotate(angle, expand=True, fillcolor=0)
    R = np.asarray(rot, np.float32)
    M = np.asarray(mrot, np.float32) / 255
    H, W = scene.shape
    x0, y0 = int(ox), int(oy)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + R.shape[1]), min(H, y0 + R.shape[0])
    if xb > xa and yb > ya:
        sub = scene[ya:yb, xa:xb]
        r = R[ya - y0:yb - y0, xa - x0:xb - x0]
        mm = M[ya - y0:yb - y0, xa - x0:xb - x0]
        scene[ya:yb, xa:xb] = sub * (1 - mm) + r * mm
    t = math.radians(-angle)
    cx, cy = w / 2, h / 2
    ncx, ncy = R.shape[1] / 2 + x0, R.shape[0] / 2 + y0

    def f(pts):
        out = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            out.append((ncx + dx * math.cos(t) - dy * math.sin(t), ncy + dx * math.sin(t) + dy * math.cos(t)))
        return out
    return f


def document(rng, fmt, mots, main, encre, test, support=None):
    """Un document : (tableau, objets, liens, cartes, info)."""
    if fmt == "seedsigner":
        a, ob, li, co = carte_seedsigner(mots, main, encre, rng)
        return a, ob, li, [dict(coins=co, n=len(mots) if mots else 0)], dict(type="carte", modele="seedsigner",
                                                                               papier="carte")
    if fmt == "modele":
        ms = modeles(test)
        m = ms[rng.integers(len(ms))]
        a, ob, li, co = carte_modele(mots, main, encre, rng, m)
        return a, ob, li, [dict(coins=co, n=len(mots) if mots else 0)], dict(type="carte", modele=m["id"],
                                                                               papier="carte")
    if fmt == "carte":
        spec = spec_carte(rng)
        if mots is not None and len(mots) > spec["slots"]:
            spec["slots"] = 24 if len(mots) <= 24 else 33
            spec["cols"] = max(spec["cols"], 2)
        a, ob, li, co, spec = carte_proc(mots, main, encre, rng, spec)
        return a, ob, li, [dict(coins=co, n=len(mots) if mots else 0)], dict(
            type="carte", modele="proc:{slots}x{cols}:{ordre}:{case}:{num}:{matiere}".format(**spec), papier="carte",
            ordre=spec["ordre"], colonnes=spec["cols"], cases=spec["slots"])
    a, ob, li, info = ecrire_liste(mots, main, encre, rng, support)
    return a, ob, li, [], dict(type="liste", **info)


def generer(rng, ex, par, fonts, test, defaut=None, style=None, fmt=None, support=None, scale=1.0):
    defaut = defaut or DEFAUTS[rng.choice(len(DEFAUTS), p=np.array(POIDS_DEFAUTS) / sum(POIDS_DEFAUTS))]
    style = style or STYLES[rng.choice(3, p=np.array(D.POIDS_STYLES) / sum(D.POIDS_STYLES))]
    main = D.Main(style, rng, ex, par, fonts)
    encre = Encre(rng)
    main.epais = int(np.clip(main.epais + encre.epais, -1, 2))
    if fmt is None:
        a_modeles = bool(modeles(test))
        p = np.array([0.2, 0.25 if a_modeles else 0, 0.2, 0.35])
        if REEL:   # réel : cartes SeedSigner 32 %, autres cartes 12 %, listes 57 %
            p = np.array([0.3, 0.1 if a_modeles else 0, 0.08, 0.52])
        fmt = str(rng.choice(["seedsigner", "modele", "carte", "liste"], p=p / p.sum()))
    if fmt == "liste" and support is None:
        sup, poids = (SUPPORTS_TEST, POIDS_SUPPORTS_TEST) if test else (SUPPORTS_TRAIN, POIDS_SUPPORTS_TRAIN)
        support = sup[rng.choice(len(sup), p=np.array(poids) / sum(poids))]
    if fmt == "seedsigner":
        n = 12 if rng.random() < (0.7 if REEL else 0.6) else 24
    elif fmt == "modele":
        ms = modeles(test)
        m = ms[rng.integers(len(ms))]
        cap = max(int(c[4]) for c in m["cases"])
        n = max([s for s in SEED_LENGTHS if s <= cap] or [12])
        if n == 24 and rng.random() < 0.3:
            n = 12
        rng_m = m
    else:
        n = int(rng.choice(SEED_LENGTHS, p=[0.68, 0.02, 0.03, 0.02, 0.25] if REEL else [0.45, 0.07, 0.1, 0.06, 0.32]))
    mots = list(mnemonique(rng.bytes(n * 4 // 3)))
    if fmt == "modele":
        a, ob, li, co = carte_modele(mots, main, encre, rng, rng_m)
        cartes = [dict(coins=co, n=n)]
        info = dict(type="carte", modele=rng_m["id"], papier="carte")
    else:
        a, ob, li, cartes, info = document(rng, fmt, mots, main, encre, test, support)
    for c in cartes:
        c["cible"] = True
    # scène : document seul, ou avec d'autres documents (planche, autre carte, autre feuille)
    autres = defaut == "planche" or rng.random() < 0.08
    ambigu = False
    if autres:
        pieces = [(a, ob, li, cartes, True)]
        for _ in range(int(rng.integers(1, 4 if info["type"] == "carte" else 2))):
            rempli = rng.random() < 0.3
            fm = info.get("modele", "") if info["type"] == "carte" and rng.random() < 0.6 else \
                str(rng.choice(["seedsigner", "carte", "liste"]))
            fm = {"seedsigner": "seedsigner"}.get(fm, "carte" if fm.startswith("proc") else
                                                  ("modele" if fm and fm not in ("liste",) and modeles(test) else fm))
            if fm not in ("seedsigner", "carte", "modele", "liste"):
                fm = "carte"
            if fm == "liste" and not rempli:
                continue
            m2 = list(mnemonique(rng.bytes(16))) if rempli else None
            if fm == "liste":
                b = document(rng, "liste", m2, main, encre, test, str(rng.choice([k for k, w in zip(SUPPORTS_TRAIN, POIDS_SUPPORTS_TRAIN) if w])))
            else:
                b = document(rng, fm, m2, main, encre, test)
            ambigu |= rempli
            pieces.append((b[0], b[1], b[2], b[3], False))
        jointes = defaut == "planche" and info["type"] == "carte"
        a, ob, li, cartes = composer_scene(pieces, rng, jointes)
        a, ob, cartes = recadrer_cible(a, ob, cartes, rng)
        # ambiguë seulement si au moins 6 mots d'un autre document rempli restent dans la scène
        H_, W_ = a.shape
        vus = sum(1 for o in ob if o.get("autre") and o["classe"] == "mot"
                  and 0 <= np.mean([p[0] for p in o["quad"]]) < W_ and 0 <= np.mean([p[1] for p in o["quad"]]) < H_)
        ambigu = vus >= 6
    page = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    est_carte = info["type"] == "carte"
    Hm = D.cadrage(page, defaut, rng, est_carte and not autres)
    img, masque = D.photographier(page, Hm, defaut, rng, scale)
    cache = None if masque is None else masque.astype(np.float32)
    occ = []
    if defaut == "ombre" or rng.random() < 0.3:
        arr, c2 = occulter(np.asarray(img, np.float32), rng)
        img = Image.fromarray(arr.astype(np.uint8))
        cache = c2 if cache is None else np.maximum(cache, c2)
        occ = True
    for o in ob:
        o["quad"] = [tuple(map(float, p)) for p in D.appliquer(Hm, o["quad"])]
    for c in cartes:
        c["coins"] = [tuple(map(float, p)) for p in D.appliquer(Hm, c["coins"])]
    for o in ob:  # mots coupés par le cadre ou cachés : « partiel » (canal dédié), tout caché : exclu
        hors = part_hors_champ(o["quad"], *D.FRAME)
        cach = 0.0
        if cache is not None:
            q = np.array(o["quad"]) * scale
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cache.shape[1]), min(y1, cache.shape[0])
            if x1 > x0 and y1 > y0:
                cach = float(cache[y0:y1, x0:x1].mean()) * (1 - hors)
        o["hors_champ"], o["cache_part"] = round(hors, 3), round(cach, 3)
        if hors + cach >= 0.9:
            o["cache"] = True
        elif hors > 0.05 or cach > 0.08:
            o["partiel"] = True
    idx = {id(o): i for i, o in enumerate(ob)}
    v = dict(info, mots=mots, n=n, rotation=0, style=style, defaut=defaut, objets=ob, cartes=cartes,
             liens=[(idx[id(p)], idx[id(q)]) for p, q in li if id(p) in idx and id(q) in idx],
             format="seedsigner" if info.get("modele") == "seedsigner" else info["type"],
             support=info.get("papier", "carte"), encre=encre.kind, ambigu=ambigu, occultation=bool(occ),
             police=os.path.basename(main.font) if main.font else "emnist")
    v["categorie_support"] = CATEGORIE_SUPPORT.get(v["support"], "autre")
    return img, v


def recadrer_cible(a, ob, cartes, rng):
    """Scène recadrée autour du document cible (l'utilisateur vise la cible) : marge de 10 à 60 % de sa taille."""
    pts = [p for o in ob if not o.get("autre") for p in o["quad"]] + \
        [p for c in cartes if c.get("cible") for p in c["coins"]]
    if not pts:
        return a, ob, cartes
    q = np.array(pts)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    w, h = x1 - x0, y1 - y0
    cx0 = int(max(0, x0 - rng.uniform(0.1, 0.6) * w))
    cy0 = int(max(0, y0 - rng.uniform(0.1, 0.6) * h))
    cx1 = int(min(a.shape[1], x1 + rng.uniform(0.1, 0.6) * w))
    cy1 = int(min(a.shape[0], y1 + rng.uniform(0.1, 0.6) * h))
    a = a[cy0:cy1, cx0:cx1].copy()
    for o in ob:
        o["quad"] = [(x - cx0, y - cy0) for x, y in o["quad"]]
    for c in cartes:
        c["coins"] = [(x - cx0, y - cy0) for x, y in c["coins"]]
    return a, ob, cartes


def composer_scene(pieces, rng, jointes):
    """Plusieurs documents sur une table (ou cartes jointives d'une planche)."""
    hs = [p[0].shape for p in pieces]
    if jointes:
        h = max(s[0] for s in hs)
        w = max(s[1] for s in hs)
        nx = min(3, len(pieces))
        ny = -(-len(pieces) // nx)
        gap = int(rng.uniform(0, 0.06) * w)
        marge = int(rng.uniform(0.03, 0.08) * w)
        scene = np.full((ny * h + (ny - 1) * gap + 2 * marge, nx * w + (nx - 1) * gap + 2 * marge),
                        rng.uniform(215, 245), np.float32)
        order = rng.permutation(len(pieces))
        pos = [(marge + (j % nx) * (w + gap), marge + (j // nx) * (h + gap)) for j in order]
        angs = [0.0] * len(pieces)
    else:
        sh = max(s[0] for s in hs)
        sw = sum(s[1] for s in hs)
        W, H = int(sw * rng.uniform(1.05, 1.3)), int(sh * rng.uniform(1.1, 1.4))
        scene = np.full((H, W), rng.uniform(30, 160), np.float32)
        pos, x = [], rng.uniform(0, 0.03) * W
        for s in hs:
            pos.append((x, rng.uniform(0, H - s[0] * 1.05)))
            x += s[1] * rng.uniform(1.0, 1.08)
        angs = [float(rng.normal(0, 4)) for _ in pieces]
    ob, li, cartes = [], [], []
    for (a, o, l_, c, cible), (ox, oy), ang in zip(pieces, pos, angs):
        f = coller(scene, a, ox, oy, ang, rng)
        for x in o:
            x["quad"] = f(x["quad"])
            x["autre"] = not cible
        if cible:
            ob += o
            li += l_
        else:
            ob += [dict(x, ignorer=True) for x in o]
        for cc in c:
            cartes.append(dict(cc, coins=f(cc["coins"]), cible=cible))
    return scene, ob, li, cartes


# ---------- cibles ----------

def shrink(q, f):
    q = np.asarray(q, float)
    c = q.mean(0)
    return c + (q - c) * f


def cibles_univ(v, scale_out):
    W, H = D.OUT_SIZE
    v2 = dict(v, objets=[o for o in v["objets"]])
    y9 = D.cibles(v2, scale_out)  # 0-7 + exclusion en 8
    ims = [Image.new("L", (W, H), 0) for _ in range(3)]
    dr = [ImageDraw.Draw(i) for i in ims]
    for c in v["cartes"] or []:
        q = shrink(np.array(c["coins"]) * scale_out, 0.88)
        dr[0].polygon([tuple(p) for p in q], fill=255)
    for i, j in v.get("liens", []):
        a, b = v["objets"][i], v["objets"][j]
        if a.get("cache") or b.get("cache"):
            continue
        qa, qb = np.array(a["quad"]) * scale_out, np.array(b["quad"]) * scale_out
        ka, ha = D.noyau(qa)
        kb, hb = D.noyau(qb)
        ca, cb = np.mean(ka, 0), np.mean(kb, 0)
        # bande du centre du numéro au début du mot
        start_b = (np.asarray(kb[0]) + np.asarray(kb[3])) / 2
        e = max(1, int(round(0.25 * min(ha, hb))))
        dr[1].line([tuple(ca), tuple(start_b)], fill=255, width=e)
    for o in v["objets"]:
        if o.get("partiel") and o["classe"] == "mot":
            q = np.array(o["quad"]) * scale_out
            if D.visible_part(q, W, H):
                k, _ = D.noyau(q)
                dr[2].polygon([tuple(p) for p in k], fill=255)
                dr[2].point(tuple(np.mean(k, 0)), fill=255)
    y = np.concatenate([y9[..., :8], np.stack([np.asarray(i) for i in ims], -1), y9[..., 8:9]], -1)
    return y


def main():
    mode, sortie, nombre, graine = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    test = "--test" in sys.argv
    opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
    rng = np.random.default_rng(graine)
    ex, par = G.charger_emnist(train=not test)
    fonts = D.polices(test)
    if mode == "train":
        scale = 2 * D.NET_SIZE[0] / D.FRAME[0]
        X = np.zeros((nombre, D.NET_SIZE[1], D.NET_SIZE[0]), np.uint8)
        Y = np.zeros((nombre, D.OUT_SIZE[1], D.OUT_SIZE[0], 12), np.uint8)
        meta = []
        for i in range(nombre):
            img, v = generer(rng, ex, par, fonts, test, scale=scale)
            X[i] = np.asarray(img.reduce(2))
            Y[i] = cibles_univ(v, D.OUT_SIZE[0] / D.FRAME[0])
            meta.append(dict(style=v["style"], defaut=v["defaut"], format=v["format"], support=v["support"],
                             n=v["n"]))
        np.savez_compressed(sortie, x=X, y=Y, meta=json.dumps(meta))
    else:
        os.makedirs(sortie, exist_ok=True)
        verite = {}
        for i in range(nombre):
            img, v = generer(rng, ex, par, fonts, test, defaut=opt("--defaut"), style=opt("--style"),
                             fmt=opt("--format"), support=opt("--support"))
            nom = f"u{graine:03d}_{i:04d}.jpg"
            img.save(os.path.join(sortie, nom), quality=int(rng.integers(70, 95)))
            verite[nom] = v
        json.dump(verite, open(os.path.join(sortie, "verite.json"), "w"), ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
