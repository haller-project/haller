"""Composition « vrais fonds + écriture simulée » sur les photos de papiers vierges annotées (v2, univ5).

Une page = une phrase de seed écrite comme on l'écrit vraiment :
  - lignes d'écriture : lignes imprimées relevées dans la photo (traits.py : polylignes qui suivent la perspective et
    la courbure) ; Seyès : lignes principales ; carreaux : 1 ou 2 carreaux par ligne ; papier uni : lignes virtuelles
    sur la page redressée, avec la dérive d'une main (pente, ondulation) ;
  - structure : 12, 18 ou 24 mots, 1 ou 2 colonnes (remplies par colonne, parfois par rangée), lignes consécutives
    (parfois une ligne sautée), numéros (3 pages sur 4) alignés dans leur colonne, à droite ou à gauche ;
  - taille : hauteur des capitales = 45 à 75 % de l'interligne (Seyès 40 à 65 %), même main sur toute la page ; un
    mot trop long pour sa colonne est écrit plus petit ;
  - position : ligne de base posée sur la ligne imprimée ; sur une grille, numéros et mots calés sur les lignes
    verticales ; marges tirées ; aucun chevauchement ; tout le texte dans la page et dans le cadre ;
  - encre : composée par multiplication avec le papier (elle prend la lumière, l'ombre et la teinte de la photo),
    floutée au flou mesuré de la photo, bruit du papier ; l'augmentation d'éclairage (lumiere.py) vient APRÈS l'encre ;
  - occultants : l'encre passe sous la main (masque) ; part cachée -> caché (>= 0,9) ou partiel (> 0,08) ;
  - distributions réelles (train/detecteur/distributions, statistiques agrégées des photos réelles, élargies x 1,5) :
    recadrage autour du texte pour que la hauteur des mots rapportée au petit côté suive le réel (médiane 0,056),
    texte redressé puis orientation tirée (droit 80 %, quarts de tour 14 %, retourné 6 %), balance des blancs tirée
    vers le froid (supports photographiés plus chauds que le réel), 50 % de listes numérotées, 12 / 24 mots ;
  - cartes officielles (cartes.py, zones de cases) : un mot par case, calé à gauche sur une ligne de base parallèle
    au bas de la case, numéros imprimés étiquetés, coins de carte ; orientation et échelle déjà tirées par
    cartes.py varier ; les cartes d'évaluation ("jeu": "eval") sont écartées.
Les supports dont les lignes imprimées n'ont pas été relevées proprement (traits.regulier faux) sont écartés s'ils
sont réglés (écrire en travers des lignes serait faux), traités en papier uni sinon. Vérité au format des bancs.

  python composer.py banc ANNOT TRAVAIL SORTIE N GRAINE [--codes sv_0001,...] [--planche CHEMIN.jpg]
  python composer.py lot  ANNOT TRAVAIL SORTIE.npz N GRAINE [--codes ...]
ANNOT : dossier des CODE.json (annoter_papiers.py puis traits.py) et CODE_masque.png ; TRAVAIL : photos CODE.jpg.
Les supports sont pris dans l'ordre de ANNOT/ordre.txt (un CODE.json par ligne) s'il existe, sinon par nom de fichier.

Code figé tel qu'il a produit les lots univ5 (détecteur Haller-1), modifications de publication seulement : noms
publics des photos (sv_*, cb_*, cv_*) au lieu des codes internes, ordre des supports lu dans ordre.txt (l'ordre des
codes internes, sous lequel les lots ont été tirés), chemins et noms de machines retirés.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

os.environ.setdefault("UNIV_REEL", "1")   # gen_det : poids des styles d'écriture réglés sur les photos réelles
os.environ.setdefault("UNIV_POLICE_RARE", "1")   # gen_det : polices minoritaires et plus fines (retouche du 20/11)
ICI = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(ICI, "..", "detecteur")
sys.path.insert(0, ICI)
sys.path.insert(0, DET)
sys.path.insert(0, os.path.join(DET, "supports"))
import lumiere  # noqa: E402
import poser_ecriture as P  # noqa: E402
from poser_ecriture import D, U, mnemonique  # noqa: E402

REGLES = {"ligne", "grands_carreaux", "petits_carreaux", "quadrille", "carnet"}
# décor pris pour des lignes (éventails), papier froissé aux lignes en éventail : papier uni
TRAITS_IGNORES = {"sv_0130", "sv_0152"}
# Distributions réglées sur les photos réelles (train/detecteur/distributions, mesures agrégées du 20/11/2026) :
# photos de supports plus chaudes que le réel (4 710 K contre 5 540 K) -> balance des blancs tirée vers le froid ;
# orientation à l'entrée du moteur : texte droit en majorité, quarts de tour et retournées en proportion
T_LUMIERE = dict(t_min=4300.0, t_max=8500.0)
P_ORIENTATION = [0.80, 0.07, 0.06, 0.07]   # 0, 90, 180, 270 degrés après redressement
DISTRIBUTIONS = os.path.join(DET, "distributions", "distributions_reelles.json")
# Composition sur la photo réduite à COTE px de côté long (l'entrée du réseau fait 384 px ; le recadrage le plus serré
# garde ~2,5 fois cette résolution), écriture rendue à K fois la résolution puis projetée
COTE = 1024
K = 2.0
# Tirage des supports par type, proportionné au réel (jeu réel : feuilles blanches et lignées dominantes, aucun
# papier coloré ni illustré) : papiers colorés, à motifs et Post-it minoritaires
POIDS_TYPES = {"blanc": 30, "ligne": 20, "carnet": 10, "grands_carreaux": 6, "petits_carreaux": 6, "quadrille": 4,
               "bristol": 5, "plie_froisse": 8, "postit": 6, "colore": 4, "autre": 1}
ENCRES_T = {"bleu": (0.16, 0.22, 0.55), "noir": (0.10, 0.10, 0.12), "bleu_clair": (0.30, 0.40, 0.75),
            "gris": (0.40, 0.40, 0.42)}


# ---------- géométrie des lignes ----------

def reechantillonner(pts, pas=2.0):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < 2 * pas:
        return None
    s = np.arange(0, d[-1], pas)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def dans_polygone(pts, poly):
    import cv2
    poly = np.float32(poly).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts])


def region_utile(a, w, h, marge):
    """Polygone de la page rétréci de `marge` px, coupé au cadre de la photo (moins la même marge)."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(a["page"]))], 1)
    m[:int(marge)] = 0
    m[h - int(marge):] = 0
    m[:, :int(marge)] = 0
    m[:, w - int(marge):] = 0
    m = cv2.erode(m, np.ones((int(2 * marge) | 1, int(2 * marge) | 1), np.uint8))
    return m > 0


class Ligne:
    """Ligne de base : points tous les 2 px (repère de la photo), sens d'écriture, interligne au-dessus."""

    def __init__(self, pts):
        self.p = pts
        t = np.gradient(pts, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-6)[:, None]
        self.t = t
        self.haut = np.c_[t[:, 1], -t[:, 0]]   # normale « vers le haut » si l'écriture va de gauche à droite
        self.H = None

    @property
    def longueur(self):
        return 2.0 * (len(self.p) - 1)

    def point(self, s):
        i = np.clip(s / 2.0, 0, len(self.p) - 1)
        i0 = int(np.floor(i))
        i1 = min(i0 + 1, len(self.p) - 1)
        f = i - i0
        return self.p[i0] * (1 - f) + self.p[i1] * f, self.t[i0], self.haut[i0]


def couper(pts, region):
    """Plus longue suite de points de la polyligne dans la région."""
    h, w = region.shape
    xi, yi = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ok[ok] = region[yi[ok], xi[ok]]
    best, cur, deb = (0, 0), 0, 0
    for i, v in enumerate(np.r_[ok, False]):
        if v:
            if cur == 0:
                deb = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (deb, deb + cur)
            cur = 0
    return pts[best[0]:best[1]] if best[1] - best[0] > 10 else None


def interligne(lig, autre):
    """Distance moyenne de `lig` à la polyligne `autre`, et signe (autre au-dessus selon lig.haut ?)."""
    from scipy.spatial import cKDTree
    tr = cKDTree(autre.p)
    idx = np.linspace(0, len(lig.p) - 1, 15).astype(int)
    d, j = tr.query(lig.p[idx])
    v = autre.p[j] - lig.p[idx]
    signe = np.sign(np.sum(v * lig.haut[idx], 1))
    return float(np.median(d)), float(np.median(signe))


def lignes_imprimees(a, region, rng, L):
    """Lignes de base sur les lignes imprimées (haut -> bas), chacune avec son interligne H ; None si inutilisable."""
    fam = a["traits"]["ecriture"]
    brutes = fam["lignes"]
    pas = fam["pas"]
    typ = a.get("support")
    if typ == "grands_carreaux":
        ph = fam.get("seyes_phase", 0)
        idx = list(range(ph, len(brutes), 4))
        step = 4
    else:
        options = [k for k in (1, 2, 3) if 0.022 * L <= k * pas <= 0.075 * L]
        if typ in ("petits_carreaux", "quadrille"):
            options = [k for k in options if k <= 2] or options
        else:
            options = [k for k in options if k == 1] or options[:1]
        step = int(rng.choice(options)) if options else (1 if pas >= 0.022 * L else 2)
        idx = list(range(int(rng.integers(step)), len(brutes), step))
    # paires régulières : de la ligne j à j + step, toutes les paires intermédiaires régulières
    ok = [bool(l.get("suivante_ok")) for l in brutes]
    res = []
    for i in idx:
        if i - step < 0:
            continue   # pas de ligne au-dessus : la première ligne imprimée ne reçoit pas d'écriture
        if not all(ok[i - step:i]):
            res.append(None)
            continue
        q = reechantillonner(brutes[i]["pts"])
        q = couper(q, region) if q is not None else None
        if q is None:
            res.append(None)
            continue
        lig = Ligne(q)
        au_dessus = Ligne(np.asarray(brutes[i - step]["pts"], float))
        d, sg = interligne(lig, au_dessus)
        lig.H = d
        lig.signe = sg
        res.append(lig)
    vals = [l for l in res if l is not None]
    if not vals:
        return None, step
    sg = np.sign(np.median([l.signe for l in vals])) or 1.0
    if sg < 0:   # la ligne d'au-dessus est du côté opposé à la normale : écrire dans l'autre sens
        for l in vals:
            H_ = l.H
            l.__init__(l.p[::-1].copy())
            l.H = H_
    return res, step


def lignes_libres(a, region, rng, L):
    """Papier uni : lignes virtuelles sur la page redressée, pas tiré, dérive de la main."""
    lib = a["lignes"] if a["lignes"].get("libre") else dict(pas_min=0.05, pas_max=0.09)
    pas = rng.uniform(lib["pas_min"], lib["pas_max"])
    Hm = P.homographie([(0, 0), (1, 0), (1, 1), (0, 1)], a["page"])
    pente0 = rng.normal(0, 0.015)
    y = rng.uniform(1.0, 1.6) * pas
    res = []
    while y < 0.97:
        xs = np.linspace(0.0, 1.0, 60)
        ys = y + (pente0 + rng.normal(0, 0.006)) * (xs - 0.5) + rng.normal(0, 0.002) * np.sin(
            xs * rng.uniform(2, 6) + rng.uniform(0, 6))
        q = reechantillonner(P.appliquer(Hm, np.c_[xs, ys]))
        q = couper(q, region) if q is not None else None
        if q is not None:
            lig = Ligne(q)
            top = P.appliquer(Hm, np.c_[xs, ys - pas])
            lig.H = float(np.median(np.hypot(*(top - P.appliquer(Hm, np.c_[xs, ys])).T)))
            res.append(lig)
        else:
            res.append(None)
        y += pas
    return res, 1


def croisements(lig, travers):
    """Abscisses curvilignes où les lignes verticales de la grille croisent la ligne de base."""
    if not travers:
        return None
    from scipy.spatial import cKDTree
    pts = np.concatenate([reechantillonner(l["pts"]) for l in travers["lignes"]
                          if reechantillonner(l["pts"]) is not None])
    d, _ = cKDTree(pts).query(lig.p)
    s = []
    for i in range(1, len(d) - 1):
        if d[i] < 2.5 and d[i] <= d[i - 1] and d[i] <= d[i + 1]:
            if not s or 2.0 * i - s[-1] > 6:
                s.append(2.0 * i)
    return np.array(s) if len(s) >= 3 else None


# ---------- encre ----------

def mesurer_photo(rgb, region):
    """Flou (sigma, px) et bruit (écart type) du papier, mesurés dans la page."""
    from scipy import ndimage
    g = rgb.mean(2)
    ys, xs = np.nonzero(region)
    if len(xs) < 1000:
        return 1.0, 2.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    c = g[y0:y1, x0:x1]
    flou = P.flou_photo(c)
    res = c - ndimage.median_filter(c, 3)
    bruit = float(1.4826 * np.median(np.abs(res[region[y0:y1, x0:x1]])))
    return flou, bruit


def rendre_texte(main, texte, h, rng, k=None, haut_min=None):
    """Mot ou numéro rendu à K fois la résolution de la photo : (alpha, base, x_encre0, x_encre1, y0, y1)."""
    k = k or K
    m, base = U.ecrire_mot(main, texte, h * k, rng)
    al = 1 - np.asarray(m, np.float32) / 255.0
    ys, xs = np.nonzero(al > 0.25)
    if len(xs) == 0:
        return None
    return al, base, xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


def poser(rgb, lig, s0, rend, encre, flou, bruit, rng, k=None):
    """Pose un texte rendu sur la ligne de base `lig` à partir de l'abscisse s0 (début de l'encre). Renvoie le
    quadrilatère d'encre dans la photo, la boîte de travail et l'alpha, sans encore écrire dans rgb."""
    from scipy import ndimage
    k = k or K
    al, base, ex0, ex1, ey0, ey1 = rend
    larg = (ex1 - ex0) / k
    (p0, t0, n0), (p1, t1, n1) = lig.point(s0), lig.point(s0 + larg)
    # rectangle d'encre -> quadrilatère qui suit la ligne de base (pente locale)
    hy0, hy1 = (base - ey0) / k, (ey1 - base) / k      # au-dessus et au-dessous de la ligne de base
    q = np.array([p0 + n0 * hy0, p1 + n1 * hy0, p1 - n1 * hy1, p0 - n0 * hy1])
    src = [(ex0, ey0), (ex1, ey0), (ex1, ey1), (ex0, ey1)]
    Hm = P.homographie(src, q)
    bx0, by0 = np.floor(q.min(0)).astype(int) - 3
    bx1, by1 = np.ceil(q.max(0)).astype(int) + 3
    H_, W_ = rgb.shape[:2]
    bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W_, bx1), min(H_, by1)
    if bx1 <= bx0 or by1 <= by0:
        return None
    gx, gy = np.meshgrid(np.arange(bx0, bx1) + 0.5, np.arange(by0, by1) + 0.5)
    uv = P.appliquer(np.linalg.inv(Hm), np.c_[gx.ravel(), gy.ravel()])
    a_l = ndimage.uniform_filter(al, size=int(k))
    a = ndimage.map_coordinates(a_l, [uv[:, 1], uv[:, 0]], order=1, cval=0).reshape(gx.shape)
    if encre["kind"] == "crayon":
        a *= np.clip(rng.normal(0.8, 0.2, a.shape), 0.3, 1)
    a = ndimage.gaussian_filter(a, max(0.35, flou * rng.uniform(0.7, 1.0)))
    a = np.clip(a * encre["opacite"], 0, 1)
    return dict(quad=q.tolist(), boite=(bx0, by0, bx1, by1), a=a)


def encrer(rgb, pose, encre, bruit, rng):
    bx0, by0, bx1, by1 = pose["boite"]
    a = pose["a"][..., None]
    T = np.asarray(encre["T"], np.float32)
    zone = rgb[by0:by1, bx0:bx1]
    out = zone * (1 - a + a * T)
    out += a * rng.normal(0, max(0.5, 0.7 * bruit), out.shape[:2])[..., None]
    rgb[by0:by1, bx0:bx1] = out


def lier(objets):
    """Liens numéro -> mot (indices dans objets) pour le canal de lien : même position k."""
    num = {o["k"]: i for i, o in enumerate(objets) if o.get("classe") == "numero" and "k" in o}
    return [(num[o["k"]], i) for i, o in enumerate(objets) if o.get("classe") == "mot" and o.get("k") in num]


def axe_ecriture(ligs):
    """Angle (degrés, repère de la photo, y vers le bas) du sens d'écriture moyen des lignes."""
    t = np.mean([l.t.mean(0) for l in ligs], 0)
    return round(float(np.degrees(np.arctan2(t[1], t[0]))), 1)


def tourner_k(im, v, k):
    """Quart(s) de tour dans le sens trigonométrique (comme P.tourner), photo et vérité."""
    for _ in range(k % 4):
        w, h = im.size
        im = im.transpose(Image.ROTATE_90)
        f = lambda pts: [[y, w - x] for x, y in pts]  # noqa: E731
        v = dict(v, objets=[dict(o, quad=f(o["quad"])) for o in v["objets"]],
                 cartes=[dict(c, coins=f(c["coins"])) for c in v.get("cartes", [])])
        if v.get("axe") is not None:
            v["axe"] = round(((v["axe"] - 90 + 180) % 360) - 180, 1)
    return im, v


def redresser_et_tourner(im, v, rng):
    """Papiers : texte remis droit (sens d'écriture vers la droite), puis orientation tirée selon P_ORIENTATION."""
    if v.get("axe") is None:
        return im, v
    k = int(round(v["axe"] / 90.0)) % 4    # ROTATE_90 retire 90 degrés à l'axe
    im, v = tourner_k(im, v, k)
    kk = int(rng.choice(4, p=P_ORIENTATION))
    im, v = tourner_k(im, v, kk)
    v["orientation"] = 90 * kk
    return im, v


def encre_presente(rgb, masque):
    """Taches sombres plus épaisses qu'une ligne imprimée (texte imprimé, logo, encre), hors occultants."""
    import cv2
    g = rgb.mean(2).astype(np.uint8)
    kf = int(31 * max(g.shape) / 2048) | 1
    fond = cv2.medianBlur(g, max(kf, 9)).astype(np.int16)
    d = fond - g
    t = (d > np.maximum(18, 0.22 * fond)).astype(np.uint8)   # seuil relatif : vaut aussi en faible lumière
    ko = 5 if max(g.shape) > 1500 else 3
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    return (t > 0) & (masque < 0.5)


# ---------- page ----------

def boites_chevauchent(q, autres, tol=0.04):
    b = np.array(q)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    for (u0, v0, u1, v1) in autres:
        iw, ih = min(x1, u1) - max(x0, u0), min(y1, v1) - max(y0, v0)
        if iw > 0 and ih > 0 and iw * ih > tol * min((x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0)):
            return True
    return False


def mise_en_page(lignes, n, rng, L):
    """Choix des lignes, colonnes et ordre : liste de (indice de mot, ligne, colonne) ou None."""
    cols_opts = ([1, 2] if rng.random() < 0.6 else [2, 1]) if n <= 12 else ([2, 1] if rng.random() < 0.7 else [1, 2])
    saut = 2 if rng.random() < 0.12 else 1
    for cols in cols_opts:
        rows = -(-n // cols)
        cands = []
        for j0 in range(len(lignes)):
            sel = [j0 + saut * r for r in range(rows)]
            if sel[-1] >= len(lignes) or any(lignes[j] is None for j in sel):
                continue
            Lmin = min(lignes[j].longueur for j in sel)
            H = np.median([lignes[j].H for j in sel])
            if Lmin < cols * 6.0 * H:
                continue
            cands.append(sel)
        if cands:
            sel = cands[int(rng.integers(len(cands)))]
            par_rangee = cols == 2 and rng.random() < 0.2
            places = []
            for k in range(n):
                r, c = (k // cols, k % cols) if par_rangee else (k % rows, k // rows)
                places.append((k, sel[r], c))
            return places, cols, rows, sel
    return None


def ouvrir_reduit(a, travail):
    """Photo réduite à COTE px de côté long et annotation mise à la même échelle."""
    im = Image.open(os.path.join(travail, a["image"])).convert("RGB")
    f = COTE / max(im.size)
    if f >= 1:
        return im, a
    im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    sc = lambda q: (np.asarray(q, float) * f).tolist() if q is not None else None  # noqa: E731
    b = dict(a)
    for cle in ("page", "coins"):
        if b.get(cle) is not None:
            b[cle] = sc(b[cle])
    b["exclure"] = [sc(q) for q in a.get("exclure", [])]
    if a.get("zones") is not None:
        b["zones"] = [dict(z, quad=sc(z["quad"]), numero_quad=sc(z.get("numero_quad"))) for z in a["zones"]]
    if a.get("traits"):
        t = dict(a["traits"])
        for nom in ("ecriture", "travers"):
            if t.get(nom):
                t[nom] = dict(t[nom], pas=t[nom]["pas"] * f,
                              lignes=[dict(l, pts=sc(l["pts"])) for l in t[nom]["lignes"]])
        b["traits"] = t
    return im, b


def rendre(a, travail, annot, rng, ex, par, fonts, essais=6):
    im, a = ouvrir_reduit(a, travail)
    w, h = im.size
    L = max(w, h)
    rgb0 = np.asarray(im, np.float32)
    masque = np.zeros((h, w), np.float32)
    if a.get("masque") and os.path.exists(os.path.join(annot, a["masque"])):
        masque = (np.asarray(Image.open(os.path.join(annot, a["masque"])).convert("L").resize((w, h)),
                             np.float32) > 127).astype(np.float32)
    region = region_utile(a, w, h, marge=0.012 * L)
    flou, bruit = mesurer_photo(rgb0, region)
    code = a["image"][:-4]
    t = a.get("traits") or {}
    imprime = bool(t.get("regulier")) and code not in TRAITS_IGNORES
    for essai in range(essais):
        if imprime:
            lignes, step = lignes_imprimees(a, region, rng, L)
        else:
            lignes, step = lignes_libres(a, region, rng, L)
        if not lignes:
            continue
        # réel : 12 mots 74 %, 24 mots 26 % ; tirage plus fort sur 24 (les pages trop petites retombent sur 12)
        n = int(rng.choice([12, 24, 18], p=[0.5, 0.42, 0.08]))
        mp = mise_en_page(lignes, n, rng, L) or (mise_en_page(lignes, 12, rng, L) if n != 12 else None)
        if mp is None:
            continue
        places, cols, rows, sel = mp
        n = len(places)
        out = composer_page(a, rgb0, masque, region, lignes, places, cols, sel, flou, bruit, rng, ex, par, fonts,
                            imprime and a.get("support") in ("petits_carreaux", "quadrille", "grands_carreaux"), L)
        if out is not None:
            return out
    return None


def composer_page(a, rgb0, masque, region, lignes, places, cols, sel, flou, bruit, rng, ex, par, fonts, grille, L):
    h, w = masque.shape
    n = len(places)
    deja_encre = encre_presente(rgb0, masque)
    style = D.STYLES[rng.choice(3, p=np.array(D.POIDS_STYLES) / sum(D.POIDS_STYLES))]
    main = D.Main(style, rng, ex, par, fonts)
    kind = str(rng.choice(["bille", "feutre", "crayon"], p=[0.6, 0.25, 0.15]))
    nom_c = str(rng.choice(list(ENCRES_T), p=[0.5, 0.35, 0.1, 0.05])) if kind != "crayon" else "gris"
    T = np.clip(np.array(ENCRES_T[nom_c]) * rng.uniform(0.75, 1.2), 0.02, 0.9)
    encre = dict(kind=kind, T=T, opacite=rng.uniform(0.8, 1.0) if kind != "crayon" else rng.uniform(0.55, 0.85))
    mots = list(mnemonique(rng.bytes(n * 4 // 3)))
    seyes = a.get("support") == "grands_carreaux"
    f = rng.uniform(0.40, 0.65) if seyes else rng.uniform(0.45, 0.75)
    Hs = np.array([lignes[j].H for j in sel])
    hcap = f * float(np.median(Hs))
    numeros = rng.random() < 0.5   # réel : 44 % des listes numérotées
    fmt = str(rng.choice(["", ".", ")"], p=[0.35, 0.4, 0.25])) if numeros else ""
    num_droite = rng.random() < 0.5
    # colonnes alignées dans le repère de la page redressée (parallèles au bord de la page, perspective comprise) :
    # coordonnée c(s) le long de chaque ligne = axe de la page le plus proche du sens d'écriture
    Hinv = np.linalg.inv(P.homographie([(0, 0), (1, 0), (1, 1), (0, 1)], a["page"]))
    cs = {}
    for j in set(sel):
        uv = P.appliquer(Hinv, lignes[j].p)
        d = uv[-1] - uv[0]
        ax = int(abs(d[1]) > abs(d[0]))
        c_ = uv[:, ax] * (1 if d[ax] >= 0 else -1)
        cs[j] = np.maximum.accumulate(c_)
    c_deb = max(cs[j][0] for j in sel)
    c_fin = min(cs[j][-1] for j in sel)
    if c_fin - c_deb <= 0:
        return None
    marge_c = c_deb + rng.uniform(0.03, 0.12) * (c_fin - c_deb)
    col2_c = c_deb + rng.uniform(0.47, 0.56) * (c_fin - c_deb)

    def s_de(j, c):
        return float(np.interp(c, cs[j], 2.0 * np.arange(len(cs[j]))))

    debut_col_c = [marge_c] + ([col2_c] if cols == 2 else [])
    largeur_num = 0.0
    rendus_num = {}
    if numeros:
        for k, _, _ in places:
            r = rendre_texte(main, f"{k + 1}{fmt}", hcap * rng.uniform(0.9, 1.0), rng)
            if r is not None:
                rendus_num[k] = r
        largeur_num = max([(r[3] - r[2]) / K for r in rendus_num.values()] or [0])
    ecart = (rng.uniform(0.5, 1.2) if fmt else rng.uniform(0.8, 1.6)) * hcap
    rgb = rgb0.copy()
    poses, boites, objets = [], [], []

    def caler(lig, s, cr):
        if cr is None:
            return s
        j = np.searchsorted(cr, s - 1)
        if j >= len(cr):
            return s
        s2 = float(cr[j]) + rng.uniform(0.1, 0.3) * hcap
        return s2 if s2 - s < 0.8 * hcap else s   # carreau proche : calé ; grande maille (Seyès) : écrit à la suite

    # grille : chaque colonne démarre sur une même ligne verticale imprimée, choisie sur la première rangée
    verticales = None
    if grille and a["traits"].get("travers"):
        from scipy.spatial import cKDTree
        verticales = [reechantillonner(l["pts"]) for l in a["traits"]["travers"]["lignes"]]
        verticales = [v for v in verticales if v is not None]

    def croise(lig, v):
        d, i = cKDTree(lig.p).query(v)
        k = int(np.argmin(d))
        return 2.0 * i[k] if d[k] < 3 else None

    col_v = {}
    if verticales:
        j0 = sel[0]
        for c, cc in enumerate(debut_col_c):
            s_cible = s_de(j0, cc)
            best = None
            for vi, v in enumerate(verticales):
                sx = croise(lignes[j0], v)
                if sx is not None and sx >= s_cible - 1 and (best is None or sx < best[0]):
                    best = (sx, vi)
            if best:
                col_v[c] = best[1]
        decal = rng.uniform(0.1, 0.3) * hcap

    for k, j, c in places:
        lig = lignes[j]
        cr = croisements(lig, a["traits"].get("travers")) if grille else None
        s_col = caler(lig, s_de(j, debut_col_c[c]), cr)
        if c in col_v:
            sx = croise(lig, verticales[col_v[c]])
            if sx is not None:
                s_col = sx + decal
        fin_col = (s_de(j, debut_col_c[1]) - 0.4 * hcap) if (cols == 2 and c == 0) else lig.longueur - 0.02 * L
        if numeros and k in rendus_num:
            rn = rendus_num[k]
            wn = (rn[3] - rn[2]) / K
            sn = s_col + (largeur_num - wn if num_droite else 0) + rng.normal(0, 0.03 * hcap)
            s_mot = caler(lig, s_col + largeur_num + ecart, cr)
            pn = poser(rgb, lig, max(0.0, sn), rn, encre, flou, bruit, rng)
            if pn is None:
                return None
            pn["objet"] = dict(classe="numero", k=k, valeur=k + 1)
            poses.append(pn)
        else:
            s_mot = s_col
        hm = hcap * rng.uniform(0.94, 1.06)
        rm = rendre_texte(main, mots[k], hm, rng)
        if rm is None:
            return None
        larg = (rm[3] - rm[2]) / K
        dispo = fin_col - s_mot
        if dispo < 3 * hcap:
            return None
        if larg > dispo:   # mot trop long pour la colonne : la main écrit plus petit
            rm = rendre_texte(main, mots[k], hm * dispo / larg * 0.97, rng)
            if rm is None or (rm[3] - rm[2]) / K > dispo:
                return None
        pm = poser(rgb, lig, s_mot + rng.normal(0, 0.05 * hcap), rm, encre, flou, bruit, rng)
        if pm is None:
            return None
        pm["objet"] = dict(classe="mot", k=k, mot=mots[k])
        poses.append(pm)
    # contrôles : tout dans la page et dans le cadre, aucun chevauchement, rien sur un texte déjà imprimé ou écrit
    for p_ in poses:
        bx0, by0, bx1, by1 = p_["boite"]
        if deja_encre[by0:by1, bx0:bx1].mean() > 0.004:
            return None
        q = np.array(p_["quad"])
        if q[:, 0].min() < 0 or q[:, 1].min() < 0 or q[:, 0].max() >= w or q[:, 1].max() >= h:
            return None
        if not dans_polygone(q, a["page"]).all():
            return None
        if boites_chevauchent(q, boites):
            return None
        boites.append((q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
    for p_ in poses:
        bx0, by0, bx1, by1 = p_["boite"]
        avant = rgb[by0:by1, bx0:bx1].copy()
        encrer(rgb, p_, encre, bruit, rng)
        m = masque[by0:by1, bx0:bx1]
        d = np.abs(rgb[by0:by1, bx0:bx1] - avant).sum(2)
        tot = d.sum()
        cach = float((d * m).sum() / tot) if tot > 0 else 0.0
        rgb[by0:by1, bx0:bx1] = rgb[by0:by1, bx0:bx1] * (1 - m[..., None]) + avant * m[..., None]
        o = dict(p_["objet"], quad=p_["quad"], cache_part=round(cach, 3))
        if cach >= 0.9:
            o["cache"] = True
        elif cach > 0.08:
            o["partiel"] = True
        objets.append(o)
    for qx in a.get("exclure", []):
        objets.append(dict(classe="mot", cache=True, quad=qx))
    rgb, aug = lumiere.augmenter(rgb, rng, **T_LUMIERE)   # éclairage après l'encre : papier et encre ensemble
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    v = dict(objets=objets, cartes=[], liens=lier(objets), mots=mots, style=style, defaut="support_reel", n=n, type="liste",
             format="liste", support=a.get("support", "reel"), encre=kind, couleur=nom_c, flou=round(flou, 2),
             bruit=round(bruit, 2), source=a["image"], ambigu=False, lumiere=aug, occultation=bool(masque.any()),
             colonnes=cols, numeros=numeros, lignes="imprimees" if a.get("traits", {}).get("regulier") else "libres",
             hauteur_rel=round(f, 2), axe=axe_ecriture([lignes[j] for j in sel]))
    return out, v


def hors_cadre(q, w, h):
    import cv2
    q = np.float32(q)
    a = abs(cv2.contourArea(q))
    if a <= 0:
        return 1.0
    r, _ = cv2.intersectConvexConvex(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return float(max(0.0, 1 - r / a))


def rendre_carte(a, travail, annot, rng, ex, par, fonts):
    """Carte officielle imprimée (cartes.py) : un mot par case dans l'ordre des numéros imprimés, posé sur une ligne
    de base parallèle au bas de la case, calé à gauche ; hauteur 45 à 70 % de la case, même main sur la carte."""
    im, a = ouvrir_reduit(a, travail)
    w, h = im.size
    rgb0 = np.asarray(im, np.float32)
    region = np.ones((h, w), bool)
    if a.get("coins"):
        import cv2
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.int32(np.round(a["coins"]))], 1)
        region = m > 0
    flou, bruit = mesurer_photo(rgb0, region)
    cases = sorted([z for z in a["zones"] if z.get("numero")], key=lambda z: z["numero"])
    if not cases:
        return None
    nmax = len(cases)
    n = nmax if nmax in (12, 18, 24) else max([k for k in (12, 18, 24) if k <= nmax] or [nmax])
    if nmax == 24 and rng.random() < 0.15:
        n = 12
    style = D.STYLES[rng.choice(3, p=np.array(D.POIDS_STYLES) / sum(D.POIDS_STYLES))]
    main = D.Main(style, rng, ex, par, fonts)
    kind = str(rng.choice(["bille", "feutre", "crayon"], p=[0.6, 0.25, 0.15]))
    nom_c = str(rng.choice(list(ENCRES_T), p=[0.5, 0.35, 0.1, 0.05])) if kind != "crayon" else "gris"
    encre = dict(kind=kind, T=np.clip(np.array(ENCRES_T[nom_c]) * rng.uniform(0.75, 1.2), 0.02, 0.9),
                 opacite=rng.uniform(0.8, 1.0) if kind != "crayon" else rng.uniform(0.55, 0.85))
    mots = list(mnemonique(rng.bytes(n * 4 // 3)))
    f = rng.uniform(0.45, 0.7)
    leve = rng.uniform(0.08, 0.2)
    depart = rng.uniform(0.03, 0.1)
    rgb = rgb0.copy()
    poses = []
    for k, z in enumerate(cases[:n]):
        q = np.array(z["quad"], float)
        hz = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        bas = np.array([q[3] + (q[0] - q[3]) * leve, q[2] + (q[1] - q[2]) * leve])
        pts = reechantillonner(bas)
        if pts is None:
            return None
        lig = Ligne(pts)
        if np.dot(lig.haut[0], q[0] - q[3]) < 0:
            return None
        hc = f * hz * rng.uniform(0.95, 1.05)
        rm = rendre_texte(main, mots[k], hc, rng)
        if rm is None:
            return None
        dispo = lig.longueur * (0.97 - depart)
        larg = (rm[3] - rm[2]) / K
        if larg > dispo:
            rm = rendre_texte(main, mots[k], hc * dispo / larg * 0.97, rng)
            if rm is None or (rm[3] - rm[2]) / K > dispo:
                return None
        pm = poser(rgb, lig, depart * lig.longueur, rm, encre, flou, bruit, rng)
        if pm is None:
            return None
        pm["objet"] = dict(classe="mot", k=k, mot=mots[k])
        poses.append(pm)
    objets = []
    for p_ in poses:
        encrer(rgb, p_, encre, bruit, rng)
        hors = hors_cadre(p_["quad"], w, h)
        o = dict(p_["objet"], quad=p_["quad"], cache_part=round(hors, 3))
        if hors >= 0.9:
            o["cache"] = True
        elif hors > 0.08:
            o["partiel"] = True
        objets.append(o)
    for z in cases:
        if z.get("numero_quad"):
            objets.append(dict(classe="numero", k=z["numero"] - 1, valeur=z["numero"], imprime=True,
                               quad=z["numero_quad"]))
    rgb, aug = lumiere.augmenter(rgb, rng, **T_LUMIERE)
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    cartes = [dict(coins=a["coins"], cible=True, n=n)] if a.get("coins") else []
    v = dict(objets=objets, cartes=cartes, liens=lier(objets), mots=mots, style=style, defaut="carte_reelle", n=n, type="carte",
             format="carte", support="carte_" + str(a.get("carte", a.get("modele", "?"))), encre=kind, couleur=nom_c,
             flou=round(flou, 2), bruit=round(bruit, 2), source=a["image"], ambigu=False, lumiere=aug,
             occultation=False, colonnes=0, numeros=True, lignes="cases", hauteur_rel=round(f, 2))
    return out, v


def tirer_quantiles(q, rng, elargir=1.5, log=True):
    """Tirage dans une distribution donnée par ses quantiles {5, 25, 50, 75, 95} (fonction de répartition linéaire
    par morceaux, queues prolongées jusqu'à 0 et 100), élargie autour de la médiane d'un facteur `elargir` (en
    logarithme si log=True) : même forme, étendues plus larges."""
    ps = np.array([0, 5, 25, 50, 75, 95, 100], float)
    v = np.array([q["q05"], q["q05"], q["q25"], q["q50"], q["q75"], q["q95"], q["q95"]], float)
    v[0] = v[1] - (v[2] - v[1]) * 0.5
    v[-1] = v[-2] + (v[-2] - v[-3]) * 0.5
    if log:
        v = np.log(np.maximum(v, 1e-6))
    x = np.interp(rng.uniform(0, 100), ps, v)
    x = v[3] + (x - v[3]) * elargir
    return float(np.exp(x)) if log else float(x)


def recadrer(im, v, cible, rng):
    """Recadre la photo pour que la hauteur médiane des mots rapportée à la hauteur de l'image approche `cible`
    (on ne peut que rapprocher : jamais d'agrandissement du champ). Le cadre garde les proportions de la photo et
    contient tout le texte avec une marge ; vérité ramenée au cadre."""
    w, h = im.size
    hs = [np.linalg.norm(np.array(o["quad"][3]) - np.array(o["quad"][0])) for o in v["objets"]
          if o.get("classe") == "mot" and "k" in o and not o.get("cache")]
    if not hs:
        return im, v
    r = float(np.median(hs)) / min(w, h)
    if r >= cible:
        return im, v
    f = r / cible                       # côté du cadre / côté de la photo
    pts = np.array([p for o in v["objets"] if "k" in o for p in o["quad"]], float)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mg = rng.uniform(0.03, 0.12)
    bw, bh = (x1 - x0) * (1 + 2 * mg), (y1 - y0) * (1 + 2 * mg)
    f = max(f, bw / w, bh / h, 0.5)   # cadre d'au moins la moitié de la photo (>= 512 px) : pas d'agrandissement flou
    if f >= 0.97:
        return im, v
    cw, ch = f * w, f * h
    def tirer_pos(lo, hi):
        return rng.uniform(lo, hi) if hi > lo else 0.5 * (lo + hi)

    cx = tirer_pos(max(0, x1 + mg * (x1 - x0) - cw), min(w - cw, x0 - mg * (x1 - x0)))
    cy = tirer_pos(max(0, y1 + mg * (y1 - y0) - ch), min(h - ch, y0 - mg * (y1 - y0)))
    cx, cy = float(np.clip(cx, 0, w - cw)), float(np.clip(cy, 0, h - ch))
    out = im.crop((int(cx), int(cy), int(cx + cw), int(cy + ch)))
    dx, dy = int(cx), int(cy)
    v = dict(v, objets=[dict(o, quad=[(px - dx, py - dy) for px, py in o["quad"]]) for o in v["objets"]],
             cartes=[dict(c, coins=[(px - dx, py - dy) for px, py in c["coins"]]) for c in v.get("cartes", [])],
             recadrage=round(f, 3))
    return out, v


def en_cibles(im, v):
    """Comme poser_ecriture.en_cibles, sans det_engine (qui exige le paquet seedsigner) : photo en gris ajustée dans
    l'entrée du réseau (288 x 384, bandes de gris médian, comme det_engine.reduce_photo), cibles 192 x 144 x 12."""
    W, H = D.NET_SIZE
    g = im.convert("L")
    s = min(W / g.width, H / g.height)
    w, h = max(1, round(g.width * s)), max(1, round(g.height * s))
    small = g.resize((w, h), Image.BOX)
    canvas = Image.new("L", (W, H), int(np.median(np.asarray(small))))
    ox, oy = (W - w) // 2, (H - h) // 2
    canvas.paste(small, (ox, oy))
    v2 = dict(v, objets=[dict(o, quad=[(px * s + ox, py * s + oy) for px, py in o["quad"]]) for o in v["objets"]],
              cartes=[dict(c, coins=[(px * s + ox, py * s + oy) for px, py in c["coins"]]) for c in v["cartes"]])
    return np.asarray(canvas), U.cibles_univ(v2, D.OUT_SIZE[0] / W)


def utilisable(a):
    """Support retenu pour la composition : carte d'entraînement, papier réglé dont les lignes sont relevées, ou papier
    uni. Les cartes d'évaluation (jeu eval) ne servent jamais ici."""
    if a.get("zones") is not None or a.get("carte"):
        return a.get("jeu", "train") == "train"
    t = a.get("traits") or {}
    if a.get("support") in REGLES:
        return bool(t.get("regulier"))
    return True


def planche(paths, verite, sortie, cols=4, cell=520):
    from PIL import ImageDraw
    rows = (len(paths) + cols - 1) // cols
    sh = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 6, cell - 24))
        r, c = divmod(i, cols)
        sh.paste(im, (c * cell + 3, r * cell + 3))
        v = verite[os.path.basename(p)]
        aug = "brut" if v["lumiere"].get("brut") else ",".join(k for k in v["lumiere"])
        npart = sum(1 for o in v["objets"] if o.get("partiel"))
        ImageDraw.Draw(sh).text((c * cell + 5, r * cell + cell - 20),
                                f"{v['source'][:-4]} {v['support']} n={v['n']} col={v['colonnes']} "
                                f"{v['lignes']} {aug} partiels={npart}", fill="black")
    sh.save(sortie, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["banc", "lot"])
    ap.add_argument("annot")
    ap.add_argument("travail")
    ap.add_argument("sortie")
    ap.add_argument("nombre", type=int)
    ap.add_argument("graine", type=int)
    ap.add_argument("--codes")
    ap.add_argument("--planche")
    ap.add_argument("--eval", action="store_true",
                    help="cartes du jeu d'évaluation seulement (banc de bout en bout, jamais d'entraînement)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.graine)
    ex, par = D.G.charger_emnist(train=True) if hasattr(D, "G") else P.G.charger_emnist(train=True)
    fonts = D.polices(False)
    fs = sorted(f for f in glob.glob(os.path.join(a.annot, "*.json")))
    ordre = os.path.join(a.annot, "ordre.txt")
    if os.path.exists(ordre):
        rang = {n: i for i, n in enumerate(open(ordre).read().split())}
        fs = sorted((f for f in fs if os.path.basename(f) in rang), key=lambda f: rang[os.path.basename(f)])
    if a.codes:
        garde = set(a.codes.split(","))
        fs = [f for f in fs if os.path.basename(f)[:-5] in garde]
    if a.eval:
        if a.mode != "banc":
            sys.exit("--eval : mode banc seulement")
        supports = [s for s in (json.load(open(f)) for f in fs)
                    if s.get("zones") and s.get("jeu") == "eval"]
    else:
        supports = [s for s in (json.load(open(f)) for f in fs) if utilisable(s)]

    q_mots = json.load(open(DISTRIBUTIONS))["jeux"]["reel"]["grandeurs"]["global"]["h_mot"]

    par_type = {}
    for x in supports:
        par_type.setdefault(x.get("support", "autre") if x.get("zones") is None else "carte", []).append(x)
    types = sorted(par_type)
    pt = np.array([POIDS_TYPES.get(t, 5) for t in types], float)
    pt /= pt.sum()

    def tirer():
        while True:
            lot = par_type[types[rng.choice(len(types), p=pt)]]
            s = lot[rng.integers(len(lot))]
            if s.get("zones") is not None:   # cartes : orientation et échelle déjà tirées par cartes.py varier
                r = rendre_carte(s, a.travail, a.annot, rng, ex, par, fonts)
                if r is not None:
                    return r
                continue
            r = rendre(s, a.travail, a.annot, rng, ex, par, fonts)
            if r is not None:
                im, v = r
                im, v = recadrer(im, v, tirer_quantiles(q_mots, rng), rng)
                return redresser_et_tourner(im, v, rng)

    if a.mode == "banc":
        os.makedirs(a.sortie, exist_ok=True)
        verite, chemins = {}, []
        for i in range(a.nombre):
            im, v = tirer()
            nom = f"c{a.graine:05d}_{i:04d}.jpg"
            im.save(os.path.join(a.sortie, nom), quality=int(rng.integers(80, 95)))
            verite[nom] = v
            chemins.append(os.path.join(a.sortie, nom))
        json.dump(verite, open(os.path.join(a.sortie, "verite.json"), "w"), ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
        if a.planche:
            planche(chemins[:16], verite, a.planche)
    else:
        X = np.zeros((a.nombre, 384, 288), np.uint8)
        Y = np.zeros((a.nombre, 192, 144, 12), np.uint8)
        meta = []
        for i in range(a.nombre):
            im, v = tirer()
            X[i], Y[i] = en_cibles(im, v)
            meta.append(dict(style=v["style"], defaut=v["defaut"], format=v["format"], support=v["support"], n=v["n"],
                             source=v["source"], lumiere=v["lumiere"], colonnes=v["colonnes"], lignes=v["lignes"]))
        np.savez_compressed(a.sortie, x=X, y=Y, meta=json.dumps(meta))


if __name__ == "__main__":
    main()
