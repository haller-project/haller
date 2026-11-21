"""Pages simulées annotées pour le détecteur de mots (Haller-1-det).

Reprend la mise en page et le tracé de train/pi/banc2/gen_sim.py (capitales EMNIST) et de
gen_cursive.py (polices), mais compose le numéro et le mot séparément et enregistre la
position exacte de chaque élément : boîte d'encre des mots et des numéros (quadrilatère dans
la photo), coins de la zone claire de chaque carte officielle, rang et colonne du mot.

Mauvais cadrages simulés (`defaut`) : normal, coupe_haut, coupe_bas, coupe_cote, proche, loin,
perspective, rotation (±35°), retournee (180°), planche (plusieurs cartes d'une planche
imprimée), ombre (ombres et objets posés), eclairage (très inégal).

Deux sorties :
  train : lots .npz pour l'entraînement (entrée 288x384, cibles 144x192, voir cibles())
      python3 gen_det.py train SORTIE.npz NOMBRE GRAINE [--test]
  banc  : photos 1944x2592 JPEG + verite.json (format des bancs, plus les boîtes)
      python3 gen_det.py banc DOSSIER NOMBRE GRAINE [--defaut D] [--style S] [--test]
--test : EMNIST test et polices réservées à l'évaluation (jamais vues à l'entraînement).
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "pi", "banc2"))
_cwd = os.getcwd()
# module `train` (chargement de modèles, hors de ce code figé) : inutile à la génération
import types  # noqa: E402
sys.modules["train"] = types.ModuleType("train")
sys.modules["train"].charger = None
import gen_sim as G  # noqa: E402  (fait os.chdir vers train/)
sys.path.insert(0, os.path.abspath(os.path.join(ICI, "..")))
from variete import journal as J  # noqa: E402  (journal des facteurs tirés, sans tirage : docs/variete.md)
from variete import resume as JR  # noqa: E402
from seed_sim import mnemonique  # noqa: E402
os.chdir(_cwd)
C = G.C

FRAME = G.FRAME                  # photo portrait 1944 x 2592
AJUSTER_LARGEUR = False          # posé par gen_univ.py (UNIV_CADRE) le temps d'une photo d'un autre format
NET_SIZE = (288, 384)            # entrée du détecteur (largeur, hauteur)
OUT_SIZE = (144, 192)            # sorties du détecteur (1/2)
DEFAUTS = ["normal", "coupe_haut", "coupe_bas", "coupe_cote", "proche", "loin", "perspective", "rotation",
           "retournee", "planche", "ombre", "eclairage"]
POIDS_DEFAUTS = [26, 9, 7, 7, 7, 7, 8, 9, 4, 6, 5, 5]
STYLES = ["capitales", "minuscules", "cursive"]
POIDS_STYLES = [5, 2, 3]
# UNIV_REEL=1 : distributions réglées sur les photos réelles (distributions/README.md, mesures du 20/11/2026,
# 76 photos du jeu réel, statistiques agrégées seulement), élargies x 1,5
REEL = bool(os.environ.get("UNIV_REEL"))
if REEL:
    POIDS_STYLES = [5.5, 2.5, 1.5]   # réel : capitales 54 %, minuscules 24 %, cursive 9 %

# Coins de la zone claire des cartes (page.py, CARD_CORNERS) dans le gabarit 1004 x 1401
CARD_CORNERS = {
    12: [(52.5, 82.96), (937.5, 77.05), (937.5, 1337.5), (52.5, 1337.5)],
    24: [(33.22, 57.67), (1002.5, 49.89), (1002.5, 1369.85), (32.04, 1373.89)],
}

# Polices : entraînement = polices2 (haller) ; test = polices réservées (gen_cursive.POLICES_EVAL)
# HALLER_FONTS : dossier des 287 polices de `python -m haller.fonts` ; HALLER_TEST_FONTS : polices réservées au mode test
POLICES_TRAIN = os.environ.get("HALLER_FONTS", os.path.join(ICI, "..", "..", "fonts"))
POLICES_TEST_DIR = os.environ.get("HALLER_TEST_FONTS", os.path.join(ICI, "..", "..", "fonts_test"))
POLICES_TEST = [
    "playwritefrtrad/PlaywriteFRTrad[wght].ttf", "playwritegbj/PlaywriteGBJ[wght].ttf",
    "playwritedesas/PlaywriteDESAS[wght].ttf", "playwriteitmoderna/PlaywriteITModerna[wght].ttf",
    "playwriteusmodern/PlaywriteUSModern[wght].ttf", "playwritees/PlaywriteES[wght].ttf",
    "leaguescript/LeagueScript-Regular.ttf", "meddon/Meddon.ttf", "fleurdeleah/FleurDeLeah-Regular.ttf",
]
CURSIVES = set(open(os.path.join(ICI, "cursive_fonts.txt")).read().split())


def polices(test):
    if test:
        rep = POLICES_TEST_DIR
        ps = [os.path.join(rep, p) for p in POLICES_TEST if os.path.exists(os.path.join(rep, p))]
        if not ps:  # dossier sans ces polices : toutes les polices du dossier
            ps = [os.path.join(rep, f, g) for f in sorted(os.listdir(rep)) if os.path.isdir(os.path.join(rep, f))
                  for g in os.listdir(os.path.join(rep, f)) if g.endswith(".ttf")]
        return {"cursive": ps, "minuscules": ps}
    fs = sorted(f for f in os.listdir(POLICES_TRAIN) if f.lower().endswith((".ttf", ".otf")))
    # polices aux minuscules minuscules (x < 0,25 em : hampes et boucles 3 à 4 fois le corps, mots de 10 à 120 px
    # selon leurs lettres) écartées : proportions d'aucune écriture réelle (docs/variete.md ; 7 polices sur 287)
    fs = [f for f in fs if hauteur_x(os.path.join(POLICES_TRAIN, f)) >= X_MIN_POLICE]
    cur = [os.path.join(POLICES_TRAIN, f) for f in fs if f in CURSIVES]
    autres = [os.path.join(POLICES_TRAIN, f) for f in fs if f not in CURSIVES]
    return {"cursive": cur or autres, "minuscules": autres}


def hauteur_x(path, cache={}):
    if path not in cache:
        try:
            bb = ImageFont.truetype(path, 100).getbbox("x", anchor="ls")
            cache[path] = max(0.2, -bb[1] / 100)
        except Exception:
            cache[path] = 0.5
    return cache[path]


# Contrôle de la variété (docs/variete.md, registre train/variete/registre.json) : bornes réalistes appliquées sans
# tirage supplémentaire.
BLANC_MAX = 0.7      # plus grand blanc entre lettres d'un mot, en hauteurs de sa boîte d'encre (réel : p99 0,66-0,71)
X_MIN_POLICE = 0.25  # hauteur des minuscules d'une police d'entraînement, au moins 0,25 em
MIN_AMINCI = 48      # hauteur (px) sous laquelle le trait EMNIST n'est pas aminci (sinon lettres en morceaux)


def limiter_blancs(m, cap=BLANC_MAX):
    """Réduit à `cap` hauteurs de la boîte d'encre tout blanc intérieur d'un mot (colonnes retirées au milieu du
    blanc). m : image du mot, 255 = papier. Encre : plus sombre qu'à mi-chemin du trait le plus sombre et du papier."""
    m = np.asarray(m)
    if m.ndim != 2 or m.size == 0:
        return m
    noir = float(m.min())
    if noir > 245:
        return m
    ink = m < 255 - 0.5 * (255 - noir)
    rows = np.nonzero(ink.any(1))[0]
    cols = np.nonzero(ink.any(0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return m
    lim = max(1, int(cap * (rows[-1] - rows[0] + 1)))
    c = ink.any(0)
    garder = np.ones(m.shape[1], bool)
    x, x1 = cols[0], cols[-1]
    while x <= x1:
        if not c[x]:
            d = x
            while x <= x1 and not c[x]:
                x += 1
            n = x - d
            if n > lim:
                k = n - lim
                a = d + (n - k) // 2
                garder[a:a + k] = False
        else:
            x += 1
    return m if garder.all() else np.ascontiguousarray(m[:, garder])


class Main:
    """Écriture d'une page : style, police, pente, épaisseur, encre."""
    def __init__(self, style, rng, ex, par, fonts):
        self.style, self.ex, self.par = style, ex, par
        self.epais = int(rng.choice([-1, 0, 0, 0, 1, 1]))
        self.pente = rng.normal(0, 5)
        self.esp = rng.uniform(0.1, 0.25)
        # UNIV_ESP_LARGE=p (nuit du 20-21/11) : sur une part p des pages, lettres espacées de 0,3 à 0,5 hauteur de mot
        # (capitales détachées écrites dans une case, « F U E L » des photos web) ; tirage seulement si la variable est
        # posée, pour ne pas changer les autres lots
        if os.environ.get("UNIV_ESP_LARGE") and rng.random() < float(os.environ["UNIV_ESP_LARGE"]):
            self.esp = rng.uniform(0.3, 0.5)  # au plus ~0,5 hauteur de lettre (réel, retour du 21/11)
        self.encre = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 110)
        J.noter("esp_lettres_tire_h", self.esp)
        J.noter("pente_ecriture_deg", self.pente)
        J.noter("epaisseur_trait", self.epais)
        J.noter("encre_niveau", self.encre)
        self.font = None
        # UNIV_POLICE_RARE=1 (compositeur, retouche du 20/11) : l'écriture réelle dominante est en capitales
        # irrégulières au stylo bille ; capitales en police 15 % au lieu de 30 %, trait des polices plus fin
        rare = bool(os.environ.get("UNIV_POLICE_RARE"))
        if style != "capitales" or rng.random() < (0.15 if rare else 0.3):
            self.font = fonts[style if style != "capitales" else "minuscules"][
                rng.integers(len(fonts[style if style != "capitales" else "minuscules"]))]
            self.x = rng.uniform(0.5, 0.65)
            self.trait = rng.uniform(0.03, 0.06) if rare else rng.uniform(0.05, 0.09)
            self.casse = "maj" if style == "capitales" else ("cap" if rng.random() < 0.15 else "min")
            J.noter("trait_police", self.trait)
        J.noter("police", os.path.basename(self.font) if self.font else "emnist")

    def composer(self, texte, haut, rng, pente=None):
        pente = self.pente if pente is None else pente
        if self.font is None:
            # trait EMNIST aminci seulement pour une écriture assez grande (sinon lettres en morceaux)
            epais = self.epais if self.epais >= 0 or haut >= MIN_AMINCI else 0
            m, base = G.composer(texte, haut, self.ex, self.par, rng, epais, pente, self.esp)
            return limiter_blancs(m), base
        if self.casse == "maj":
            texte = texte.upper()
            taille = max(8, int(haut / 0.72))
        else:
            texte = texte.lower() if self.casse == "min" else texte.capitalize()
            taille = max(8, int(self.x * haut / hauteur_x(self.font)))
        f = ImageFont.truetype(self.font, taille)
        W = int(taille * (0.9 * len(texte) + 3))
        H = int(taille * 2.4)
        base = int(taille * 1.5)
        im = Image.new("L", (W, H), 255)
        ImageDraw.Draw(im).text((int(taille * 0.4), base), texte, font=f, fill=0, anchor="ls")
        bb = Image.eval(im, lambda v: 255 - v).getbbox()
        if bb:
            im = im.crop((0, 0, min(W, bb[2] + int(taille * 0.3)), H))
        trait = int(round(self.trait * haut * (self.x if self.casse != "maj" else 0.7) / 2)) + max(0, self.epais)
        if trait > 0:
            im = im.filter(ImageFilter.MinFilter(2 * trait + 1))
        s = math.tan(math.radians(pente))
        if abs(s) > 1e-3:
            marge = int(abs(s) * H) + 2
            im2 = Image.new("L", (im.width + 2 * marge, H), 255)
            im2.paste(im, (marge, 0))
            im = im2.transform(im2.size, Image.AFFINE, (1, s, -s * base, 0, 1, 0), Image.BILINEAR, fillcolor=255)
        return limiter_blancs(np.asarray(im)), base


def poser(page, m, x, y, encre, objets, info):
    """Encre m (255 papier) en (x, y) et enregistre la boîte d'encre dans objets."""
    G.encrer(page, m, x, y, encre)
    ys, xs = np.nonzero(np.asarray(m) < 128)
    if info.get("classe") == "mot" and len(ys):
        JR.noter_mot(m)
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
    objets.append(dict(info, quad=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))


def page_liste(mots, main, rng, opts):
    W, H = G.PAGE
    fond = rng.uniform(205, 245)
    page = np.full((H, W), fond, np.float32)
    n, ncol = len(mots), opts["colonnes"]
    par_col = -(-n // ncol)
    haut = rng.uniform(50, 80) if ncol == 1 else rng.uniform(45, 65)
    pas = haut * rng.uniform(1.45, 2.1)
    ytop = rng.uniform(0.06, 0.18) * H
    J.noter("interligne_h", pas / haut)
    if par_col * pas + ytop > 0.95 * H:
        f = 0.95 * H / (par_col * pas + ytop)
        haut, pas = haut * f, pas * f
    if opts["papier"] == "ligne":
        im = Image.fromarray(page.astype(np.uint8))
        d = ImageDraw.Draw(im)
        gris = int(rng.uniform(140, 200))
        for yy in np.arange((ytop + haut * 1.2) % pas, H, pas):
            d.line([(0, yy), (W, yy)], fill=gris, width=int(rng.integers(1, 4)))
        if rng.random() < 0.5:
            xm = int(rng.uniform(0.08, 0.14) * W)
            d.line([(xm, 0), (xm, H)], fill=gris, width=2)
        page = np.asarray(im, np.float32).copy()
    fmt = rng.choice(["{}.", "{})", "{}"])
    xg = rng.uniform(0.07, 0.15) * W
    largeur_col = (W - xg) / ncol
    objets = []
    for k, mot in enumerate(mots):
        col, row = divmod(k, par_col)
        pente = main.pente + rng.normal(0, 1)
        yb = ytop + haut * 1.2 + row * pas + rng.normal(0, haut * 0.03)
        if opts["papier"] == "ligne":
            yb -= rng.uniform(0, 0.12) * haut
        x = int(xg + col * largeur_col + rng.normal(0, haut * 0.06))
        if opts["numeros"]:
            mn, bn = main.composer(fmt.format(k + 1), haut, rng, pente)
            poser(page, mn, x, int(yb - bn), main.encre, objets, dict(classe="numero", k=k, rang=row, col=col))
            e = rng.uniform(0.25, 0.7)
            J.ajouter("ecart_numero_mot_h", e)
            x += mn.shape[1] + int(haut * e)
        m, base = main.composer(mot.upper() if main.font is None else mot, haut, rng, pente)
        if m.shape[1] > 0.95 * largeur_col:
            f = 0.95 * largeur_col / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), max(5, int(m.shape[0] * f)))))
            base = int(base * f)
        m = np.asarray(Image.fromarray(m).rotate(rng.normal(0, 0.8), expand=False, fillcolor=255))
        poser(page, m, x, int(yb - base), main.encre, objets, dict(classe="mot", k=k, rang=row, col=col, mot=mot))
    return Image.fromarray(np.clip(page, 0, 255).astype(np.uint8)), objets, None


_numeros_cache = {}


def numeros_imprimes(n):
    """Boîtes d'encre des numéros imprimés sur le gabarit de carte vierge."""
    if n not in _numeros_cache:
        sup = C._support("carte12" if n == 12 else "carte24")
        a = np.asarray(sup["img"])
        boites = []
        # UNIV_NUMEROS_CORRIGES=1 : la zone commence après le bord noir vertical de la carte (sinon les numéros
        # 1 à 9, plus loin de la case, prennent le bord dans leur boîte : x 40 au lieu de 114 sur la carte 12)
        bord = 0
        if os.environ.get("UNIV_NUMEROS_CORRIGES"):
            col = (a[300:1200] < 128).mean(0)
            bord = int(np.nonzero(col[:120] > 0.5)[0].max()) + 12
        for x0, x1, y in sup["lignes"]:
            X0 = max(bord, x0 - 140)
            z = a[y - 50:y + 6, X0:x0 - 4] < 128
            ys, xs = np.nonzero(z)
            boites.append((X0 + xs.min(), y - 50 + ys.min(), X0 + xs.max() + 1,
                           y - 50 + ys.max() + 1) if len(xs) else None)
        _numeros_cache[n] = boites
    return _numeros_cache[n]


def face_carte(mots, main, rng, n):
    """Recto de carte (gabarit 1004 x 1401 avec bord noir), objets dans ses coordonnées."""
    sup = C._support("carte12" if n == 12 else "carte24")
    a = np.asarray(sup["img"], np.float32).copy()
    objets = []
    # réel : les mots remplissent ~90 % de l'interligne de la carte (hauteur des mots 0,053 du petit côté)
    haut0 = sup["ecart"] * (rng.uniform(0.75, 1.0) if REEL else rng.uniform(0.45, 0.85))
    for k, (x0, x1, yl) in enumerate(sup["lignes"]):
        if mots is None:
            break
        haut = haut0 * rng.uniform(0.96, 1.04)
        m, base = main.composer(mots[k].upper() if main.font is None else mots[k], haut, rng)
        if m.shape[1] > x1 - x0 + 10:
            f = (x1 - x0 + 10) / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), m.shape[0])))
        x = x0 + int(rng.uniform(-0.25, 0.1) * haut)
        poser(a, m, x, int(yl - rng.uniform(0, 0.15) * haut - base), main.encre, objets,
              dict(classe="mot", k=k, rang=k % 12, col=k // 12, mot=mots[k]))
    for k, b in enumerate(numeros_imprimes(n)):
        if b:
            objets.append(dict(classe="numero", k=k, rang=k % 12, col=k // 12, imprime=True,
                               quad=[(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]))
    w, h = sup["img"].size
    bord = int(0.07 * w)
    carte = np.full((h + 2 * bord, w + 2 * bord), rng.uniform(20, 45), np.float32)
    carte[bord:bord + h, bord:bord + w] = a
    decal = lambda q: [(x + bord, y + bord) for x, y in q]  # noqa: E731
    for o in objets:
        o["quad"] = decal(o["quad"])
    return carte, objets, decal(CARD_CORNERS[n])


def page_carte(mots, main, rng, planche=False):
    n = len(mots)
    carte, objets, coins = face_carte(mots, main, rng, n)
    cartes = [dict(coins=coins, cible=True, n=n)]
    if not planche:
        return Image.fromarray(np.clip(carte, 0, 255).astype(np.uint8)), objets, cartes
    # planche imprimée : grille de cartes jointives sur une feuille, la cible à une position au hasard
    h, w = carte.shape
    nx, ny = int(rng.integers(2, 4)), int(rng.integers(1, 3))
    ix, iy = int(rng.integers(nx)), int(rng.integers(ny))
    gap = int(rng.uniform(0, 0.08) * w)
    marge = int(rng.uniform(0.03, 0.1) * w)
    feuille = np.full((ny * h + (ny - 1) * gap + 2 * marge, nx * w + (nx - 1) * gap + 2 * marge),
                      rng.uniform(215, 245), np.float32)
    tous, cartes = [], []
    for j in range(ny):
        for i in range(nx):
            ox, oy = marge + i * (w + gap), marge + j * (h + gap)
            if (i, j) == (ix, iy):
                c, ob, co, cible = carte, objets, coins, True
            else:
                autres = list(mnemonique(rng.bytes(n * 4 // 3))) if rng.random() < 0.25 else None
                c, ob, co = face_carte(autres, main, rng, n)
                cible = False
            feuille[oy:oy + h, ox:ox + w] = c
            for o in ob:
                tous.append(dict(o, quad=[(x + ox, y + oy) for x, y in o["quad"]], autre=not cible))
            cartes.append(dict(coins=[(x + ox, y + oy) for x, y in co], cible=cible, n=n))
    return Image.fromarray(np.clip(feuille, 0, 255).astype(np.uint8)), tous, cartes


def dlt(src, dst):
    """Homographie 3x3 src -> dst (4 points)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, float))
    Hm = vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def appliquer(Hm, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def cadrage(page, defaut, rng, carte):
    """Homographie page -> photo (coordonnées de la photo pleine 1944 x 2592) selon le défaut."""
    W, H = FRAME
    remplir = (not carte) and rng.random() < 0.25
    if carte:
        hp = H * rng.uniform(0.42, 0.62) if page.height < 2000 else H * rng.uniform(0.75, 0.95)
        if REEL and page.height < 2000:   # réel : carte = 0,43 de l'image [0,19-0,95] (généré avant : 0,24)
            hp = H * float(np.clip(rng.normal(0.72, 0.13), 0.42, 0.98))
    else:
        if REEL:
            remplir = rng.random() < 0.35   # réel : feuille = 0,77 de l'image (généré avant : 0,67)
        hp = H * (rng.uniform(1.0, 1.25) if remplir else rng.uniform(0.7, 0.95) if not REEL else rng.uniform(0.8, 1.0))
    ang = rng.normal(0, 1.5 if REEL else 4)   # réel : inclinaison 0,8° [0-5,9]
    persp = 0.035
    cx, cy = W / 2 + rng.normal(0, 0.04 * W), H / 2 + rng.normal(0, 0.03 * H)
    if defaut == "proche":
        hp *= rng.uniform(1.3, 1.9)
        # gros plan : au plus 1,3 hauteur de cadre pour une carte (~70 % visible), 1,5 pour une feuille ou une scène
        # de plusieurs documents (docs/variete.md : 1/3 d'une carte seulement à 2,1 hauteurs, implausible)
        hp = min(hp, (1.3 if carte else 1.5) * H)
        cx += rng.normal(0, 0.1 * W)
        cy += rng.normal(0, 0.1 * H)
    elif defaut == "loin":
        hp *= rng.uniform(0.35, 0.55)
        cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    elif defaut == "perspective":
        persp = rng.uniform(0.08, 0.16)
    elif defaut == "rotation":
        # rotation forte, cas rare : 6-18° (réel : max 10,9°, supports vierges 17° ; jamais 30-45°, 21/11)
        ang = rng.choice([-1, 1]) * rng.uniform(6, 18)
    elif defaut == "retournee":
        ang = 180 + rng.normal(0, 6)
    J.noter("rotation_deg", ang)
    J.noter("perspective_amp", persp)
    J.noter("remplir", remplir)
    wp = hp * page.width / page.height
    if (AJUSTER_LARGEUR or carte) and defaut not in ("proche", "coupe_cote") and wp > 0.95 * W:
        # cadre plus étroit que 3:4 (UNIV_CADRE de gen_univ.py), ou carte en paysage : le document est cadré pour
        # tenir en largeur (une carte n'est coupée des deux côtés que de près ; docs/variete.md)
        hp, wp = hp * 0.95 * W / wp, 0.95 * W
    if defaut in ("rotation", "retournee"):  # la page tournée doit tenir dans le cadre
        a = math.radians(ang)
        ext_w = abs(wp * math.cos(a)) + abs(hp * math.sin(a))
        ext_h = abs(wp * math.sin(a)) + abs(hp * math.cos(a))
        f = min(1.0, 0.95 * W / ext_w, 0.95 * H / ext_h) if not remplir else 1.0
        wp, hp = wp * f, hp * f
    coupe = rng.uniform(0.08, 0.35)
    if defaut == "coupe_haut":
        cy = hp / 2 - coupe * hp
    elif defaut == "coupe_bas":
        cy = H - hp / 2 + coupe * hp
    elif defaut == "coupe_cote":
        c = rng.uniform(0.06, 0.25)
        J.noter("coupe_part", c)
        cx = (wp / 2 - c * wp) if rng.random() < 0.5 else (W - wp / 2 + c * wp)
    J.noter("hauteur_doc_cadre", hp / H)
    if defaut in ("coupe_haut", "coupe_bas"):
        J.noter("coupe_part", coupe)
    a = math.radians(ang)
    coins = np.array([(-wp / 2, -hp / 2), (wp / 2, -hp / 2), (wp / 2, hp / 2), (-wp / 2, hp / 2)])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = coins @ rot.T + (cx, cy) + rng.uniform(-persp, persp, (4, 2)) * hp
    return dlt([(0, 0), (page.width, 0), (page.width, page.height), (0, page.height)], dst)


def photographier(page, Hm, defaut, rng, scale):
    """Photo (taille FRAME * scale) de la page par l'homographie Hm (page -> photo pleine)."""
    W, H = int(round(FRAME[0] * scale)), int(round(FRAME[1] * scale))
    S = np.diag([scale, scale, 1.0]) @ Hm
    inv = np.linalg.inv(S)
    inv = inv / inv[2, 2]
    table = rng.uniform(15, 110)
    if defaut == "planche" or rng.random() < 0.15:
        table = rng.uniform(110, 200)  # table claire
    J.noter("table_gris", table)
    out = page.transform((W, H), Image.PERSPECTIVE, tuple(inv.ravel()[:8]), Image.BILINEAR, fillcolor=int(table))
    a = np.asarray(out, np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx /= W
    yy /= H
    g = 0.5 if defaut != "eclairage" else rng.uniform(0.9, 1.6)
    gx, gy = rng.uniform(-g, g, 2)
    eclair = 1 + gx * (xx - 0.5) + gy * (yy - 0.5)
    if defaut == "eclairage":
        cx, cy, r = rng.uniform(0, 1, 2).tolist() + [rng.uniform(0.2, 0.6)]
        eclair = eclair * (1 - rng.uniform(0.3, 0.6) * np.clip(((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2, 0, 1))
        if rng.random() < 0.5:  # reflet
            hx, hy = rng.uniform(0.2, 0.8, 2)
            a += rng.uniform(40, 120) * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * rng.uniform(0.03, 0.1) ** 2))
    gain = rng.uniform(0.55, 1.0)
    J.noter("eclairage_gradient", max(abs(gx), abs(gy)))
    J.noter("exposition_gain", gain)
    a = a * np.clip(eclair, 0.15, 2.0) * gain
    masque = None
    if defaut == "ombre":
        a, masque = ombres_objets(a, rng, W, H)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if REEL:   # réel : flou 1,1 px à 1024 [0,4-3,4], longue queue ; bruit plus faible que les pages simulées
        sig = rng.uniform(2.5, 4.0) if rng.random() < 0.1 else float(np.clip(np.exp(rng.normal(np.log(1.15), 0.4)),
                                                                            0.35, 2.5))
        bruit = rng.uniform(1, 4)
    else:
        sig, bruit = rng.uniform(0.4, 1.6), rng.uniform(2, 5)
    J.noter("flou_sigma_1024", sig * 1024 / FRAME[1])   # écart type du flou, px à 1024 de côté long
    J.noter("bruit_sigma", bruit)
    out = out.filter(ImageFilter.GaussianBlur(sig * scale))
    a = np.asarray(out, np.float32) + rng.normal(0, bruit * min(1.0, math.sqrt(scale * 2)), (H, W))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), masque


def ombres_objets(a, rng, W, H):
    """Ombres portées (assombrissement flou) et objets opaques ; renvoie le masque des objets."""
    masque = Image.new("L", (W, H), 0)
    for _ in range(int(rng.integers(1, 3))):
        om = Image.new("L", (W, H), 0)
        pts = [(rng.uniform(-0.2, 1.2) * W, rng.uniform(-0.2, 1.2) * H) for _ in range(int(rng.integers(3, 6)))]
        ImageDraw.Draw(om).polygon(pts, fill=255)
        om = np.asarray(om.filter(ImageFilter.GaussianBlur(rng.uniform(0.01, 0.05) * W)), np.float32) / 255
        a = a * (1 - rng.uniform(0.3, 0.65) * om)
    if rng.random() < 0.7:
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d, dm = ImageDraw.Draw(im), ImageDraw.Draw(masque)
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:  # stylo : bande allongée
                cx, cy, L, e, t = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.3, 0.7) * W, \
                    rng.uniform(0.02, 0.04) * W, rng.uniform(0, math.pi)
                dx, dy, nx, ny = math.cos(t) * L / 2, math.sin(t) * L / 2, -math.sin(t) * e / 2, math.cos(t) * e / 2
                pts = [(cx - dx + nx, cy - dy + ny), (cx + dx + nx, cy + dy + ny), (cx + dx - nx, cy + dy - ny),
                       (cx - dx - nx, cy - dy - ny)]
            else:  # doigt ou objet au bord du cadre
                bord = int(rng.integers(4))
                r = rng.uniform(0.05, 0.12) * W
                t = rng.uniform(0.1, 0.9)
                cx, cy = [(t * W, 0), (W, t * H), (t * W, H), (0, t * H)][bord]
                pts = [(cx + r * math.cos(u), cy + 1.6 * r * math.sin(u)) for u in np.linspace(0, 2 * math.pi, 16)]
            gris = int(rng.uniform(30, 200))
            d.polygon(pts, fill=gris)
            dm.polygon(pts, fill=255)
        a = np.asarray(im, np.float32)
    return a, np.asarray(masque) > 127


def generer(rng, ex, par, fonts, defaut=None, style=None, scale=1.0, type_=None):
    """Une photo annotée : (image, vérité). Les quadrilatères sont dans les coordonnées de la photo pleine."""
    defaut = defaut or DEFAUTS[rng.choice(len(DEFAUTS), p=np.array(POIDS_DEFAUTS) / sum(POIDS_DEFAUTS))]
    style = style or STYLES[rng.choice(3, p=np.array(POIDS_STYLES) / sum(POIDS_STYLES))]
    main = Main(style, rng, ex, par, fonts)
    if type_ is None:
        type_ = "carte" if defaut == "planche" else rng.choice(["carte12", "carte24", "liste1", "liste2"],
                                                               p=[0.27, 0.13, 0.38, 0.22])
        if type_ == "carte":
            type_ = "carte12" if rng.random() < 0.65 else "carte24"
    if type_.startswith("carte"):
        n = 12 if type_ == "carte12" else 24
        mots = list(mnemonique(rng.bytes(n * 4 // 3)))
        page, objets, cartes = page_carte(mots, main, rng, planche=defaut == "planche")
        info = dict(type="carte", colonnes=1 if n == 12 else 2, numeros=True, papier="carte")
    else:
        ncol = 1 if type_ == "liste1" else 2
        n = int(rng.choice([12, 15, 18, 21, 24])) if ncol == 1 else int(rng.choice([24, 12, 18]))
        if ncol == 1 and rng.random() < 0.5:
            n = 12
        mots = list(mnemonique(rng.bytes(n * 4 // 3)))
        opts = dict(colonnes=ncol, numeros=bool(rng.random() < 0.5), papier=str(rng.choice(["blanc", "ligne"])))
        page, objets, cartes = page_liste(mots, main, rng, opts)
        info = dict(type="liste", **opts)
    Hm = cadrage(page, defaut, rng, type_.startswith("carte") and defaut != "planche")
    img, masque = photographier(page, Hm, defaut, rng, scale)
    for o in objets:
        o["quad"] = [tuple(map(float, p)) for p in appliquer(Hm, o["quad"])]
    for c in cartes or []:
        c["coins"] = [tuple(map(float, p)) for p in appliquer(Hm, c["coins"])]
    if masque is not None:  # objets posés : mots cachés à plus de 25 % marqués
        for o in objets:
            q = np.array(o["quad"]) * scale
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, masque.shape[1]), min(y1, masque.shape[0])
            if x1 > x0 and y1 > y0 and masque[y0:y1, x0:x1].mean() > 0.25:
                o["cache"] = True
    v = dict(info, mots=mots, n=n, rotation=0, style=style, defaut=defaut, objets=objets, cartes=cartes,
             police=os.path.basename(main.font) if main.font else "emnist")
    return img, v


# ---------- cibles d'entraînement ----------

def visible_part(q, W, H):
    q = np.asarray(q)
    return not (q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H)


def noyau(q, f_h=0.4, f_w=0.25):
    """Quadrilatère réduit : bande centrale (f_h de la hauteur), bouts rentrés de f_w hauteur."""
    q = np.asarray(q, float)
    a, b, c, d = q
    h = (np.linalg.norm(d - a) + np.linalg.norm(c - b)) / 2
    w = (np.linalg.norm(b - a) + np.linalg.norm(c - d)) / 2
    ux = ((b - a) + (c - d)) / 2
    ux /= max(1e-6, np.linalg.norm(ux))
    uy = ((d - a) + (c - b)) / 2
    uy /= max(1e-6, np.linalg.norm(uy))
    centre = q.mean(0)
    hw = max(w / 2 - f_w * h, min(w / 2, 0.15 * h))
    hh = f_h * h / 2
    return [centre - ux * hw - uy * hh, centre + ux * hw - uy * hh, centre + ux * hw + uy * hh,
            centre - ux * hw + uy * hh], h


def cibles(v, scale_out, ignore_cut=True):
    """Cartes cibles à la résolution de sortie (OUT_SIZE) :
    0 noyau de mot, 1 noyau de numéro, 2 zone claire de carte, 3-6 coins (HG, HD, BD, BG, gaussiennes),
    7 hauteur de la boîte (pixels de sortie x 8), 8 masque d'exclusion (255 : non noté)."""
    W, H = OUT_SIZE
    ims = [Image.new("L", (W, H), 0) for _ in range(9)]
    dr = [ImageDraw.Draw(i) for i in ims]
    for o in v["objets"]:
        q = np.array(o["quad"]) * scale_out
        if not visible_part(q, W, H):
            continue
        k, h = noyau(q)
        pts = [tuple(p) for p in k]
        canal = 0 if o["classe"] == "mot" else 1
        if o.get("cache"):
            dr[8].polygon([tuple(p) for p in q], fill=255)
            continue
        dr[canal].polygon(pts, fill=255)
        c = np.mean(k, 0)
        dr[canal].point(tuple(c), fill=255)
        dr[7].polygon(pts, fill=int(min(255, h * 8)))
        dr[7].point(tuple(c), fill=int(min(255, h * 8)))
    for c in v["cartes"] or []:
        q = np.array(c["coins"]) * scale_out
        dr[2].polygon([tuple(p) for p in q], fill=255)
    y = np.stack([np.asarray(i) for i in ims], -1).copy()
    yy, xx = np.mgrid[0:H, 0:W]
    for c in v["cartes"] or []:
        q = np.array(c["coins"]) * scale_out
        s = max(1.0, 0.012 * np.linalg.norm(q[2] - q[0]))
        for j in range(4):
            g = np.exp(-((xx - q[j, 0]) ** 2 + (yy - q[j, 1]) ** 2) / (2 * s * s))
            y[..., 3 + j] = np.maximum(y[..., 3 + j], (g * 255).astype(np.uint8))
    return y


VARIABLES_ENV = ("UNIV_REEL", "UNIV_ESP_LARGE", "UNIV_POLICE_RARE", "UNIV_NUMEROS_CORRIGES", "UNIV_ROT", "UNIV_ALTERE",
                 "UNIV_VERSO", "UNIV_MULTI", "UNIV_CADRE", "UNIV_SANS_LIGNES", "MODELES_CARTES")


def config_journal(graine, nombre, test, generateur="gen_det"):
    srcs = [os.path.join(ICI, "gen_det.py"), G.__file__]
    if generateur == "gen_univ":
        srcs.append(os.path.join(ICI, "gen_univ.py"))
    c = J.config_lot(generateur, srcs, VARIABLES_ENV + tuple(k for k in os.environ if k.startswith("UNIV_")))
    c.update(graine=graine, nombre=nombre, test=bool(test), argv=sys.argv[1:])
    return c


def main():
    mode, sortie, nombre, graine = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    test = "--test" in sys.argv
    opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None  # noqa: E731
    rng = np.random.default_rng(graine)
    ex, par = G.charger_emnist(train=not test)
    fonts = polices(test)
    if mode == "train":
        scale = 2 * NET_SIZE[0] / FRAME[0]
        X = np.zeros((nombre, NET_SIZE[1], NET_SIZE[0]), np.uint8)
        Y = np.zeros((nombre, OUT_SIZE[1], OUT_SIZE[0], 9), np.uint8)
        meta = []
        jw = J.Ecrivain(sortie, config_journal(graine, nombre, test))
        for i in range(nombre):
            J.ouvrir("gen_det", i)
            img, v = generer(rng, ex, par, fonts, scale=scale)
            jw.ecrire(JR.finir_page(v, FRAME))
            X[i] = np.asarray(img.reduce(2))
            Y[i] = cibles(v, OUT_SIZE[0] / FRAME[0])
            meta.append(dict(style=v["style"], defaut=v["defaut"], type=v["type"], n=v["n"]))
        jw.close()
        np.savez_compressed(sortie, x=X, y=Y, meta=json.dumps(meta))
    else:
        os.makedirs(sortie, exist_ok=True)
        verite = {}
        jw = J.Ecrivain(sortie, config_journal(graine, nombre, test))
        for i in range(nombre):
            J.ouvrir("gen_det", i)
            img, v = generer(rng, ex, par, fonts, defaut=opt("--defaut"), style=opt("--style"), type_=opt("--type"))
            nom = f"det{graine:03d}_{i:04d}.jpg"
            q = int(rng.integers(70, 95))
            J.noter("jpeg_qualite", q)
            J.noter("image", nom)
            jw.ecrire(JR.finir_page(v, FRAME))
            img.save(os.path.join(sortie, nom), quality=q)
            verite[nom] = v
        jw.close()
        json.dump(verite, open(os.path.join(sortie, "verite.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
