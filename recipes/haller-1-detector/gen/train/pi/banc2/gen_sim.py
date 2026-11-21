"""Photos simulées de seeds manuscrites (EMNIST : data/emnist_train.npz, sinon torchvision).

Supports : liste libre (papier blanc ou ligné ; 1 ou 2 colonnes ; 12, 15, 18,
21 ou 24 mots ; avec ou sans numéros « 1. » / « 1) » / « 1 »), cartes
officielles SeedSigner 12 et 24 mots.
Écriture soignée (quelqu'un qui recopie sa seed) : lettres et chiffres EMNIST
(partie test par défaut, jamais vus à l'entraînement), capitales, taille,
espacement et pente constants à peu de chose près sur la page, épaisseur du
trait (filtres min/max), encre surtout foncée.
Prise de vue : page posée sur une table (ou remplissant le cadre), rotation,
perspective, cadrage, éclairage en dégradé, flou, bruit, JPEG ; image portrait
1944x2592 (orientation déjà corrigée : rotation 0 dans la vérité).

Usage : python gen_sim.py <dossier> <nombre> [graine] [--train]
Sortie : <dossier>/*.jpg et <dossier>/verite.json
  {nom: {type, groupe, mots, rotation, colonnes, numeros, papier, n}}
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ICI = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(ICI, "..", ".."))  # dossier train/ du dépôt
sys.path.insert(0, TRAIN)
os.chdir(TRAIN)
import cartes as C  # noqa: E402
from officiel import homographie  # noqa: E402
from seed_sim import mnemonique  # noqa: E402

PAGE = (1400, 1980)  # feuille A5 à environ 190 dpi
FRAME = (1944, 2592)  # photo portrait


def charger_emnist(train):
    # Export numpy (data/emnist_<train|test>.npz) sur les machines sans torchvision
    chemin = os.path.join(TRAIN, "data", f"emnist_{'train' if train else 'test'}.npz")
    if os.path.exists(chemin):
        d = np.load(chemin)
        x, y = d["x"], d["y"]
        return x, {k: np.nonzero(y == k)[0] for k in range(36)}
    from torchvision.datasets import EMNIST
    d = EMNIST(os.path.join(TRAIN, "data"), split="byclass", train=train, download=False)
    x = d.data.numpy().transpose(0, 2, 1)
    y = d.targets.numpy()
    garder = y < 36  # 0-9 chiffres, 10-35 capitales
    x, y = x[garder], y[garder]
    return x, {k: np.nonzero(y == k)[0] for k in range(36)}


def glyphe(ex, par, cls, haut, rng, epais):
    i = rng.choice(par[cls])
    g = Image.fromarray(255 - ex[i]).crop(Image.fromarray(ex[i]).getbbox())
    hh = max(5, int(haut * rng.uniform(0.95, 1.04)))
    g = g.resize((max(3, int(g.width * hh / g.height)), hh), Image.BILINEAR)
    if epais > 0:  # trait plus épais (encre sombre sur fond clair : minimum)
        g = g.filter(ImageFilter.MinFilter(2 * epais + 1))
    elif epais < 0:  # trait plus fin
        g = g.filter(ImageFilter.MaxFilter(3))
    return np.asarray(g.rotate(rng.uniform(-3, 3), expand=True, fillcolor=255))


def composer(texte, haut, ex, par, rng, epais, pente, esp_moy):
    """Image d'un texte en capitales et chiffres (fond 255) et rangée de base.
    « . » et « ) » sont dessinés ; « » (espace) est un blanc."""
    elems = []
    for ch in texte:
        if ch == " ":
            elems.append(("esp", None))
        elif ch == ".":
            r = max(2, int(haut * rng.uniform(0.06, 0.1)))
            a = np.full((2 * r + 2, 2 * r + 2), 255, np.uint8)
            Image.fromarray(a).copy()
            im = Image.fromarray(a); ImageDraw.Draw(im).ellipse([1, 1, 2 * r, 2 * r], fill=0)
            elems.append(("point", np.asarray(im)))
        elif ch == ")":
            im = Image.new("L", (max(6, int(haut * 0.3)), int(haut)), 255)
            ImageDraw.Draw(im).arc([-int(haut * 0.3), 0, int(haut * 0.25), int(haut) - 1], -70, 70, fill=0, width=max(2, int(haut * 0.07)))
            elems.append(("lettre", np.asarray(im)))
        else:
            cls = int(ch) if ch.isdigit() else 10 + ord(ch.upper()) - 65
            elems.append(("lettre", glyphe(ex, par, cls, haut, rng, epais)))
    largeurs = [e.shape[1] if e is not None else int(haut * 0.6) for _, e in elems]
    esp = [int(haut * max(0.0, rng.normal(esp_moy, 0.035))) for _ in elems]
    W = sum(largeurs) + sum(esp) + int(haut)
    base = int(haut * 1.2)
    H = base + int(haut * 0.4)
    out = np.full((H, W), 255, np.uint8)
    x = int(haut * 0.3)
    for k, ((t, e), w) in enumerate(zip(elems, largeurs)):
        if k:
            x += esp[k]
        if e is not None:
            y = base - e.shape[0] + int(rng.integers(-1, 2))
            y = max(0, min(H - e.shape[0], y))
            ww = min(e.shape[1], W - x)
            out[y:y + e.shape[0], x:x + ww] = np.minimum(out[y:y + e.shape[0], x:x + ww], e[:, :ww])
        x += w
    im = Image.fromarray(out)
    if pente:  # cisaillement : haut des lettres décalé vers la droite pour pente > 0
        s = np.tan(np.radians(pente))
        marge = int(abs(s) * H) + 2
        im2 = Image.new("L", (W + 2 * marge, H), 255)
        im2.paste(im, (marge, 0))
        im = im2.transform(im2.size, Image.AFFINE, (1, s, -s * base, 0, 1, 0), Image.BILINEAR, fillcolor=255)
    return np.asarray(im), base


def encrer(page, m, x, y, encre):
    """Pose le texte m (255 = papier) sur la page en (x, y), encre de niveau « encre »."""
    m = np.asarray(m, np.float32) / 255
    h, w = m.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = m[y0 - y:y1 - y, x0 - x:x1 - x]
    zone = page[y0:y1, x0:x1]
    page[y0:y1, x0:x1] = np.minimum(zone, zone * sub + encre * (1 - sub))


def page_liste(mots, ex, par, rng, opts):
    W, H = PAGE
    fond = rng.uniform(205, 245)
    page = np.full((H, W), fond, np.float32)
    n = len(mots)
    ncol = opts["colonnes"]
    par_col = -(-n // ncol)
    haut = rng.uniform(50, 80) if ncol == 1 else rng.uniform(45, 65)
    pas = haut * rng.uniform(1.45, 2.1)
    papier = opts["papier"]
    ytop = rng.uniform(0.06, 0.18) * H
    if par_col * pas + ytop > 0.95 * H:  # page trop petite : écriture réduite
        f = 0.95 * H / (par_col * pas + ytop)
        haut, pas = haut * f, pas * f
    if papier == "ligne":
        pas_l = pas
        y0l = ytop + haut * 1.2
        im = Image.fromarray(page.astype(np.uint8))
        d = ImageDraw.Draw(im)
        gris = int(rng.uniform(140, 200))
        for yy in np.arange(y0l % pas_l, H, pas_l):
            d.line([(0, yy), (W, yy)], fill=gris, width=int(rng.integers(1, 4)))
        if rng.random() < 0.5:
            xm = int(rng.uniform(0.08, 0.14) * W)
            d.line([(xm, 0), (xm, H)], fill=gris, width=2)
        page = np.asarray(im, np.float32)
    # écriture soignée : encre plutôt foncée, trait régulier, pente constante
    encre = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 100)
    epais = int(rng.choice([-1, 0, 0, 0, 1, 1]))
    pente = rng.normal(0, 5)
    esp = rng.uniform(0.1, 0.25)
    numeros = opts["numeros"]
    fmt = rng.choice(["{}.", "{})", "{}"])
    xg = rng.uniform(0.07, 0.15) * W
    largeur_col = (W - xg) / ncol
    for k, mot in enumerate(mots):
        col, row = divmod(k, par_col)
        texte = mot.upper()
        if numeros:
            texte = fmt.format(k + 1) + " " + texte
        m, base = composer(texte, haut, ex, par, rng, epais, pente + rng.normal(0, 1), esp)
        if m.shape[1] > 0.95 * largeur_col:
            f = 0.95 * largeur_col / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), max(5, int(m.shape[0] * f)))))
            base = int(base * f)
        yb = ytop + haut * 1.2 + row * pas + rng.normal(0, haut * 0.03)
        if papier == "ligne":
            yb -= rng.uniform(0, 0.12) * haut
        x = int(xg + col * largeur_col + rng.normal(0, haut * 0.06))
        im = Image.fromarray(m).rotate(rng.normal(0, 0.8), expand=False, fillcolor=255)
        encrer(page, np.asarray(im), x, int(yb - base), encre)
    return Image.fromarray(np.clip(page, 0, 255).astype(np.uint8))


def page_carte(mots, ex, par, rng):
    nom = "carte12" if len(mots) == 12 else "carte24"
    sup = C._support(nom)
    img = sup["img"].copy()
    encre = rng.uniform(15, 55) if rng.random() < 0.8 else rng.uniform(55, 100)
    epais = int(rng.choice([-1, 0, 0, 0, 1, 1]))
    pente = rng.normal(0, 5)
    a = np.asarray(img, np.float32).copy()
    haut0 = sup["ecart"] * rng.uniform(0.45, 0.85)  # taille d'écriture propre à la carte
    esp_carte = rng.uniform(0.1, 0.25)
    for mot, (x0, x1, yl) in zip(mots, sup["lignes"]):
        haut = haut0 * rng.uniform(0.96, 1.04)
        m, base = composer(mot.upper(), haut, ex, par, rng, epais, pente, esp_carte)
        if m.shape[1] > x1 - x0 + 10:
            f = (x1 - x0 + 10) / m.shape[1]
            m = np.asarray(Image.fromarray(m).resize((int(m.shape[1] * f), m.shape[0])))
        x = x0 + int(rng.uniform(-0.25, 0.1) * haut)
        encrer(a, m, x, int(yl - rng.uniform(0, 0.15) * haut - base), encre)
    # la carte physique : bord noir autour de l'intérieur blanc
    w, h = img.size
    bord = int(0.07 * w)
    carte = np.full((h + 2 * bord, w + 2 * bord), rng.uniform(20, 45), np.float32)
    carte[bord:bord + h, bord:bord + w] = a
    return Image.fromarray(np.clip(carte, 0, 255).astype(np.uint8))


def photographier(page, rng, remplir=False):
    W, H = FRAME
    if remplir:  # la feuille dépasse du cadre
        hp = H * rng.uniform(1.0, 1.25)
    else:
        hp = H * rng.uniform(0.7, 0.95)
    wp = hp * page.width / page.height
    ang = np.radians(rng.normal(0, 4))
    cx, cy = W / 2 + rng.normal(0, 0.04 * W), H / 2 + rng.normal(0, 0.03 * H)
    coins = np.array([(-wp / 2, -hp / 2), (wp / 2, -hp / 2), (wp / 2, hp / 2), (-wp / 2, hp / 2)])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    persp = rng.uniform(-0.035, 0.035, (4, 2)) * hp
    dst = [tuple(p) for p in coins @ rot.T + (cx, cy) + persp]
    coef = homographie([(0, 0), (page.width, 0), (page.width, page.height), (0, page.height)], dst)
    table = rng.uniform(15, 110)
    out = page.transform((W, H), Image.PERSPECTIVE, coef, Image.BILINEAR, fillcolor=int(table))
    a = np.asarray(out, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx, gy = rng.uniform(-0.5, 0.5, 2)
    eclair = 1 + gx * (xx / W - 0.5) + gy * (yy / H - 0.5)
    a = a * eclair * rng.uniform(0.55, 1.0)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    a = np.asarray(out, np.float32) + rng.normal(0, rng.uniform(2, 5), (H, W))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


GROUPES = [
    # (groupe, poids, options)
    ("sim_liste", 3, dict(type="liste", n=[12], colonnes=1, numeros=False, papier="blanc")),
    ("sim_liste_num", 3, dict(type="liste", n=[12], colonnes=1, numeros=True, papier="blanc")),
    ("sim_ligne", 3, dict(type="liste", n=[12], colonnes=1, numeros=None, papier="ligne")),
    ("sim_2col", 2, dict(type="liste", n=[24, 12, 18], colonnes=2, numeros=None, papier=None)),
    ("sim_longueurs", 2, dict(type="liste", n=[15, 18, 21, 24], colonnes=1, numeros=None, papier=None)),
    ("sim_carte12", 2, dict(type="carte", n=[12])),
    ("sim_carte24", 1, dict(type="carte", n=[24])),
]


def main():
    dossier, nombre = sys.argv[1], int(sys.argv[2])
    graine = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
    rng = np.random.default_rng(graine)
    ex, par = charger_emnist(train="--train" in sys.argv)
    os.makedirs(dossier, exist_ok=True)
    poids = np.array([g[1] for g in GROUPES], float)
    verite = {}
    for i in range(nombre):
        g, _, o = GROUPES[rng.choice(len(GROUPES), p=poids / poids.sum())]
        n = int(rng.choice(o["n"]))
        mots = list(mnemonique(rng.bytes(n * 4 // 3)))
        if o["type"] == "carte":
            page = page_carte(mots, ex, par, rng)
            img = photographier(page, rng)
            info = dict(colonnes=1 if n == 12 else 2, numeros=True, papier="carte")
        else:
            opts = dict(colonnes=o["colonnes"], numeros=bool(rng.random() < 0.5) if o["numeros"] is None else o["numeros"],
                        papier=(rng.choice(["blanc", "ligne"]) if o["papier"] is None else o["papier"]))
            page = page_liste(mots, ex, par, rng, opts)
            img = photographier(page, rng, remplir=rng.random() < 0.25)
            info = opts
        nom = f"sim{graine:02d}_{i:04d}.jpg"
        img.save(os.path.join(dossier, nom), quality=int(rng.integers(70, 95)))
        verite[nom] = dict(type=o["type"], groupe=g, mots=mots, rotation=0, n=n, **info)
    json.dump(verite, open(os.path.join(dossier, "verite.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
