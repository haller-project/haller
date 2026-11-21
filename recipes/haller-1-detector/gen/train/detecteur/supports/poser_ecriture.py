"""Écriture simulée posée sur des photos de supports vierges réels (cartes, carnets, feuilles lignées ou à carreaux,
Post-it, modèles officiels imprimés), pour entraîner et évaluer le détecteur sur un fond réel.

Annotation d'un support (JSON à côté de la photo, NOM.json) :
    {"image": "NOM.jpg",
     "zones": [{"quad": [[x, y] x 4], "type": "case" | "ligne", "numero": 3,
                "numero_quad": [[x, y] x 4]}, ...],          # numéro imprimé de la case (facultatif)
     "modele": "seedsigner_mnemonic_12w", "coins": [[x, y] x 4],   # ou : modèle annoté (cases.json) et coins de
                                                                   # la carte dans la photo -> zones calculées
     "page": [[x, y] x 4], "lignes": {"y0": 0.1, "pas": 0.05, "n": 16, "marge": 0.12},   # ou : feuille lignée,
                                                                   # lignes en fraction de la page redressée
     "cartes": [[[x, y] x 4]],                                    # coins de carte pour la vérité (facultatif)
     "exclure": [[[x, y] x 4]]}                                    # encre déjà présente : zone non notée
Les quadrilatères sont dans le sens horaire depuis le haut à gauche, dans le sens d'écriture.

Rendu d'une zone : mot composé par gen_det.Main (même écriture que les pages simulées), posé dans le rectangle
redressé de la zone (hauteur 45 à 80 % de la case, ou de l'interligne, ligne de base près du bas), puis projeté
dans la photo par l'homographie de la zone. Encre : couleur (bleu, noir), opacité, épaisseur ; éclairage : l'encre
suit la luminosité locale du papier (ombres, reflets) ; netteté : flou appris de la photo (largeur des bords des
traits imprimés), bruit, recompression JPEG. Vérité au format des bancs (gen_univ) : objets mot / numéro (imprimé ou
écrit) avec leur quadrilatère d'encre, cartes.

    python3 supports/poser_ecriture.py banc DOSSIER_SUPPORTS SORTIE N GRAINE     photos + verite.json
    python3 supports/poser_ecriture.py lot DOSSIER_SUPPORTS SORTIE.npz N GRAINE   lot d'entraînement (x, y, meta)
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
import gen_det as D  # noqa: E402
import gen_univ as U  # noqa: E402
from gen_det import G, mnemonique  # noqa: E402

ENCRES = {"bleu": (35, 45, 120), "noir": (25, 25, 30), "bleu_clair": (60, 80, 170), "gris": (70, 70, 75)}


def homographie(src, dst):
    """Matrice 3 x 3 qui envoie les points src sur dst (4 points)."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(a, float), np.array(b, float))
    return np.append(h, 1).reshape(3, 3)


def appliquer(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def zones_support(a):
    """Zones d'écriture d'un support annoté : liste de dict(quad, type, numero, numero_quad)."""
    zones = list(a.get("zones", []))
    if a.get("modele") and a.get("coins"):
        cases = {m["id"]: m for m in json.load(open(os.path.join(U.MODELES, "cases.json")))}[a["modele"]]
        x0, y0, x1, y1 = cases["carte"]
        H = homographie([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], a["coins"])
        nums = {int(b[4]): b for b in cases["numeros"]}
        for c in cases["cases"]:
            q = appliquer(H, [(c[0], c[1]), (c[2], c[1]), (c[2], c[3]), (c[0], c[3])]).tolist()
            b = nums.get(int(c[4]))
            nq = appliquer(H, [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]).tolist() if b else None
            zones.append(dict(quad=q, type="case", numero=int(c[4]), numero_quad=nq))
    if a.get("page") and a.get("lignes"):
        L = a["lignes"]
        if L == "auto":
            L = lignes_auto(a)
        H = homographie([(0, 0), (1, 0), (1, 1), (0, 1)], a["page"])
        for k in range(L["n"]):
            yb = L["y0"] + L["pas"] * (k + 1)
            q = appliquer(H, [(L.get("marge", 0.08), yb - L["pas"]), (0.95, yb - L["pas"]), (0.95, yb),
                              (L.get("marge", 0.08), yb)]).tolist()
            zones.append(dict(quad=q, type="ligne", numero=None, numero_quad=None))
    return zones


def lignes_auto(a, support_dir=None):
    """Lignes d'une feuille lignée ou à carreaux : page redressée (1000 px de haut), profil vertical des pixels plus
    sombres que le papier, pas des lignes par autocorrélation, puis phase ; renvoie dict(y0, pas, n, marge) en
    fraction de la page (y0 = première ligne moins un pas)."""
    im = Image.open(os.path.join(support_dir or a["_dir"], a["image"])).convert("L")
    Hp = 1000
    Wp = int(Hp * 0.75)
    Hm = homographie([(0, 0), (Wp, 0), (Wp, Hp), (0, Hp)], a["page"])
    gx, gy = np.meshgrid(np.arange(Wp) + 0.5, np.arange(Hp) + 0.5)
    uv = appliquer(Hm, np.c_[gx.ravel(), gy.ravel()])
    from scipy import ndimage
    g = np.asarray(im, np.float32)
    r = ndimage.map_coordinates(g, [uv[:, 1], uv[:, 0]], order=1).reshape(Hp, Wp)
    fond = ndimage.uniform_filter(r, 25)
    sombre = (fond - r) > 8
    prof = sombre[:, int(0.1 * Wp):int(0.9 * Wp)].mean(1)
    prof = prof - ndimage.uniform_filter1d(prof, 61)
    ac = np.correlate(prof, prof, "full")[Hp - 1:]
    lo = 12
    pas = lo + int(np.argmax(ac[lo:250]))
    phase = int(np.argmax([prof[k::pas].sum() for k in range(pas)]))
    ys = np.arange(phase, Hp, pas)
    ys = ys[(ys > 0.04 * Hp) & (ys < 0.97 * Hp)]
    return dict(y0=(ys[0] - pas) / Hp, pas=pas / Hp, n=len(ys), marge=a.get("marge", 0.1))


def flou_photo(g):
    """Écart type (px) du flou de la photo, estimé sur les bords nets les plus raides."""
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    fort = np.percentile(m, 99.5)
    contraste = np.percentile(g, 95) - np.percentile(g, 5)
    # bord d'un échelon de hauteur C flouté par une gaussienne sigma : pente max C / (sigma sqrt(2 pi))
    return float(np.clip(contraste / max(fort, 1e-3) / 2.5, 0.4, 4.0))


def poser_mot(rgb, lum, zone, mot, main, encre, rng, flou):
    """Pose `mot` dans la zone ; renvoie le quadrilatère d'encre dans la photo, ou None."""
    q = np.array(zone["quad"], float)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if w < 8 or h < 6:
        return None
    k = 3.0  # rendu sur-échantillonné puis réduit par la projection
    W, Hh = int(w * k), int(h * k)
    haut = Hh * (rng.uniform(0.45, 0.8) if zone["type"] == "case" else rng.uniform(0.5, 0.85))
    m, base = U.ecrire_mot(main, mot, haut, rng, maxw=int(W * 0.95))
    m = np.asarray(m, np.float32)
    canvas = np.full((Hh, W), 255, np.float32)
    x = int(rng.uniform(0.02, max(0.03, 0.98 - m.shape[1] / W)) * W)
    yb = int(Hh * (rng.uniform(0.78, 0.95) if zone["type"] == "case" else rng.uniform(0.85, 1.0)))
    y = yb - base
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(Hh, y + m.shape[0]), min(W, x + m.shape[1])
    if y1 <= y0 or x1 <= x0:
        return None
    canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], m[y0 - y:y1 - y, x0 - x:x1 - x])
    alpha = 1 - canvas / 255
    if encre["kind"] == "crayon":
        alpha *= np.clip(rng.normal(0.8, 0.25, alpha.shape), 0.2, 1)
    ys, xs = np.nonzero(alpha > 0.5)
    if len(xs) == 0:
        return None
    # projection rectangle sur-échantillonné -> zone de la photo
    src = [(0, 0), (W, 0), (W, Hh), (0, Hh)]
    Hm = homographie(src, q)
    Hinv = np.linalg.inv(Hm)
    bx0, by0 = np.floor(q.min(0)).astype(int) - 2
    bx1, by1 = np.ceil(q.max(0)).astype(int) + 2
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(rgb.shape[1], bx1), min(rgb.shape[0], by1)
    if bx1 <= bx0 or by1 <= by0:
        return None
    gx, gy = np.meshgrid(np.arange(bx0, bx1) + 0.5, np.arange(by0, by1) + 0.5)
    uv = appliquer(Hinv, np.c_[gx.ravel(), gy.ravel()])
    # moyenne sur 3 x 3 sous-pixels de la source (anticrénelage simple)
    from scipy import ndimage
    a_lisse = ndimage.uniform_filter(alpha, size=int(k))
    a = ndimage.map_coordinates(a_lisse, [uv[:, 1], uv[:, 0]], order=1, cval=0).reshape(gx.shape)
    if flou > 0.5:
        a = ndimage.gaussian_filter(a, flou * 0.8)
    a = np.clip(a * encre["opacite"], 0, 1)
    # l'encre suit la lumière locale du papier
    lumiere = np.clip(lum[by0:by1, bx0:bx1] / max(1.0, np.percentile(lum, 95)), 0.15, 1.1)
    for c in range(3):
        cible = encre["couleur"][c] * lumiere
        rgb[by0:by1, bx0:bx1, c] = rgb[by0:by1, bx0:bx1, c] * (1 - a) + cible * a
    # quadrilatère d'encre dans la photo
    pts = [(xs.min(), ys.min()), (xs.max() + 1, ys.min()), (xs.max() + 1, ys.max() + 1), (xs.min(), ys.max() + 1)]
    return appliquer(Hm, pts).tolist()


def rendre(support_dir, a, rng, ex, par, fonts, style=None):
    """Une page : photo du support avec l'écriture, vérité au format des bancs."""
    im = Image.open(os.path.join(support_dir, a["image"])).convert("RGB")
    rgb = np.asarray(im, np.float32).copy()
    lum = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(9)), np.float32)
    flou = flou_photo(np.asarray(im.convert("L"), np.float32))
    zones = zones_support(dict(a, _dir=support_dir))
    cases = sorted([z for z in zones if z.get("numero")], key=lambda z: z["numero"])
    libres = [z for z in zones if not z.get("numero")]
    n_max = len(cases) or len(libres)
    longueurs = [n for n in U.SEED_LENGTHS if n <= n_max] or [n_max]
    n = int(rng.choice(longueurs))
    choisies = cases[:n] if cases else libres[:n]
    style = style or D.STYLES[rng.choice(3, p=np.array(D.POIDS_STYLES) / sum(D.POIDS_STYLES))]
    main = D.Main(style, rng, ex, par, fonts)
    kind = str(rng.choice(["bille", "feutre", "crayon"], p=[0.6, 0.25, 0.15]))
    nom_c = str(rng.choice(list(ENCRES), p=[0.5, 0.35, 0.1, 0.05])) if kind != "crayon" else "gris"
    couleur = np.array(ENCRES[nom_c], float) * rng.uniform(0.8, 1.15)
    encre = dict(kind=kind, couleur=couleur, opacite=rng.uniform(0.75, 1.0) if kind != "crayon" else rng.uniform(0.5, 0.8))
    mots = list(mnemonique(rng.bytes(n * 4 // 3))) if n in U.SEED_LENGTHS else \
        list(mnemonique(rng.bytes(16)))[:n]
    objets = []
    ecrire_numeros = not cases and rng.random() < 0.6
    for k, (z, mot) in enumerate(zip(choisies, mots)):
        qz = np.array(z["quad"], float)
        if ecrire_numeros:   # numéro écrit à la main devant le mot, sur le premier cinquième de la zone
            f = rng.uniform(0.1, 0.16)
            zn = dict(type=z["type"], quad=[qz[0].tolist(), (qz[0] + f * (qz[1] - qz[0])).tolist(),
                                             (qz[3] + f * (qz[2] - qz[3])).tolist(), qz[3].tolist()])
            qn = poser_mot(rgb, lum, zn, f"{k + 1}" + str(rng.choice(["", ".", ")"])), main, encre, rng, flou)
            if qn:
                objets.append(dict(classe="numero", k=k, valeur=k + 1, quad=qn))
            z = dict(z, quad=[zn["quad"][1], qz[1].tolist(), qz[2].tolist(), zn["quad"][2]])
        q = poser_mot(rgb, lum, z, mot, main, encre, rng, flou)
        if q:
            objets.append(dict(classe="mot", k=k, mot=mot, quad=q))
    for z in cases:   # numéros imprimés : toutes les cases, remplies ou non (comme les modèles simulés)
        if z.get("numero_quad"):
            objets.append(dict(classe="numero", k=z["numero"] - 1, valeur=z["numero"], imprime=True,
                               quad=z["numero_quad"]))
    for qx in a.get("exclure", []):   # encre déjà présente sur le support : zone non notée
        objets.append(dict(classe="mot", cache=True, quad=qx))
    rgb += rng.normal(0, rng.uniform(1, 4), rgb.shape)
    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    cartes = [dict(coins=c, cible=True, n=n) for c in a.get("cartes", [])]
    if a.get("coins") and not cartes:
        cartes = [dict(coins=a["coins"], cible=True, n=n)]
    v = dict(objets=objets, cartes=cartes, liens=[], style=style, defaut="support_reel", n=n,
             type="carte" if cartes else "liste", format="carte" if cartes else "liste",
             support=a.get("support", "reel"), encre=kind, couleur=nom_c, flou=round(flou, 2),
             source=a["image"], ambigu=False)
    return out, v


def tourner(im, v, rng):
    """Rotation de 0, 90, 180 ou 270 degrés (photo et vérité)."""
    k = int(rng.choice(4, p=[0.7, 0.1, 0.1, 0.1]))
    for _ in range(k):
        w, h = im.size
        im = im.transpose(Image.ROTATE_90)
        f = lambda pts: [[y, w - x] for x, y in pts]  # noqa: E731
        for o in v["objets"]:
            o["quad"] = f(o["quad"])
        for c in v["cartes"]:
            c["coins"] = f(c["coins"])
    return im, v


def en_cibles(im, v):
    """Image réduite au format du réseau (gris, 288 x 384, bandes de gris médian) et cibles (192 x 144 x 12)."""
    import det_engine as E
    x, s, (ox, oy) = E.reduce_photo(im.convert("L"))
    v2 = dict(v, objets=[dict(o, quad=[(px * s + ox, py * s + oy) for px, py in o["quad"]]) for o in v["objets"]],
              cartes=[dict(c, coins=[(px * s + ox, py * s + oy) for px, py in c["coins"]]) for c in v["cartes"]])
    return x, U.cibles_univ(v2, 0.5)


def main():
    mode, d, sortie, nombre, graine = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(graine)
    ex, par = G.charger_emnist(train=True)
    fonts = D.polices(False)
    supports = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))
                if os.path.basename(f) != "verite.json"]
    if mode == "banc":
        os.makedirs(sortie, exist_ok=True)
        verite = {}
        for i in range(nombre):
            a = supports[rng.integers(len(supports))]
            im, v = rendre(d, a, rng, ex, par, fonts)
            nom = f"s{graine:05d}_{i:04d}.jpg"
            im.convert("L").save(os.path.join(sortie, nom), quality=int(rng.integers(75, 95)))
            verite[nom] = v
        json.dump(verite, open(os.path.join(sortie, "verite.json"), "w"), ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    else:
        X = np.zeros((nombre, 384, 288), np.uint8)
        Y = np.zeros((nombre, 192, 144, 12), np.uint8)
        meta = []
        for i in range(nombre):
            a = supports[rng.integers(len(supports))]
            im, v = rendre(d, a, rng, ex, par, fonts)
            im, v = tourner(im, v, rng)
            X[i], Y[i] = en_cibles(im, v)
            meta.append(dict(style=v["style"], defaut=v["defaut"], format=v["format"], support=v["support"],
                             n=v["n"], source=v["source"]))
        np.savez_compressed(sortie, x=X, y=Y, meta=json.dumps(meta))


if __name__ == "__main__":
    main()
