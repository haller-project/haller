"""Lignes imprimées d'un papier photographié (ligné, Seyès, carreaux, carnet), relevées dans la photo elle-même.

Remplace le modèle « pas constant sur la page redressée » d'annoter_papiers.py, qui supposait un quadrilatère de page
exact : quand la page déborde du cadre ou se courbe, les lignes calculées dérivent des lignes imprimées.

Méthode (image réduite à 1600 px de côté long) :
  1. réponse « trait fin sombre » : médiane locale moins l'image, limitée à la page et hors occultants ;
  2. familles de traits : segments HoughLinesP, histogramme des angles pondéré par la longueur ; la famille
     d'écriture est la plus forte (papier ligné) ou, sur une grille, celle qui est la plus proche de l'horizontale ;
  3. image tournée pour rendre la famille horizontale ; profils verticaux par bandes (8 bandes), pas fin par
     autocorrélation, pics par bande, pics chaînés d'une bande à l'autre ; chaque chaîne d'au moins 3 bandes donne
     une ligne y(x) ajustée en degré 2 (perspective, page courbée) ;
  4. lignes ramenées dans la photo en polylignes (coordonnées de la photo de travail), force moyenne de chaque ligne.
La famille perpendiculaire (grilles) est relevée de la même façon, pour aligner numéros et mots sur les colonnes.

    python traits.py ANNOT TRAVAIL [--codes sv_0001,...] [--planche SORTIE.jpg]   ajoute "traits" à ANNOT/CODE.json
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy import ndimage, signal

COTE = 1600
GRILLES = {"petits_carreaux", "quadrille", "grands_carreaux"}


def reponse(g, region):
    """Traits fins plus sombres que le papier (0..40), nuls hors de la région."""
    fond = cv2.medianBlur(g, 15).astype(np.float32)
    r = np.clip(fond - g.astype(np.float32), 0, 40)
    r[~region] = 0
    return r


def familles(r, region):
    """Angles (degrés, dans [-90, 90[) des familles de traits, du plus fort au plus faible, avec leur poids."""
    b = (r > 6).astype(np.uint8) * 255
    L = max(r.shape)
    segs = cv2.HoughLinesP(b, 1, np.pi / 360, 80, minLineLength=int(0.12 * L), maxLineGap=int(0.01 * L))
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float32)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 90) % 180 - 90
    lon = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    h, _ = np.histogram(ang, bins=180, range=(-90, 90), weights=lon)
    h = ndimage.convolve1d(h, np.ones(5), mode="wrap")
    out = []
    for _ in range(2):
        k = int(np.argmax(h))
        if h[k] <= 0:
            break
        sel = np.abs((ang - (k - 90 + 0.5) + 90) % 180 - 90) < 4
        a = float(np.degrees(np.arctan2(np.sum(lon[sel] * np.sin(np.radians(2 * ang[sel]))),
                                        np.sum(lon[sel] * np.cos(np.radians(2 * ang[sel]))))) / 2)
        out.append((a, float(h[k])))
        for d in range(-12, 13):   # efface le voisinage du pic (angles modulo 180)
            h[(k + d) % 180] = 0
    return out


def pas_fin(p, lo):
    """Plus petit pas répété du profil : premier maximum local net de l'autocorrélation."""
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    hi = min(len(ac) - 1, len(p) // 3)
    pics, _ = signal.find_peaks(ac[:hi], height=0.12)
    pics = [int(k) for k in pics if k >= lo]
    if not pics:
        return None
    best = max(ac[k] for k in pics)
    for k in pics:   # le premier pic assez haut (le pas fin), pas le plus haut (souvent un multiple)
        if ac[k] >= 0.55 * best:
            return float(k)
    return None


def relever(r, region, angle, nb=16):
    """Lignes de la famille d'angle `angle` : liste de dict(pts=[[x, y]...] dans r, force), pas fin (px)."""
    h, w = r.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)   # tourne de +angle : la famille devient horizontale
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += W / 2 - c[0]
    M[1, 2] += H / 2 - c[1]
    rr = cv2.warpAffine(r, M, (W, H))
    reg = cv2.warpAffine(region.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    cols = np.nonzero(reg.sum(0) > 0.05 * reg.sum(0).max())[0]
    if len(cols) < 50:
        return [], None
    x0, x1 = cols.min(), cols.max()
    bords = np.linspace(x0, x1, nb + 1).astype(int)
    profils = []
    for a, b in zip(bords[:-1], bords[1:]):
        n = reg[:, a:b].sum(1)
        p = rr[:, a:b].sum(1) / np.maximum(n, 1)
        p[n < 0.3 * (b - a)] = 0
        profils.append(p)
    centre = np.mean(profils[nb // 4:3 * nb // 4], 0)
    pas = pas_fin(centre - ndimage.uniform_filter1d(centre, 81), lo=6)
    if pas is None:
        return [], None
    pics = []
    for p in profils:
        q = p - ndimage.uniform_filter1d(p, int(3 * pas) | 1)
        k, pr = signal.find_peaks(q, distance=max(2, int(0.6 * pas)), prominence=1e-6)
        if len(k):
            seuil = 0.25 * np.percentile(pr["prominences"], 90)
            k = k[pr["prominences"] >= seuil]
        pics.append(k.astype(float))
    xm = 0.5 * (bords[:-1] + bords[1:])
    # chaînage : chaque pic d'une bande prolonge la chaîne la plus proche de la bande précédente (pente suivie)
    chaines = []
    ouvertes = []
    for j, ks in enumerate(pics):
        nouv = []
        libres = list(ks)
        for ch in ouvertes:
            (xa, ya) = ch[-1]
            pente = (ch[-1][1] - ch[-2][1]) / (ch[-1][0] - ch[-2][0]) if len(ch) > 1 else 0.0
            yp = ya + pente * (xm[j] - xa)
            if libres:
                d = np.abs(np.array(libres) - yp)
                i = int(np.argmin(d))
                if d[i] < 0.3 * pas:
                    ch.append((xm[j], libres.pop(i)))
                    nouv.append(ch)
                    continue
            if xm[j] - xa > 2.5 * (xm[1] - xm[0]):   # chaîne interrompue depuis plus de 2 bandes : close
                chaines.append(ch)
            else:
                nouv.append(ch)
        for y in libres:
            nouv.append([(xm[j], y)])
        ouvertes = nouv
    chaines += ouvertes
    lignes = []
    for ch in chaines:
        if len(ch) < 4:
            continue
        xs, ys = np.array(ch).T
        coef = np.polyfit(xs, ys, 2 if len(ch) >= 5 else 1)
        # étendue : là où la ligne traverse la région
        xx = np.arange(x0, x1, 8.0)
        yy = np.polyval(coef, xx)
        ok = (yy >= 0) & (yy < H)
        xx, yy = xx[ok], yy[ok]
        dedans = reg[yy.astype(int), xx.astype(int)]
        if dedans.sum() < 5:
            continue
        i0, i1 = np.nonzero(dedans)[0][[0, -1]]
        xx, yy = xx[i0:i1 + 1], yy[i0:i1 + 1]
        f = float(np.mean(rr[np.clip(yy.astype(int), 0, H - 1), xx.astype(int)]))
        Minv = cv2.invertAffineTransform(M)
        pts = np.c_[xx, yy, np.ones(len(xx))] @ Minv.T
        xc = (x0 + x1) / 2
        lignes.append(dict(pts=pts, force=f, y=float(np.polyval(coef, xc)), coef=coef, xa=xx[0], xb=xx[-1], xs=xx, ys=yy,
                           pente=float(np.polyval(np.polyder(coef), xc)) if len(coef) > 1 else 0.0))
    lignes.sort(key=lambda l: l["y"])
    garde = []
    for l in lignes:   # doublons (deux chaînes sur la même ligne) : garder la plus longue
        if garde and abs(l["y"] - garde[-1]["y"]) < 0.4 * pas:
            if len(l["pts"]) > len(garde[-1]["pts"]):
                garde[-1] = l
            continue
        garde.append(l)
    # pente cohérente avec les voisines (la perspective varie lentement) : une chaîne qui saute d'une ligne à
    # l'autre a une pente aberrante
    if len(garde) >= 3:
        pentes = np.array([l["pente"] for l in garde])
        med = ndimage.median_filter(pentes, size=min(7, len(garde)), mode="nearest")
        garde = [l for l, p_, m_ in zip(garde, pentes, med) if abs(p_ - m_) < 0.025]
    # trous : un écart de k pas (k >= 2) reçoit k - 1 lignes interpolées entre ses deux voisines
    if len(garde) >= 3:
        ys = np.array([l["y"] for l in garde])
        p0 = float(np.median(np.diff(ys)))
        plein = [garde[0]]
        for a_, b_ in zip(garde[:-1], garde[1:]):
            k = (b_["y"] - a_["y"]) / p0
            kr = int(round(k))
            if 2 <= kr <= 3 and abs(k - kr) < 0.2 and abs(a_["pente"] - b_["pente"]) < 0.015:
                xa, xb = max(a_["xa"], b_["xa"]), min(a_["xb"], b_["xb"])
                if xb - xa > 80:
                    xx = np.arange(xa, xb, 8.0)
                    ya, yb = np.polyval(a_["coef"], xx), np.polyval(b_["coef"], xx)
                    Minv = cv2.invertAffineTransform(M)
                    for j in range(1, kr):
                        yy = ya + (yb - ya) * j / kr
                        pts = np.c_[xx, yy, np.ones(len(xx))] @ Minv.T
                        plein.append(dict(pts=pts, force=0.0, y=float(np.mean(yy)), interpolee=True, xs=xx, ys=yy,
                                          pente=0.5 * (a_["pente"] + b_["pente"])))
            plein.append(b_)
        garde = plein
        # groupes de lignes consécutives (écart <= 1,6 pas) ; une ligne isolée est une fausse chaîne
        groupes, cur = [], [garde[0]]
        for a_, b_ in zip(garde[:-1], garde[1:]):
            if b_["y"] - a_["y"] > 1.6 * p0:
                groupes.append(cur)
                cur = []
            cur.append(b_)
        groupes.append(cur)
        garde = [l for g_ in groupes if len(g_) >= 3 for l in g_]
        # paire régulière : écart avec la ligne suivante entre 0,6 et 1,5 pas, stable à +-35 % le long de la ligne
        for a_, b_ in zip(garde[:-1], garde[1:]):
            xa, xb = max(a_["xs"][0], b_["xs"][0]), min(a_["xs"][-1], b_["xs"][-1])
            ok = False
            if xb - xa > 80:
                xx = np.linspace(xa, xb, 20)
                g = np.interp(xx, b_["xs"], b_["ys"]) - np.interp(xx, a_["xs"], a_["ys"])
                m_ = g.mean()
                ok = bool(0.6 * p0 < m_ < 1.5 * p0 and g.min() > 0.65 * m_ and g.max() < 1.35 * m_)
            a_["suivante_ok"] = ok
    return garde, pas


def detecter(img, page, masque, typ):
    """Familles de lignes imprimées de la photo `img` (BGR) : dict(ecriture=..., travers=...) ; chaque famille :
    dict(angle, pas, lignes=[dict(pts, force)]) en coordonnées de la photo, ou None si rien de régulier."""
    h0, w0 = img.shape[:2]
    s = COTE / max(h0, w0)
    g = cv2.cvtColor(cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    region = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(region, [np.int32(np.round(np.asarray(page) * s))], 1)
    region = cv2.erode(region, np.ones((15, 15), np.uint8)) > 0
    if masque is not None:
        mk = cv2.resize(masque.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        region &= ~(cv2.dilate(mk.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    r = reponse(g, region)
    fam = familles(r, region)
    if not fam:
        return dict(ecriture=None, travers=None)
    if typ in ("petits_carreaux", "quadrille") and len(fam) == 2 and fam[1][1] > 0.3 * fam[0][1]:
        fam.sort(key=lambda f: abs(f[0]))   # grille : écrire le long de la famille la plus horizontale
    res = {}
    for nom, (angle, poids) in zip(["ecriture", "travers"], fam):
        lignes, pas = relever(r, region, angle)
        if nom == "travers" and (typ not in GRILLES or abs(abs(angle - fam[0][0]) - 90) > 20):
            res[nom] = None
            continue
        ok = pas is not None and len(lignes) >= 4
        res[nom] = dict(angle=round(angle, 2), pas=round(pas / s, 2),
                        lignes=[dict(pts=np.round(l["pts"] / s, 1).tolist(), force=round(l["force"], 2),
                                     suivante_ok=bool(l.get("suivante_ok", False)),
                                     **({"interpolee": True} if l.get("interpolee") else {}))
                                for l in lignes]) if ok else None
        if ok and nom == "ecriture" and typ == "grands_carreaux":
            # Seyès : une ligne sur 4 est une ligne principale (plus foncée) ; phase de plus forte force moyenne
            f = np.array([l["force"] for l in lignes])
            ph = int(np.argmax([f[k::4][f[k::4] > 0].mean() if (f[k::4] > 0).any() else 0 for k in range(4)]))
            res[nom]["seyes_phase"] = ph
    if "travers" not in res:
        res["travers"] = None
    return res


def regularite(fam, hauteur):
    """Vrai si la famille ressemble à un réglage imprimé : au moins 6 lignes, 85 % de paires voisines régulières,
    lignes longues, pas d'au moins 0,4 % de la photo."""
    if not fam or len(fam["lignes"]) < 6:
        return False
    lon = [np.hypot(*(np.array(l["pts"][-1]) - l["pts"][0])) for l in fam["lignes"]]
    longues = np.mean(np.array(lon) > 0.4 * np.median(lon))
    paires = np.mean([l["suivante_ok"] for l in fam["lignes"][:-1]])
    fam["paires_ok"] = round(float(paires), 2)
    return bool(longues > 0.6 and paires >= 0.85 and fam["pas"] > 0.004 * hauteur)


def annoter(chemin_json, travail):
    a = json.load(open(chemin_json))
    img = cv2.imread(os.path.join(travail, a["image"]))
    m = None
    d = os.path.dirname(chemin_json)
    if a.get("masque") and os.path.exists(os.path.join(d, a["masque"])):
        m = cv2.imread(os.path.join(d, a["masque"]), cv2.IMREAD_GRAYSCALE) > 127
    t = detecter(img, a["page"], m, a.get("support"))
    t["regulier"] = regularite(t["ecriture"], max(img.shape[:2]))
    a["traits"] = t
    json.dump(a, open(chemin_json, "w"), ensure_ascii=False)
    return a


def dessiner(a, travail, taille=520):
    from PIL import Image, ImageDraw
    im = Image.open(os.path.join(travail, a["image"])).convert("RGB")
    d = ImageDraw.Draw(im)
    d.polygon([tuple(p) for p in a["page"]], outline=(0, 200, 0), width=5)
    t = a["traits"]
    for nom, coul in (("ecriture", (255, 0, 0)), ("travers", (0, 120, 255))):
        if t.get(nom):
            for l in t[nom]["lignes"]:
                d.line([tuple(p) for p in l["pts"]], fill=coul, width=3)
    im.thumbnail((taille, taille))
    txt = f"{a['image'][:-4]} {a.get('support')} " + (
        f"pas {t['ecriture']['pas']:.0f} n {len(t['ecriture']['lignes'])} reg {int(t['regulier'])}"
        if t.get("ecriture") else "rien")
    ImageDraw.Draw(im).text((4, 4), txt, fill=(255, 0, 255))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("annot")
    ap.add_argument("travail")
    ap.add_argument("--codes")
    ap.add_argument("--planche")
    ap.add_argument("--proc", type=int, default=1)
    x = ap.parse_args()
    fs = sorted(f for f in glob.glob(os.path.join(x.annot, "*.json")))
    if x.codes:
        garde = set(x.codes.split(","))
        fs = [f for f in fs if os.path.basename(f)[:-5] in garde]
    if x.proc > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(x.proc) as ex:
            res = list(ex.map(annoter, fs, [x.travail] * len(fs)))
    else:
        res = [annoter(f, x.travail) for f in fs]
    if x.planche:
        from PIL import Image
        ims = [dessiner(a, x.travail) for a in res]
        cols = 4
        rows = (len(ims) + cols - 1) // cols
        sh = Image.new("RGB", (cols * 520, rows * 520), "white")
        for i, im in enumerate(ims):
            sh.paste(im, ((i % cols) * 520, (i // cols) * 520))
        sh.save(x.planche, quality=80)


if __name__ == "__main__":
    main()
