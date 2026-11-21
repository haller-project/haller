"""Cartes officielles imprimées (planches docs/supports_planches.pdf), photographiées vierges : modèles, repérage,
annotation, détourage, variantes de fond, de perspective et de lumière.

Sous-commandes (venv avec opencv, numpy, scipy, pillow) :

  modeles  BASE PLUS SORTIE
      Modèles utilisés pour le repérage : PNG de data/modeles_cartes (BASE) et cases de BASE/cases.json et
      PLUS/cases.json (essais comparables, --plus d'annoter.py) ; modèles dérivés (rognés ou tournés pour que
      l'écriture soit horizontale) annotés ici même avec la convention « bloc » d'annoter.py (RELEVES_CARTES).
      Écrit SORTIE/png/*.png, SORTIE/cases.json et SORTIE/_controle.jpg.
  travail  IMPORT CLASSEMENT SORTIE
      Versions de travail (côté long 2048) des photos gardées, nommées CODE.jpg.
  annoter  TRAVAIL CLASSEMENT MODELES SORTIE [--proc N]
      Repérage SIFT du modèle connu (classement) dans chaque photo, homographie RANSAC, coins de la carte, cases
      et numéros imprimés projetés ; masque du papier (GrabCut) et masque de découpe (papier limité au contour
      de la carte : les marges de papier en trop sont retirées). SORTIE/CODE.json, CODE_papier.png,
      CODE_decoupe.png, contrôles SORTIE/_controle_NN.jpg.
  varier   TRAVAIL ANNOT FONDS FONDS_ANNOT FONDS_INDEX SORTIE N GRAINE [--proc N] [--codes cb_0001,...]
      N variantes par photo d'entraînement : carte détourée posée sur une photo de support vierge (table, carton
      gris, table noire, papiers), perspective modérée, ombre portée, balance des blancs et exposition accordées
      au fond, flou accordé ; puis éclairage (lumiere.augmenter). SORTIE/base/*.jpg (avant éclairage, pour
      composer.py qui éclaire lui-même), SORTIE/eclairees/*.jpg (publiées), SORTIE/annot/*.json.
  brut     TRAVAIL ANNOT SORTIE
      Versions brutes recadrées sur la carte (marge de 6 % du côté de la carte), annotations ramenées au cadre.

Annotation (format de train/detecteur/supports/poser_ecriture.py, coordonnées de l'image de travail) :
  image, code (Kxxx), carte (Cnn), modele, jeu (train | eval), coins (4 coins de la carte, sens horaire depuis le
  haut à gauche dans le sens d'écriture), zones [{quad, type: "case", numero, numero_quad}], H (modèle -> photo),
  inliers, err_px, papier / decoupe (masques PNG).
Numéros imprimés : boîte d'encre du numéro seul, trouvée dans une colonne qui commence après le bord ou le cadre de
la carte (défaut corrigé par UNIV_NUMEROS_CORRIGES, docs/detecteur-universel.md § 3).
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "detecteur", "modeles"))

LONG = 2048

# Modèles dérivés : source, rognage (x0, y0, x1, y1) dans le PNG source, rotation (degrés, sens trigonométrique,
# appliquée après le rognage), relevé « bloc » d'annoter.py dans le PNG dérivé (premier numéro, nombre, x0 et x1
# des numéros, y0 et y1 de la colonne, x0 et x1 de la case), base, contour de la carte. None = sans cases.
RELEVES_CARTES = {
    # Carte SeedSigner « SEED WORDS » (planche tc21 : recto mots, verso SeedQR) ; colonne des numéros après le bord
    # noir et le cadre gris arrondi (x 281-290). 10 à 12 ont un numéro plus large : bloc à part.
    "seedsigner_tc21_w12": dict(src="seedsigner_tc21_w12", base="ligne", carte=[261, 150, 635, 676],
                                blocs=[(1, 9, 296, 327, 266, 522, 330, 566), (10, 3, 296, 341, 522, 614, 344, 566)]),
    "seedsigner_tc25_w24": dict(src="seedsigner_tc25_w24", base="ligne", carte=[261, 150, 637, 676],
                                blocs=[(1, 9, 288, 320, 266, 522, 321, 440), (10, 3, 288, 331, 522, 614, 332, 440),
                                       (13, 12, 455, 494, 266, 614, 495, 604)]),
    # Trezor 12 mots (intérieur) : imprimé couché, pastilles en bas ; tourné d'un quart de tour horaire, pastille
    # à gauche, lettres de la case à droite. Numéro imprimé = pastille noire (comme trezor_seed20_inside).
    "trezor_seed12_inside@r": dict(src="trezor_seed12_inside", rot=-90, base="case", carte=[57, 57, 417, 777],
                                   blocs=[(1, 12, 66, 112, 60, 772, 119, 404)]),
    # Trezor multi-share : première part seulement (les 5 parts sont identiques), même orientation.
    "trezor_multishare5@part": dict(src="trezor_multishare5", crop=[200, 0, 1000, 400], rot=-90, base="case",
                                    carte=[84, 19, 341, 787], blocs=[(1, 20, 93, 122, 50, 792, 127, 330)]),
    # BitBox02 : volet « RECOVERY WORDS » (tourné d'un quart de tour horaire). Le trait d'écriture passe sous le
    # numéro : un bloc par numéro, de la ligne précédente au trait (exclu), comme les modèles Ledger.
    "bitbox02_backupcard@r": dict(src="bitbox02_backupcard", rot=-90, base="case", carte=[207, 25, 625, 640],
                                  lignes_bitbox=True),
    # Penlock 12 mots : bande « Seed Phrase » seulement (tournée), comme penlock_worksheet24_front.
    "penlock_worksheet12_front@r": dict(src="penlock_worksheet12_front", rot=-90, crop_apres=[140, 70, 545, 1140],
                                        base="ligne", carte=[25, 27, 377, 1043],
                                        blocs=[(1, 12, 46, 101, 180, 922, 104, 340)]),
    # Seed XOR : cases sous le numéro, sur plusieurs tableaux ; convention « bloc » inapplicable (essais
    # comparables, 20/11) : contour seulement.
    "seedxor_worksheet12": dict(src="seedxor_worksheet12", base=None, carte=None),
    "seedxor_worksheet24": dict(src="seedxor_worksheet24", base=None, carte=None),
}


def charger_png(base, rel):
    src = rel.get("src")
    im = Image.open(os.path.join(base, "png", src + ".png")).convert("L")
    if rel.get("crop"):
        im = im.crop(tuple(rel["crop"]))
    if rel.get("rot"):
        im = im.rotate(rel["rot"], expand=True, fillcolor=255)
    if rel.get("crop_apres"):
        im = im.crop(tuple(rel["crop_apres"]))
    return im


# Modèles annotés (cases.json) rognés sur une seule copie ou sur la zone photographiée en gros plan, quand les
# motifs répétés du PNG complet égarent le repérage. Cases, numéros et contour sont décalés du rognage.
ROGNES = {
    "trezor_legacy_recovery_card@1": dict(src="trezor_legacy_recovery_card", crop=[480, 110, 1180, 600]),
    "psst_worksheet_share@table": dict(src="psst_worksheet_share", crop=[300, 1150, 960, 1757]),
}


def lignes_bitbox(a):
    """Traits d'écriture du volet RECOVERY WORDS (PNG tourné) : rangées sombres sur x 270-400, entre y 150 et 560."""
    prof = (a[:, 270:400] < 200).mean(1)
    ys = [y for y in range(150, 560) if prof[y] > 0.6 and prof[y - 1] <= 0.6]
    return ys


def cmd_modeles(a):
    import annoter as AN
    os.makedirs(os.path.join(a.sortie, "png"), exist_ok=True)
    cases = {m["id"]: m for m in json.load(open(os.path.join(a.base, "cases.json")))}
    cases.update({m["id"]: m for m in json.load(open(os.path.join(a.plus, "cases.json")))})
    manifest = {m["id"]: m for m in json.load(open(os.path.join(a.base, "manifest.json")))}
    out = []
    for mid, e in manifest.items():
        if mid in cases:
            m = dict(cases[mid])
            Image.open(os.path.join(a.base, m["png"])).convert("L").save(os.path.join(a.sortie, "png", mid + ".png"))
            m["png"] = f"png/{mid}.png"
            out.append(m)
    for mid, r in ROGNES.items():
        b = cases[r["src"]]
        x0, y0, x1, y1 = r["crop"]
        Image.open(os.path.join(a.base, b["png"])).convert("L").crop((x0, y0, x1, y1)).save(
            os.path.join(a.sortie, "png", mid.replace("@", "_") + ".png"))
        dec = lambda bb: [bb[0] - x0, bb[1] - y0, bb[2] - x0, bb[3] - y0, *bb[4:]]  # noqa: E731
        carte = dec(b["carte"])
        carte = [max(0, carte[0]), max(0, carte[1]), min(x1 - x0, carte[2]), min(y1 - y0, carte[3])]
        out.append(dict(id=mid, png=f"png/{mid.replace('@', '_')}.png", split=b["split"], carte=carte,
                        numeros=[dec(x) for x in b["numeros"] if y0 <= x[1] and x[3] <= y1],
                        cases=[dec(x) for x in b["cases"] if y0 <= x[1] and x[3] <= y1],
                        derive=dict(src=r["src"], crop=r["crop"], rot=0)))
    for mid, r in RELEVES_CARTES.items():
        im = charger_png(a.base, r)
        arr = np.asarray(im, np.float32)
        im.save(os.path.join(a.sortie, "png", mid.replace("@", "_") + ".png"))
        numeros, cs = [], []
        if r.get("lignes_bitbox"):
            ys = lignes_bitbox(arr)
            assert len(ys) == 12, ys
            for k, ly in enumerate(ys):
                for c, (nx, sx) in enumerate([((228, 258), (258, 405)), ((418, 445), (445, 592))]):
                    n_, c_ = AN.bloc(arr, k + 1 + 12 * c, 1, *nx, ly - 30, ly - 2, *sx, "case")
                    numeros += n_
                    cs += c_
        elif r.get("blocs"):
            for b in r["blocs"]:
                n_, c_ = AN.bloc(arr, *b, r["base"])
                numeros += n_
                cs += c_
        carte = r["carte"]
        if carte is None:
            ys, xs = np.where(arr < 200)
            carte = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        numeros.sort(key=lambda b: b[4])
        cs.sort(key=lambda b: b[4])
        out.append(dict(id=mid, png=f"png/{mid.replace('@', '_')}.png", split="train", carte=carte,
                        numeros=numeros, cases=cs, derive=dict(src=r["src"], crop=r.get("crop"), rot=r.get("rot", 0))))
    json.dump(out, open(os.path.join(a.sortie, "cases.json"), "w"), indent=1)
    # contrôle des modèles dérivés
    ims = []
    for m in out:
        if "derive" not in m:
            continue
        im = Image.open(os.path.join(a.sortie, m["png"])).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in m["numeros"]:
            d.rectangle(b[:4], outline=(255, 0, 0), width=1)
        for b in m["cases"]:
            d.rectangle(b[:4], outline=(0, 160, 0), width=1)
        d.rectangle(m["carte"], outline=(0, 0, 255), width=2)
        im.thumbnail((700, 700))
        ims.append(im)
    s = Image.new("RGB", (4 * 700, (len(ims) + 3) // 4 * 700), "gray")
    for i, im in enumerate(ims):
        s.paste(im, ((i % 4) * 700, (i // 4) * 700))
    s.save(os.path.join(a.sortie, "_controle.jpg"), quality=88)
    print(len(out), "modèles")


# --- photos -----------------------------------------------------------------------------------------------------

def cmd_travail(a):
    cl = json.load(open(a.classement))
    os.makedirs(a.sortie, exist_ok=True)
    for code, c in cl["photos"].items():
        if c["statut"] != "gardee":
            continue
        dst = os.path.join(a.sortie, code + ".jpg")
        if os.path.exists(dst):
            continue
        im = Image.open(os.path.join(a.importe, c["fichier"]))
        im.draft("RGB", (LONG, LONG))
        im = ImageOps.exif_transpose(im).convert("RGB")
        s = LONG / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dst, quality=93)
    print("travail :", a.sortie)


def appliquer(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:]


def rect(b):
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def reperer(photo, modele_png):
    """Homographie modèle -> photo par SIFT (test du ratio) et RANSAC, affinée sur les inliers (moindres carrés)."""
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    g = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(modele_png, cv2.IMREAD_GRAYSCALE)
    sm = min(1.0, 1600 / max(m.shape))
    ms = cv2.resize(m, None, fx=sm, fy=sm, interpolation=cv2.INTER_AREA) if sm < 1 else m
    if max(ms.shape) < 900:   # petits modèles (Ledger, 389 px) : agrandis pour que SIFT trouve des points
        f = 900 / max(ms.shape)
        ms = cv2.resize(ms, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        sm *= f
    kp1, d1 = sift.detectAndCompute(ms, None)
    kp2, d2 = sift.detectAndCompute(g, None)
    mt = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    bons = [p[0] for p in mt if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(bons) < 10:
        return None, dict(appariements=len(bons))
    src = np.float32([kp1[b.queryIdx].pt for b in bons]) / sm
    dst = np.float32([kp2[b.trainIdx].pt for b in bons])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=20000, confidence=0.9995)
    if H is None:
        return None, dict(appariements=len(bons))
    inl = mask.ravel() == 1
    H2, _ = cv2.findHomography(src[inl], dst[inl], 0)
    if H2 is not None:
        H = H2
    err = np.linalg.norm(appliquer(H, src[inl]) - dst[inl], axis=1)
    return H, dict(appariements=len(bons), inliers=int(inl.sum()), err_px=round(float(np.median(err)), 2))


def masque_papier(img, coins):
    """Masque du papier (GrabCut sur l'image réduite) : sûr dans la carte rétrécie, probable jusqu'à 25 % autour."""
    h, w = img.shape[:2]
    s = 640 / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    q = np.float32(coins) * s
    c = q.mean(0)
    mk = np.full(small.shape[:2], cv2.GC_BGD, np.uint8)
    cv2.fillPoly(mk, [np.int32(c + (q - c) * 1.3)], cv2.GC_PR_BGD)
    cv2.fillPoly(mk, [np.int32(c + (q - c) * 1.06)], cv2.GC_PR_FGD)
    cv2.fillPoly(mk, [np.int32(c + (q - c) * 0.85)], cv2.GC_FGD)
    if (mk == cv2.GC_BGD).sum() < 500 or (mk == cv2.GC_FGD).sum() < 500:   # carte qui remplit le cadre
        return np.full((h, w), 255, np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mk, None, bg, fg, 6, cv2.GC_INIT_WITH_MASK)
    p = np.where((mk == cv2.GC_FGD) | (mk == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(p)
    if n > 2:   # composante qui contient le centre de la carte
        p = np.where(lab == lab[int(c[1]), int(c[0])], 255, 0).astype(np.uint8)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    big = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return (cv2.GaussianBlur(big, (0, 0), 1.5) > 127).astype(np.uint8) * 255


def annoter_une(args):
    code, c, travail, modeles, sortie = args
    mods = {m["id"]: m for m in json.load(open(os.path.join(modeles, "cases.json")))}
    m = mods[c["modele"]]
    photo = os.path.join(travail, code + ".jpg")
    H, info = reperer(photo, os.path.join(modeles, m["png"]))
    out = dict(image=code + ".jpg", code=code, carte=c["carte"], modele=c["modele"], jeu=c["jeu"], support="carte",
               **info)
    if H is None:
        out["echec"] = True
        json.dump(out, open(os.path.join(sortie, code + ".json"), "w"), indent=1)
        return code, out
    img = cv2.imread(photo)
    coins = appliquer(H, rect(m["carte"]))
    nums = {int(b[4]): b for b in m["numeros"]}
    zones = []
    for b in m["cases"]:
        nb = nums.get(int(b[4]))
        zones.append(dict(quad=appliquer(H, rect(b)).round(1).tolist(), type="case", numero=int(b[4]),
                          numero_quad=appliquer(H, rect(nb)).round(1).tolist() if nb else None))
    pap = masque_papier(img, coins)
    # découpe : papier limité au contour imprimé de la carte agrandi de 1 % de sa diagonale
    diag = np.linalg.norm(coins[2] - coins[0])
    cc = coins.mean(0)
    agrandi = [cc + (p - cc) * (1 + 0.012 * diag / max(1, np.linalg.norm(p - cc))) for p in coins]
    dec = np.zeros_like(pap)
    cv2.fillPoly(dec, [np.int32(np.round(agrandi))], 255)
    dec = cv2.bitwise_and(dec, pap)
    cv2.imwrite(os.path.join(sortie, code + "_papier.png"), pap)
    cv2.imwrite(os.path.join(sortie, code + "_decoupe.png"), dec)
    marge = float(1 - dec.sum() / max(1, pap.sum()))
    out.update(coins=coins.round(1).tolist(), zones=zones, cartes=[coins.round(1).tolist()], H=H.tolist(),
               papier=code + "_papier.png", decoupe=code + "_decoupe.png", marge_retiree=round(marge, 3),
               largeur=img.shape[1], hauteur=img.shape[0])
    json.dump(out, open(os.path.join(sortie, code + ".json"), "w"), indent=1)
    return code, out


def controle(img, an, taille=700):
    """Image de contrôle : carte (bleu), cases (vert), numéros imprimés (rouge), contour de découpe (jaune)."""
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 600)
    if an.get("decoupe_img") is not None:
        cs, _ = cv2.findContours(an["decoupe_img"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in cs:
            d.line([tuple(p[0]) for p in ct] + [tuple(ct[0][0])], fill=(255, 220, 0), width=lw)
    if an.get("coins"):
        d.polygon([tuple(p) for p in an["coins"]], outline=(0, 80, 255), width=lw + 1)
    for z in an.get("zones", []):
        d.polygon([tuple(p) for p in z["quad"]], outline=(0, 200, 0), width=lw)
        if z.get("numero_quad"):
            d.polygon([tuple(p) for p in z["numero_quad"]], outline=(255, 0, 0), width=lw)
    im.thumbnail((taille, taille))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 330, 16], fill=(0, 0, 0))
    d.text((3, 2), f"{an['code']} {an['carte']} {an.get('jeu')} inl={an.get('inliers')} err={an.get('err_px')}"
                   f" marge={an.get('marge_retiree')}", fill=(255, 255, 255))
    return im


def planche(images, chemin, cols=4, taille=700):
    rows = (len(images) + cols - 1) // cols
    s = Image.new("RGB", (cols * taille, rows * taille), (90, 90, 90))
    for i, im in enumerate(images):
        s.paste(im, ((i % cols) * taille, (i // cols) * taille))
    s.save(chemin, quality=85)


def cmd_annoter(a):
    cl = json.load(open(a.classement))["photos"]
    os.makedirs(a.sortie, exist_ok=True)
    taches = [(k, c, a.travail, a.modeles, a.sortie) for k, c in sorted(cl.items()) if c["statut"] == "gardee"
              and (not a.codes or k in a.codes.split(","))]
    with ProcessPoolExecutor(a.proc) as ex:
        res = dict(ex.map(annoter_une, taches))
    ims = []
    for k in sorted(res):
        an = res[k]
        print(k, an["carte"], an["modele"], an.get("inliers"), an.get("err_px"), an.get("marge_retiree"),
              "ECHEC" if an.get("echec") else "")
        img = cv2.imread(os.path.join(a.travail, k + ".jpg"))
        if not an.get("echec"):
            an["decoupe_img"] = cv2.imread(os.path.join(a.sortie, an["decoupe"]), cv2.IMREAD_GRAYSCALE)
        ims.append(controle(img, an))
    for i in range(0, len(ims), 12):
        planche(ims[i:i + 12], os.path.join(a.sortie, f"_controle_{i // 12 + 1:02d}.jpg"))


# --- variantes --------------------------------------------------------------------------------------------------

def flou_photo(g):
    """Écart type du flou (px), estimé sur les bords les plus raides (même méthode que poser_ecriture.flou_photo)."""
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    contraste = np.percentile(g, 95) - np.percentile(g, 5)
    return float(np.clip(contraste / max(np.percentile(m, 99.5), 1e-3) / 2.5, 0.4, 4.0))


BLANCS = {"ligne", "grands_carreaux", "petits_carreaux", "quadrille", "carnet", "bristol", "blanc", "plie_froisse"}


def charger_fonds(fonds_annot, fonds_index):
    """Photos de supports vierges utilisables comme fond : pas d'encre préexistante, pas de main ; couleur du
    papier blanc (référence de balance des blancs) quand le support est blanc."""
    ix = {p["code"]: p for p in json.load(open(fonds_index))["photos"] if p.get("code")}
    out = []
    for code, p in sorted(ix.items()):
        f = os.path.join(fonds_annot, code + ".json")
        if not os.path.exists(f):
            continue
        an = json.load(open(f))
        if an.get("exclure") or an.get("auto", {}).get("masque_frac", 0) > 0.003:
            continue
        out.append(dict(code=code, type=p["type"], page=an.get("page"), appareil=p.get("appareil"),
                        ambiance="lampe" if p["prise"][11:16] < "17:55" else "table_noire"))
    return out


def blanc_fond(rgb, f, refs):
    """Couleur (RVB linéaire 0-1) qu'aurait un papier blanc posé sur ce fond."""
    import lumiere
    if f["type"] in BLANCS and f["page"]:
        h, w = rgb.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.int32(np.round(f["page"]))], 1)
        px = rgb[m > 0]
        if len(px) > 1000:
            lum = px.mean(1)
            sel = px[lum > np.percentile(lum, 40)]
            return lumiere.srgb_lin(np.median(sel, 0))
    # sinon : pixels clairs et peu saturés du fond lui-même (papier blanc, motifs blancs, reflets)
    mx, mn = rgb.max(2), rgb.min(2)
    neutre = (rgb.mean(2) > np.percentile(rgb.mean(2), 80)) & ((mx - mn) / np.maximum(mx, 1) < 0.25)
    if neutre.mean() > 0.01:
        return lumiere.srgb_lin(np.median(rgb[neutre], 0))
    ref = refs.get(f["ambiance"])
    if ref is None:
        ref = refs.get("lampe", np.ones(3) * 0.6)
    lum = rgb.mean(2)
    niveau = lumiere.srgb_lin(np.percentile(lum, 97))
    return ref / ref.mean() * niveau


def placement(wm, hm, W, H, rng):
    """Quadrilatère de la carte dans le fond : orientation, taille, position, perspective modérée (trapèze)."""
    # orientation (décision du 20/11 et mesures réelles, train/detecteur/distributions) : majorité droite,
    # quarts de tour (appareil en paysage) 18 %, retournées 7 % ; inclinaison réelle 0,8° [0-5,9] : écart type 3,5°,
    # 5 % de fortes inclinaisons jusqu'à 18° (contrôle de la variété du 21/11, docs/variete.md : réel au plus 10,9°,
    # supports vierges 17° ; 25° auparavant)
    incl = rng.uniform(-18, 18) if rng.random() < 0.05 else float(np.clip(rng.normal(0, 3.5), -18, 18))
    rot = float(rng.choice([0, 90, 180, 270], p=[0.75, 0.09, 0.07, 0.09])) + incl
    t = math.radians(rot)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c0 = np.array([[-wm / 2, -hm / 2], [wm / 2, -hm / 2], [wm / 2, hm / 2], [-wm / 2, hm / 2]])
    # trapèze : les coins d'un côté (direction u) se rapprochent, ceux de l'autre s'écartent
    phi = rng.uniform(0, 2 * math.pi)
    u = np.array([math.cos(phi), math.sin(phi)])
    kappa = rng.uniform(0.0, 0.14)
    q = c0 @ R.T
    proj = q @ u / (np.abs(q @ u).max() + 1e-6)
    q = q * (1 + kappa * proj)[:, None]
    ext = q.max(0) - q.min(0)
    # appareil tenu dans le sens de la carte (paysage pour une carte couchée), 85 % des cas : le cadre est tourné
    tourne = bool((ext[0] > ext[1]) != (W > H) and rng.random() < 0.85)
    if tourne:
        W, H = H, W
    # échelle (décision du 20/11) : la carte remplit le cadre comme dans un scan, part de l'image tirée
    # autour des photos brutes (médiane 57 %, quartiles 39-76 %) : 85 % entre 35 et 90 %, 15 % de plans plus larges
    # (15-35 %) ; bornée pour que la carte reste dans le cadre
    # (la borne du cadre rabote les grandes valeurs : tirage centré plus haut pour retrouver la médiane des brutes)
    part = float(np.clip(rng.normal(0.66, 0.17), 0.35, 0.92)) if rng.random() < 0.85 else rng.uniform(0.15, 0.35)
    aire = abs(cv2.contourArea(np.float32(q)))
    s = min(math.sqrt(part * W * H / max(aire, 1e-6)), 0.98 * min(W / ext[0], H / ext[1]))
    q = q * s
    ext = q.max(0) - q.min(0)
    marge = 0.02 * np.array([W, H])
    cx = rng.uniform(marge[0] + ext[0] / 2, max(marge[0] + ext[0] / 2 + 1, W - marge[0] - ext[0] / 2))
    cy = rng.uniform(marge[1] + ext[1] / 2, max(marge[1] + ext[1] / 2 + 1, H - marge[1] - ext[1] / 2))
    part_reelle = abs(cv2.contourArea(np.float32(q))) / (W * H)
    return q + [cx, cy], dict(rotation=round(rot, 1), trapeze=round(kappa, 3), echelle=round(float(s), 3),
                              part_image=round(float(part_reelle), 3), fond_tourne=tourne)


def hors_cadre(q, w, h):
    q = np.float32(q)
    a = abs(cv2.contourArea(q))
    r, _ = cv2.intersectConvexConvex(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return float(max(0.0, 1 - r / max(a, 1e-6)))


def touche_cadre(dec, n=20):
    """Vrai si la découpe (papier de la carte) touche le bord de la photo sur plus de n pixels."""
    bord = np.concatenate([dec[0], dec[-1], dec[:, 0], dec[:, -1]])
    return int((bord > 127).sum()) > n


def varier_une(args):
    code, i, graine, travail, annot, fonds, fonds_dir, refs, sortie, modeles = args
    import lumiere
    rng = np.random.default_rng([graine, int(code[1:]), i])
    an = json.load(open(os.path.join(annot, code + ".json")))
    m = {x["id"]: x for x in json.load(open(os.path.join(modeles, "cases.json")))}[an["modele"]]
    src = cv2.cvtColor(cv2.imread(os.path.join(travail, an["image"])), cv2.COLOR_BGR2RGB).astype(np.float32)
    dec = cv2.imread(os.path.join(annot, an["decoupe"]), cv2.IMREAD_GRAYSCALE)
    # carte coupée par le cadre à travers l'impression : pas de variante (bord droit artificiel dans le dessin) ;
    # une coupure dans la marge blanche passe pour un bord de papier découpé et reste admise
    if touche_cadre(dec) and hors_cadre(an["coins"], src.shape[1], src.shape[0]) > 0.03:
        return None
    f = fonds[rng.integers(len(fonds))]
    bg = cv2.cvtColor(cv2.imread(os.path.join(fonds_dir, f["code"] + ".jpg")), cv2.COLOR_BGR2RGB).astype(np.float32)
    Hh, Ww = bg.shape[:2]
    x0, y0, x1, y1 = m["carte"]
    dst, geo = placement(x1 - x0, y1 - y0, Ww, Hh, rng)
    bg_orig = bg   # repère des annotations du fond (page), pour la balance des blancs
    if geo["fond_tourne"]:   # fond photographié en paysage (ou en portrait) : quart de tour du fond
        bg = np.ascontiguousarray(np.rot90(bg))
        Hh, Ww = bg.shape[:2]
    H2 = cv2.getPerspectiveTransform(np.float32(rect(m["carte"])), np.float32(dst))
    T = H2 @ np.linalg.inv(np.array(an["H"]))
    # réduction préalable de la source si la carte rétrécit beaucoup (anticrénelage)
    ech = math.sqrt(abs(cv2.contourArea(np.float32(dst))) / max(1.0, abs(cv2.contourArea(np.float32(an["coins"])))))
    ech_tot = ech
    T_annot = T.copy()   # photo de travail -> fond (la réduction ci-dessous ne concerne que l'image)
    qx = np.array(an["coins"])
    bx0, by0 = np.maximum(qx.min(0), 0).astype(int)
    bx1, by1 = np.minimum(qx.max(0), src.shape[1::-1]).astype(int)
    flou_src = flou_photo(src[by0:by1, bx0:bx1].mean(2))
    if ech < 0.75:
        S = np.diag([ech, ech, 1.0])
        src = cv2.resize(src, None, fx=ech, fy=ech, interpolation=cv2.INTER_AREA)
        dec = cv2.resize(dec, None, fx=ech, fy=ech, interpolation=cv2.INTER_AREA)
        T = T @ np.linalg.inv(S)
    carte = cv2.warpPerspective(src, T, (Ww, Hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    alpha = cv2.warpPerspective(dec.astype(np.float32) / 255, T, (Ww, Hh), flags=cv2.INTER_LINEAR)
    valide = cv2.warpPerspective(np.ones(dec.shape, np.float32), T, (Ww, Hh), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.GaussianBlur(alpha * valide, (0, 0), 0.6)
    # balance des blancs et exposition de la carte accordées au fond
    lin_c = lumiere.srgb_lin(carte)
    sel = alpha > 0.98
    lc = lin_c[sel]
    blanc_c = np.median(lc[lc.mean(1) > np.percentile(lc.mean(1), 70)], 0)
    blanc_f = blanc_fond(bg_orig, f, refs) * rng.uniform(0.9, 1.04)
    gain = np.clip(blanc_f / np.maximum(blanc_c, 1e-4), 0.2, 4.0)
    lin_c = lin_c * gain
    # flou accordé : flou de la carte dans la source, mis à l'échelle de la pose, contre celui du fond
    fc = flou_src * ech_tot
    ff = flou_photo(bg.mean(2))
    sig = math.sqrt(max(0.0, ff ** 2 - fc ** 2)) * 0.8 + rng.uniform(0, 0.5)
    if sig > 0.3:
        lin_c = cv2.GaussianBlur(lin_c, (0, 0), sig)
    # ombre portée du bord de la carte sur le fond
    lin_b = lumiere.srgb_lin(bg)
    diag = float(np.linalg.norm(np.float32(dst[2]) - np.float32(dst[0])))
    a_ = rng.uniform(0, 2 * math.pi)
    d_ = rng.uniform(0.002, 0.01) * diag
    M = np.float32([[1, 0, d_ * math.cos(a_)], [0, 1, d_ * math.sin(a_)]])
    om = cv2.warpAffine(alpha, M, (Ww, Hh))
    om = cv2.GaussianBlur(om, (0, 0), rng.uniform(0.004, 0.015) * diag)
    force = rng.uniform(0.15, 0.4)
    lin_b = lin_b * (1 - force * om)[..., None]
    lin = lin_b * (1 - alpha[..., None]) + lin_c * alpha[..., None]
    base = np.clip(lumiere.lin_srgb(lin), 0, 255).astype(np.float32)
    base += rng.normal(0, 0.8, base.shape)
    eclairee, lum = lumiere.augmenter(base.copy(), rng)
    nom = f"{code}_v{i:02d}"
    q = int(rng.integers(86, 96))
    for d, im in (("base", base), ("eclairees", eclairee)):
        Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(os.path.join(sortie, d, nom + ".jpg"), quality=q)
    tr = lambda pts: appliquer(T_annot, pts).round(1).tolist()  # noqa: E731
    zones = [dict(z, quad=tr(z["quad"]), numero_quad=tr(z["numero_quad"]) if z.get("numero_quad") else None)
             for z in an["zones"]]
    coins = tr(an["coins"])
    v = dict(image=nom + ".jpg", code=nom, source=code, carte=an["carte"], modele=an["modele"], jeu=an["jeu"],
             support="carte", fond=f["code"], fond_type=f["type"], coins=coins, cartes=[coins], zones=zones,
             geometrie=geo, gain_blanc=[round(float(x), 3) for x in gain], flou_ajoute=round(sig, 2),
             ombre=dict(decalage=round(d_, 1), force=round(force, 2)), lumiere=lum,
             largeur=Ww, hauteur=Hh)
    json.dump(v, open(os.path.join(sortie, "annot", nom + ".json"), "w"), indent=1)
    return nom


def cmd_varier(a):
    import lumiere
    fonds = charger_fonds(a.fonds_annot, a.fonds_index)
    refs = {}
    for amb in ("lampe", "table_noire"):
        bl = []
        for f in fonds:
            if f["ambiance"] == amb and f["type"] in BLANCS and f["page"] and f["appareil"] == "pixel":
                bg = cv2.cvtColor(cv2.imread(os.path.join(a.fonds, f["code"] + ".jpg")), cv2.COLOR_BGR2RGB)
                bl.append(blanc_fond(bg.astype(np.float32), f, {}))
        if bl:
            refs[amb] = np.median(bl, 0)
    print(len(fonds), "fonds ;", {k: np.round(v, 3).tolist() for k, v in refs.items()})
    for d in ("base", "eclairees", "annot"):
        os.makedirs(os.path.join(a.sortie, d), exist_ok=True)
    codes = sorted(os.path.basename(f)[:-5] for f in os.listdir(a.annot) if f.endswith(".json") and f.startswith(("K", "cb_")))
    if a.codes:
        codes = [c for c in codes if c in a.codes.split(",")]
    taches = []
    for c in codes:
        an = json.load(open(os.path.join(a.annot, c + ".json")))
        if an["jeu"] != "train" or an.get("echec") or not an.get("zones"):
            continue
        taches += [(c, i, a.graine, a.travail, a.annot, fonds, a.fonds, refs, a.sortie, a.modeles)
                   for i in range(a.n)]
    with ProcessPoolExecutor(a.proc) as ex:
        noms = [x for x in ex.map(varier_une, taches) if x]
    print(len(noms), "variantes ;", len({t[0] for t in taches}) - len({x[:4] for x in noms}),
          "photos sans variante (carte coupée par le cadre)")
    ims = []
    for nom in noms[:: max(1, len(noms) // 24)][:24]:
        v = json.load(open(os.path.join(a.sortie, "annot", nom + ".json")))
        img = cv2.imread(os.path.join(a.sortie, "eclairees", nom + ".jpg"))
        v["code"] = f"{nom} {v['fond']}"
        v["carte"] = v["carte"]
        ims.append(controle(img, v))
    for i in range(0, len(ims), 12):
        planche(ims[i:i + 12], os.path.join(a.sortie, f"_controle_{i // 12 + 1:02d}.jpg"))


# --- versions brutes recadrées ----------------------------------------------------------------------------------

def cmd_brut(a):
    os.makedirs(os.path.join(a.sortie, "annot"), exist_ok=True)
    os.makedirs(os.path.join(a.sortie, "images"), exist_ok=True)
    for f in sorted(os.listdir(a.annot)):
        if not (f.startswith("K") and f.endswith(".json")):
            continue
        an = json.load(open(os.path.join(a.annot, f)))
        if an.get("echec"):
            continue
        im = Image.open(os.path.join(a.travail, an["image"]))
        q = np.array(an["coins"])
        ext = q.max(0) - q.min(0)
        mg = 0.06 * ext.max()
        x0, y0 = np.maximum(q.min(0) - mg, 0).astype(int)
        x1, y1 = np.minimum(q.max(0) + mg, im.size).astype(int)
        im.crop((x0, y0, x1, y1)).save(os.path.join(a.sortie, "images", an["image"]), quality=93)
        sh = lambda pts: (np.array(pts) - [x0, y0]).round(1).tolist()  # noqa: E731
        out = dict(an, coins=sh(an["coins"]), cartes=[sh(an["coins"])],
                   zones=[dict(z, quad=sh(z["quad"]), numero_quad=sh(z["numero_quad"]) if z.get("numero_quad")
                               else None) for z in an.get("zones", [])],
                   recadrage=[int(x0), int(y0), int(x1), int(y1)], largeur=int(x1 - x0), hauteur=int(y1 - y0))
        H = np.array(an["H"])
        out["H"] = (np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ H).tolist()
        out.pop("papier", None)
        out.pop("decoupe", None)
        json.dump(out, open(os.path.join(a.sortie, "annot", f), "w"), indent=1)
    print("brut :", a.sortie)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("modeles")
    p.add_argument("base")
    p.add_argument("plus")
    p.add_argument("sortie")
    p = sp.add_parser("travail")
    p.add_argument("importe")
    p.add_argument("classement")
    p.add_argument("sortie")
    p = sp.add_parser("annoter")
    p.add_argument("travail")
    p.add_argument("classement")
    p.add_argument("modeles")
    p.add_argument("sortie")
    p.add_argument("--proc", type=int, default=6)
    p.add_argument("--codes")
    p = sp.add_parser("varier")
    for k in ("travail", "annot", "fonds", "fonds_annot", "fonds_index", "sortie"):
        p.add_argument(k)
    p.add_argument("n", type=int)
    p.add_argument("graine", type=int)
    p.add_argument("--modeles", required=True)
    p.add_argument("--proc", type=int, default=6)
    p.add_argument("--codes")
    p = sp.add_parser("brut")
    p.add_argument("travail")
    p.add_argument("annot")
    p.add_argument("sortie")
    a = ap.parse_args()
    dict(modeles=cmd_modeles, travail=cmd_travail, annoter=cmd_annoter, varier=cmd_varier, brut=cmd_brut)[a.cmd](a)


if __name__ == "__main__":
    main()
