"""Audit de la variété d'un lot de données synthétiques (docs/variete.md).

    python -m variete.audit LOT [--registre R] [--extremes N] [--sans-planches]
    python -m variete.audit valider LOT --par QUI --note "ce qui a été regardé et jugé"

LOT : lot NPZ (journal <lot>.variete.jsonl à côté) ou dossier de banc (variete.jsonl, verite.json, images).

Audit : pour chaque facteur du registre (train/variete/registre.json) produit par le générateur du lot :
  - quantiles (médiane, p90, p99, max) et histogramme, comparés aux statistiques réelles du registre ;
  - alertes : part des valeurs hors de la plage plausible au-delà de la tolérance, part des cas rares au-delà de la
    part maximale admise, proportions des catégories hors de leurs fourchettes ;
  - planches des pages aux extrêmes de chaque facteur (les N plus extrêmes et quelques médianes, valeur indiquée) ;
    pour les blancs entre lettres d'un banc, découpes des mots aux plus grands blancs mesurés sur l'image.
Sorties à côté du lot : <lot>.audit.json, <lot>.audit.md, <lot>.audit/ (planches). Statut « passe » si aucune
alerte bloquante.

Validation : `valider` inscrit la configuration du lot (signature : sources du générateur et variables
d'environnement) dans train/variete/validations.json, après examen des planches par une personne ou une session.
Un lot entre à l'entraînement si son audit est « passe », s'il porte sur ce fichier exact (sha256) et si sa
configuration est validée (variete/garde.py).
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
from variete import journal as J  # noqa: E402

REGISTRE = os.path.join(ICI, "registre.json")
VALIDATIONS = os.path.join(ICI, "validations.json")


def sha256(chemin, bloc=1 << 22):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while True:
            b = f.read(bloc)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def chemins_audit(lot):
    if os.path.isdir(lot):
        return os.path.join(lot, "audit.json"), os.path.join(lot, "audit.md"), os.path.join(lot, "audit")
    return lot + ".audit.json", lot + ".audit.md", lot + ".audit"


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [1, 5, 50, 90, 95, 99])
    return dict(n=int(len(x)), min=round(float(x.min()), 4), q01=round(q[0], 4), q05=round(q[1], 4),
                q50=round(q[2], 4), q90=round(q[3], 4), q95=round(q[4], 4), q99=round(q[5], 4),
                max=round(float(x.max()), 4), moyenne=round(float(x.mean()), 4))


def garde_ligne(ligne, filtre):
    for k, vals in (filtre or {}).items():
        if str(ligne.get(k)) not in [str(v) for v in vals]:
            return False
    return True


def valeurs(lignes, nom, f):
    """[(valeur, index de ligne)] du facteur, pages filtrées ; les listes (une valeur par mot) sont aplaties."""
    out = []
    transf = f.get("transformation")
    for j, l in enumerate(lignes):
        if nom not in l and not (transf and f.get("champ") in l):
            continue
        if not garde_ligne(l, f.get("filtre")):
            continue
        v = l.get(f.get("champ", nom))
        vs = v if isinstance(v, list) else [v]
        for x in vs:
            if x is None or isinstance(x, (str, dict)):
                continue
            x = float(x)
            if transf == "abs":
                x = abs(x)
            elif transf == "ecart_90":   # écart au multiple de 90° le plus proche
                x = abs(((x + 45) % 90) - 45)
            out.append((x, j))
    return out


def marge(p, n):
    """Marge d'échantillonnage (2 écarts types binomiaux) sur une part p estimée sur n valeurs ; nulle si p = 0 :
    une valeur hors d'une plage à tolérance nulle est toujours une alerte."""
    return 0.0 if p <= 0 or n <= 0 else 2 * (p * (1 - p) / n) ** 0.5


def verifier_numerique(nom, f, vals):
    x = np.array([v for v, _ in vals], float)
    s = stats(x)
    alertes = []
    if len(x) == 0:
        return s, alertes
    lo, hi = f.get("plage", [None, None])
    hors = np.zeros(len(x), bool)
    if lo is not None:
        hors |= x < lo
    if hi is not None:
        hors |= x > hi
    p_hors = float(hors.mean())
    s["part_hors_plage"] = round(p_hors, 5)
    tol = f.get("part_hors_plage_max", 0.0)
    if p_hors > tol + marge(tol, len(x)):
        alertes.append(dict(facteur=nom, gravite="bloquant",
                            texte=f"{p_hors:.2%} des valeurs hors de la plage plausible [{lo}, {hi}] "
                                  f"(tolérance {tol:.2%}) ; max {x.max():.3g}, min {x.min():.3g}"))
    for r in f.get("rares", []):
        sens = r.get("sens", ">")
        p = float((x > r["seuil"]).mean() if sens == ">" else (x < r["seuil"]).mean())
        s[f"part_{'sup' if sens == '>' else 'inf'}_{r['seuil']}"] = round(p, 5)
        if p > r["part_max"] + marge(r["part_max"], len(x)):
            alertes.append(dict(facteur=nom, gravite=r.get("gravite", "bloquant"),
                                texte=f"{p:.2%} des valeurs {sens} {r['seuil']} (part maximale admise "
                                      f"{r['part_max']:.2%})"))
    return s, alertes


def verifier_proportions(nom, f, lignes):
    champ = f.get("champ", nom)
    vals = [str(l.get(champ)) for l in lignes if champ in l and garde_ligne(l, f.get("filtre"))]
    s = {"n": len(vals)}
    alertes = []
    if not vals:
        return s, alertes
    from collections import Counter
    c = Counter(vals)
    props = {k: round(v / len(vals), 4) for k, v in sorted(c.items(), key=lambda t: -t[1])}
    s["proportions"] = props
    fourch = f.get("valeurs", {})
    for k, (lo, hi) in fourch.items():
        p = props.get(k, 0.0)
        if not lo <= p <= hi:
            alertes.append(dict(facteur=nom, gravite=f.get("gravite", "bloquant"),
                                texte=f"« {k} » : {p:.1%} hors de la fourchette [{lo:.0%}, {hi:.0%}]"))
    autres = sum(p for k, p in props.items() if k not in fourch)
    if fourch and autres > f.get("autres_max", 0.0) + 1e-9:
        alertes.append(dict(facteur=nom, gravite=f.get("gravite", "bloquant"),
                            texte=f"valeurs hors liste : {autres:.1%} (max {f.get('autres_max', 0.0):.0%})"))
    return s, alertes


# ---------- planches ----------

def vignette_page(lot, lignes, j, X, haut=300):
    from PIL import Image
    l = lignes[j]
    if X is not None:
        im = Image.fromarray(X[l["i"]])
        if X.shape[1] <= 64:   # lignes des lecteurs (32 x 256) : agrandies, pas réduites
            haut = 64
    else:
        im = Image.open(os.path.join(lot, l["image"])).convert("RGB")
        im.thumbnail((4 * haut, 4 * haut))
    return im.resize((max(1, int(im.width * haut / im.height)), haut))


def planche(cellules, chemin, titre, cols=4):
    """cellules : [(image PIL, légende)]."""
    from PIL import Image, ImageDraw
    if not cellules:
        return None
    lw = max(c[0].width for c in cellules)
    lh = max(c[0].height for c in cellules) + 26
    rows = -(-len(cellules) // cols)
    out = Image.new("RGB", (cols * (lw + 8), rows * lh + 34), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.text((6, 8), titre, fill=(0, 0, 0))
    for k, (im, leg) in enumerate(cellules):
        x, y = (k % cols) * (lw + 8), 34 + (k // cols) * lh
        out.paste(im.convert("RGB"), (x, y))
        d.text((x + 2, y + im.height + 4), leg, fill=(0, 0, 0))
    out.save(chemin, quality=88)
    return chemin


def planche_facteur(lot, lignes, nom, vals, X, dossier, n=12, n_med=4):
    """Pages aux extrêmes du facteur (valeur la plus haute de la page pour un facteur par mot) et médianes."""
    par_page, mini = {}, {}
    for v, j in vals:
        par_page[j] = max(par_page.get(j, -np.inf), v)
        mini[j] = min(mini.get(j, np.inf), v)
    if not par_page:
        return None
    items = sorted(par_page.items(), key=lambda t: t[1])
    choix = items[-n:][::-1]
    # minima : plus petite valeur de la page (facteurs par mot : le mot le plus petit, le blanc le plus étroit)
    bas = sorted(mini.items(), key=lambda t: t[1])[:max(4, n // 2)]
    m = len(items) // 2
    med = items[max(0, m - n_med // 2):m + n_med - n_med // 2]
    cell = []
    vus = set()
    for groupe, lst in (("max", choix), ("med", med), ("min", bas)):
        for j, v in lst:
            if j in vus:
                continue
            vus.add(j)
            try:
                im = vignette_page(lot, lignes, j, X)
            except Exception:
                continue
            l = lignes[j]
            cell.append((im, f"{groupe} {v:.3g} | {l.get('style', '')[:4]} {l.get('defaut', '')[:10]} p{l['i']}"))
    return planche(cell, os.path.join(dossier, f"{nom}.jpg"), f"{os.path.basename(lot)} : {nom} (max, médianes, min)")


def blancs_banc(lot, n=24):
    """Blancs entre lettres mesurés sur l'image finale des mots d'un banc (même mesure que sur les photos réelles)."""
    from PIL import Image
    from variete.mesures_image import max_blanc, redresser
    ver = json.load(open(os.path.join(lot, "verite.json")))
    res = []
    for nom, v in ver.items():
        g = np.asarray(Image.open(os.path.join(lot, nom)).convert("L"))
        for o in v.get("objets", []):
            if o.get("classe") != "mot" or o.get("cache") or o.get("partiel") or o.get("autre"):
                continue
            q = np.asarray(o["quad"], float)
            if min(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[1] - q[0])) < 24:
                continue   # mot de moins de 24 px : mesure du blanc sans objet (réel : mots de 37 px au moins)
            b = max_blanc(g, o["quad"])
            if b is not None:
                res.append((b, nom, o["quad"], v.get("style"), o.get("alteration")))
    return res


def planche_blancs(lot, res, dossier, n=28):
    from PIL import Image
    from variete.mesures_image import redresser
    if not res:
        return None
    res = sorted(res, key=lambda t: -t[0])
    m = len(res) // 2
    sel = [("max", r) for r in res[:n]] + [("med", r) for r in res[m - 4:m + 4]]
    cache, cell = {}, []
    for g_, (b, nom, q, st, alt) in sel:
        if nom not in cache:
            cache[nom] = np.asarray(Image.open(os.path.join(lot, nom)).convert("L"))
        c = redresser(cache[nom], q)
        if c is None:
            continue
        im = Image.fromarray(c)
        im = im.resize((min(520, int(im.width * 64 / im.height)), 64))
        fond = Image.new("L", (520, 64), 255)
        fond.paste(im, (0, 0))
        cell.append((fond, f"{g_} {b:.2f} {str(st)[:4]} {alt or ''} {nom}"))
    return planche(cell, os.path.join(dossier, "blanc_lettres_image.jpg"),
                   f"{os.path.basename(lot)} : plus grands blancs entre lettres (mesure image, hauteur du mot)", cols=2)


# ---------- audit ----------

def lire_lot(lot):
    """(config, lignes) : journal du lot, ou, pour un dossier de variantes de cartes (cartes.py varier), les
    géométries réalisées enregistrées dans annot/*.json."""
    if os.path.exists(J.chemin_pour(lot)):
        return J.lire(lot)
    if not os.path.isdir(lot) and os.path.exists(lot + ".variety.jsonl"):   # lecteurs (haller.generate)
        config, lignes = None, []
        for l in open(lot + ".variety.jsonl"):
            d = json.loads(l)
            if "config" in d and "i" not in d:
                config = d["config"]
            else:
                lignes.append(d)
        return config, lignes
    ann = os.path.join(lot, "annot")
    if os.path.isdir(ann):
        import glob
        lignes = []
        for k, f in enumerate(sorted(glob.glob(os.path.join(ann, "K*.json")))):
            a = json.load(open(f))
            g = a.get("geometrie") or {}
            if "rotation" not in g:
                continue
            r = float(g["rotation"])
            lu = a.get("lumiere") or {}
            l = dict(i=k, gen="cartes.varier", image=os.path.join("eclairees", a["image"]), rotation_deg=r,
                     quart=int(round((r % 360) / 90)) % 4 * 90, trapeze=g.get("trapeze"), part_carte=g.get("part_image"),
                     flou_ajoute=a.get("flou_ajoute"), modele=a.get("modele"), fond_type=a.get("fond_type"))
            if "temperature" in lu:
                l["lumiere_temperature"] = lu["temperature"]
            lignes.append(l)
        return {"generateur": "cartes.varier", "signature": None}, lignes
    raise FileNotFoundError(f"aucun journal pour {lot}")


def auditer(lot, registre=REGISTRE, extremes=12, planches=True):
    reg = json.load(open(registre))
    config, lignes = lire_lot(lot)
    gen = (config or {}).get("generateur") or (lignes[0]["gen"] if lignes else "?")
    f_json, f_md, dossier = chemins_audit(lot)
    os.makedirs(dossier, exist_ok=True)
    X = None
    if not os.path.isdir(lot) and planches:
        X = np.load(lot)["x"]
    res = dict(lot=os.path.abspath(lot), date=datetime.datetime.now().isoformat(timespec="seconds"),
               generateur=gen, config=config, pages=len(lignes), registre_version=reg.get("version"),
               facteurs={}, alertes=[], planches={})
    if not os.path.isdir(lot):
        res["sha256"] = sha256(lot)
        res["taille"] = os.path.getsize(lot)
    if os.path.exists(J.chemin_pour(lot)):
        res["sha256_journal"] = sha256(J.chemin_pour(lot))
    presents = set(k for l in lignes for k in l)
    couverts = set()
    for nom, f in reg["facteurs"].items():
        if gen not in f.get("generateurs", []):
            continue
        champ = f.get("champ", nom)
        couverts.add(champ)
        if champ not in presents:
            if f.get("requis", True):
                res["alertes"].append(dict(facteur=nom, gravite="bloquant",
                                           texte=f"facteur absent du journal (champ {champ})"))
            continue
        if f.get("type") == "proportions":
            s, al = verifier_proportions(nom, f, lignes)
        else:
            vals = valeurs(lignes, nom, f)
            s, al = verifier_numerique(nom, f, vals)
            if planches and vals and f.get("planche", True):
                p = planche_facteur(lot, lignes, nom, vals, X, dossier, n=extremes)
                if p:
                    res["planches"][nom] = p
        res["facteurs"][nom] = dict(stats=s, reel=f.get("reel"), plage=f.get("plage"))
        res["alertes"] += al
    res["non_couverts"] = sorted(presents - couverts - {"i", "gen", "image"})
    if os.path.isdir(lot) and os.path.exists(os.path.join(lot, "verite.json")) and \
            any(gen in f.get("generateurs", []) for n_, f in reg["facteurs"].items() if n_ == "blanc_lettres_max"):
        try:
            bl = blancs_banc(lot)
            f = reg["facteurs"].get("blanc_lettres_image", {})
            vals = [(b, 0) for b, *_ in bl]
            s, al = verifier_numerique("blanc_lettres_image", f, vals) if f else (stats([b for b, *_ in bl]), [])
            res["facteurs"]["blanc_lettres_image"] = dict(stats=s, reel=f.get("reel"), plage=f.get("plage"))
            res["alertes"] += al
            if planches:
                p = planche_blancs(lot, bl, dossier)
                if p:
                    res["planches"]["blanc_lettres_image"] = p
        except ImportError as e:
            res["alertes"].append(dict(facteur="blanc_lettres_image", gravite="info",
                                       texte=f"mesure image impossible ({e}) : lancer avec opencv"))
    res["statut"] = "echec" if any(a["gravite"] == "bloquant" for a in res["alertes"]) else "passe"
    json.dump(res, open(f_json, "w"), indent=1, ensure_ascii=False, default=str)
    open(f_md, "w").write(rapport(res))
    return res


def fmt(v):
    return "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def rapport(res):
    L = [f"# Audit de variété : `{os.path.basename(res['lot'])}`", "",
         f"Générateur `{res['generateur']}`, {res['pages']} pages, signature de configuration "
         f"`{(res.get('config') or {}).get('signature')}`, registre {res.get('registre_version')}, {res['date']}.", "",
         f"**Statut : {res['statut']}**", ""]
    if res["alertes"]:
        L += ["## Alertes", ""] + [f"- {a['gravite']} — `{a['facteur']}` : {a['texte']}" for a in res["alertes"]] + [""]
    L += ["## Facteurs", "", "| Facteur | n | médiane | p90 | p99 | max | réel (médiane / p99 / max) | plage |",
          "|---|---|---|---|---|---|---|---|"]
    for nom, d in res["facteurs"].items():
        s = d["stats"]
        r = d.get("reel") or {}
        if "proportions" in s:
            props = ", ".join(f"{k} {v:.0%}" for k, v in list(s["proportions"].items())[:6])
            L.append(f"| {nom} | {s['n']} | {props} | | | | {r.get('texte', '') if isinstance(r, dict) else ''} | |")
        else:
            L.append(f"| {nom} | {s.get('n')} | {fmt(s.get('q50'))} | {fmt(s.get('q90'))} | {fmt(s.get('q99'))} | "
                     f"{fmt(s.get('max'))} | {fmt(r.get('q50'))} / {fmt(r.get('q99'))} / {fmt(r.get('max'))} | "
                     f"{d.get('plage')} |")
    if res.get("non_couverts"):
        L += ["", "Champs du journal sans entrée au registre : " + ", ".join(f"`{k}`" for k in res["non_couverts"])]
    if res["planches"]:
        L += ["", "## Planches", ""] + [f"- `{k}` : `{v}`" for k, v in res["planches"].items()]
    return "\n".join(L) + "\n"


def valider(lot, par, note):
    f_json, _, _ = chemins_audit(lot)
    res = json.load(open(f_json))
    if res["statut"] != "passe":
        sys.exit(f"audit en échec : {f_json} ; corriger le générateur avant de valider")
    sig = (res.get("config") or {}).get("signature")
    if not sig:
        sys.exit("lot sans signature de configuration (journal sans en-tête)")
    val = json.load(open(VALIDATIONS)) if os.path.exists(VALIDATIONS) else {}
    val[sig] = dict(generateur=res["generateur"], env=res["config"].get("env"), sources=res["config"].get("sources"),
                    date=datetime.date.today().isoformat(), par=par, note=note, lot_examine=res["lot"],
                    planches=sorted(res["planches"]))
    json.dump(val, open(VALIDATIONS, "w"), indent=1, ensure_ascii=False, sort_keys=True)
    print(f"configuration {sig} validée ({res['generateur']})")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "valider":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("lot")
        ap.add_argument("--par", required=True)
        ap.add_argument("--note", required=True)
        a = ap.parse_args()
        return valider(a.lot, a.par, a.note)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lot")
    ap.add_argument("--registre", default=REGISTRE)
    ap.add_argument("--extremes", type=int, default=12)
    ap.add_argument("--sans-planches", action="store_true")
    a = ap.parse_args()
    res = auditer(a.lot, a.registre, a.extremes, not a.sans_planches)
    print(rapport(res))
    sys.exit(0 if res["statut"] == "passe" else 1)


if __name__ == "__main__":
    main()
