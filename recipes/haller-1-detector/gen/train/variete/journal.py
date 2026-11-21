"""Journal des facteurs de variété réellement tirés par les générateurs de données synthétiques.

Les générateurs appellent `noter` (valeur scalaire de la page) et `ajouter` (une valeur par mot, par ligne, par
objet) au point où la valeur est tirée ou réalisée. Le journal ne consomme aucun tirage aléatoire : un lot produit
avec ou sans journal est identique au bit près. Hors d'une page ouverte (`ouvrir`), les appels ne font rien.

Format : une ligne JSON par page ou image, dans l'ordre du lot, écrite à côté du lot :
  lot NPZ  -> <lot>.variete.jsonl ;  dossier de banc -> <dossier>/variete.jsonl.
Chaque ligne : {"i": index, "gen": générateur, facteur: valeur | [valeurs], ...}. Les noms de facteurs sont ceux du
registre (train/variete/registre.json) ; un facteur absent du registre est conservé et signalé par l'audit.

    from variete import journal as J
    J.ouvrir("gen_univ", i)        # début de page
    J.noter("rotation_deg", ang)   # valeur de la page (la dernière écrite l'emporte)
    J.ajouter("blanc_lettres_h", g)  # une valeur de plus pour la page
    ligne = J.fermer()             # dict de la page, journal remis à zéro
"""
import json
import os

import numpy as np

_page = None


def ouvrir(gen, i):
    global _page
    _page = {"i": int(i), "gen": gen}


def reouvrir():
    """Remet à zéro la page en cours (essai raté d'un générateur qui recommence la même page)."""
    if _page is not None:
        ouvrir(_page["gen"], _page["i"])


def actif():
    return _page is not None


def _py(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    return v


def noter(facteur, valeur):
    if _page is not None:
        _page[facteur] = _py(valeur)


def ajouter(facteur, valeur):
    if _page is not None:
        _page.setdefault(facteur, []).append(_py(valeur))


def fermer():
    global _page
    p, _page = _page, None
    return p


def chemin_pour(lot):
    """Chemin du journal d'un lot (fichier NPZ ou dossier de banc)."""
    if os.path.isdir(lot):
        return os.path.join(lot, "variete.jsonl")
    return lot + ".variete.jsonl"


class Ecrivain:
    """Écrit les lignes de journal d'un lot, une page à la fois."""
    def __init__(self, lot, config=None):
        self.f = open(chemin_pour(lot), "w")
        if config is not None:
            self.f.write(json.dumps({"config": config}, ensure_ascii=False) + "\n")

    def ecrire(self, ligne):
        if ligne is not None:
            self.f.write(json.dumps(ligne, ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()


def lire(lot):
    """(config, lignes) du journal d'un lot."""
    config, lignes = None, []
    with open(chemin_pour(lot)) as f:
        for l in f:
            d = json.loads(l)
            if "config" in d and "i" not in d:
                config = d["config"]
            else:
                lignes.append(d)
    return config, lignes


# ---------- mesures sur l'encre propre (avant photo), sans tirage ----------

def blancs_internes(m, haut, seuil=128, min_encre=1):
    """Blancs entre lettres d'un mot : liste des largeurs des colonnes vides intérieures (rapportées à `haut`).

    m : image du mot (255 = papier). Une colonne est vide si elle compte moins de `min_encre` pixels d'encre.
    """
    m = np.asarray(m)
    if m.ndim != 2 or m.size == 0 or haut <= 0:
        return []
    col = (m < seuil).sum(0) >= min_encre
    xs = np.nonzero(col)[0]
    if len(xs) < 2:
        return []
    c = col[xs[0]:xs[-1] + 1]
    runs, n = [], 0
    for v in c:
        if v:
            if n:
                runs.append(n)
            n = 0
        else:
            n += 1
    return [r / haut for r in runs]


def config_lot(generateur, fichiers, variables):
    """Signature de configuration d'un lot : sha256 des sources du générateur et variables d'environnement."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(fichiers, key=os.path.basename):   # ordre indépendant de la forme du chemin (machine)
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    env = {k: os.environ[k] for k in sorted(set(variables)) if k in os.environ}
    h.update(json.dumps(env, sort_keys=True).encode())
    return {"generateur": generateur, "sources": sorted(os.path.basename(f) for f in fichiers), "env": env,
            "signature": h.hexdigest()[:16]}
