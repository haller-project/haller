"""Retire d'un lot NPZ les pages dont un facteur sort d'une plage plausible à tolérance nulle du registre
(docs/variete.md). Le générateur et sa signature ne changent pas : le lot filtré est un sous-ensemble des
pages tirées, son journal garde l'en-tête de configuration (avec la liste des pages retirées) et les lignes des
pages conservées, renumérotées. Le lot d'origine est gardé à côté (<lot>.brut.npz et son journal) ; l'audit est à
refaire sur le lot filtré.

    python -m variete.filtrer LOT.npz [LOT.npz ...]

Pourquoi : les plages à tolérance nulle (perspective ≤ 0,6, part de carte ≤ 3) sont dépassées par ~0,2 % des pages
de gen_univ en configuration univ5 ; un banc de 360 pages ne le montre pas, des lots de 1 000 pages si.
"""
import json
import os
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
from variete import audit as A  # noqa: E402
from variete import journal as J  # noqa: E402


def pages_hors_plage(config, lignes, registre=A.REGISTRE):
    """{index de ligne : [facteurs]} des pages hors d'une plage à tolérance nulle."""
    reg = json.load(open(registre))["facteurs"]
    gen = (config or {}).get("generateur")
    hors = {}
    for nom, f in reg.items():
        if gen not in f.get("generateurs", [gen]) or f.get("part_hors_plage_max", 0.0) > 0 or "plage" not in f:
            continue
        lo, hi = f["plage"]
        for x, j in A.valeurs(lignes, nom, f):
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                hors.setdefault(j, []).append(nom)
    return hors


def filtrer(lot):
    config, lignes = A.lire_lot(lot)
    hors = pages_hors_plage(config, lignes)
    if not hors:
        print(f"{lot} : aucune page hors plage")
        return 0
    jr = J.chemin_pour(lot)
    brut = lot[:-4] + ".brut.npz"
    if not os.path.exists(brut):
        shutil.copy2(lot, brut)
        shutil.copy2(jr, J.chemin_pour(brut))
    d = np.load(brut)
    meta = json.loads(str(d["meta"]))
    retirees = sorted(int(lignes[j]["i"]) for j in hors)
    garder = [i for i in range(len(d["x"])) if i not in set(retirees)]
    np.savez_compressed(lot[:-4] + ".tmp.npz", x=d["x"][garder], y=d["y"][garder],
                        meta=json.dumps([meta[i] for i in garder]))
    os.replace(lot[:-4] + ".tmp.npz", lot)
    config = dict(config, filtre_plages=dict(retirees=retirees, facteurs=sorted({n for v in hors.values() for n in v}),
                                             pages_brutes=len(d["x"])))
    with open(jr, "w") as f:
        f.write(json.dumps({"config": config}, ensure_ascii=False) + "\n")
        k = 0
        for l in lignes:
            if int(l["i"]) in set(retirees):
                continue
            f.write(json.dumps(dict(l, i=k), ensure_ascii=False) + "\n")
            k += 1
    print(f"{lot} : {len(retirees)} page(s) retirée(s) sur {len(d['x'])} ({', '.join(config['filtre_plages']['facteurs'])})")
    return len(retirees)


if __name__ == "__main__":
    for lot in sys.argv[1:]:
        filtrer(lot)
