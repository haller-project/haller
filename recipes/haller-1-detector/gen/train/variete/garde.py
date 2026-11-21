"""Garde-fou d'entrée en entraînement : un lot synthétique n'entre que si son audit de variété est passé.

    python3 garde.py LOT.npz [LOT.npz ...] [--essai-court "raison"]

Pour chaque lot : <lot>.audit.json présent (variete/audit.py), statut « passe », empreinte sha256 du lot égale à
celle de l'audit (le lot n'a pas changé depuis), configuration du générateur (signature) validée après examen des
planches dans train/variete/validations.json. Sinon : refus, code 3, avec la raison.

--essai-court "raison" (ou VARIETE_ESSAI_COURT="raison") : passe outre pour un essai court explicite ; le forçage est
inscrit dans train/variete/forcages.log. Un modèle entraîné ainsi ne peut pas être publié ni intégré.

Bibliothèque standard seulement : tourne sur toute machine d'entraînement (python3 du système).
"""
import datetime
import hashlib
import json
import os
import socket
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
VALIDATIONS = os.path.join(ICI, "validations.json")
FORCAGES = os.path.join(ICI, "forcages.log")


def sha256(chemin, bloc=1 << 22):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while True:
            b = f.read(bloc)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verifier(lot, validations):
    """Liste des raisons de refus (vide si le lot est admis)."""
    if not os.path.exists(lot):
        return [f"lot introuvable : {lot}"]
    f = (os.path.join(lot, "audit.json") if os.path.isdir(lot) else lot + ".audit.json")
    if not os.path.exists(f):
        return [f"pas d'audit de variété ({os.path.basename(f)}) : python -m variete.audit {lot}"]
    a = json.load(open(f))
    r = []
    if a.get("statut") != "passe":
        al = [x["texte"] for x in a.get("alertes", []) if x.get("gravite") == "bloquant"][:3]
        r.append("audit en échec : " + " ; ".join(al))
    if not os.path.isdir(lot):
        if a.get("taille") != os.path.getsize(lot) or a.get("sha256") != sha256(lot):
            r.append("le lot a changé depuis son audit (sha256 différent) : refaire l'audit")
    sig = (a.get("config") or {}).get("signature")
    if not sig:
        r.append("journal sans signature de configuration")
    elif sig not in validations:
        r.append(f"configuration {sig} ({a.get('generateur')}) non validée : examiner les planches d'un banc de même "
                 f"configuration puis python -m variete.audit valider BANC --par … --note …")
    return r


def main():
    args = sys.argv[1:]
    raison = os.environ.get("VARIETE_ESSAI_COURT")
    if "--essai-court" in args:
        k = args.index("--essai-court")
        raison = args[k + 1] if k + 1 < len(args) else "sans raison"
        args = args[:k] + args[k + 2:]
    validations = json.load(open(VALIDATIONS)) if os.path.exists(VALIDATIONS) else {}
    refus = {}
    for lot in args:
        r = verifier(lot, validations)
        if r:
            refus[lot] = r
    if not refus:
        print(f"variété : {len(args)} lot(s) admis")
        return 0
    for lot, r in refus.items():
        for x in r:
            print(f"variété : REFUS {lot} : {x}", file=sys.stderr)
    if raison:
        with open(FORCAGES, "a") as f:
            f.write(json.dumps(dict(date=datetime.datetime.now().isoformat(timespec="seconds"), machine=socket.gethostname(),
                                    raison=raison, lots=sorted(refus)), ensure_ascii=False) + "\n")
        print(f"variété : ESSAI COURT FORCÉ ({raison}) : {len(refus)} lot(s) sans audit validé ; modèle non publiable",
              file=sys.stderr)
        return 0
    print("variété : entraînement refusé (essai court : --essai-court \"raison\" ou VARIETE_ESSAI_COURT)",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
