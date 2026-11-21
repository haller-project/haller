#!/bin/bash
# Pour chaque lot dont l'audit échoue, retire les pages hors plage (filtrer.py) et refait l'audit.
#   bash filtrer_lots.sh PY DOSSIER_LOTS        (lots non vides seulement ; les .brut.npz sont ignorés)
PY=$1; cd "$(dirname "$(readlink -f "$0")")/.."
for f in $2/*.npz; do
  case $f in *.brut.npz|*.tmp.npz) continue;; esac
  [ -s $f ] || continue
  grep -q '"statut": "passe"' $f.audit.json 2>/dev/null && continue
  $PY -m variete.filtrer $f && $PY -m variete.audit $f --sans-planches | grep -o "Statut : [a-zé]*" | sed "s#^#$(basename $f) #"
done
