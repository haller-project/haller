#!/bin/bash
# Lots d'entraînement f<i> et de validation fv0 du détecteur du 21/11/2026 (gen_univ.py, configuration validée
# e9dc866d13842135) : configuration univ5 (numéros imprimés de la carte SeedSigner corrigés, UNIV_NUMEROS_CORRIGES=1 ;
# distributions réglées sur les photos réelles, UNIV_REEL=1 ; listes multicolonnes 15 %) plus numéros sur pastille
# sombre (UNIV_PASTILLE=0.4 des cartes procédurales, ≈ 5 % des pages) ; graines 50000 + i ; fv = validation, graine 59999.
# Chaque lot est écrit avec son journal de variété (<lot>.variete.jsonl) puis audité (<lot>.audit.json) ; les lots
# dont l'audit échoue sont filtrés ensuite par ../variete/filtrer_lots.sh.
#   OUT=DOSSIER bash lots_univ6.sh PROCESSUS SPEC...    SPEC = prefixe:debut:fin (f = pages simulées, fv = validation)
#   sortie $OUT/<prefixe><i>.npz ; un lot existant n'est pas refait.
#   Polices : HALLER_FONTS (287 polices de `python -m haller.fonts`) ; EMNIST : ../data/emnist_train.npz ;
#   modèles de cartes : ../../data/modeles_cartes (ou MODELES_CARTES) ; PYTHON : interpréteur (Pillow ≥ 11 ; lots
#   d'origine : Python 3.12.3, numpy 2.5.5, Pillow 13.0.0).
cd "$(dirname "$0")"
: "${OUT:?OUT=dossier de sortie}"; mkdir -p "$OUT"; export OUT PY=${PYTHON:-python3}
export UNIV_ALTERE=0.3 UNIV_VERSO=0.12 UNIV_MULTI=0.15 UNIV_ROT=1 UNIV_NUM=0.6 UNIV_SERRE=0.4 UNIV_NUMEROS_CORRIGES=1 UNIV_REEL=1 UNIV_PASTILLE=0.4
P=$1; shift
for spec in "$@"; do
  IFS=: read pre a b <<< "$spec"
  for i in $(seq $a $b); do echo "$pre $i"; done
done | xargs -P $P -L 1 sh -c '
  pre=$0; i=$1; out=$OUT/$pre$i.npz; test -f $out && exit 0
  g=$((50000 + i)); [ $pre = fv ] && g=59999
  nice $PY gen_univ.py train $out.tmp.npz 1000 $g && mv $out.tmp.npz.variete.jsonl $out.variete.jsonl && mv $out.tmp.npz $out &&
  PYTHONPATH=.. nice $PY -m variete.audit $out --sans-planches > /dev/null'
