#!/bin/bash
# Lots d'entraînement f<i> et de validation fv0 de univ5 (gen_univ.py) : numéros imprimés de la carte SeedSigner
# corrigés (UNIV_NUMEROS_CORRIGES=1), distributions réglées sur les photos réelles (UNIV_REEL=1, statistiques
# agrégées de distributions/distributions_reelles.json ; listes multicolonnes 15 %), graines 50000 + i ;
# fv = validation, graine 59999.
#   OUT=DOSSIER bash lots_univ5.sh PROCESSUS SPEC...    SPEC = prefixe:debut:fin (f = pages simulées, fv = validation)
#   sortie $OUT/<prefixe><i>.npz ; un lot existant n'est pas refait.
#   Polices : HALLER_FONTS (287 polices de `python -m haller.fonts`) ; EMNIST : ../data/emnist_train.npz ;
#   modèles de cartes : ../../data/modeles_cartes (ou MODELES_CARTES) ; PYTHON : interpréteur (python3 par défaut).
cd "$(dirname "$0")"
: "${OUT:?OUT=dossier de sortie}"; mkdir -p "$OUT"; export OUT PY=${PYTHON:-python3}
export UNIV_ALTERE=0.3 UNIV_VERSO=0.12 UNIV_MULTI=0.15 UNIV_ROT=1 UNIV_NUM=0.6 UNIV_SERRE=0.4 UNIV_NUMEROS_CORRIGES=1 UNIV_REEL=1
P=$1; shift
for spec in "$@"; do
  IFS=: read pre a b <<< "$spec"
  for i in $(seq $a $b); do echo "$pre $i"; done
done | xargs -P $P -L 1 sh -c '
  pre=$0; i=$1; out=$OUT/$pre$i.npz; test -f $out && exit 0
  g=$((50000 + i)); [ $pre = fv ] && g=59999
  nice $PY gen_univ.py train $out.tmp.npz 1000 $g && mv $out.tmp.npz $out'
