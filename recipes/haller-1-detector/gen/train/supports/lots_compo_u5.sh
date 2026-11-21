#!/bin/bash
# Lots « vrais supports + écriture simulée » de univ5 : composer.py lot, 1 000 pages par lot.
#   s<i>  : papiers vierges (annotations avec traits.py), graine 70000 + i
#   k<i>  : variantes de cartes officielles (cartes.py varier), graine 71000 + i
#   kb<i> : photos des cartes officielles (jeu train), graine 72000 + i
#   OUT=DOSSIER IN=ENTREES bash lots_compo_u5.sh PROCESSUS SPEC...   SPEC = prefixe:debut:fin ; sortie $OUT/<prefixe><i>.npz
#   ENTREES : supports/annot, supports/travail, cartes/annot, cartes/travail, variantes/annot, variantes/base (README).
#   PYTHON : interpréteur avec numpy, Pillow, scipy et opencv (python3 par défaut).
cd "$(dirname "$0")"
: "${OUT:?OUT=dossier de sortie}" "${IN:?IN=dossier des entrées}"; mkdir -p "$OUT"; export OUT IN PY=${PYTHON:-python3}
P=$1; shift
for spec in "$@"; do
  IFS=: read pre a b <<< "$spec"
  for i in $(seq $a $b); do echo "$pre $i"; done
done | xargs -P $P -L 1 sh -c '
  pre=$0; i=$1; out=$OUT/$pre$i.npz; test -f $out && exit 0
  case $pre in
    s)  A=$IN/supports/annot; I=$IN/supports/travail; g=$((70000 + i));;
    k)  A=$IN/variantes/annot; I=$IN/variantes/base; g=$((71000 + i));;
    kb) A=$IN/cartes/annot; I=$IN/cartes/travail; g=$((72000 + i));;
  esac
  nice $PY composer.py lot $A $I $out.tmp.npz 1000 $g && mv $out.tmp.npz $out'
