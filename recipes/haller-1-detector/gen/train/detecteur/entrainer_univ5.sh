#!/bin/bash
# Entraînement de univ5 (RTX 3060, 12 Go ; mémoire de l'hôte plafonnée à 7 Go) : réseau base, lots f* (gen_univ),
# s* (papiers réels), k* et kb* (cartes officielles), plus les fenêtres GNHK d'entraînement répétées GNHK_FOIS fois,
# graine 1, lot de 24, TRAIN_OPTS=--fast pour univ5. Validation : fv0, u40, GNHK test et le premier lot de la liste
# (sans effet sur les poids). Puis quantification simple sur fv0 (non retenue pour univ5 : voir quantifier.py --photo).
#   LOTS=DOSSIER GNHK=DOSSIER U40=FICHIER RES=DOSSIER HALLER=RACINE_DU_DEPOT PYTHON=... \
#     TRAIN_OPTS=--fast GNHK_FOIS=4 bash entrainer_univ5_lee.sh NOM EPOQUES LOT...   -> $RES/NOM.npz, NOM_q.npz, NOM.log
set -e
NOM=$1; EP=$2; shift 2
L=$LOTS; G=$GNHK; PY=${PYTHON:-python3}
mkdir -p $RES
SH=$(for l in "$@"; do echo $L/$l.npz; done)
GN=$(for i in $(seq ${GNHK_FOIS:-4}); do echo $G/gnhk_train.npz; done)
cd $HALLER
nice $PY -m haller.train_detector \
  $RES/$NOM.npz $SH $GN --arch base --epochs $EP --seed 1 --shards-in-memory 1 ${TRAIN_OPTS:-} \
  --val $L/fv0.npz $U40 $G/gnhk_test.npz $L/$1.npz > $RES/$NOM.log 2>&1
nice $PY -m haller.detector $RES/$NOM.npz $L/fv0.npz $RES/${NOM}_q.npz --count 400 >> $RES/$NOM.log 2>&1
echo fini >> $RES/$NOM.log
