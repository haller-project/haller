#!/bin/bash
# Entraînement du détecteur du 21/11/2026 (GPU NVIDIA, une RTX 3090 pour le modèle publié) : même recette qu'univ5
# (réseau base, 10 passes, lot de 24, --fast, fenêtres GNHK d'entraînement répétées GNHK_FOIS
# fois, validation fv0, u40, GNHK test et premier lot, sans effet sur les poids), graine GRAINE (2 pour le modèle
# publié). Le garde-fou de variété (../variete/garde.py) refuse tout lot non audité ou de configuration non validée.
# Puis quantification simple sur fv0 (non retenue : le modèle publié est quantifié par canal, voir le README).
#   LOTS=DOSSIER GNHK=DOSSIER U40=FICHIER RES=DOSSIER HALLER=RACINE_DU_DEPOT PYTHON=... GRAINE=2 \
#     bash entrainer_univ6.sh NOM EPOQUES LOT...      -> $RES/NOM.npz, NOM_q.npz, NOM.log
set -e
NOM=$1; EP=$2; shift 2
L=${LOTS:?}; R=${RES:?}; G=${GNHK:?}; U40=${U40:?}; PY=${PYTHON:-python3}
mkdir -p $R
SH=$(for l in "$@"; do echo $L/$l.npz; done)
python3 "$(dirname "$(readlink -f "$0")")/../variete/garde.py" $SH || exit 3
GN=$(for i in $(seq ${GNHK_FOIS:-4}); do echo $G/gnhk_train.npz; done)
cd ${HALLER:?}
nice $PY -m haller.train_detector $R/$NOM.npz $SH $GN --arch base --epochs $EP --seed ${GRAINE:-1} \
  --shards-in-memory 1 --fast --val $L/fv0.npz $U40 $G/gnhk_test.npz $L/$1.npz > $R/$NOM.log 2>&1
nice $PY -m haller.detector $R/$NOM.npz $L/fv0.npz $R/${NOM}_q.npz --count 400 >> $R/$NOM.log 2>&1
echo fini >> $R/$NOM.log
