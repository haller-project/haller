"""Mesures de variété sur l'image finale (photo réelle ou page générée), identiques pour les deux.

blancs_mot(gris, quad) : redresse la boîte du mot (quadrilatère, coin haut gauche en premier) à 64 px de haut,
binarise (Otsu), retire les traits horizontaux longs (ligne imprimée sous le mot, carreaux), puis renvoie la liste des
blancs intérieurs entre colonnes d'encre, rapportés à la hauteur de la boîte d'encre. Une colonne porte de l'encre si
elle compte au moins 2 pixels noirs (à 64 px de haut), pour ne pas compter le bruit isolé comme une lettre.
"""
import math

import cv2
import numpy as np

H_NORM = 64


def redresser(gris, quad):
    q = np.asarray(quad, np.float32)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if h < 4 or w < 4:
        return None
    if h > w * 1.3:   # mot vertical : le grand côté porte le texte
        q = np.roll(q, -1, 0)
        w, h = h, w
    s = H_NORM / h
    W2 = max(8, int(round(w * s)))
    dst = np.float32([[0, 0], [W2 - 1, 0], [W2 - 1, H_NORM - 1], [0, H_NORM - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(gris, M, (W2, H_NORM), flags=cv2.INTER_AREA, borderValue=255)


def encre_mot(crop):
    """Masque d'encre binaire d'un mot redressé, sans traits horizontaux longs."""
    c = cv2.GaussianBlur(crop, (3, 3), 0)
    _, b = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if b.mean() > 0.6:      # mot très pâle ou boîte sans fond : mesure impossible
        return None
    lon = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, crop.shape[1] // 3), 1)))
    return (b & (1 - lon)).astype(np.uint8)


def blancs_mot(gris, quad):
    crop = redresser(gris, quad)
    if crop is None:
        return None
    b = encre_mot(crop)
    if b is None:
        return None
    ys = np.nonzero(b.sum(1) >= 2)[0]
    if len(ys) < 8:
        return None
    hb = ys[-1] - ys[0] + 1
    col = b.sum(0) >= 2
    xs = np.nonzero(col)[0]
    if len(xs) < 4:
        return None
    c = col[xs[0]:xs[-1] + 1]
    runs, n = [], 0
    for v in c:
        if v:
            if n:
                runs.append(n)
            n = 0
        else:
            n += 1
    return [r / hb for r in runs]


def max_blanc(gris, quad):
    b = blancs_mot(gris, quad)
    if b is None:
        return None
    return max(b) if b else 0.0
