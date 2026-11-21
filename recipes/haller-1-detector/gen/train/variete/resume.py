"""Facteurs de variété déduits de la vérité d'une page générée (géométrie des mots, cartes, coupures), sans tirage.

Les définitions reprennent celles de train/detecteur/distributions/mesurer.py, pour que les valeurs du journal
soient comparables aux statistiques mesurées sur les photos réelles :
  h_mot          hauteur médiane des boîtes de mots / petit côté de l'image ;
  axe_deg        axe du texte modulo 180 (médiane circulaire du grand côté des boîtes de mots) ;
  inclinaison_deg  écart de l'axe au multiple de 90° le plus proche ;
  part_carte     aire du quadrilatère de la carte cible / aire de l'image ;
  perspective    rapport des côtés opposés du quadrilatère de carte, moins 1 (max des deux paires) ;
  colonnes       nombre médian de mots par ligne de texte.
Les blancs entre lettres (`noter_mot`) sont mesurés sur l'encre propre du mot, avant la photo :
  blanc_lettres_max  plus grand blanc intérieur du mot / hauteur de la boîte d'encre du mot (une valeur par mot).
"""
import math

import numpy as np

from . import journal as J


def noter_mot(m):
    """Plus grand blanc entre lettres du mot m (255 = papier), rapporté à la hauteur de sa boîte d'encre.

    Seuil d'encre relatif au trait le plus sombre du mot (mi-chemin entre ce trait et le papier) : un trait fin ou
    pâle compte comme de l'encre ; la pâleur est un facteur à part (encre_niveau, alterations)."""
    if not J.actif():
        return
    m = np.asarray(m)
    if m.size == 0:
        return
    noir = float(m.min())
    if noir > 245:
        return
    seuil = 255 - 0.5 * (255 - noir)
    ys = np.nonzero((m < seuil).any(1))[0]
    if len(ys) < 2:
        return
    h = ys[-1] - ys[0] + 1
    b = J.blancs_internes(m, h, seuil)
    J.ajouter("blanc_lettres_max", max(b) if b else 0.0)


def geometrie_mots(quads, cote):
    if len(quads) == 0:
        return {}
    qs = np.array(quads, float)
    e1, e2 = qs[:, 1] - qs[:, 0], qs[:, 3] - qs[:, 0]
    l1, l2 = np.hypot(*e1.T), np.hypot(*e2.T)
    lon = np.where((l1 >= l2)[:, None], e1, e2)
    h = np.minimum(l1, l2)
    ang = (np.degrees(np.arctan2(lon[:, 1], lon[:, 0])) + 90) % 180 - 90
    a2 = np.radians(2 * ang)
    axe = float(np.degrees(np.arctan2(np.median(np.sin(a2)), np.median(np.cos(a2)))) / 2)
    c = qs.mean(1)
    t = np.array([math.cos(math.radians(axe)), math.sin(math.radians(axe))])
    v = c @ np.array([-t[1], t[0]])
    o = np.argsort(v)
    hm = float(np.median(h))
    lignes, cur = [], [o[0]]
    for a, b in zip(o[:-1], o[1:]):
        if v[b] - v[a] > 0.6 * hm:
            lignes.append(cur)
            cur = []
        cur.append(b)
    lignes.append(cur)
    return dict(h_mot=hm / cote, axe_deg=axe, inclinaison_deg=abs(((axe + 45) % 90) - 45),
                colonnes=int(round(float(np.median([len(l) for l in lignes])))))


def tailles_relatives(quads):
    """Hauteur de chaque mot (petit côté de sa boîte) / hauteur médiane des mots de la page."""
    if len(quads) < 3:
        return []
    qs = np.array(quads, float)
    h = np.minimum(np.hypot(*(qs[:, 1] - qs[:, 0]).T), np.hypot(*(qs[:, 3] - qs[:, 0]).T))
    m = float(np.median(h))
    return [float(x / m) for x in h] if m > 0 else []


def aire_quad(q):
    q = np.asarray(q, float)
    return 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))


def perspective(q):
    q = np.asarray(q, float)
    c = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return float(max(max(c[0], c[2]) / max(min(c[0], c[2]), 1e-6), max(c[1], c[3]) / max(min(c[1], c[3]), 1e-6)) - 1)


def finir_page(v, frame):
    """Ajoute au journal de la page les facteurs déduits de la vérité v (image de taille `frame`) et le ferme."""
    if not J.actif():
        return None
    W, H = frame
    for k in ("style", "defaut", "format", "type", "support", "categorie_support", "encre", "n", "ambigu",
              "occultation", "cadre", "ordre", "modele"):
        if k in v and v[k] is not None and not isinstance(v[k], (list, dict)):
            J.noter(k, v[k])
    for k in ("orientation", "recadrage", "hauteur_rel", "colonnes_tirees", "lignes", "couleur", "source"):
        if k in v and isinstance(v[k], (int, float, str, bool)):
            J.noter(k, v[k])
    lu = v.get("lumiere")
    if isinstance(lu, dict):
        J.noter("lumiere_brut", bool(lu.get("brut")))
        for k in ("temperature", "exposition"):
            if k in lu:
                J.noter("lumiere_" + k, lu[k])
        J.noter("lumiere_ombre", bool(lu.get("ombre")))
        J.noter("lumiere_faible", bool(lu.get("faible")))
    ob = [o for o in v.get("objets") or [] if not o.get("autre")]
    mots = [o["quad"] for o in ob if o.get("classe") == "mot" and not o.get("cache")]
    J.noter("n_mots_vus", len(mots))
    g = geometrie_mots(mots, min(W, H))
    for r in tailles_relatives(mots):
        J.ajouter("taille_mot_rel", r)
    for k, x in g.items():
        J.noter(k, x)
    nm = max(1, sum(1 for o in ob if o.get("classe") == "mot"))
    J.noter("part_mots_partiels", sum(1 for o in ob if o.get("classe") == "mot" and o.get("partiel")) / nm)
    J.noter("part_mots_caches", sum(1 for o in ob if o.get("classe") == "mot" and o.get("cache")) / nm)
    alt = [o["alteration"] for o in ob if o.get("alteration")]
    J.noter("n_alterations", len(alt))
    for a in alt:
        J.ajouter("alterations", a)
    ne = sum(1 for o in ob if o.get("classe") == "numero" and not o.get("imprime"))
    J.noter("numeros_ecrits", bool(ne >= 0.5 * nm))
    for c in v.get("cartes") or []:
        if c.get("cible", True) and c.get("coins"):
            J.noter("part_carte", aire_quad(c["coins"]) / (W * H))
            J.noter("perspective", perspective(c["coins"]))
            break
    return J.fermer()
