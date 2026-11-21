"""Capture « une ligne à la fois », sans grille imprimée.

Simulation : un mot BIP39 complet écrit en capitales (lettres EMNIST de test) sur
papier blanc, ligné ou quadrillé ; espacement variable (lettres parfois collées),
traits parfois interrompus, ligne inclinée, photo 720x480 dont seule la bande
du cadre affiché à l'écran est lue.
Lecture (numpy + Pillow) : bande -> binarisation locale -> redressement de
l'inclinaison -> découpe par projection verticale -> scission des blocs trop
larges -> classification -> mot du dictionnaire (alignement avec insertion /
suppression) -> seed (checksum).
Usage : python ligne.py <modele.npz> [nb_seeds] [--photos]
"""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from infer_np import conv3x3_relu_pool, mm
from seed_sim import MOTS, checksum_ok, mnemonique
from train import charger

BANDE = (40, 160, 680, 320)  # cadre affiché à l'écran, dans l'image 720x480


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def logits(p, x):
    x = (x.astype(np.float32) / 255.0)[:, None]
    for k in ("c1", "c2", "c3"):
        x = conv3x3_relu_pool(x, p[k + ".weight"], p[k + ".bias"])
    return mm(x.reshape(len(x), -1), p["fc.weight"].T) + p["fc.bias"]


# ---------- simulation ----------

def papier(rng, w=720, h=480):
    img = Image.new("L", (w, h), int(rng.integers(215, 250)))
    d = ImageDraw.Draw(img)
    genre = rng.choice(["blanc", "ligne", "carreaux"])
    if genre == "ligne":
        pas = int(rng.integers(28, 40)); y0 = int(rng.integers(0, pas))
        for y in range(y0, h, pas):
            d.line([(0, y), (w, y)], fill=175, width=1)
    elif genre == "carreaux":
        pas = int(rng.integers(18, 26))
        for y in range(int(rng.integers(0, pas)), h, pas):
            d.line([(0, y), (w, y)], fill=190, width=1)
        for x in range(int(rng.integers(0, pas)), w, pas):
            d.line([(x, 0), (x, h)], fill=190, width=1)
    return img, genre


def ecrire(img, mot, ex, par_lettre, rng):
    """Écrit le mot centré dans la bande ; renvoie le nombre de lettres posées."""
    haut = int(rng.integers(48, 78))
    lettres = []
    for ch in mot.upper():
        i = rng.choice(par_lettre[ord(ch) - 65])
        g = Image.fromarray(255 - ex[i])
        bb = Image.fromarray(ex[i]).getbbox()
        g = g.crop(bb)
        s = haut / g.height * rng.uniform(0.9, 1.08)
        g = g.resize((max(4, int(g.width * s)), max(4, int(g.height * s))), Image.BILINEAR)
        g = g.rotate(rng.uniform(-6, 6), expand=True, fillcolor=255)
        a = np.asarray(g).copy()
        if rng.random() < 0.15:  # trait interrompu
            r = int(rng.integers(a.shape[0] // 4, 3 * a.shape[0] // 4))
            a[r : r + 3] = 255
        lettres.append(a)
    # espacement : parfois négatif (lettres qui se touchent)
    esp = [int(rng.choice([-3, 0, 2, 4, 6, 8, 10, 14], p=[.04, .06, .12, .18, .2, .18, .12, .1])) for _ in lettres]
    larg = sum(l.shape[1] for l in lettres) + sum(esp[1:])
    bx0, by0, bx1, by1 = BANDE
    if larg > bx1 - bx0 - 10:
        f = (bx1 - bx0 - 10) / larg
        lettres = [np.asarray(Image.fromarray(l).resize((max(3, int(l.shape[1] * f)), max(3, int(l.shape[0] * f))))) for l in lettres]
        larg = sum(l.shape[1] for l in lettres) + sum(esp[1:])
    x = (bx0 + bx1) // 2 - larg // 2 + int(rng.integers(-20, 21))
    yb = (by0 + by1) // 2 + int(rng.integers(-15, 16))
    couche = Image.new("L", img.size, 255)
    for k, l in enumerate(lettres):
        if k:
            x += esp[k]
        y = yb - l.shape[0] // 2 + int(rng.integers(-4, 5))
        enc = Image.fromarray(l)
        couche.paste(Image.fromarray(np.minimum(np.asarray(couche.crop((x, y, x + l.shape[1], y + l.shape[0]))), l)), (x, y))
        x += l.shape[1]
    couche = couche.rotate(rng.uniform(-5, 5), center=((bx0 + bx1) / 2, yb), fillcolor=255)
    encre = 255 - np.asarray(couche, np.float32)
    fond = np.asarray(img, np.float32)
    c = rng.uniform(20, 60)
    return Image.fromarray(np.clip(fond - encre / 255 * (fond - c), 0, 255).astype(np.uint8))


def degrader_photo(img, rng, echelle=1):
    a = np.asarray(img, np.float32)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = a * (0.65 + 0.35 * (xx / w * rng.uniform(0, 1) + yy / h * rng.uniform(0, 0.6)))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(echelle * rng.uniform(0.6, 1.3)))
    a = np.asarray(out, np.float32) + rng.normal(0, 5, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------- lecture ----------

def moyenne_locale(a, r):
    """Moyenne sur une fenêtre (2r+1)^2. Calcul par le flou-boîte de Pillow (C),
    bien plus rapide qu'en numpy sur un petit processeur ; précision à l'unité
    de niveau de gris près, suffisante pour les seuils utilisés."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.BoxBlur(r)), np.float32)


def rayon_fond(a):
    """Rayon de la fenêtre de fond local : proportionnel à la hauteur de la bande
    (un tiers), au moins 15 px. Une fenêtre fixe trop petite ne voit plus que de
    l'encre au centre d'un trait épais ou flou (photo réelle agrandie)."""
    return max(15, a.shape[0] // RAYON_FRACTION)


def binariser(a):
    # Encre = nettement plus sombre que le voisinage. Un pixel isolé verticalement
    # ou horizontalement (trait d'un pixel : ligne de cahier, carreau) est retiré.
    m = a < 0.72 * moyenne_locale(a, rayon_fond(a))
    v = np.zeros_like(m); v[1:-1] = m[1:-1] & (m[:-2] | m[2:])
    h = np.zeros_like(m); h[:, 1:-1] = m[:, 1:-1] & (m[:, :-2] | m[:, 2:])
    return v & h


def effacer_lignes_imprimees(a):
    """Efface les lignes horizontales imprimées (soulignement, cahier) avant
    binarisation : une rangée dont l'assombrissement médian sur toute la largeur
    est net sur au moins 85 % de la largeur est une ligne ; le texte ne couvre jamais autant.
    Les rangées concernées sont remplacées par le fond local."""
    fond = moyenne_locale(a, rayon_fond(a))
    d = np.percentile(fond - a, 15, axis=1)  # sombre sur au moins 85 % de la largeur
    lig = d > 6
    lig = lig | np.roll(lig, 1) | np.roll(lig, -1)
    if lig.any():
        a = a.copy()
        a[lig] = np.maximum(a[lig], fond[lig] * 1.02)
    return a


def retirer_traits_longs(m, frac=0.25):
    """Retire les segments horizontaux d'encre plus longs que frac x largeur de la
    bande : traits imprimés, même légèrement inclinés (chaque rangée en contient
    un long morceau). Les barres de lettres (E, T, H) sont bien plus courtes."""
    lmin = max(40, int(frac * m.shape[1]))
    debut = m.copy()
    debut[:, 1:] &= ~m[:, :-1]
    ids = np.cumsum(debut.ravel()).reshape(m.shape) * m
    longueur = np.bincount(ids.ravel())
    longueur[0] = 0
    return m & ~(longueur >= lmin)[ids]


def nettoyer_lignes(m):
    # Supprime les lignes (rangées / colonnes presque pleines sur la bande)
    m = retirer_traits_longs(m)
    lig = m.mean(1) > 0.5
    m[lig] = False
    col = m.mean(0) > 0.6
    m[:, col] = False
    return m


def redresser(a, m):
    # Inclinaison estimée par la droite qui passe par le centre de l'encre de chaque colonne
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return a, m
    pente = np.polyfit(xs, ys, 1)[0]
    ang = np.degrees(np.arctan(pente))
    if abs(ang) < 0.4:  # inclinaison négligeable : pas de rotation ni de nouvelle binarisation
        return a, m
    im = Image.fromarray(a.astype(np.uint8)).rotate(ang, resample=Image.BILINEAR, fillcolor=int(np.median(a)))
    a2 = np.asarray(im, np.float32)
    return a2, nettoyer_lignes(binariser(a2))


def segmenter(m):
    # Rangées d'encre : garder la bande de texte principale
    # Bloc de rangées d'encre d'un seul tenant le plus chargé (les jambages de la
    # ligne du dessus ou les hampes de celle du dessous en sont exclus)
    prof = m.sum(1)
    actif = prof > max(2, 0.08 * prof.max())
    if not actif.any():
        return [], (0, 0)
    blocs, d = [], None
    for y, v in enumerate(np.append(actif, False)):
        if v and d is None:
            d = y
        elif not v and d is not None:
            if blocs and d - blocs[-1][1] <= 2:
                blocs[-1][1] = y
            else:
                blocs.append([d, y])
            d = None
    y0, y1 = max(blocs, key=lambda b: prof[b[0] : b[1]].sum())
    # Hampes, jambages et points peu chargés autour du bloc : étendre tant que l'encre continue
    while y0 > 0 and prof[y0 - 1] > 0:
        y0 -= 1
    while y1 < len(prof) and prof[y1] > 0:
        y1 += 1
    h = y1 - y0
    col = m[y0:y1].sum(0)
    encre = col > 0
    segs, x = [], 0
    while x < len(encre):
        if encre[x]:
            d = x
            while x < len(encre) and encre[x]:
                x += 1
            segs.append([d, x])
        x += 1
    # Fusionne seulement les fragments quasi jointifs (écart < 2 px) ; les autres
    # regroupements sont laissés au décodage (lire_ligne / mots_candidats)
    fus = []
    for s in segs:
        if fus and s[0] - fus[-1][1] < 2:
            fus[-1][1] = s[1]
        else:
            fus.append(s)
    fus = [s for s in fus if m[y0:y1, s[0]:s[1]].sum() > 0.01 * h * h]
    # Scinde les blocs trop larges (lettres collées) au creux de la projection
    larg = np.median([s[1] - s[0] for s in fus]) if fus else h
    ref = max(larg, 0.55 * h)
    out = []
    for d, f in fus:
        n = int(round((f - d) / ref))
        if (f - d) > 1.55 * ref and n >= 2:
            bornes = [d]
            for k in range(1, n):
                c = d + (f - d) * k // n
                fen = slice(max(d + 2, c - int(0.25 * ref)), min(f - 2, c + int(0.25 * ref)))
                cs = col[fen]
                bornes.append(fen.start + int(cs.argmin()) if len(cs) else c)
            bornes.append(f)
            out += [[bornes[k], bornes[k + 1]] for k in range(n)]
        else:
            out.append([d, f])
    return out, (y0, y1)


def vers_28(bloc):
    if bloc.sum() < 10:
        return None
    ys, xs = np.nonzero(bloc)
    e = bloc[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    s = max(e.shape)
    carre = np.zeros((s, s), np.float32)
    oy, ox = (s - e.shape[0]) // 2, (s - e.shape[1]) // 2
    carre[oy : oy + e.shape[0], ox : ox + e.shape[1]] = e
    g = Image.fromarray((carre * 255).astype(np.uint8)).resize((24, 24), Image.BILINEAR)
    out = np.zeros((28, 28), np.uint8)
    out[2:26, 2:26] = np.asarray(g)
    return out


DERNIERS = {}  # fragments et hauteur de ligne de la dernière lecture
RAYON_FRACTION = 2  # rayon du fond local = hauteur de bande / RAYON_FRACTION
STATS = {}  # nombre de groupes classés (mesure du coût)
LARGEUR_MAX_GROUPE = 2.5  # largeur maximale d'un regroupement de fragments (hauteurs de ligne)
KMAX = 3  # nombre maximal de fragments regroupés en une lettre
GAMMA = -3.26  # dictionnaire : score de chaque lettre relatif au hasard (+log 26), sinon les mots courts sont favorisés


def lire_ligne(p, T, photo, kmax=None):
    """Renvoie G : dict (i, k) -> log-probabilités (26) du groupe des fragments
    i .. i+k-1 pris comme une seule lettre, et le nombre de fragments."""
    kmax = kmax or KMAX
    LARGEUR.clear()
    a = effacer_lignes_imprimees(np.asarray(photo.crop(BANDE), np.float32))
    m = nettoyer_lignes(binariser(a))
    a, m = redresser(a, m)
    segs, (y0, y1) = segmenter(m)
    h = max(y1 - y0, 1)
    DERNIERS.update(segs=segs, h=h)
    fond = moyenne_locale(a, rayon_fond(a))
    encre = np.clip((fond - a) / np.maximum(fond * 0.6, 1), 0, 1) * m
    cles, xs = [], []
    for i in range(len(segs)):
        for k in range(1, kmax + 1):
            if i + k > len(segs):
                break
            d, f = segs[i][0], segs[i + k - 1][1]
            if k > 1 and f - d > LARGEUR_MAX_GROUPE * h:
                break
            x = vers_28(encre[y0:y1, d:f])
            if x is not None:
                cles.append((i, k)); xs.append(x)
                STATS["groupes"] = STATS.get("groupes", 0) + 1
                LARGEUR[(i, k)] = (f - d) / h
    G = {}
    LARGEUR_REF[0] = np.median([(f - d) / h for d, f in segs]) if segs else 1.0
    if xs:
        pr = softmax(logits(p, np.stack(xs)) / T)
        if pr.shape[1] == 52:  # modèle majuscules + minuscules : casse fusionnée
            pr = pr[:, :26] + pr[:, 26:]
        lp = np.log(pr + 1e-9)
        G = dict(zip(cles, lp))
    return G, len(segs)


LARGEUR, LARGEUR_REF = {}, [1.0]  # largeur des groupes (en hauteurs de ligne), dernière lecture
LAMBDA_LARGEUR = 6  # dictionnaire : pénalité par hauteur de ligne au-delà de LARGEUR_LETTRE
LARGEUR_LETTRE = 0.8
LARGEUR_MAX_BRUT = 1.6  # texte libre : largeur maximale d'un groupe de plusieurs fragments


def brut(G, n):
    # Meilleur découpage sans dictionnaire (texte libre) : Viterbi sur les fragments
    best = [(-1e9, "")] * (n + 1)
    best[0] = (0.0, "")
    for i in range(n):
        if best[i][0] <= -1e9:
            continue
        for k in range(1, KMAX + 1):
            if (i, k) in G and (k == 1 or LARGEUR[(i, k)] <= LARGEUR_MAX_BRUT):
                lp = G[(i, k)]
                c = int(lp.argmax())
                v = best[i][0] + lp[c]
                if v > best[i + k][0]:
                    best[i + k] = (v, best[i][1] + chr(97 + c))
    return best[n][1]


PAR_LONGUEUR = {}
for _w in MOTS:
    PAR_LONGUEUR.setdefault(len(_w), []).append(_w)
IDX_LONGUEUR = {L: np.array([[ord(c) - 97 for c in w] for w in ws]) for L, ws in PAR_LONGUEUR.items()}


def mots_candidats(G, n):
    """Score de chaque mot du dictionnaire : meilleur découpage des n fragments
    en len(mot) groupes consécutifs, vectorisé sur tous les mots d'une longueur."""
    cands, scores = [], []
    for L, idx in IDX_LONGUEUR.items():
        if L > n or L * KMAX < n:
            continue
        D = np.full((n + 1, L + 1, len(idx)), -1e9)
        D[0, 0] = 0
        for i in range(1, n + 1):
            for k in range(1, KMAX + 1):
                g = G.get((i - k, k))
                if g is None:
                    continue
                # Un groupe plus large qu'une lettre (en hauteurs de ligne) est pénalisé
                g = g - LAMBDA_LARGEUR * max(0.0, LARGEUR.get((i - k, k), 0.0) - LARGEUR_LETTRE)
                # D[i, j] = max(D[i-k, j-1] + g[lettre j du mot])
                cand = D[i - k, :L] + g[idx.T] - GAMMA
                np.maximum(D[i, 1:], cand, out=D[i, 1:])
        cands += PAR_LONGUEUR[L]
        scores.append(D[n, L])
    if not cands:
        return list(MOTS), np.full(len(MOTS), -1e9)
    return cands, np.concatenate(scores)


def corriger_seed(liste):
    # liste : (candidats, scores) par mot ; checksum, puis substitution d'un mot si besoin
    base = [c[int(np.argmax(s))] for c, s in liste]
    if checksum_ok(base):
        return base, "direct"
    alts = []
    for k, (c, s) in enumerate(liste):
        o = np.argsort(-s)[1:4]
        alts += [(float(s.max() - s[i]), k, c[i]) for i in o]
    alts.sort()
    for cout, k, w in alts:
        e = base.copy(); e[k] = w
        if checksum_ok(e):
            return e, "corrigé"
    return base, "échec"


def distance(a, b):
    # Distance d'édition (Levenshtein)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def main():
    global GAMMA
    for arg in sys.argv:
        if arg.startswith("--gamma="):
            GAMMA = float(arg.split("=")[1])
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[1]).items()}
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
    photos = "--photos" in sys.argv
    if photos:
        import os
        os.makedirs("lignes_demo", exist_ok=True)
    ex, ey = charger(False)
    ex, ey = ex.numpy(), ey.numpy()
    par_lettre = {k: np.nonzero(ey == k)[0] for k in range(26)}
    # Température ajustée sur le test EMNIST (comme calib.py)
    z = np.concatenate([logits(p, ex[i : i + 1024]) for i in range(0, 8192, 1024)])
    Ts = np.linspace(0.4, 2.5, 43)
    T = Ts[np.argmin([-np.log(softmax(z / t)[np.arange(len(z)), ey[:8192]] + 1e-12).mean() for t in Ts])]
    rng = np.random.default_rng(21)
    st = dict(err_car=0, car=0, lignes=0, nb_ok=0, brut=0, top1=0, top3=0, seeds_ok=0, seeds_faux=0, temps=0.0)
    par_papier = {}
    for f in range(nseeds):
        mots = mnemonique(rng.bytes(32))
        liste = []
        for k, mot in enumerate(mots):
            img, genre = papier(rng)
            img = degrader_photo(ecrire(img, mot, ex, par_lettre, rng), rng)
            if photos and f == 0 and k < 8:
                img.save(f"lignes_demo/ligne_{k}_{mot}.png")
            t0 = time.perf_counter()
            G, nf = lire_ligne(p, T, img)
            cands, s = mots_candidats(G, nf)
            st["temps"] += time.perf_counter() - t0
            liste.append((cands, s))
            lu = brut(G, nf)
            top = [cands[i] for i in np.argsort(-s)[:3]]
            st["lignes"] += 1
            st["nb_ok"] += len(lu) == len(mot)
            st["brut"] += lu == mot
            st["err_car"] += distance(lu, mot); st["car"] += len(mot)
            st["top1"] += top[0] == mot
            st["top3"] += mot in top
            g = par_papier.setdefault(genre, [0, 0]); g[0] += 1; g[1] += top[0] == mot
        final, mode = corriger_seed(liste)
        st["seeds_ok"] += final == mots
        st["seeds_faux"] += (final != mots) and checksum_ok(final)
    n = st["lignes"]
    print(f"modèle T={T:.2f} ; {nseeds} seeds, {n} lignes")
    print(f"nombre de lettres trouvé exact : {st['nb_ok']/n*100:.1f} %")
    print(f"mot lu exact sans dictionnaire : {st['brut']/n*100:.1f} % ; erreurs par lettre : {st['err_car']/st['car']*100:.1f} %")
    print(f"mot juste (dictionnaire) : {st['top1']/n*100:.1f} % ; dans les 3 premiers : {st['top3']/n*100:.1f} %")
    print("par papier :", {k: f"{v[1]/v[0]*100:.1f} % ({v[0]})" for k, v in par_papier.items()})
    print(f"seeds exactes après checksum : {st['seeds_ok']}/{nseeds} ; valides mais fausses : {st['seeds_faux']}")
    print(f"temps de lecture moyen par ligne (x86) : {st['temps']/n*1000:.0f} ms")


if __name__ == "__main__":
    main()
