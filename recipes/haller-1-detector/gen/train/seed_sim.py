"""Simulation « seed » : grilles remplies avec les 4 premières lettres de 24 mots BIP39.

Compare la lecture brute (lettre la plus probable) et la lecture contrainte par
le dictionnaire (mot qui maximise la somme des log-probabilités de ses lettres ;
case vide pour les mots de 3 lettres).
Usage : python seed_sim.py <modele.npz> [nb_feuilles]
"""
import hashlib
import itertools
import sys

import numpy as np
from PIL import Image, ImageDraw

import make_sheet as ms
import read_sheet as rs
from infer_np import conv3x3_relu_pool, mm
from train import charger

MOTS = [m.strip() for m in open("bip39_english.txt")]


def mnemonique(entropie: bytes):
    # BIP39 : entropie + premiers bits de SHA-256, découpés en mots de 11 bits
    n = len(entropie) * 8
    bits = int.from_bytes(entropie, "big") << (n // 32)
    bits |= hashlib.sha256(entropie).digest()[0] >> (8 - n // 32)
    nmots = (n + n // 32) // 11
    return [MOTS[(bits >> (11 * (nmots - 1 - i))) & 2047] for i in range(nmots)]


def checksum_ok(mots):
    idx = [MOTS.index(m) for m in mots]
    nb = len(idx) * 11
    cs = nb // 33
    v = 0
    for i in idx:
        v = (v << 11) | i
    ent = (v >> cs).to_bytes((nb - cs) // 8, "big")
    return (v & ((1 << cs) - 1)) == hashlib.sha256(ent).digest()[0] >> (8 - cs)


def corriger(scores):
    # scores : liste de 24 vecteurs (2048). Meilleur choix, puis si le checksum échoue,
    # substitution d'un ou deux mots par leurs suivants, au moindre coût de score.
    ordre = [np.argsort(-s)[:4] for s in scores]
    base = [int(o[0]) for o in ordre]
    if checksum_ok([MOTS[i] for i in base]):
        return base, "direct"
    cands = []
    for k in range(len(base)):
        for alt in ordre[k][1:]:
            cands.append((float(scores[k][base[k]] - scores[k][alt]), k, int(alt)))
    cands.sort()
    meilleur = None
    for cout, k, alt in cands:
        essai = base.copy(); essai[k] = alt
        if checksum_ok([MOTS[i] for i in essai]):
            meilleur = (cout, essai)
            break
    for (c1, k1, a1), (c2, k2, a2) in itertools.combinations(cands[:40], 2):
        if k1 == k2 or (meilleur and c1 + c2 >= meilleur[0]):
            continue
        essai = base.copy(); essai[k1] = a1; essai[k2] = a2
        if checksum_ok([MOTS[i] for i in essai]):
            meilleur = (c1 + c2, essai)
    return (meilleur[1], "corrigé") if meilleur else (base, "échec")


def logits(p, x):
    x = (x.astype(np.float32) / 255.0)[:, None]
    for k in ("c1", "c2", "c3"):
        x = conv3x3_relu_pool(x, p[k + ".weight"], p[k + ".bias"])
    z = mm(x.reshape(len(x), -1), p["fc.weight"].T) + p["fc.bias"]
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def feuille_mots(x, y, mots, rng):
    par_lettre = {k: np.nonzero(y == k)[0] for k in range(26)}
    W, H = ms.COLS * ms.CELL + 2 * ms.MARGE, ms.ROWS * ms.CELL + 2 * ms.MARGE
    img = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(img)
    for cx, cy in [(0, 0), (W - ms.REP, 0), (0, H - ms.REP), (W - ms.REP, H - ms.REP)]:
        d.rectangle([cx + 6, cy + 6, cx + ms.REP - 6, cy + ms.REP - 6], fill=0)
    for n, mot in enumerate(mots):
        r, c0 = n % 12, (n // 12) * 4
        for j in range(4):
            x0, y0 = ms.MARGE + (c0 + j) * ms.CELL, ms.MARGE + r * ms.CELL
            d.rectangle([x0, y0, x0 + ms.CELL, y0 + ms.CELL], outline=150, width=2)
            if j >= len(mot):
                continue
            i = rng.choice(par_lettre[ord(mot[j].upper()) - 65])
            g = Image.fromarray(255 - x[i]).resize((48, 48), Image.BILINEAR).rotate(rng.uniform(-8, 8), fillcolor=255)
            ox, oy = rng.integers(-4, 5, 2)
            box = (int(x0 + 8 + ox), int(y0 + 8 + oy))
            fond = img.crop((box[0], box[1], box[0] + 48, box[1] + 48))
            img.paste(Image.composite(Image.new("L", g.size, 30), fond, Image.fromarray(255 - np.array(g))), box)
    return img


def main():
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[1]).items()}
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    x, y = charger(False)
    x, y = x.numpy(), y.numpy()
    rng = np.random.default_rng(11)
    pref = [m[:4] for m in MOTS]
    idx = np.array([[ord(c) - 97 for c in m.ljust(4, "{")] for m in pref])  # '{' -> 26 = vide
    bilan = {}
    mots_ok_brut = mots_ok_dico = seeds_ok = lettres_ok = lettres = 0
    for f in range(n):
        mots = mnemonique(rng.bytes(32))
        assert checksum_ok(mots)
        scores_seed = []
        img = feuille_mots(x, y, mots, rng)
        ph, _ = ms.photo(img, rng)
        ph.save("/dev/shm/seed_sim.png")
        a = np.asarray(ph.convert("L"), np.float32)
        rep = rs.reperes(a)
        cw, ch, h2 = rs.SH, rs.SW, rs.REP / 2
        coef = rs.homographie(rep, [(h2, h2), (cw - h2, h2), (cw - h2, ch - h2), (h2, ch - h2)])
        fe = np.asarray(ph.transform((cw, ch), Image.PERSPECTIVE, coef, Image.BILINEAR).rotate(-90, expand=True))
        ok_seed = True
        for k, mot in enumerate(mots):
            r, c0 = k % 12, (k // 12) * 4
            cases = []
            for j in range(4):
                x0, y0 = rs.MARGE + (c0 + j) * rs.CELL, rs.MARGE + r * rs.CELL
                cases.append(rs.vers_emnist(fe[y0 + 9 : y0 + rs.CELL - 9, x0 + 9 : x0 + rs.CELL - 9]))
            vide = [c is None for c in cases]
            lp = logits(p, np.stack([c if c is not None else np.zeros((28, 28), np.uint8) for c in cases]))
            brut = "".join("" if v else chr(97 + int(l.argmax())) for v, l in zip(vide, lp))
            lettres += len(mot[:4])
            lettres_ok += sum(a == b for a, b in zip(brut, mot[:4]))
            # Score de chaque mot du dictionnaire ; case vide imposée si mot de 3 lettres
            lp5 = np.concatenate([lp, np.full((4, 1), -50.0)], 1)
            for j, v in enumerate(vide):
                if v:
                    lp5[j, :26] = -50.0
                    lp5[j, 26] = 0.0
                else:
                    lp5[j, 26] = -50.0
            score = lp5[np.arange(4)[None, :], idx].sum(1)
            choix = MOTS[int(score.argmax())]
            scores_seed.append(score)
            mots_ok_brut += brut == mot[:4]
            mots_ok_dico += choix == mot
            ok_seed &= choix == mot
        seeds_ok += ok_seed
        res, mode = corriger(scores_seed)
        juste = [MOTS[i] for i in res] == mots
        bilan[(mode, juste)] = bilan.get((mode, juste), 0) + 1
    nm = 24 * n
    print(f"lettres justes (brut) : {lettres_ok}/{lettres} = {lettres_ok/lettres*100:.2f} %")
    print(f"mots justes, brut : {mots_ok_brut}/{nm} = {mots_ok_brut/nm*100:.2f} %")
    print(f"mots justes, avec dictionnaire : {mots_ok_dico}/{nm} = {mots_ok_dico/nm*100:.2f} %")
    print(f"seeds de 24 mots entièrement justes : {seeds_ok}/{n}")
    print("avec checksum (mode, juste) :", bilan)


if __name__ == "__main__":
    main()
