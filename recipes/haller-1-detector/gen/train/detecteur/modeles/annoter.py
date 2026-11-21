"""Cases des modèles de cartes du commerce (data/modeles_cartes) : à partir d'un relevé grossier par modèle
(colonne de numéros, plage verticale, plage horizontale de la case), trouve la boîte d'encre de chaque numéro
imprimé et le rectangle d'écriture de chaque case ; écrit cases.json (lu par gen_univ.carte_modele).

    python3 -I modeles/annoter.py DOSSIER_MODELES [--controle IMAGE.jpg] [--plus]
Bloc : (premier numéro, nombre, x0 et x1 des numéros, y0 et y1 de la colonne, x0 et x1 de la case).
base : « ligne » (on écrit sur une ligne au niveau du bas du numéro) ou « case » (case centrée sur le numéro).
split : « test » = modèle tenu à l'écart de l'entraînement.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

RELEVES = {
    "seedsigner_mnemonic_12w": dict(split="train", base="ligne", carte=[90, 20, 1188, 1647],
                                    blocs=[(1, 12, 313, 396, 280, 1403, 400, 950)]),
    "seedsigner_mnemonic_24w": dict(split="train", base="ligne", carte=[86, 20, 1188, 1634],
                                    blocs=[(1, 12, 201, 266, 272, 1394, 272, 591), (13, 12, 668, 740, 272, 1394, 746, 1064)]),
    "trezor_seed24_inside": dict(split="train", base="case", carte=[56, 56, 1185, 492],
                                 blocs=[(1, 12, 445, 480, 72, 481, 490, 788), (13, 12, 823, 860, 72, 481, 868, 1166)]),
    "coldcard_backup_v3": dict(split="train", base="case", carte=[0, 0, 300, 525],
                               blocs=[(1, 12, 14, 27, 212, 508, 28, 143), (13, 12, 150, 168, 212, 508, 169, 283)]),
    "coldcard_backup_v2": dict(split="train", base="case", carte=[0, 0, 300, 525],
                               blocs=[(1, 12, 14, 27, 212, 508, 28, 143), (13, 12, 150, 168, 212, 508, 169, 283)]),
    "bitcoinkeys_worksheet24": dict(split="train", base="case", carte=[0, 0, 1275, 1650],
                                    blocs=[(1, 24, 95, 132, 298, 1425, 840, 1223)]),
    "coldcard_microsd_password": dict(split="train", base="case", carte=[0, 0, 225, 375],
                                      blocs=[(1, 12, 5, 25, 38, 364, 27, 212)]),
    "trezor_legacy_recovery_card": dict(split="test", base="case", carte=[520, 150, 1140, 560],
                                        blocs=[(1, 12, 555, 588, 172, 538, 591, 816), (13, 12, 850, 884, 172, 538, 886, 1111)]),
    "foundation_passport_card": dict(split="test", base="ligne", carte=[196, 684, 993, 1049],
                                     blocs=[(1, 6, 206, 230, 755, 968, 232, 340), (7, 6, 347, 370, 755, 968, 372, 480),
                                            (13, 6, 487, 512, 755, 968, 514, 620), (19, 6, 626, 652, 755, 968, 654, 760)]),
    "trezor_seed20_inside": dict(split="test", base="case", carte=[56, 53, 417, 1126],
                                 blocs=[(1, 20, 75, 111, 115, 1117, 122, 397)]),
    "keybackup_seedwords": dict(split="test", base="case", carte=[0, 0, 1816, 2474],
                                blocs=[(1, 12, 160, 203, 480, 1500, 205, 762), (13, 12, 945, 985, 480, 1500, 1000, 1557)]),
    # Moitié gauche (24 mots). Le modèle imprime « 21 » deux fois et pas de « 22 » : la case en 22e position
    # porte le numéro 22 (ordre des mots), son numéro imprimé se lit « 21 ».
    "bitcoinerguide_recovery_sheet": dict(split="train", base="ligne", carte=[10, 10, 875, 1050],
                                          blocs=[(1, 12, 40, 85, 215, 875, 90, 391),
                                                 (13, 12, 470, 515, 215, 875, 520, 822)]),
    # Filigrane gris (195) sous les numéros 5-6 : sous le seuil d'encre, d'où une boîte serrée par numéro.
    "mnemonicbackup_12w_btc": dict(split="train", base="case", carte=[83, 140, 835, 745],
                                   blocs=[(1, 4, 115, 150, 320, 483, 157, 414), (5, 1, 126, 142, 494, 514, 157, 414),
                                          (6, 1, 126, 143, 534, 556, 157, 414), (7, 6, 480, 535, 320, 565, 542, 799)]),
    # Filigrane sous les numéros 8-12 : boîte serrée par numéro.
    "mnemonicbackup_24w_btc": dict(split="train", base="case", carte=[83, 140, 835, 745],
                                   blocs=[(1, 7, 105, 160, 200, 483, 167, 426), (8, 1, 136, 152, 493, 514, 167, 426),
                                          (9, 1, 135, 152, 534, 555, 167, 426), (10, 1, 115, 152, 575, 596, 167, 426),
                                          (11, 1, 116, 152, 616, 637, 167, 426), (12, 1, 117, 153, 657, 677, 167, 426),
                                          (13, 12, 480, 535, 200, 690, 541, 801)]),
    "mnemonicbackup_shamir33": dict(split="train", base="case", carte=[83, 140, 835, 745],
                                    blocs=[(1, 16, 125, 175, 180, 684, 179, 407), (17, 17, 500, 550, 170, 712, 555, 783)]),
    # Numérotation par lignes (1 2 / 3 4 …) : un bloc d'un numéro par case, hauteur de zone = pas des lignes.
    "padawans_seed_backup": dict(split="train", base="ligne", carte=[70, 70, 873, 1168],
                                 blocs=[(2 * r + c + 1, 1, *nx, int(409 + 48.4 * r) - 24, int(409 + 48.4 * r) + 24, *sx)
                                        for r in range(6) for c, (nx, sx) in enumerate([((95, 135), (136, 453)),
                                                                                       ((485, 527), (527, 842))])]),
    # Premier tableau (« Write the secret down here ») seulement ; 4 colonnes numérotées par lignes.
    "psst_worksheet_secret": dict(split="test", base="ligne", carte=[0, 0, 1242, 1755],
                                  blocs=[(4 * r + c + 1, 1, *nx, int(347.5 + 37.2 * r) - 18, int(347.5 + 37.2 * r) + 18, *sx)
                                         for r in range(6) for c, (nx, sx) in enumerate([((50, 82), (92, 254)),
                                                                                        ((265, 297), (308, 484)),
                                                                                        ((495, 527), (538, 701)),
                                                                                        ((712, 744), (755, 917))])]),
    "psst_worksheet_share": dict(split="train", base="ligne", carte=[0, 0, 1242, 1757],
                                 blocs=[(2 * r + c + 1, 1, *nx, int(1240 + 39.45 * r) - 19, int(1240 + 39.45 * r) + 19, *sx)
                                        for r in range(12) for c, (nx, sx) in enumerate([((336, 376), (392, 603)),
                                                                                        ((634, 674), (690, 901))])]),
    # Colonne « Seed Phrase » seulement ; on écrit le mot sur la ligne pointillée segmentée.
    "penlock_worksheet24_front": dict(split="train", base="ligne", carte=[291, 50, 642, 1697],
                                      blocs=[(1, 24, 310, 365, 210, 1555, 372, 610)]),
    # Cases arrondies : numéro en haut à gauche, pointillé 33 px sous son centre ; zone de 84 px pour que le bas
    # de la case (centre + 0,4 × 84) tombe sur le pointillé.
    "rapidtoolset_18w_grid": dict(split="train", base="case", carte=[0, 0, 1800, 1338],
                                  blocs=[(3 * r + c + 1, 1, *nx, int(318.5 + 162 * r) - 42, int(318.5 + 162 * r) + 42, *sx)
                                         for r in range(6) for c, (nx, sx) in enumerate([((100, 200), (216, 573)),
                                                                                        ((655, 755), (771, 1125)),
                                                                                        ((1210, 1310), (1323, 1680))])]),
    "rapidtoolset_24w_list": dict(split="train", base="case", carte=[0, 0, 1800, 4254],
                                  blocs=[(r + 1, 1, 100, 210, int(318.5 + 162 * r) - 42, int(318.5 + 162 * r) + 42, 216, 1680)
                                         for r in range(24)]),
    # Épreuve InDesign : carte = traits de coupe intérieurs. 4 colonnes de 6 cases grises, numéro à gauche de la case.
    "trezor_one_card_2017": dict(split="train", base="case", carte=[339, 428, 2039, 1411],
                                 blocs=[(1, 6, 360, 395, 700, 1345, 401, 737), (7, 6, 745, 790, 700, 1345, 795, 1130),
                                        (13, 6, 1205, 1246, 700, 1345, 1253, 1588),
                                        (19, 6, 1595, 1640, 700, 1345, 1645, 1980)]),
    # Les modèles suivants ont le trait d'écriture sous le numéro (et non au niveau de son bas) et ce trait passe sous
    # la colonne des numéros : un bloc par numéro, zone = de la ligne précédente (exclue) à la ligne du mot (exclue),
    # base « case » pour que le bas de la case (centre + 0,4 × zone) tombe juste au-dessus du trait.
    # Basse résolution (numéros de 10 px de haut).
    "ledger_nanos_my_recovery_phrase": dict(split="train", base="case", carte=[10, 9, 379, 689],
                                            blocs=[(r + 1 + 12 * c, 1, *nx, 251 + 36 * r - 34, 251 + 36 * r, *sx)
                                                   for c, (nx, sx) in enumerate([((36, 62), (62, 178)),
                                                                                 ((205, 232), (232, 348))])
                                                   for r in range(12)]),
    # Page de droite (24 mots) seulement. Très basse résolution (numéros de 7 à 8 px de haut).
    "ledger_nanox_recovery_sheet": dict(split="train", base="case", carte=[368, 17, 674, 534],
                                        blocs=[(r + 1 + 12 * c, 1, *nx, ly - 25, ly, *sx)
                                               for c, (nx, sx) in enumerate([((390, 412), (413, 504)),
                                                                             ((532, 553), (555, 646))])
                                               for r, ly in enumerate([149, 177, 205, 233, 261, 290, 318, 346,
                                                                       374, 402, 430, 459])]),
    "onekey_recovery_sheet": dict(split="test", base="case", carte=[0, 0, 1679, 704],
                                  blocs=[(r + 1 + 6 * c, 1, *nx, ly - 78, ly, *sx)
                                         for c, (nx, sx) in enumerate([((50, 115), (115, 398)), ((435, 500), (500, 783)),
                                                                       ((890, 955), (955, 1240)),
                                                                       ((1280, 1345), (1345, 1626))])
                                         for r, ly in enumerate([250, 330, 410, 490, 569, 649])]),
}

# Essais comparables (20/11) : modèles ajoutés seulement avec --plus (dossier modeles_cartes_plus), pour mesurer
# l'effet de modèles supplémentaires avant de les verser dans l'entraînement.
RELEVES_PLUS = {
    "bitcoinkeys_worksheet12": dict(split="train", base="case", carte=[0, 0, 1275, 1650],
                                    blocs=[(1, 12, 95, 132, 285, 855, 843, 1225)]),
    "coldcard_ultimate_recovery": dict(split="train", base="case", carte=[0, 0, 500, 440],
                                       blocs=[(1, 12, 16, 36, 96, 434, 37, 126), (13, 12, 130, 154, 96, 434, 155, 249)]),
}


def runs(profile, gap=3):
    on = np.nonzero(profile)[0]
    if len(on) == 0:
        return []
    out, s, p = [], on[0], on[0]
    for v in on[1:]:
        if v - p > gap:
            out.append((s, p + 1))
            s = v
        p = v
    out.append((s, p + 1))
    return out


def bloc(a, first, count, nx0, nx1, y0, y1, sx0, sx1, base):
    zone = a[y0:y1, nx0:nx1]
    ink = zone < min(200, np.median(a) - 40)
    rs = [r for r in runs(ink.any(1)) if r[1] - r[0] >= 3]
    if len(rs) != count:  # partage régulier si les numéros ne se séparent pas
        edges = np.linspace(0, y1 - y0, count + 1)
        rs = []
        for i in range(count):
            e0, e1 = int(edges[i]), int(edges[i + 1])
            rr = runs(ink[e0:e1].any(1))
            rs.append((e0 + rr[0][0], e0 + rr[-1][1]) if rr else (e0, e1))
    centers = [(r[0] + r[1]) / 2 + y0 for r in rs]
    pitch = float(np.median(np.diff(centers))) if count > 1 else (y1 - y0)
    numeros, cases = [], []
    for i, (r0, r1) in enumerate(rs):
        cols = np.nonzero(ink[r0:r1].any(0))[0]
        bx0, bx1 = nx0 + cols.min(), nx0 + cols.max() + 1
        by0, by1 = y0 + r0, y0 + r1
        numeros.append([int(bx0), int(by0), int(bx1), int(by1), first + i])
        cy = (by0 + by1) / 2
        if base == "ligne":
            bot = by1 + 0.06 * pitch
        else:
            bot = cy + 0.4 * pitch
        cases.append([int(sx0), int(bot - 0.85 * pitch), int(sx1), int(bot), first + i])
    return numeros, cases


def main():
    d = sys.argv[1]
    out = []
    releves = dict(RELEVES, **RELEVES_PLUS) if "--plus" in sys.argv else RELEVES
    for mid, r in releves.items():
        p = os.path.join("png", mid + ".png")
        a = np.asarray(Image.open(os.path.join(d, p)).convert("L"), np.float32)
        numeros, cases = [], []
        for b in r["blocs"]:
            n, c = bloc(a, *b, r["base"])
            numeros += n
            cases += c
        out.append(dict(id=mid, png=p, split=r["split"], carte=r["carte"], numeros=numeros, cases=cases))
    json.dump(out, open(os.path.join(d, "cases.json"), "w"), indent=1)
    if "--controle" in sys.argv:
        ims = []
        for m in out:
            im = Image.open(os.path.join(d, m["png"])).convert("RGB")
            dr = ImageDraw.Draw(im)
            for b in m["numeros"]:
                dr.rectangle(b[:4], outline=(255, 0, 0), width=2)
            for b in m["cases"]:
                dr.rectangle(b[:4], outline=(0, 160, 0), width=2)
            dr.rectangle(m["carte"], outline=(0, 0, 255), width=3)
            x0, y0, x1, y1 = m["carte"]
            im = im.crop((max(0, x0 - 10), max(0, y0 - 10), x1 + 10, y1 + 10))
            im.thumbnail((480, 480))
            ims.append(im)
        s = Image.new("RGB", (4 * 480, (len(ims) + 3) // 4 * 480), "gray")
        for i, im in enumerate(ims):
            s.paste(im, ((i % 4) * 480, (i // 4) * 480))
        s.save(sys.argv[sys.argv.index("--controle") + 1], quality=85)
    for m in out:
        print(m["id"], m["split"], len(m["numeros"]), len(m["cases"]))


if __name__ == "__main__":
    main()
