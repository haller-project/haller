"""
Entraînement du détecteur pour les essais de la nuit du 20-21/11/2026 : boucle de haller.train_detector
(ext/haller-det, branche detector) avec des options supplémentaires, une hypothèse par option.

    python train_nuit.py OUT.npz SHARD.npz... --val VAL.npz... [options]

  --aug base|photo     augmentations : base (haller : contraste, luminosité, bruit) ou photo (calibrées sur les
                       photos réelles du DEV, nuit/stats_photo.py : flou, contraste bas, gamma, éclairage en
                       pente, bruit faible)
  --kd POIDS.npz       distillation : sorties d'un enseignant (poids exportés, topologie lue dans le fichier) ;
  --kd-w W             poids de la perte de distillation (BCE sur les probabilités des canaux mots et numéros)
  --ohem R             pertes des mots et numéros : BCE sur les positifs + les R x plus dures des négatives
  --ema D              moyenne exponentielle des poids (exportée à la place des poids courants)
  --epochs, --lr, --batch, --seed, --arch, --init : comme haller.train_detector
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# HALLER_DET : racine du dépôt haller (paquet haller) ; par défaut, celle qui contient ce code figé
sys.path.insert(0, os.environ.get("HALLER_DET", os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6)))
from haller import detector as D  # noqa: E402
from haller.train_detector import UNKNOWN, evaluate, initialize, load, losses  # noqa: E402


def archs():
    a = dict(D.ARCHS)
    L = D.LAYERS
    # bas de l'encodeur élargi (1/16 et 1/32 à 96 canaux) : peu de MAC, plus de contexte
    a["wide_low"] = tuple((n, i, 96 if n in ("e5", "e6", "e7", "e8") else c, p) for n, i, c, p in L)
    # deux convolutions de plus au niveau le plus bas (1/32, 9 x 12 : un niveau 1/64 ne tombe pas juste) : champ
    # réceptif plus large pour peu de MAC (Doc-UFCN, Boillet et al. 2020 : le champ réceptif compte plus que la largeur)
    a["deeper"] = L[:8] + (("e9", ("e8",), 64, False), ("e10", ("e9",), 64, False),
                           ("d4", ("e6", "up:e10"), 32, False)) + L[9:]
    return a


D.ARCHS.update(archs())


def macs(layers):
    return D.multiply_accumulates(layers)


def teacher_forward(w, x):
    layers = json.loads(str(w["layers"]))
    head_in = layers[-1][0]
    t = {"x": x}
    for name, inputs, _, pool in layers[:-1]:
        y = F.relu(F.conv2d(torch.cat([D.fetch(t, i) for i in inputs], 1), w[name + ".weight"], w[name + ".bias"],
                            padding=1))
        t[name] = F.max_pool2d(y, 2) if pool else y
    return F.conv2d(t[head_in], w["head.weight"], w["head.bias"], padding=1)


def augment_photo(x, rng):
    """Écart mesuré (stats_photo.py, DEV contre lot b0) : photos réelles plus floues (netteté médiane 347 contre
    507), moins bruitées (0,6 contre 1,3 niveau de gris), contraste parfois très bas (p10 24 contre 91), plus claires.
    Tirages bornés par ces mesures."""
    b, dev = x.shape[0], x.device
    T = lambda v: torch.tensor(v, dtype=x.dtype, device=dev).view(b, 1, 1, 1)  # noqa: E731
    # flou gaussien (70 % des pages), sigma 0,3 à 1,2 pixel d'entrée
    sig = np.where(rng.random(b) < 0.7, rng.uniform(0.3, 1.2, b), 0.0)
    if sig.max() > 0:
        k = torch.arange(-3, 4, device=dev, dtype=x.dtype)
        out = []
        for i in range(b):
            if sig[i] == 0:
                out.append(x[i:i + 1])
                continue
            g = torch.exp(-k ** 2 / (2 * sig[i] ** 2))
            g = g / g.sum()
            y = F.conv2d(F.pad(x[i:i + 1], (3, 3, 0, 0), mode="replicate"), g.view(1, 1, 1, 7))
            y = F.conv2d(F.pad(y, (0, 0, 3, 3), mode="replicate"), g.view(1, 1, 7, 1))
            out.append(y)
        x = torch.cat(out)
    # gamma 0,7 à 1,5
    x = x.clamp(1e-4, 1) ** T(np.exp(rng.uniform(math.log(0.7), math.log(1.5), b)))
    # contraste 0,35 à 1,3 autour de la moyenne de l'image, luminosité ±0,15
    m = x.mean((2, 3), keepdim=True)
    x = (x - m) * T(rng.uniform(0.35, 1.3, b)) + m + T(rng.uniform(-0.15, 0.15, b))
    # éclairage en pente (50 %) : jusqu'à ±0,15 d'un bord à l'autre
    h, w = x.shape[2:]
    yy = torch.linspace(-0.5, 0.5, h, device=dev).view(1, 1, h, 1)
    xx = torch.linspace(-0.5, 0.5, w, device=dev).view(1, 1, 1, w)
    a = rng.uniform(0, 2 * math.pi, b)
    s = np.where(rng.random(b) < 0.5, rng.uniform(0, 0.3, b), 0.0)
    x = x + T(s * np.cos(a)) * xx + T(s * np.sin(a)) * yy
    # bruit faible
    x = x + torch.randn_like(x) * T(rng.uniform(0, 0.012, b))
    return x.clamp(0, 1)


BANQUE = {}


def banque():
    if not BANQUE:
        d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "imprime.npz"))
        plat, forme = d["plat"], d["forme"]
        off = np.concatenate([[0], np.cumsum(forme[:, 0] * forme[:, 1])])
        BANQUE["t"] = [plat[off[k]:off[k + 1]].reshape(forme[k]) for k in range(len(forme))]
        BANQUE["taille"] = d["taille"]
    return BANQUE


def dilate(m, r):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return (F.max_pool2d(t, 2 * r + 1, 1, r)[0, 0] > 0).numpy()


def distracteurs_imprimes(x, y, rng, part):
    """Texte imprimé sans chiffre (consignes, titres de carte ; banque_imprime.py) posé sur le papier, hors des mots et des numéros,
    cible nulle : le détecteur doit apprendre que l'écriture imprimée n'est pas un mot manuscrit. x (B, 384, 288)
    uint8 modifié en place, y cibles (B, 192, 144, 12). Corps du texte tiré autour de la hauteur médiane des mots."""
    b = banque()
    for i in range(len(x)):
        if rng.random() >= part:
            continue
        occupe = (y[i, :, :, 0] > 127) | (y[i, :, :, 1] > 127) | (y[i, :, :, 9] > 127)
        hauteurs = y[i, :, :, 7][occupe]
        h = float(np.median(hauteurs)) / 8 * 2 if hauteurs.size else 14.0   # hauteur des mots en pixels d'entrée
        interdit = dilate(occupe, max(2, int(h / 4))).repeat(2, 0).repeat(2, 1)
        fond = x[i].astype(np.float32)
        papier = fond > max(np.percentile(fond, 55), 60)   # zones claires : feuille, carte (jamais la table)
        for _ in range(int(rng.integers(1, 6))):
            cible = np.clip(h * rng.uniform(0.35, 0.9), 6, 40)
            choix = np.nonzero(np.abs(b["taille"] - cible) <= 1.5)[0]
            if not len(choix):
                continue
            t = b["t"][int(choix[rng.integers(len(choix))])]
            th, tw = t.shape
            for _essai in range(8):
                py, px = int(rng.integers(0, 384 - th)), int(rng.integers(0, 288 - tw))
                a = t.astype(np.float32) / 255
                if not (interdit[py:py + th, px:px + tw] & (a > 0.05)).any() \
                        and papier[py:py + th, px:px + tw].mean() > 0.97:
                    fond[py:py + th, px:px + tw] *= 1 - a * rng.uniform(0.45, 0.9)
                    break
        x[i] = np.clip(fond, 0, 255).astype(np.uint8)


def augment_flou(x, rng):
    """Augmentations de base (haller) puis flou gaussien seul sur la moitié des pages (sigma 0,3 à 1,0) : l'écart de
    netteté est le plus net des écarts mesurés entre réel et simulé."""
    from haller.train_detector import augment
    x = augment(x, rng)
    b, dev = x.shape[0], x.device
    sig = np.where(rng.random(b) < 0.5, rng.uniform(0.3, 1.0, b), 0.0)
    k = torch.arange(-3, 4, device=dev, dtype=x.dtype)
    out = []
    for i in range(b):
        if sig[i] == 0:
            out.append(x[i:i + 1])
            continue
        g = torch.exp(-k ** 2 / (2 * sig[i] ** 2))
        g = g / g.sum()
        y = F.conv2d(F.pad(x[i:i + 1], (3, 3, 0, 0), mode="replicate"), g.view(1, 1, 1, 7))
        out.append(F.conv2d(F.pad(y, (0, 0, 3, 3), mode="replicate"), g.view(1, 1, 7, 1)))
    return torch.cat(out)


def augment_base(x, rng):
    from haller.train_detector import augment
    return augment(x, rng)


class AvecSeuil(torch.nn.Module):
    """Réseau du détecteur avec, à l'entraînement seulement, une tête de carte de seuil (DBNet, Liao et al. 2020,
    arXiv:1911.08947) pour les mots et les numéros : conv 3x3 sur d1 à 2 sorties, abandonnée à l'export (le réseau
    exporté est celui de base, même coût sur le Pi)."""

    def __init__(self, arch):
        super().__init__()
        self.net = D.DetNet(arch)
        c = dict((n, o) for n, _, o, _ in self.net.layers)[D.HEAD_INPUT]
        self.seuil = torch.nn.Conv2d(c, 2, 3, padding=1)
        self.d1 = {}
        self.net.blocks[D.HEAD_INPUT].register_forward_hook(lambda m, i, o: self.d1.__setitem__("o", o))

    def forward(self, x):
        out = self.net(x)
        return out, self.seuil(self.d1["o"])


def db_terms(out, seuil, y, r=3):
    """Cible de seuil : 0,3 dans le noyau ; dans la bande autour (dilatation de r pixels de sortie), 0,7 contre le bord
    du noyau décroissant vers 0,3 à r pixels, comme la carte de seuil de DB (haute au bord du texte) ; L1 sur la bande (poids 10). Carte binaire approchée B = sigmoid(50 (P - T)),
    BCE contre le noyau (poids 1)."""
    y = y.permute(0, 3, 1, 2).float()
    keep = (y[:, 11] < 128).float()
    terms = {}
    for k, c in enumerate((D.WORD, D.NUMBER)):
        t = (y[:, c] > 127).float()
        dist = torch.zeros_like(t)
        cur = t
        for i in range(1, r + 1):
            nxt = F.max_pool2d(cur.unsqueeze(1), 3, 1, 1).squeeze(1)
            dist = dist + (nxt - cur) * i
            cur = nxt
        bande = (cur > 0).float()
        cible = torch.where(t > 0, torch.full_like(t, 0.3), 0.7 - 0.4 * (dist - 1) / r) * bande + 0.3 * (1 - bande)
        ts = torch.sigmoid(seuil[:, k])
        terms[f"db_t{k}"] = 10 * (torch.abs(ts - cible) * bande * keep).sum() / (bande * keep).sum().clamp(min=1)
        b = torch.sigmoid(50 * (torch.sigmoid(out[:, c]) - ts)).clamp(1e-6, 1 - 1e-6)
        terms[f"db_b{k}"] = (F.binary_cross_entropy(b, t, reduction="none") * keep).sum() / keep.sum().clamp(min=1)
    return terms


def ohem_terms(out, y, ratio):
    """Mots et numéros : BCE sur tous les positifs et les ratio x plus dures négatives (par lot), plus le dice."""
    y = y.permute(0, 3, 1, 2).float()
    keep = y[:, 11] < 128
    terms = {}
    for name, c in (("word", D.WORD), ("number", D.NUMBER)):
        t = (y[:, c] > 127).float()
        bce = F.binary_cross_entropy_with_logits(out[:, c], t, reduction="none")
        pos = (t > 0) & keep
        neg = (t == 0) & keep
        npos = int(pos.sum())
        k = min(int(neg.sum()), max(ratio * npos, 1000))
        hard = torch.topk(bce[neg], k).values if k > 0 else bce.new_zeros(1)
        l = (bce[pos].sum() + hard.sum()) / max(npos + k, 1)
        p = torch.sigmoid(out[:, c])
        kf = keep.float()
        inter = (p * t * kf).sum()
        dice = 1 - (2 * inter + 1) / ((p * kf).sum() + (t * kf).sum() + 1)
        terms[name] = l * 5 + dice
    return terms


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("shards", nargs="+")
    a.add_argument("--val", nargs="*", default=[])
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--batch", type=int, default=24)
    a.add_argument("--lr", type=float, default=3e-3)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--init")
    a.add_argument("--arch", default="base")
    a.add_argument("--aug", default="base")
    a.add_argument("--kd")
    a.add_argument("--kd-w", type=float, default=1.0)
    a.add_argument("--ohem", type=int, default=0)
    a.add_argument("--ema", type=float, default=0)
    a.add_argument("--db", action="store_true", help="carte de seuil DB à l'entraînement")
    a.add_argument("--imprime", type=float, default=0, help="part des pages avec distracteurs imprimés")
    a = a.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = AvecSeuil(a.arch).to(device) if a.db else None
    model = full.net if a.db else D.DetNet(a.arch).to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), "MAC", macs(D.ARCHS[a.arch]), vars(a), flush=True)
    if a.init:
        x, _ = load(a.shards[0])
        initialize(model, dict(np.load(a.init)), torch.from_numpy(x[:96]).to(device).float().unsqueeze(1) / 255)
        del x
    teacher = None
    if a.kd:
        d = np.load(a.kd)
        teacher = {k: (torch.from_numpy(d[k]).to(device) if k != "layers" else d[k]) for k in d.files}
    ema = None
    if a.ema:
        ema = D.DetNet(a.arch).to(device)
        ema.load_state_dict(model.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
    aug = {"photo": augment_photo, "flou": augment_flou}.get(a.aug, augment_base)
    val = {}
    for v in a.val:
        x, y = load(v)
        val[v.split("/")[-1]] = (x[:600], y[:600])
    sizes = [len(json.loads(str(np.load(s)["meta"]))) for s in a.shards]
    steps = a.epochs * sum(sizes) // a.batch
    opt = torch.optim.AdamW((full if full is not None else model).parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps + 10, pct_start=0.05)
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(a.shards))
        sums, n = {}, 0
        for g in order:
            x, y = load(a.shards[g])
            perm = rng.permutation(len(x))
            for i in range(0, len(perm) - a.batch + 1, a.batch):
                idx = np.sort(perm[i:i + a.batch])
                if a.imprime:
                    xi = x[idx].copy()
                    distracteurs_imprimes(xi, y[idx], rng, a.imprime)
                else:
                    xi = x[idx]
                xb = torch.from_numpy(xi).to(device).float().unsqueeze(1) / 255
                yb = torch.from_numpy(y[idx]).to(device)
                xa = aug(xb, rng)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    if full is not None:
                        out, seuil = full(xa)
                        seuil = seuil.float()
                    else:
                        out = model(xa)
                out = out.float()
                terms = losses(out, yb)
                if full is not None:
                    terms.update(db_terms(out, seuil, yb))
                if a.ohem:
                    terms.update(ohem_terms(out, yb, a.ohem))
                if teacher is not None:
                    with torch.no_grad():
                        tp = torch.sigmoid(teacher_forward(teacher, xa))
                    keep = (yb[..., 11] < 128).float()
                    kd = 0
                    for c in (D.WORD, D.NUMBER):
                        kd = kd + (F.binary_cross_entropy_with_logits(out[:, c], tp[:, c], reduction="none")
                                   * keep).sum() / keep.sum().clamp(min=1)
                    terms["kd"] = kd * a.kd_w * 5
                loss = sum(terms.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if ema is not None:
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                for k, v in terms.items():
                    sums[k] = sums.get(k, 0) + float(v.detach())
                n += 1
            del x, y
        m = ema if ema is not None else model
        res = evaluate(m, val, device)
        print(f"epoch {epoch}/{a.epochs} " + " ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
              + f" {time.time() - t0:.0f} s " + json.dumps({k: {q: round(u, 4) for q, u in r.items()}
                                                             for k, r in res.items()}), flush=True)
        np.savez(a.out, **m.export())


if __name__ == "__main__":
    main()
