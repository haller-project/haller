"""Inférence numpy pure du classifieur A-Z (aucune dépendance hors numpy).

Destinée à SeedSigner OS (Pi Zero : ARMv6, numpy sans BLAS).
Usage : python infer_np.py <modele.npz> <test.npz> [taille_lot]
Affiche : concordance avec les prédictions torch, précision, temps par lettre.
"""
import sys
import time
import tracemalloc

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# NOBLAS=1 : produit matriciel par einsum (boucles scalaires de numpy),
# pour reproduire un numpy compilé sans BLAS comme celui de SeedSigner OS.
if os.environ.get("NOBLAS"):
    def mm(a, b):
        return np.einsum("ij,jk->ik", a, b)
else:
    def mm(a, b):
        return a @ b


def conv3x3_relu_pool(x, w, b):
    # x : (N, C, H, W) ; w : (O, C, 3, 3) ; padding 1, puis ReLU et max-pool 2x2
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(xp, (3, 3), axis=(2, 3))  # N, C, H, W, 3, 3
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * wd, c * 9)
    y = mm(cols, w.reshape(w.shape[0], -1).T) + b  # (N*H*W, O)
    y = np.maximum(y, 0).reshape(n, h, wd, -1)
    h2, w2 = h // 2, wd // 2
    y = y[:, : h2 * 2, : w2 * 2].reshape(n, h2, 2, w2, 2, -1).max(axis=(2, 4))
    return y.transpose(0, 3, 1, 2)


def predire(p, x):
    # x : (N, 28, 28) uint8
    x = (x.astype(np.float32) / 255.0)[:, None]
    for k in ("c1", "c2", "c3"):
        x = conv3x3_relu_pool(x, p[k + ".weight"], p[k + ".bias"])
    logits = mm(x.reshape(len(x), -1), p["fc.weight"].T) + p["fc.bias"]
    return logits.argmax(1)


def main():
    p = {k: v.astype(np.float32) for k, v in np.load(sys.argv[1]).items()}
    t = np.load(sys.argv[2])
    lot = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    x, y, ref = t["x"][:n], t["y"][:n], t["pred"][:n]
    predire(p, x[:2])  # échauffement
    tracemalloc.start()
    t0 = time.perf_counter()
    pred = np.concatenate([predire(p, x[i : i + lot]) for i in range(0, n, lot)])
    dt = time.perf_counter() - t0
    _, pic = tracemalloc.get_traced_memory()
    print(f"lettres={n} lot={lot}")
    print(f"concordance torch/numpy : {(pred == ref).mean()*100:.2f} %")
    print(f"précision : {(pred == y).mean()*100:.2f} %")
    print(f"temps : {dt:.2f} s, soit {dt/n*1000:.1f} ms par lettre")
    print(f"pic mémoire numpy : {pic/1e6:.1f} Mo")


if __name__ == "__main__":
    main()
