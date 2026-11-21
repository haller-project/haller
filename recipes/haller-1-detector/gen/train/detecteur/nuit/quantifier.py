"""Quantification Haller-1 (haller.detector.calibrate / quantize, inchangés) avec des variantes de calibration des
activations : percentile, nombre de pages, pages de calibration passées par les augmentations photo (train_nuit.py)
pour ressembler aux photos réelles (aucune photo réelle n'est utilisée).

    python quantifier.py POIDS.npz LOT.npz SORTIE.npz [--pct 99.99] [--count 400] [--photo]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.argv, argv = sys.argv[:1], sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_nuit.py")).read().split("def main")[0])
sys.argv = argv

a = argparse.ArgumentParser()
a.add_argument("poids")
a.add_argument("lot")
a.add_argument("out")
a.add_argument("--pct", type=float, default=99.99)
a.add_argument("--count", type=int, default=400)
a.add_argument("--photo", action="store_true")
a.add_argument("--canal", action="store_true", help="échelle d'activation par canal (même format, même C)")
a = a.parse_args()
d = np.load(a.poids)
w = {k: d[k] for k in d.files}
x = np.load(a.lot)["x"][:a.count]
if a.photo:
    rng = np.random.default_rng(3)
    t = torch.from_numpy(x).float().unsqueeze(1) / 255
    x = (augment_photo(t, rng)[:, 0].numpy() * 255).round().astype(np.uint8)


def calibrer_canaux(weights, images, pct):
    """a_max de chaque canal de chaque couche."""
    wt = {k: torch.from_numpy(v) for k, v in weights.items() if k != "layers"}
    acts = {}
    with torch.no_grad():
        for i in range(0, len(images), 16):
            t = {}
            D.folded_forward(wt, torch.from_numpy(images[i:i + 16, None].astype(np.float32) / 255), t)
            for name, _, _, _ in D.LAYERS:
                v = t[name].permute(1, 0, 2, 3).reshape(t[name].shape[1], -1).numpy()
                sel = np.random.default_rng(i).choice(v.shape[1], min(v.shape[1], 50000))
                acts.setdefault(name, []).append(v[:, sel])
    return {k: np.maximum(np.percentile(np.concatenate(v, 1), pct, axis=1), 1e-3) for k, v in acts.items()}


def quantifier_canaux(weights, amax):
    """D.quantize avec une échelle de sortie par canal : s_out[c] = AMAX / amax[c] ; la couche suivante reçoit le
    vecteur d'échelles de ses entrées (w_eff = w / s_in par canal d'entrée), comme déjà pour la concaténation."""
    scale = {"x": np.array([D.INPUT_SCALE])}
    out = {}
    chans = {"x": 1}
    for n, _, c, _ in D.LAYERS:
        chans[n] = c

    def s_in_of(inputs):
        return np.concatenate([np.broadcast_to(scale[i.split(":")[-1]], (chans[i.split(":")[-1]],)) for i in inputs])

    def quant_conv(name, inputs):
        w = weights[name + ".weight"].astype(np.float64)
        b = weights[name + ".bias"].astype(np.float64)
        w_eff = w / s_in_of(inputs)[None, :, None, None]
        o = w.shape[0]
        sw = D.AMAX / np.maximum(np.abs(w_eff.reshape(o, -1)).max(1), 1e-12)
        out[name + ".weight"] = np.round(w_eff * sw[:, None, None, None]).transpose(0, 2, 3, 1).astype(np.int16)
        return sw, b

    for name, inputs, _, _ in D.LAYERS:
        sw, b = quant_conv(name, inputs)
        s_out = D.AMAX / amax[name]
        out[name + ".bias"] = np.round(b * sw).astype(np.int32)
        out[name + ".mul"] = np.round(s_out / sw * (1 << D.SHIFT)).astype(np.int32)
        scale[name] = s_out
    sw, b = quant_conv("head", (D.HEAD_INPUT,))
    out["head.scale"] = (1.0 / sw).astype(np.float32)
    out["head.bias"] = b.astype(np.float32)
    out["amax"] = np.array([amax[n].max() for n, _, _, _ in D.LAYERS], np.float32)
    out["layers"] = weights["layers"]
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


if a.canal:
    amax = calibrer_canaux(w, x, a.pct)
    q = quantifier_canaux(w, amax)
    assert max(int(np.abs(q[n + ".mul"]).max()) for n, _, _, _ in D.LAYERS) < 2 ** 31
else:
    amax = D.calibrate(w, x, a.pct)
    q = D.quantize(w, amax)
np.savez(a.out, **q)
amax = {k: float(np.max(v)) for k, v in amax.items()}
print(a.out, {k: round(v, 2) for k, v in amax.items()})
