"""
Quantizes the float weights written by haller.train into the integer model
computed by SeedSigner's C library (resources/handwriting/haller-*.npz), and
writes its reference vectors (haller.vectors) next to it.

    python -m haller.quantize MODEL.npz SHARD.npz OUT.npz --count 3000 --seed 1 \
        [--strips STYLE=STRIPS.npz ...]

Arithmetic (SeedSigner, native/handwriting.c):
- input: ink 0-255 -> ink << 3 (2040 stands for 1.0);
- weights per output channel: w_q = round(w * s_w[o]), s_w[o] = 2047 / max|w[o]|;
- activations after ReLU and pooling: 12-bit integers, scale 2047 / a_max, a_max
  being the 99.99th percentile of the layer's float activations over COUNT word
  images of SHARD (a training shard written by haller.generate), computed by
  the float model (PyTorch, haller.model.folded_forward);
- each layer: int32 accumulator = bias_q + sum(w_q * a_q), then
  a_q' = min(2047, (acc * mul[o] + 2^15) >> 16) if acc > 0 else 0;
- output layer: logit = acc * scale[k] + bias[k] (float32).

Output arrays, little-endian, in the memory layout read by the C code:
- b1.weight (c1, 9) int16; b2-b4.weight (O, 3, 3, C) int16;
- t1.weight (hidden, 3, 2 * c3) int16, input index h * c3 + c;
- t2.weight (hidden, 3, hidden) int16;
- <layer>.bias (O,) int32, <layer>.mul (O,) int32;
- out.weight (37, hidden) int16, out.scale (37,) float32, out.bias (37,) float32;
- amax (6,) float32: calibrated activation ranges, for reference.
The same inputs give the same file on a given processor. The calibrated ranges
are float32 percentiles of PyTorch activations, which can differ by about
1e-6 from one processor to another (vector instructions), enough to change a
rounding: a bias can then differ by one unit. `--ranges WEIGHTS.npz` takes the
ranges stored in an existing quantized file instead of calibrating them; the
rest is computed in float64 and gives the same file on every machine.
"""
import argparse

import numpy as np
import torch

from . import vectors
from .model import folded_forward

LAYERS = ("b1", "b2", "b3", "b4", "t1", "t2")
AMAX = 2047
SHIFT = 16



def calibration_inks(shard_path: str, count: int, seed: int) -> list:
    """COUNT ink maps drawn from a training shard."""
    shard = np.load(shard_path)
    x, w = shard["x"], shard["w"]
    pick = np.sort(np.random.default_rng(seed).choice(len(w), count, replace=False))
    return [x[i, :, :w[i]].copy() for i in pick if w[i] > 0]



def calibrate(p: dict, inks: list, percentile: float = 99.99) -> np.ndarray:
    """
    a_max of each layer over a list of ink maps, each padded on the right with
    16 empty columns, the width rounded up to a multiple of 4, as in training.
    """
    weights = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in p.items()}
    acts = [[] for _ in LAYERS]
    with torch.no_grad():
        for ink in inks:
            width = ink.shape[1]
            x = torch.zeros(1, 1, ink.shape[0], (width + 16 + 3) // 4 * 4)
            x[0, 0, :, :width] = torch.tensor(ink, dtype=torch.float32) / 255
            layers = []
            folded_forward(weights, x, layers)
            for k, a in enumerate(layers):
                acts[k].append(a.numpy().ravel())
    return np.array([np.percentile(np.concatenate(a), percentile) for a in acts], np.float32)



def quantize(p, amax):
    out = {}
    scale_in = 2040.0
    c3 = p["b3.weight"].shape[0]
    for i, layer in enumerate(LAYERS):
        w = p[layer + ".weight"].astype(np.float64)
        b = p[layer + ".bias"].astype(np.float64)
        o = w.shape[0]
        sw = AMAX / np.maximum(np.abs(w.reshape(o, -1)).max(1), 1e-8)
        if layer == "b1":
            wq = (w * sw[:, None, None, None]).reshape(o, 9)
        elif layer.startswith("b"):
            wq = (w * sw[:, None, None, None]).transpose(0, 2, 3, 1)  # O, ky, kx, C
        elif layer == "t1":  # input index c * 2 + h in training -> h * c3 + c
            wq = (w * sw[:, None, None]).reshape(o, c3, 2, 3).transpose(0, 3, 2, 1).reshape(o, 3, 2 * c3)
        else:
            wq = (w * sw[:, None, None]).transpose(0, 2, 1)  # O, k, C
        scale_out = AMAX / max(float(amax[i]), 1e-6)
        out[layer + ".weight"] = np.round(wq).astype(np.int16)
        out[layer + ".bias"] = np.round(b * sw * scale_in).astype(np.int32)
        out[layer + ".mul"] = np.round(scale_out / (sw * scale_in) * (1 << SHIFT)).astype(np.int32)
        scale_in = scale_out
    wo = p["out.weight"].astype(np.float64)[:, :, 0]
    so = AMAX / np.maximum(np.abs(wo).max(1), 1e-8)
    out["out.weight"] = np.round(wo * so[:, None]).astype(np.int16)
    out["out.scale"] = (1.0 / (so * scale_in)).astype(np.float32)
    out["out.bias"] = p["out.bias"].astype(np.float32)
    out["amax"] = np.asarray(amax, np.float32)
    return {k: np.ascontiguousarray(v) for k, v in out.items()}



def main():
    a = argparse.ArgumentParser()
    a.add_argument("model")
    a.add_argument("shard")
    a.add_argument("out")
    a.add_argument("--count", type=int, default=3000)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--strips", action="append", default=[],
                   help="STYLE=STRIPS.npz: source of reference vectors (haller.vectors), repeatable")
    a.add_argument("--ranges", help="quantized weights whose activation ranges (amax) are used instead of "
                                    "calibrating them; the calibrated ranges are still printed for comparison")
    a = a.parse_args()

    p = {k: v.astype(np.float32) for k, v in np.load(a.model).items()}
    amax = calibrate(p, calibration_inks(a.shard, a.count, a.seed))
    print("amax", amax)
    if a.ranges:
        calibrated, amax = amax, np.load(a.ranges)["amax"].astype(np.float32)
        print("amax of", a.ranges, amax, "largest difference", float(np.abs(calibrated - amax).max()))
    np.savez(a.out, **quantize(p, amax))
    if a.strips:
        print(vectors.write(a.out, vectors.parse_sources(a.strips), float_weights=p))



if __name__ == "__main__":
    main()
