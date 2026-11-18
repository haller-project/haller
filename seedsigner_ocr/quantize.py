"""
Quantizes the float weights written by seedsigner_ocr.train into the integer
model shipped with SeedSigner (resources/handwriting/word_model.npz).

    python -m seedsigner_ocr.quantize MODEL.npz SHARD.npz OUT.npz --count 3000 --seed 1

Arithmetic (seedsigner.helpers.handwriting.word_model, native/handwriting.c):
- input: ink 0-255 -> ink << 3 (2040 stands for 1.0);
- weights per output channel: w_q = round(w * s_w[o]), s_w[o] = 2047 / max|w[o]|;
- activations after ReLU and pooling: 12-bit integers, scale 2047 / a_max, a_max
  being the 99.99th percentile of the layer's float activations over COUNT word
  images of SHARD (a training shard written by seedsigner_ocr.generate);
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
The same inputs always give the same file.
"""
import argparse

import numpy as np

LAYERS = ("b1", "b2", "b3", "b4", "t1", "t2")
AMAX = 2047
SHIFT = 16



def conv2d(x, w, b):
    """x (C, H, W), w (O, C, 3, 3), zero padding 1."""
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2)).transpose(0, 3, 4, 1, 2)
    return (w.reshape(w.shape[0], -1) @ cols.reshape(c * 9, h * wd) + b[:, None]).reshape(-1, h, wd)



def conv1d(x, w, b, dilation=1):
    """x (C, T), w (O, C, 3), "same" zero padding."""
    t = x.shape[1]
    xp = np.pad(x, ((0, 0), (dilation, dilation)))
    cols = np.concatenate([xp[:, j * dilation:j * dilation + t] for j in range(3)], 0)
    return w.transpose(0, 2, 1).reshape(w.shape[0], -1) @ cols + b[:, None]



def max_pool(x, ph, pw):
    c, h, w = x.shape
    return x[:, :h // ph * ph, :w // pw * pw].reshape(c, h // ph, ph, w // pw, pw).max(axis=(2, 4))



def forward(p, ink):
    """
    Float reference: (activations of the six layers after ReLU and pooling,
    logits (37, frames)). The ink map is padded on the right with 16 empty
    columns, the width rounded up to a multiple of 4, as in training.
    """
    width = ink.shape[1]
    x = np.zeros((1, ink.shape[0], (width + 16 + 3) // 4 * 4), np.float32)
    x[0, :, :width] = ink / np.float32(255)
    acts = []
    for i, pool in enumerate(((2, 2), (2, 2), (2, 1), (2, 1))):
        x = max_pool(np.maximum(conv2d(x, p[f"b{i + 1}.weight"], p[f"b{i + 1}.bias"]), 0), *pool)
        acts.append(x)
    x = x.reshape(-1, x.shape[2])
    x = np.maximum(conv1d(x, p["t1.weight"], p["t1.bias"]), 0)
    acts.append(x)
    x = np.maximum(conv1d(x, p["t2.weight"], p["t2.bias"], 2), 0)
    acts.append(x)
    z = p["out.weight"][:, :, 0] @ x + p["out.bias"][:, None]
    return acts, z[:, :max(1, width // 4)]



def calibrate(p, inks, percentile=99.99):
    """a_max of each layer over a list of ink maps."""
    acts = [[] for _ in LAYERS]
    for ink in inks:
        for k, a in enumerate(forward(p, ink)[0]):
            acts[k].append(a.ravel())
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
    a = a.parse_args()

    p = {k: v.astype(np.float32) for k, v in np.load(a.model).items()}
    shard = np.load(a.shard)
    x, w = shard["x"], shard["w"]
    pick = np.sort(np.random.default_rng(a.seed).choice(len(w), a.count, replace=False))
    inks = [x[i, :, :w[i]].copy() for i in pick if w[i] > 0]
    del x, shard
    amax = calibrate(p, inks)
    print("amax", amax)
    np.savez(a.out, **quantize(p, amax))



if __name__ == "__main__":
    main()
