"""
Word detector (Haller-1-det prototype): a small fully convolutional network
that finds the handwritten words, the line numbers and the seed cards on a
photo reduced to 288 x 384 pixels (portrait, grayscale).

Outputs, at half the input resolution (144 x 192), one map per channel:
  0 word kernel: central band of every word box (40 % of its height, ends
    moved in by a quarter of the height), so that neighboring words stay apart;
  1 number kernel: the same for line numbers, handwritten or printed;
  2 card: light inner area of an official seed card;
  3-6 card corners: top left, top right, bottom right, bottom left of that
    area (Gaussian peaks), in the reading orientation of the card;
  7 height of the word or number box, on the kernels (output pixels / 8).
Channels 0-6 are logits, channel 7 a value.

Layers: 3x3 convolutions with batch normalization folded at export, ReLU,
2x2 max pooling, nearest upsampling and channel concatenation, which are the
operations of SeedSigner's integer word network (16-bit weights, 12-bit
activations) plus upsampling and concatenation.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = (288, 384)   # width, height
OUTPUT_SIZE = (144, 192)
CHANNELS = 8
WORD, NUMBER, CARD, CORNERS, HEIGHT = 0, 1, 2, (3, 4, 5, 6), 7
HEIGHT_SCALE = 8.0

# (name, input tensors, output channels, max pooling after the convolution)
# Tensors: "x" is the input; every layer output is named after the layer;
# "up:<name>" is a layer output upsampled by 2.
LAYERS = (
    ("e1", ("x",), 8, True),        # 1/2
    ("e2", ("e1",), 16, True),      # 1/4
    ("e3", ("e2",), 32, True),      # 1/8
    ("e4", ("e3",), 48, True),      # 1/16
    ("e5", ("e4",), 64, False),
    ("e6", ("e5",), 64, False),
    ("e7", ("e6",), 64, True),      # 1/32
    ("e8", ("e7",), 64, False),
    ("d4", ("e6", "up:e8"), 32, False),   # 1/16
    ("d3", ("e3", "up:d4"), 32, False),   # 1/8
    ("d2", ("e2", "up:d3"), 16, False),   # 1/4
    ("d1", ("e1", "up:d2"), 8, False),    # 1/2
)
HEAD_INPUT = "d1"



def layer_inputs(name):
    return dict((n, i) for n, i, _, _ in LAYERS)[name]



class DetNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = {"x": 1}
        self.blocks = nn.ModuleDict()
        for name, inputs, out, _ in LAYERS:
            c = sum(channels[i.split(":")[-1]] for i in inputs)
            self.blocks[name] = nn.Sequential(nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU())
            channels[name] = out
        self.head = nn.Conv2d(channels[HEAD_INPUT], CHANNELS, 3, padding=1)


    def forward(self, x):
        t = {"x": x}
        for name, inputs, _, pool in LAYERS:
            y = self.blocks[name](torch.cat([fetch(t, i) for i in inputs], 1))
            t[name] = F.max_pool2d(y, 2) if pool else y
        return self.head(t[HEAD_INPUT])


    def export(self) -> dict:
        weights = {}
        for name, _, _, _ in LAYERS:
            conv, bn = self.blocks[name][0], self.blocks[name][1]
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            weights[name + ".weight"] = (conv.weight * s.view(-1, 1, 1, 1)).detach().cpu().numpy()
            weights[name + ".bias"] = (bn.bias - bn.running_mean * s).detach().cpu().numpy()
        weights["head.weight"] = self.head.weight.detach().cpu().numpy()
        weights["head.bias"] = self.head.bias.detach().cpu().numpy()
        weights = {k: v.astype(np.float32) for k, v in weights.items()}
        weights["layers"] = np.array(json.dumps([[n, list(i), c, p] for n, i, c, p in LAYERS] + [[HEAD_INPUT]]))
        return weights



def fetch(t, name):
    if name.startswith("up:"):
        return F.interpolate(t[name[3:]], scale_factor=2, mode="nearest")
    return t[name]



def folded_forward(weights: dict, x: torch.Tensor, activations: dict = None) -> torch.Tensor:
    """Output maps of the exported model (torch tensors); x in 0-1, (N, 1, 384, 288)."""
    t = {"x": x}
    for name, inputs, _, pool in LAYERS:
        y = F.relu(F.conv2d(torch.cat([fetch(t, i) for i in inputs], 1), weights[name + ".weight"],
                            weights[name + ".bias"], padding=1))
        t[name] = F.max_pool2d(y, 2) if pool else y
    if activations is not None:
        activations.update(t)
    return F.conv2d(t[HEAD_INPUT], weights["head.weight"], weights["head.bias"], padding=1)



def count_parameters(weights: dict) -> int:
    return int(sum(v.size for k, v in weights.items() if k != "layers"))



def multiply_accumulates() -> int:
    """Multiply-accumulates of one inference."""
    w, h = INPUT_SIZE
    size, channels, total = {"x": (w, h)}, {"x": 1}, 0
    for name, inputs, out, pool in LAYERS:
        first = inputs[0].split(":")[-1]
        sw, sh = size[first] if not inputs[0].startswith("up:") else (size[first][0] * 2, size[first][1] * 2)
        c = sum(channels[i.split(":")[-1]] for i in inputs)
        total += sw * sh * 9 * c * out
        size[name] = (sw // 2, sh // 2) if pool else (sw, sh)
        channels[name] = out
    sw, sh = size[HEAD_INPUT]
    return total + sw * sh * 9 * channels[HEAD_INPUT] * CHANNELS



# ---------- quantization (same arithmetic as native/handwriting.c) ----------

AMAX = 2047
SHIFT = 16
INPUT_SCALE = 2040.0     # gray level g -> g << 3



def calibrate(weights: dict, images: np.ndarray, percentile: float = 99.99) -> dict:
    """a_max of every layer output over a batch of photos (N, 384, 288) uint8."""
    w = {k: torch.from_numpy(v) for k, v in weights.items() if k != "layers"}
    acts = {}
    with torch.no_grad():
        for i in range(0, len(images), 16):
            t = {}
            folded_forward(w, torch.from_numpy(images[i:i + 16, None].astype(np.float32) / 255), t)
            for name, _, _, _ in LAYERS:
                v = t[name].numpy().ravel()
                acts.setdefault(name, []).append(v[np.random.default_rng(i).choice(len(v), min(len(v), 200000))])
    return {k: float(np.percentile(np.concatenate(v), percentile)) for k, v in acts.items()}



def quantize(weights: dict, amax: dict) -> dict:
    """Integer model: <layer>.weight (O, 3, 3, C) int16, .bias int32, .mul int32 (16 fractional bits);
    head.weight int16, head.scale and head.bias float32; amax per layer."""
    scale = {"x": INPUT_SCALE}
    out = {}

    def quant_conv(name, inputs):
        w = weights[name + ".weight"].astype(np.float64)
        b = weights[name + ".bias"].astype(np.float64)
        s_in = np.concatenate([np.full(weights_in_channels(i), scale[i.split(":")[-1]]) for i in inputs])
        w_eff = w / s_in[None, :, None, None]
        o = w.shape[0]
        sw = AMAX / np.maximum(np.abs(w_eff.reshape(o, -1)).max(1), 1e-12)
        out[name + ".weight"] = np.round(w_eff * sw[:, None, None, None]).transpose(0, 2, 3, 1).astype(np.int16)
        return sw, b

    channels = {"x": 1}
    for name, inputs, c, _ in LAYERS:
        channels[name] = c

    def weights_in_channels(i):
        return channels[i.split(":")[-1]]

    for name, inputs, _, _ in LAYERS:
        sw, b = quant_conv(name, inputs)
        s_out = AMAX / max(amax[name], 1e-6)
        out[name + ".bias"] = np.round(b * sw).astype(np.int32)
        out[name + ".mul"] = np.round(s_out / sw * (1 << SHIFT)).astype(np.int32)
        scale[name] = s_out
    sw, b = quant_conv("head", (HEAD_INPUT,))
    out["head.scale"] = (1.0 / sw).astype(np.float32)
    out["head.bias"] = b.astype(np.float32)
    out["amax"] = np.array([amax[n] for n, _, _, _ in LAYERS], np.float32)
    out["layers"] = weights["layers"]
    return {k: np.ascontiguousarray(v) for k, v in out.items()}



def integer_forward(q: dict, image: np.ndarray) -> np.ndarray:
    """Reference of the integer computation in numpy (slow): output maps (8, 192, 144) of a photo (384, 288)."""
    t = {"x": image.astype(np.int64)[:, :, None] << 3}

    def conv(x, w):
        h, wd, _ = x.shape
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        acc = np.zeros((h, wd, w.shape[0]), np.int64)
        for dy in range(3):
            for dx in range(3):
                acc += p[dy:dy + h, dx:dx + wd] @ w[:, dy, dx, :].T.astype(np.int64)
        return acc

    def get(name):
        if name.startswith("up:"):
            return t[name[3:]].repeat(2, 0).repeat(2, 1)
        return t[name]

    for name, inputs, _, pool in LAYERS:
        acc = conv(np.concatenate([get(i) for i in inputs], 2), q[name + ".weight"]) + q[name + ".bias"]
        if pool:
            h, w, c = acc.shape
            acc = acc.reshape(h // 2, 2, w // 2, 2, c).max((1, 3))
        v = (acc * q[name + ".mul"] + (1 << (SHIFT - 1))) >> SHIFT
        t[name] = np.where(acc > 0, np.minimum(v, AMAX), 0)
    acc = conv(t[HEAD_INPUT], q["head.weight"])
    # float32 products and sums, as in the C library
    return (acc.astype(np.float32) * q["head.scale"] + q["head.bias"]).transpose(2, 0, 1)



def main():
    """python -m haller.detector MODEL.npz SHARD.npz OUT.npz [--count N]: quantized detector, calibrated on
    COUNT photos of a training shard."""
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("model")
    a.add_argument("shard")
    a.add_argument("out")
    a.add_argument("--count", type=int, default=400)
    a = a.parse_args()
    d = np.load(a.model)
    weights = {k: d[k] for k in d.files}
    images = np.load(a.shard)["x"][:a.count]
    amax = calibrate(weights, images)
    print("amax", {k: round(v, 3) for k, v in amax.items()})
    np.savez(a.out, **quantize(weights, amax))



if __name__ == "__main__":
    main()
