"""
Trains the word detector (haller.detector) on shards of photos with target maps
(x: N x 384 x 288 uint8, y: N x 192 x 144 x 12 uint8, see haller.detector and
the page generator), and writes the float weights (batch normalization folded).

    python -m haller.train_detector OUT.npz SHARD.npz... --val VAL.npz... [--epochs 12]

Shards are read a few at a time (each holds about 360 MB uncompressed). Target
channels: 0 word kernel, 1 number kernel, 2 card, 3-6 card corners (0-255),
7 box height x 8 in output pixels, 8 card kernel, 9 link, 10 partial word,
11 ignored area (255). Shards of 9 channels (no card kernel, link nor partial
word, ignored area in 8) are read as well: their missing channels are left out
of the loss.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .detector import CARD, CARD_KERNEL, CORNERS, HEIGHT, HEIGHT_SCALE, LINK, NUMBER, PARTIAL, WORD, DetNet

SHARDS_IN_MEMORY = 2



def load(path):
    """x, y of a shard; y always has 12 channels, those of a 9-channel shard being filled with UNKNOWN in 8-10."""
    d = np.load(path)
    x, y = d["x"], d["y"]
    if y.shape[-1] == 9:
        pad = np.full(y.shape[:-1] + (3,), UNKNOWN, np.uint8)
        y = np.concatenate([y[..., :8], pad, y[..., 8:9]], -1)
    return x, y


UNKNOWN = 1   # value of a target channel absent from a shard (an actual target is 0 or 255)



def losses(out, y):
    """Per-term losses of a batch: out (B, 11, H, W) logits/values, y (B, H, W, 12) uint8 on the device."""
    y = y.permute(0, 3, 1, 2).float()
    keep = (y[:, 11] < 128).float()
    terms = {}
    for name, c in (("word", WORD), ("number", NUMBER)):
        t = (y[:, c] > 127).float()
        p = torch.sigmoid(out[:, c])
        bce = F.binary_cross_entropy_with_logits(out[:, c], t, reduction="none")
        bce = (bce * keep).sum() / keep.sum().clamp(min=1)
        inter = (p * t * keep).sum()
        dice = 1 - (2 * inter + 1) / ((p * keep).sum() + (t * keep).sum() + 1)
        terms[name] = bce * 5 + dice
    terms["card"] = F.binary_cross_entropy_with_logits(out[:, CARD], (y[:, CARD] > 127).float())
    known = (y[:, CARD_KERNEL, :1, :1] != UNKNOWN).float()   # per image
    for name, c, weight in (("kernel", CARD_KERNEL, 1.0), ("link", LINK, 3.0), ("partial", PARTIAL, 3.0)):
        t = (y[:, c] > 127).float()
        bce = F.binary_cross_entropy_with_logits(out[:, c], t, reduction="none") * (1 + weight * t)
        terms[name] = (bce * known).sum() / (known.sum() * t.shape[1] * t.shape[2]).clamp(min=1)
    tc = y[:, list(CORNERS)] / 255
    w = 1 + 30 * tc
    terms["corners"] = (F.binary_cross_entropy_with_logits(out[:, list(CORNERS)], tc, reduction="none") * w).mean() * 4
    on = ((y[:, WORD] > 127) | (y[:, NUMBER] > 127)).float()
    terms["height"] = (torch.abs(out[:, HEIGHT] - y[:, HEIGHT] / (8 * HEIGHT_SCALE)) * on).sum() / on.sum().clamp(min=1)
    return terms



def augment(x, rng):
    """Contrast, brightness and noise per image (x float 0-1, (B, 1, H, W))."""
    b = x.shape[0]
    c = torch.tensor(rng.uniform(0.6, 1.3, b), dtype=x.dtype, device=x.device).view(b, 1, 1, 1)
    m = torch.tensor(rng.uniform(-0.15, 0.15, b), dtype=x.dtype, device=x.device).view(b, 1, 1, 1)
    x = (x - 0.5) * c + 0.5 + m
    x = x + torch.randn_like(x) * torch.tensor(rng.uniform(0, 0.03, b), dtype=x.dtype, device=x.device).view(b, 1, 1, 1)
    return x.clamp(0, 1)



def evaluate(model, sets, device):
    """Pixel precision and recall of the word and number kernels, corner peak hit rate, per set."""
    model.eval()
    res = {}
    with torch.no_grad():
        for name, (x, y) in sets.items():
            tp = fp = fn = 0
            ntp = nfp = nfn = 0
            for i in range(0, len(x), 32):
                xb = torch.from_numpy(x[i:i + 32]).to(device).float().unsqueeze(1) / 255
                yb = torch.from_numpy(y[i:i + 32]).to(device)
                out = model(xb)
                keep = yb[..., 11] < 128
                for c, acc in ((WORD, "w"), (NUMBER, "n")):
                    p = (out[:, c] > 0) & keep
                    t = (yb[..., c] > 127) & keep
                    a, b_, c_ = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
                    if acc == "w":
                        tp, fp, fn = tp + a, fp + b_, fn + c_
                    else:
                        ntp, nfp, nfn = ntp + a, nfp + b_, nfn + c_
            res[name] = dict(word_p=tp / max(1, tp + fp), word_r=tp / max(1, tp + fn),
                             num_p=ntp / max(1, ntp + nfp), num_r=ntp / max(1, ntp + nfn))
    model.train()
    return res



@torch.no_grad()
def initialize(model, weights: dict, x: torch.Tensor):
    """
        Starts from exported weights, where each batch normalization is folded
        into its convolution: the convolutions take the folded weights, and
        each normalization is set, layer by layer, so that on the batch `x` it
        gives back the folded bias (identity in training mode on that batch).
    """
    for name, block in model.blocks.items():
        block[0].weight.copy_(torch.from_numpy(weights[name + ".weight"]))
    model.head.weight.copy_(torch.from_numpy(weights["head.weight"]))
    model.head.bias.copy_(torch.from_numpy(weights["head.bias"]))
    model.train()
    for name, block in model.blocks.items():
        conv, bn = block[0], block[1]
        seen = {}
        hook = conv.register_forward_hook(lambda m, i, o: seen.setdefault("o", o))
        model(x)
        hook.remove()
        o = seen["o"].double()
        mean, var = o.mean((0, 2, 3)), o.var((0, 2, 3), unbiased=False)
        bn.weight.copy_(torch.sqrt(var + bn.eps).float())
        bn.bias.copy_((mean + torch.from_numpy(weights[name + ".bias"]).to(o.device).double()).float())
        bn.running_mean.copy_(mean.float())
        bn.running_var.copy_(var.float())



def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("shards", nargs="+")
    a.add_argument("--val", nargs="*", default=[])
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--batch", type=int, default=24)
    a.add_argument("--lr", type=float, default=3e-3)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--shards-in-memory", type=int, default=SHARDS_IN_MEMORY)
    a.add_argument("--init", help="exported weights (.npz) to start from")
    a.add_argument("--arch", default="base", help="network variant (haller.detector.ARCHS)")
    a = a.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetNet(a.arch).to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    if a.init:
        x, _ = load(a.shards[0])
        initialize(model, dict(np.load(a.init)), torch.from_numpy(x[:96]).to(device).float().unsqueeze(1) / 255)
        del x
    val = {}
    for v in a.val:
        x, y = load(v)
        val[v.split("/")[-1]] = (x[:600], y[:600])
    sizes = [len(json.loads(str(np.load(s)["meta"]))) for s in a.shards]
    steps = a.epochs * sum(sizes) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps + 10, pct_start=0.05)
    best = None
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(a.shards))
        sums, n = {}, 0
        for g in range(0, len(order), a.shards_in_memory):
            parts = [load(a.shards[k]) for k in order[g:g + a.shards_in_memory]]
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            del parts
            perm = rng.permutation(len(x))
            for i in range(0, len(perm) - a.batch + 1, a.batch):
                idx = np.sort(perm[i:i + a.batch])
                xb = torch.from_numpy(x[idx]).to(device).float().unsqueeze(1) / 255
                yb = torch.from_numpy(y[idx]).to(device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = model(augment(xb, rng))
                terms = losses(out.float(), yb)
                loss = sum(terms.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                for k, v in terms.items():
                    sums[k] = sums.get(k, 0) + float(v.detach())
                n += 1
            del x, y
        res = evaluate(model, val, device)
        print(f"epoch {epoch}/{a.epochs} " + " ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
              + f" {time.time() - t0:.0f} s " + json.dumps({k: {m: round(u, 4) for m, u in r.items()}
                                                             for k, r in res.items()}), flush=True)
        weights = model.export()
        np.savez(a.out.replace(".npz", f"_e{epoch}.npz"), **weights)
        np.savez(a.out, **weights)



if __name__ == "__main__":
    main()
