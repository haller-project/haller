"""
Trains the word recognizer on shards written by haller.generate.

    python -m haller.train OUT.npz SHARD.npz [SHARD.npz ...] --eval EVAL.npz

Shards are read one at a time (each fits in memory), in a new random order at
every epoch. After each epoch the model is evaluated on EVAL.npz (word images
cut from photos by SeedSigner's page detection: x, w, y as in the shards, and
a boolean `reel` field marking the strips of real photos): share of words whose
most probable BIP39 word is the right one, on simulated and on real strips.
The float weights of the best epoch on simulated strips are written to OUT.npz.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dictionary import ctc_word_scores, dictionary_tables
from .model import WordNet, encode



def load_shard(path):
    d = np.load(path)
    return d["x"], d["w"].astype(np.int64), [str(y) for y in d["y"]]



def batch(x, w, labels, idx, device, augment):
    xb = torch.from_numpy(x[idx]).to(device).float().div_(255).unsqueeze(1)
    wb = torch.from_numpy(w[idx]).to(device)
    if augment:
        # horizontal stretch 0.8-1.2 (left edge fixed), vertical shift, ink intensity
        n = len(idx)
        sx = torch.empty(n, device=device).uniform_(0.8, 1.2)
        theta = torch.zeros(n, 2, 3, device=device)
        theta[:, 0, 0] = 1 / sx
        theta[:, 0, 2] = 1 / sx - 1
        theta[:, 1, 1] = 1
        theta[:, 1, 2] = torch.empty(n, device=device).uniform_(-2, 2) / 16
        grid = F.affine_grid(theta, xb.shape, align_corners=False)
        xb = F.grid_sample(xb, grid, align_corners=False)
        wb = torch.clamp((wb.float() * sx).round().long(), 8, x.shape[2])
        xb = (xb * torch.empty(n, 1, 1, 1, device=device).uniform_(0.6, 1.2)).clamp_(0, 1)
    targets = [torch.tensor(encode(labels[i])) for i in idx]
    return xb, wb, targets



def evaluate(net, x, w, labels, words, tables, device, selection=None):
    net.eval()
    index = {wd: i for i, wd in enumerate(words)}
    selection = np.arange(len(labels)) if selection is None else selection
    good = 0
    with torch.no_grad():
        for start in range(0, len(selection), 64):
            idx = selection[start:start + 64]
            xb, wb, _ = batch(x, w, labels, idx, device, False)
            lp = net(xb)
            scores = ctc_word_scores(lp, torch.clamp(wb // 4, 1, lp.shape[2]), tables)
            best = scores.argmax(1).cpu().numpy()
            good += sum(int(b == index.get(labels[i], -1)) for b, i in zip(best, idx))
    net.train()
    return good / max(len(selection), 1)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("shards", nargs="+")
    p.add_argument("--eval", required=True)
    p.add_argument("--bip39", default="data/bip39_english.txt")
    p.add_argument("--channels", type=int, nargs=3, default=[16, 32, 64])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val", help="second validation set (x, w, y): the best epoch maximizes the mean of the "
                                 "simulated EVAL score and of this one")
    p.add_argument("--keep-epochs", action="store_true", help="also write the weights of every epoch (OUT_eN.npz)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    words = [l.strip() for l in open(a.bip39) if l.strip()]
    tables = dictionary_tables(words, device)
    ev = np.load(a.eval)
    ev_x, ev_w, ev_y, ev_real = ev["x"], ev["w"].astype(np.int64), [str(y) for y in ev["y"]], ev["reel"].astype(bool)
    if a.val:
        va = np.load(a.val)
        va_x, va_w, va_y = va["x"], va["w"].astype(np.int64), [str(y) for y in va["y"]]

    net = WordNet(a.channels, a.hidden).to(device)
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    sizes = [len(np.load(s)["w"]) for s in a.shards]
    steps = a.epochs * sum(n // a.batch for n in sizes)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best, t0 = -1.0, time.time()
    for epoch in range(a.epochs):
        loss_sum, n_batches = 0.0, 0
        for s in rng.permutation(len(a.shards)):
            x, w, labels = load_shard(a.shards[s])
            order = rng.permutation(len(labels))
            for k in range(len(order) // a.batch):
                idx = np.sort(order[k * a.batch:(k + 1) * a.batch])
                xb, wb, targets = batch(x, w, labels, idx, device, True)
                width = min(x.shape[2], (int(wb.max()) + 19) // 4 * 4)
                lp = net(xb[..., :width])
                frames = torch.clamp(wb // 4, 1, lp.shape[2])
                lengths = torch.tensor([len(t) for t in targets], device=device)
                loss = ctc(lp.permute(2, 0, 1), torch.cat(targets).to(device), frames, lengths)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
                loss_sum += loss.item()
                n_batches += 1
        sim = evaluate(net, ev_x, ev_w, ev_y, words, tables, device, np.nonzero(~ev_real)[0])
        real = evaluate(net, ev_x, ev_w, ev_y, words, tables, device, np.nonzero(ev_real)[0])
        val = evaluate(net, va_x, va_w, va_y, words, tables, device, np.arange(len(va_y))) if a.val else None
        print(f"epoch {epoch + 1}/{a.epochs} loss {loss_sum / n_batches:.4f} "
              f"words: simulated {100 * sim:.2f} % real {100 * real:.2f} %"
              + (f" val {100 * val:.2f} %" if a.val else "") + f" {time.time() - t0:.0f} s", flush=True)
        if a.keep_epochs:
            np.savez(a.out.replace(".npz", f"_e{epoch + 1}.npz"), **net.export())
        score = (sim + val) / 2 if a.val else sim
        if score >= best:  # selection on simulated photos only; real photos are a test set
            best = score
            np.savez(a.out, **net.export())
            json.dump(dict(channels=a.channels, hidden=a.hidden, epoch=epoch + 1, simulated=sim, real=real, val=val),
                      open(a.out.replace(".npz", ".json"), "w"), indent=1)



if __name__ == "__main__":
    main()
