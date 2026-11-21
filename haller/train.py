"""
Trains the word recognizer on shards written by haller.generate.

    python -m haller.train OUT.npz SHARD.npz [SHARD.npz ...] --eval EVAL.npz

Shards are read one at a time (each fits in memory), in a new random order at
every epoch. After each epoch the model is quantized (haller.quantize,
calibration on 3000 words of the first shard) and evaluated by SeedSigner's C
library (haller.evaluate), as it will run on the device, on EVAL.npz (word
images cut from photos by SeedSigner's page detection: x, w, y as in the
shards, and a boolean `reel` field marking the strips of real photos): share
of words whose most probable BIP39 word is the right one, on simulated and on
real strips (EVAL.npz may have no `reel` field: every strip is then simulated,
and the real score printed is 0). The float weights of the last epoch are
written to OUT.npz; with --select best, those of the epoch with the best
simulated score (with --val, the best mean of the simulated and VAL scores).

The learning rate follows a one-cycle schedule over --epochs; --stop-after N
ends the run after epoch N with the schedule unchanged, so that epoch N of a
longer schedule is reproduced without training the remaining epochs.
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import evaluate, quantize
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



def cached_shard(path: str):
    """
    The shard decompressed once into PATH.x.npy (ink maps, read as a memory map), PATH.w.npy (widths), PATH.t.npy
    (labels encoded, padded with 0) and PATH.n.npy (label lengths) next to it, so that later epochs read it
    without decompressing.
    """
    files = [path + suffix for suffix in (".x.npy", ".w.npy", ".t.npy", ".n.npy")]
    if not all(os.path.exists(f) for f in files):
        x, w, labels = load_shard(path)
        codes = [encode(label) for label in labels]
        t = np.zeros((len(codes), max(len(c) for c in codes)), np.int64)
        for i, c in enumerate(codes):
            t[i, :len(c)] = c
        for f, a in zip(files, (x, w, t, np.array([len(c) for c in codes], np.int64))):
            np.save(f[:-4] + ".tmp.npy", a)
            os.replace(f[:-4] + ".tmp.npy", f)
    return np.load(files[0], mmap_mode="r"), np.load(files[1]), np.load(files[2]), np.load(files[3])



class ShardFeeder:
    """
    Shards copied to the device one at a time (a shard of 50 000 ink maps is 410 MB of uint8): the next one is
    read and copied by a thread while the current one is trained on.
    """
    def __init__(self, paths: list, order, device: str):
        self.paths, self.order, self.device = paths, list(order), device
        self.next, self.thread = None, None
        self._start(0)

    def _load(self, k):
        x, w, t, n = cached_shard(self.paths[self.order[k]])
        xs = torch.from_numpy(np.ascontiguousarray(x))
        pin = self.device != "cpu"
        if pin:
            xs = xs.pin_memory()
        self.next = tuple(a.to(self.device, non_blocking=pin)
                          for a in (xs, torch.from_numpy(w), torch.from_numpy(t), torch.from_numpy(n)))

    def _start(self, k):
        if k < len(self.order):
            self.thread = threading.Thread(target=self._load, args=(k,))
            self.thread.start()

    def __iter__(self):
        for k in range(len(self.order)):
            self.thread.join()
            shard, self.next = self.next, None
            self._start(k + 1)
            yield shard



def device_batch(x, w, t, n, idx, augment):
    """batch() on a shard already on the device (ShardFeeder); targets padded with 0, and their lengths."""
    xb = x[idx].float().div_(255).unsqueeze(1)
    wb = w[idx]
    if augment:
        m, device = len(idx), xb.device
        sx = torch.empty(m, device=device).uniform_(0.8, 1.2)
        theta = torch.zeros(m, 2, 3, device=device)
        theta[:, 0, 0] = 1 / sx
        theta[:, 0, 2] = 1 / sx - 1
        theta[:, 1, 1] = 1
        theta[:, 1, 2] = torch.empty(m, device=device).uniform_(-2, 2) / 16
        grid = F.affine_grid(theta, xb.shape, align_corners=False)
        xb = F.grid_sample(xb, grid, align_corners=False)
        wb = torch.clamp((wb.float() * sx).round().long(), 8, x.shape[2])
        xb = (xb * torch.empty(m, 1, 1, 1, device=device).uniform_(0.6, 1.2)).clamp_(0, 1)
    return xb, wb, t[idx], n[idx]



def load_folded(net, weights: dict):
    """Starts `net` from exported weights (batch normalization folded in): each convolution takes the folded
    weights, its batch normalization the identity and the folded bias."""
    with torch.no_grad():
        for name in ("b1", "b2", "b3", "b4", "t1", "t2"):
            conv, bn = getattr(net, name)[0], getattr(net, name)[1]
            conv.weight.copy_(torch.from_numpy(weights[name + ".weight"]))
            bn.weight.fill_(1.0)
            bn.bias.copy_(torch.from_numpy(weights[name + ".bias"]))
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - bn.eps)
        net.out.weight.copy_(torch.from_numpy(weights["out.weight"]))
        net.out.bias.copy_(torch.from_numpy(weights["out.bias"]))



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("shards", nargs="+")
    p.add_argument("--eval", required=True,
                   help="evaluation strips (x, w, y, optional boolean `reel`: strips of real photos, a test set)")
    p.add_argument("--channels", type=int, nargs=3, default=[16, 32, 64])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val", help="second validation set (x, w, y), scored at every epoch; with --select best, the "
                                 "best epoch maximizes the mean of the simulated EVAL score and of this one")
    p.add_argument("--select", choices=("last", "best"), default="last",
                   help="weights written to OUT.npz: those of the last epoch (default), or of the epoch with the best "
                        "simulated score. The simulated strips do not rank the epochs of a fine-tuning on the reading "
                        "of real photos (an epoch in the middle of the schedule can score best and read real strips "
                        "worse), so the last epoch, at the end of the schedule, is the default")
    p.add_argument("--stop-after", type=int,
                   help="stop after this epoch, the learning-rate schedule still spanning --epochs")
    p.add_argument("--eval-processes", type=int, default=1, help="processes of the evaluation by the C library")
    p.add_argument("--fast", action="store_true",
                   help="shards decompressed once (cached next to them) and copied to the GPU one at a time, the next "
                        "one prepared by a thread; the evaluation of an epoch runs while the next one trains. Same "
                        "batches and targets as without it (padded targets: the same CTC loss)")
    p.add_argument("--init", help="float weights (OUT.npz of a previous training) to start from")
    p.add_argument("--keep-epochs", action="store_true", help="also write the weights of every epoch (OUT_eN.npz)")
    p.add_argument("--deterministic", action="store_true",
                   help="bit-identical runs on the same hardware and software: deterministic GPU kernels, "
                        "CTC loss computed on the CPU (about 10 %% slower)")
    a = p.parse_args()
    last_epoch = a.epochs if a.stop_after is None else a.stop_after
    if not 1 <= last_epoch <= a.epochs:
        p.error("--stop-after must be between 1 and --epochs")

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read when cuBLAS is first used
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = not a.deterministic
    # the CUDA CTC loss has no deterministic backward pass
    ctc_device = "cpu" if a.deterministic else device
    calibration = quantize.calibration_inks(a.shards[0], 3000, 1)
    ev = np.load(a.eval)
    ev_x, ev_w, ev_y, ev_real = ev["x"], ev["w"].astype(np.int64), [str(y) for y in ev["y"]], (ev["reel"].astype(bool) if "reel" in ev.files else np.zeros(len(ev["w"]), bool))
    if a.val:
        va = np.load(a.val)
        va_x, va_w, va_y = va["x"], va["w"].astype(np.int64), [str(y) for y in va["y"]]

    net = WordNet(a.channels, a.hidden).to(device)
    if a.init:
        load_folded(net, dict(np.load(a.init)))
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    sizes = [len(np.load(s)["w"]) for s in a.shards]
    steps = a.epochs * sum(n // a.batch for n in sizes)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best, t0 = -1.0, time.time()
    pending = []

    def evaluate_epoch(epoch, weights, loss):
        q = quantize.quantize(weights, quantize.calibrate(weights, calibration))
        n = a.eval_processes
        sim = evaluate.accuracy(q, ev_x, ev_w, ev_y, n, np.nonzero(~ev_real)[0])
        real = evaluate.accuracy(q, ev_x, ev_w, ev_y, n, np.nonzero(ev_real)[0])
        val = evaluate.accuracy(q, va_x, va_w, va_y, n) if a.val else None
        print(f"epoch {epoch + 1}/{a.epochs} loss {loss:.4f} "
              f"words: simulated {100 * sim:.2f} % real {100 * real:.2f} %"
              + (f" val {100 * val:.2f} %" if a.val else "") + f" {time.time() - t0:.0f} s", flush=True)
        pending.append((epoch, weights, sim, real, val))

    def keep_best():
        nonlocal best
        for epoch, weights, sim, real, val in sorted(pending, key=lambda item: item[0]):
            if a.select == "last":
                score = float(epoch)  # epochs are kept in order: the last one is written last
            else:
                score = (sim + val) / 2 if a.val else sim
            if score >= best:  # selection on simulated photos only; real photos are a test set
                best = score
                np.savez(a.out, **weights)
                json.dump(dict(channels=a.channels, hidden=a.hidden, epoch=epoch + 1, epochs=a.epochs, lr=a.lr,
                               select=a.select, simulated=sim, real=real, val=val),
                          open(a.out.replace(".npz", ".json"), "w"), indent=1)
        pending.clear()

    evaluation = None
    for epoch in range(last_epoch):
        loss_sum, n_batches = torch.zeros((), device=device), 0
        order_shards = rng.permutation(len(a.shards))
        if a.fast:
            shards = ShardFeeder(a.shards, order_shards, device)
        else:
            shards = (load_shard(a.shards[s]) for s in order_shards)
        for shard in shards:
            count = len(shard[1]) if a.fast else len(shard[2])
            order = rng.permutation(count)
            for k in range(count // a.batch):
                idx = np.sort(order[k * a.batch:(k + 1) * a.batch])
                if a.fast:
                    xb, wb, padded, lengths = device_batch(*shard, torch.from_numpy(idx).to(device), True)
                    targets_ctc, lengths_ctc = padded.to(ctc_device), lengths.to(ctc_device)
                    width = min(shard[0].shape[2], (int(wb.max()) + 19) // 4 * 4)
                else:
                    x, w, labels = shard
                    xb, wb, targets = batch(x, w, labels, idx, device, True)
                    width = min(x.shape[2], (int(wb.max()) + 19) // 4 * 4)
                    lengths_ctc = torch.tensor([len(t) for t in targets], device=ctc_device)
                    targets_ctc = torch.cat(targets).to(ctc_device)
                lp = net(xb[..., :width])
                frames = torch.clamp(wb // 4, 1, lp.shape[2])
                loss = ctc(lp.permute(2, 0, 1).to(ctc_device), targets_ctc, frames.to(ctc_device), lengths_ctc)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sched.step()
                loss_sum += loss.detach().to(device)
                n_batches += 1
            del shard
        weights = net.export()
        if a.keep_epochs:
            np.savez(a.out.replace(".npz", f"_e{epoch + 1}.npz"), **weights)
        loss = float(loss_sum) / n_batches
        if a.fast:
            # the C library releases the GIL: the evaluation runs while the next epoch trains
            if evaluation is not None:
                evaluation.join()
                keep_best()
            evaluation = threading.Thread(target=evaluate_epoch, args=(epoch, weights, loss))
            evaluation.start()
        else:
            evaluate_epoch(epoch, weights, loss)
            keep_best()
    if evaluation is not None:
        evaluation.join()
        keep_best()


if __name__ == "__main__":
    main()
