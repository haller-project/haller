"""
CTC score of every dictionary word, for many images at once (PyTorch, used for
evaluation during training; SeedSigner has its own implementation).

State 0 of every word is a prefix state (blank or digit, looping: the line
number); then l1, b, l2, b, ..., lL, b as in standard CTC.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import encode

NEG = -1e30



def dictionary_tables(words, device):
    max_len = max(len(w) for w in words)
    states = 1 + 2 * max_len
    labels = np.zeros((len(words), states), np.int64)
    skip = np.zeros((len(words), states), bool)
    valid = np.zeros((len(words), states), bool)
    last = np.zeros((len(words), 2), np.int64)
    for i, w in enumerate(words):
        c = encode(w)
        for j, l in enumerate(c):
            labels[i, 1 + 2 * j] = l
            skip[i, 1 + 2 * j] = j > 0 and c[j - 1] != l
        valid[i, :2 * len(c) + 1] = True
        last[i] = (2 * len(c) - 1, 2 * len(c))
    t = lambda a: torch.tensor(a, device=device)
    return t(labels), t(skip), t(valid), t(last)



def ctc_word_scores(lp, frames, tables):
    """lp: (B, 37, T) log-probabilities; frames: (B,) valid lengths.
    Returns (B, M) log-likelihoods of the M words."""
    labels, skip, valid, last = tables
    b = lp.shape[0]
    m, s = labels.shape
    lpt = lp.permute(0, 2, 1)
    prefix = torch.logsumexp(torch.cat([lpt[..., :1], lpt[..., 27:]], -1), -1)
    emit = torch.gather(lpt, 2, labels.view(1, 1, -1).expand(b, lpt.shape[1], -1)).view(b, -1, m, s)
    emit[..., 0] = prefix[..., None]
    alpha = torch.full((b, m, s), NEG, device=lp.device)
    alpha[:, :, 0] = emit[:, 0, :, 0]
    alpha[:, :, 1] = emit[:, 0, :, 1]
    result = torch.full((b, m), NEG, device=lp.device)
    for t in range(lpt.shape[1]):
        if t > 0:
            one = F.pad(alpha, (1, 0), value=NEG)[..., :-1]
            two = torch.where(skip[None], F.pad(alpha, (2, 0), value=NEG)[..., :-2], torch.full_like(alpha, NEG))
            alpha = torch.logsumexp(torch.stack([alpha, one, two]), 0) + emit[:, t]
            alpha = torch.where(valid[None], alpha, torch.full_like(alpha, NEG))
        done = frames - 1 == t
        if done.any():
            result[done] = torch.logsumexp(torch.gather(alpha, 2, last[None].expand(b, m, 2)), -1)[done]
    return result
