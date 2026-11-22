"""
Keeps the reliable samples of a converted real dataset (haller.names)
or of generated handwriting (haller.emuru).

Each source image (field `group`; without it, each sample is its own image)
has one or several candidate crops. A trained model reads every candidate
without dictionary (greedy CTC decoding) and with the BIP39 dictionary:

- a candidate read exactly as its label is kept (the first one of its image),
  as is one where a single letter is read as a look-alike digit (labels have
  no digits: "0" for "o", "1" for "i"...), or one whose label is a BIP39 word
  that the dictionary decoding (as in SeedSigner) gives as the most probable;
- otherwise, the candidate read with one edit from its label is set aside for
  verification by a vision model (`verify`), which sees the ink map;
- images without such a candidate are dropped (wrong crop, unreadable scan or
  wrong label).

    python -m haller.filter read MODEL.npz OUT_PREFIX CANDIDATES.npz [...]
        -> OUT_PREFIX_exact.npz, OUT_PREFIX_near.npz (with field `read`)
    python -m haller.filter verify NEAR.npz OUT.npz --url http://127.0.0.1:8097/v1/vision

GNHK crops (haller.gnhk) have a third filter, without candidates: a sample is
dropped when the model's greedy reading has a character error rate above 0.8
against its label (another line, a word cropped upside down) or when it is
almost blank (less than 2 % of its pixels darker than 96 out of 255):

    python -m haller.filter gnhk MODEL.npz GNHK.npz OUT.npz [--mask MASK.npy]

The readings are computed on the GPU when there is one; a few samples whose
error rate is at the threshold can be kept on one device and not on the other.
`--mask` applies a selection written earlier (`--write-mask`) instead.
"""
import argparse
import base64
import io
import json
import sys
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dictionary import ctc_word_scores, dictionary_tables
from .model import ALPHABET, folded_forward

BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def network(weights: dict, x: np.ndarray) -> torch.Tensor:
    """
    Output logits (N, classes, frames) of the network for ink maps x (N, 32, 256).
    `weights` are the exported (batch normalization folded) torch tensors.
    """
    return folded_forward(weights, torch.from_numpy(x).to(DEVICE).float().div_(255).unsqueeze(1))



def greedy_read(logits: torch.Tensor, widths: np.ndarray) -> list[str]:
    """
    Text read without dictionary, as in training: best class per frame,
    repeats and blanks removed, frames limited to width / 4.
    """
    best = logits.argmax(1).cpu().numpy()
    texts = []
    for frames, w in zip(best, widths):
        frames = frames[:max(1, int(w) // 4)]
        texts.append("".join(ALPHABET[k - 1] for j, k in enumerate(frames) if k and (j == 0 or frames[j - 1] != k)))
    return texts



def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]



def dictionary_read(logits: torch.Tensor, widths: np.ndarray, words: list[str], tables) -> list[str]:
    """
    Most probable BIP39 word, as SeedSigner reads it.
    """
    frames = torch.clamp(torch.from_numpy(widths.astype(np.int64)).to(DEVICE) // 4, 1, logits.shape[2])
    lp = F.log_softmax(logits, 1)
    best = []
    for i in range(0, len(lp), 64):  # scores of all the words: (64, 2048, states) at a time
        best += ctc_word_scores(lp[i:i + 64], frames[i:i + 64], tables).argmax(1).tolist()
    return [words[k] for k in best]



def read_all(weights: dict, x: np.ndarray, w: np.ndarray, words: list[str], tables) -> tuple[list[str], list[str]]:
    """
    Greedy readings and most probable BIP39 words of ink maps x with widths w.
    """
    read, top = [], []
    with torch.no_grad():
        for start in range(0, len(w), BATCH):
            logits = network(weights, x[start:start + BATCH])
            read += greedy_read(logits, w[start:start + BATCH])
            top += dictionary_read(logits, w[start:start + BATCH], words, tables)
    return read, top



# Digits the model reads in place of capital letters, and those letters
LOOKALIKES = {"0": "o", "1": "il", "2": "z", "4": "a", "5": "s", "6": "gb", "7": "t", "8": "b", "9": "gq"}



def lookalike(read: str, label: str) -> bool:
    """
    True if `read` is `label` with one letter replaced by a look-alike digit.
    """
    diff = [(r, l) for r, l in zip(read, label) if r != l]
    return len(read) == len(label) and len(diff) == 1 and diff[0][1] in LOOKALIKES.get(diff[0][0], "")



def select(y: np.ndarray, group: np.ndarray, read: list[str], top: list[str]) -> tuple[list[int], list[int]]:
    """
    Indices of the kept candidates (exact or look-alike readings, or BIP39
    label read as the most probable BIP39 word) and of the candidates to
    verify (one-edit readings of images without a kept one), at most one per
    image.
    """
    exact, near = {}, {}
    for i, (label, g, r, t) in enumerate(zip(y, group, read, top)):
        if r == label or lookalike(r, str(label)) or t == label:
            exact.setdefault(g, i)
        elif edit_distance(r, str(label)) == 1:
            near.setdefault(g, i)
    return sorted(exact.values()), sorted(i for g, i in near.items() if g not in exact)



def save(path: str, x: np.ndarray, w: np.ndarray, y: np.ndarray, **extra):
    np.savez_compressed(path, x=x, w=w, y=y, **extra)



def read_command(a):
    weights = {k: torch.from_numpy(v).to(DEVICE) for k, v in np.load(a.model).items()}
    words = [w.strip() for w in open(a.bip39) if w.strip()]
    tables = dictionary_tables(words, DEVICE)
    exact_parts, near_parts, near_reads = [], [], []
    images = 0
    for path in a.candidates:
        d = np.load(path)
        x, w, y = d["x"], d["w"], d["y"]
        group = d["group"] if "group" in d.files else np.arange(len(y))
        read, top = read_all(weights, x, w, words, tables)
        exact, near = select(y, group, read, top)
        images += len(np.unique(group))
        exact_parts.append((x[exact], w[exact], y[exact]))
        near_parts.append((x[near], w[near], y[near]))
        near_reads += [read[i] for i in near]
    save(a.out + "_exact.npz", *(np.concatenate(c) for c in zip(*exact_parts)))
    save(a.out + "_near.npz", *(np.concatenate(c) for c in zip(*near_parts)), read=np.array(near_reads))
    print(f"{images} images: {sum(len(p[2]) for p in exact_parts)} exact, {len(near_reads)} one edit away")



def ink_map_png(x: np.ndarray, w: int) -> str:
    """
    Ink map as a dark-on-white image, enlarged, base64 PNG.
    """
    img = Image.fromarray(255 - x[:, :w]).resize((w * 3, x.shape[0] * 3), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()



def ask(url: str, images: list[str], word: str) -> list[float]:
    """
    Probability, for each image, that the vision model reads `word` on it.
    """
    question = {"type": "noul", "instructions": f"The handwritten word in this image is {word.upper()}."}
    body = json.dumps({"images": images, "questions": {"q": question}}).encode()
    request = urllib.request.Request(url, body, {"content-type": "application/json"})
    answer = json.load(urllib.request.urlopen(request, timeout=600))
    results = answer if isinstance(answer, list) else answer.get("results", [answer])
    return [r["answers"]["q"]["noul"] for r in results]



def is_local(url: str) -> bool:
    """True if `url` names this machine (localhost or a loopback address)."""
    host = urllib.parse.urlsplit(url).hostname or ""
    return host == "localhost" or host == "::1" or host.startswith("127.")



def verify_command(a):
    if not is_local(a.url):
        print(f"warning: --url {a.url} is not on this machine: every ink map of {a.near} and its word are sent to it",
              file=sys.stderr, flush=True)
    d = np.load(a.near)
    x, w, y = d["x"], d["w"], d["y"]
    keep = []
    for i in range(len(y)):
        if ask(a.url, [ink_map_png(x[i], int(w[i]))], str(y[i]))[0] >= a.threshold:
            keep.append(i)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(y)}: {len(keep)} confirmed", flush=True)
    save(a.out, x[keep], w[keep], y[keep])
    print(f"{len(keep)}/{len(y)} confirmed")



def gnhk_command(a):
    weights = {k: torch.from_numpy(v).to(DEVICE) for k, v in np.load(a.model).items()}
    d = np.load(a.samples)
    x, w, y = d["x"], d["w"], np.array([str(t) for t in d["y"]])
    read = []
    with torch.no_grad():
        for s in range(0, len(y), BATCH):
            read += greedy_read(network(weights, x[s:s + BATCH]), w[s:s + BATCH])
    cer = np.array([edit_distance(r, t) / max(1, len(t)) for r, t in zip(read, y)])
    ink = np.array([(x[i, :, :w[i]] > 96).mean() for i in range(len(y))])
    keep = (cer <= a.max_cer) & (ink >= a.min_ink)
    if a.mask:
        mask = np.unpackbits(np.load(a.mask))[:len(y)].astype(bool)
        print(f"selection of {a.mask}: {int((mask != keep).sum())} samples differ from this reading")
        keep = mask
    if a.write_mask:
        np.save(a.write_mask, np.packbits(keep))
    extra = {"page": np.array([str(p) for p in d["page"]])[keep]} if "page" in d.files else {}
    save(a.out, x[keep], w[keep], y[keep], **extra)
    print(f"{int(keep.sum())}/{len(y)} kept ({int((ink < a.min_ink).sum())} almost blank, "
          f"{int((cer > a.max_cer).sum())} with a character error rate above {a.max_cer})")



def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("read")
    r.add_argument("model")
    r.add_argument("out")
    r.add_argument("candidates", nargs="+")
    r.add_argument("--bip39", default="data/bip39_english.txt")
    v = sub.add_parser("verify")
    v.add_argument("near")
    v.add_argument("out")
    v.add_argument("--url", default="http://127.0.0.1:8097/v1/vision")
    v.add_argument("--threshold", type=float, default=0.9)
    g = sub.add_parser("gnhk")
    g.add_argument("model")
    g.add_argument("samples")
    g.add_argument("out")
    g.add_argument("--max-cer", type=float, default=0.8)
    g.add_argument("--min-ink", type=float, default=0.02)
    g.add_argument("--mask", help="packed boolean selection (numpy .npy) applied instead of the computed one")
    g.add_argument("--write-mask", help="writes the selection as a packed boolean array (numpy .npy)")
    a = p.parse_args()
    {"read": read_command, "verify": verify_command, "gnhk": gnhk_command}[a.command](a)



if __name__ == "__main__":
    main()
