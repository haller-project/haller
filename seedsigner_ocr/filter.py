"""
Keeps the reliable samples of a converted real dataset (seedsigner_ocr.names)
or of generated handwriting (seedsigner_ocr.emuru).

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

    python -m seedsigner_ocr.filter read MODEL.npz OUT_PREFIX CANDIDATES.npz [...]
        -> OUT_PREFIX_exact.npz, OUT_PREFIX_near.npz (with field `read`)
    python -m seedsigner_ocr.filter verify NEAR.npz OUT.npz --url http://127.0.0.1:8097/v1/vision
"""
import argparse
import base64
import io
import json
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .dictionary import ctc_word_scores, dictionary_tables
from .model import ALPHABET

BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def network(weights: dict, x: np.ndarray) -> torch.Tensor:
    """
    Output logits (N, classes, frames) of the network for ink maps x (N, 32, 256).
    `weights` are the exported (batch normalization folded) torch tensors.
    """
    a = torch.from_numpy(x).to(DEVICE).float().div_(255).unsqueeze(1)
    for i, (ph, pw) in ((1, (2, 2)), (2, (2, 2)), (3, (2, 1)), (4, (2, 1))):
        a = F.max_pool2d(F.relu(F.conv2d(a, weights[f"b{i}.weight"], weights[f"b{i}.bias"], padding=1)), (ph, pw))
    a = a.flatten(1, 2)
    a = F.relu(F.conv1d(a, weights["t1.weight"], weights["t1.bias"], padding=1))
    a = F.relu(F.conv1d(a, weights["t2.weight"], weights["t2.bias"], padding=2, dilation=2))
    return F.conv1d(a, weights["out.weight"], weights["out.bias"])



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



def verify_command(a):
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
    a = p.parse_args()
    read_command(a) if a.command == "read" else verify_command(a)



if __name__ == "__main__":
    main()
