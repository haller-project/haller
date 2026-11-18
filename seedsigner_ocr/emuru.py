"""
BIP39 words written in real people's hands by Emuru, a styled handwritten text
generator (Pippi et al., "Zero-Shot Styled Text Image Generation, but Make It
Autoregressive", CVPR 2025; model blowing-up-groundhogs/emuru, MIT license).

Styles are lines of GNHK pages (GoodNotes Handwriting in the Wild, CC BY 4.0;
Lee et al., ICDAR 2021) with their transcription: one style per page, so one
writer per style. Each style writes a batch of BIP39 words, in lowercase or in
capitals (labels stay lowercase). Each generated word then goes through the
same chain as the synthetic data: placed on a line with two other words of the
same style as neighbors (or none), photographed, normalized.

Generation needs a GPU (torch, torchvision, transformers < 5, diffusers,
einops); the raw generated words are kept per style in RAW_DIR, so an
interrupted run resumes where it stopped (with the same arguments: the words
given to each style depend on SEED, --styles and --per-style).

    python -m seedsigner_ocr.emuru OUT.npz GNHK_SPLIT_DIR RAW_DIR SEED \
        --bip39 data/bip39_english.txt --styles 500 --per-style 60 --batch 12

Output: x (N, 32, 256) uint8 ink maps padded with zeros, w (N,) widths,
y (N,) labels, style (N,) style index (GNHK page).
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH, locate_word, remove_long_runs
from .synth import compose_line, crop_ink, legible, photograph

STYLE_HEIGHT = 64          # Emuru works on lines 64 px high
STYLE_WIDTH = (160, 512)   # style line width range, in pixels at that height
MAX_NEW_TOKENS = 112       # one token = 8 px of width



# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def clean(gray: Image.Image) -> Image.Image:
    """Paper made white and ink dark: divide by the background (large blur),
    stretch the contrast."""
    a = np.asarray(gray, np.float32)
    background = np.asarray(gray.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(12)), np.float32)
    a = np.clip(a / np.maximum(background, 1), 0, 1)
    low = np.percentile(a, 1)
    a = np.clip((a - low) / max(0.85 - low, 0.05), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))



def is_word(text: str) -> bool:
    """Letters only, punctuation aside (no digits, symbols or GNHK special
    tokens such as %math%)."""
    letters = text.strip(".,;:!?'\"")
    return letters.isascii() and letters.isalpha()



def style_line(split_dir: str, name: str, rng: np.random.Generator):
    """(image 64 px high, transcription) of one handwritten line of the page,
    or None: words of a line, letters only (punctuation aside), roughly
    horizontal, cut at a word boundary to fit STYLE_WIDTH, with a clean
    background."""
    items = json.load(open(os.path.join(split_dir, name + ".json")))
    lines = {}
    for item in items:
        lines.setdefault(item.get("line_idx"), []).append(item)
    candidates = []
    for words in lines.values():
        if len(words) < 2 or not all(is_word(w["text"]) for w in words):
            continue
        boxes = []
        for w in words:
            p = w["polygon"]
            xs, ys = [p[f"x{i}"] for i in range(4)], [p[f"y{i}"] for i in range(4)]
            boxes.append((min(xs), min(ys), max(xs), max(ys), w["text"]))
        boxes.sort()
        heights = np.array([b[3] - b[1] for b in boxes])
        top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
        if bottom - top > 1.8 * np.median(heights) or (heights < 12).any():
            continue  # curved or tilted line, or tiny writing
        candidates.append((boxes, top, bottom))
    if not candidates:
        return None
    boxes, top, bottom = candidates[rng.integers(len(candidates))]
    scale = STYLE_HEIGHT / (bottom - top)
    left = boxes[0][0]
    kept = []
    for b in boxes:  # words from the left, up to the maximum style width
        if (b[2] - left) * scale > STYLE_WIDTH[1]:
            break
        kept.append(b)
    if len(kept) < 2 or sum(len(b[4]) for b in kept) < 8 or (kept[-1][2] - left) * scale < STYLE_WIDTH[0]:
        return None
    page = ImageOps.exif_transpose(Image.open(os.path.join(split_dir, name + ".jpg"))).convert("L")
    pad = 0.1 * (bottom - top)
    crop = page.crop((max(0, int(left - pad)), int(top), min(page.width, int(kept[-1][2] + pad)), int(bottom)))
    a = np.array(clean(crop.resize((max(8, int(crop.width * scale)), STYLE_HEIGHT), Image.BILINEAR)))
    ink = a < 128
    a[ink & ~remove_long_runs(ink, a.shape[1] // 3)] = 255   # ruled lines
    ink = a < 128
    margin = STYLE_HEIGHT // 8
    if not 0.03 <= ink.mean() <= 0.2 or ink[margin:-margin].sum() < 0.85 * ink.sum():
        return None  # blank, noisy background, or pieces of the neighboring lines
    return Image.fromarray(a), " ".join(b[4] for b in kept)



def pick_styles(split_dir: str, count: int, rng: np.random.Generator, pages: list = None) -> list:
    """[(page name, image, transcription)], one line per page, among `pages`
    (default: all the pages of the split)."""
    pages = sorted(pages or (n[:-5] for n in os.listdir(split_dir) if n.endswith(".json")))
    styles = []
    for i in rng.permutation(len(pages)):
        found = style_line(split_dir, pages[i], rng)
        if found is not None:
            styles.append((pages[i], *found))
        if len(styles) == count:
            break
    return styles



# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class Emuru:
    """Batched generation of several texts in one style, with the encoder run
    once and the decoder's key/value cache (the reference generate_batch
    recomputes everything at each step)."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained("blowing-up-groundhogs/emuru", trust_remote_code=True)
        self.model.T5.to(device, torch.bfloat16)
        self.model.to(device)
        self.model.eval()
        self.device = device


    def write(self, style: Image.Image, style_text: str, texts: list) -> list:
        """One image (grayscale, 64 px high) per text, None where generation
        did not stop."""
        torch, m = self.torch, self.model
        from transformers.modeling_outputs import BaseModelOutput
        from torchvision.transforms import functional as F
        width = -(-style.width // 8) * 8
        img = Image.new("RGB", (width, STYLE_HEIGHT), "white")
        img.paste(style.convert("RGB"))
        x = F.normalize(F.to_tensor(img), [0.5], [0.5])[None].to(self.device)
        b = len(texts)
        with torch.inference_mode():
            z = m.query_rearrange(m.vae.encode(x).latent_dist.sample())          # (1, L, 8)
            tok = m.tokenizer([style_text + " " + t for t in texts], return_tensors="pt", padding=True).to(self.device)
            enc = m.T5.encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask).last_hidden_state
            enc = BaseModelOutput(last_hidden_state=enc)
            sos = m.sos.weight[None].expand(b, 1, -1)
            embeds = torch.cat([sos, m.vae_to_t5(z).expand(b, -1, -1)], 1).to(torch.bfloat16)
            pad = m.padding_token.float()
            canvas, past = [], None
            run = torch.zeros(b, dtype=torch.int, device=self.device)
            stops = torch.full((b,), -1, dtype=torch.int, device=self.device)
            for step in range(MAX_NEW_TOKENS):
                out = m.T5(encoder_outputs=enc, attention_mask=tok.attention_mask, decoder_inputs_embeds=embeds,
                           past_key_values=past, use_cache=True)
                past = out.past_key_values
                latent = m.t5_to_vae(out.logits[:, -1:].float())                 # (b, 1, 8)
                canvas.append(latent)
                is_pad = torch.nn.functional.cosine_similarity(latent[:, 0], pad, dim=-1) > m.padding_token_threshold
                run = torch.where(is_pad, run + 1, 0)
                stops = torch.where((stops < 0) & (run >= 10), step - 9, stops)
                if (stops >= 0).all():
                    break
                embeds = m.vae_to_t5(latent).to(torch.bfloat16)
            gen = torch.cat(canvas, 1)
            context = min(z.size(1), 8)  # a few style columns so the decoder starts in context
            full = torch.cat([z[:, -context:].expand(b, -1, -1), gen], 1)
            images = torch.clamp(m.vae.decode(m.z_rearrange(full)).sample, -1, 1)
            images = ((images + 1) * 127.5).mean(1).byte().cpu().numpy()
        result = []
        for i in range(b):
            s = int(stops[i])
            result.append(None if s <= 0 else images[i, :, 8 * context: 8 * (context + s)])
        return result



def is_written(a: np.ndarray) -> bool:
    """A generated word, not a blank or a smudge."""
    ink = a < 128
    if ink.mean() < 0.01 or ink.mean() > 0.5 or a.shape[1] < 16:
        return False
    cols = np.nonzero(ink.any(0))[0]
    return cols[-1] - cols[0] >= 12



def generate(styles: list, words: list, raw_dir: str, per_style: int, batch: int, rng: np.random.Generator,
             cases: tuple = ("lower", "upper")):
    """Writes RAW_DIR/<style index>.npz: x (n, 64, W) generated words padded
    with white, w (n,) widths, text (n,) as written (case kept)."""
    os.makedirs(raw_dir, exist_ok=True)
    # every (word, case) pair about the same number of times, spread over the styles
    pairs = [(w, case) for w in words for case in cases]
    order = np.concatenate([rng.permutation(len(pairs)) for _ in range(-(-len(styles) * per_style // len(pairs)))])
    model = None
    t0, done = time.time(), 0
    for s, (page, image, transcription) in enumerate(styles):
        chunk = order[s * per_style:(s + 1) * per_style]
        path = os.path.join(raw_dir, f"{s}.npz")
        if os.path.exists(path):
            continue
        model = model or Emuru()
        texts = [pairs[i][0].upper() if pairs[i][1] == "upper" else pairs[i][0] for i in chunk]
        texts.sort(key=len)  # similar lengths stop together
        out, kept = [], []
        # The batch of median length first: if half of it fails, the other
        # batches of the style are skipped (generation that does not stop
        # costs the most)
        starts = list(range(0, len(texts), batch))
        starts.insert(0, starts.pop(len(starts) // 2))
        for n, i in enumerate(starts):
            for t, a in zip(texts[i:i + batch], model.write(image, transcription, texts[i:i + batch])):
                if a is not None and is_written(a):
                    out.append(a)
                    kept.append(t)
            done += len(texts[i:i + batch])
            if n == 0 and 2 * len(out) < len(texts[i:i + batch]):
                break
        x = np.full((len(out), STYLE_HEIGHT, max([a.shape[1] for a in out] + [1])), 255, np.uint8)
        for i, a in enumerate(out):
            x[i, :, :a.shape[1]] = a
        np.savez_compressed(path + ".tmp.npz", x=x, w=np.array([a.shape[1] for a in out], np.int16),
                            text=np.array(kept), page=page, style_text=transcription, style=np.asarray(image))
        os.replace(path + ".tmp.npz", path)  # a run killed while writing leaves no truncated file
        print(f"style {s + 1}/{len(styles)} {page}: {len(out)}/{len(texts)} kept, "
              f"{done / (time.time() - t0):.1f} words/s", flush=True)



# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def to_line(a: np.ndarray) -> Image.Image:
    """Generated word as dark ink on white paper, cropped to its ink."""
    a = a.astype(np.float32)
    paper, ink = np.percentile(a, 90), np.percentile(a, 1)
    a = np.clip((a - ink) / max(paper - ink, 1), 0, 1) * 255
    a[a > 220] = 255
    return crop_ink(Image.fromarray(a.astype(np.uint8)), 8)



def compose(raw_dir: str, count: int, rng: np.random.Generator):
    samples, labels, style_ids = [], [], []
    for s in range(count):
        d = np.load(os.path.join(raw_dir, f"{s}.npz"))
        words = [to_line(a[:, :w]) for a, w in zip(d["x"], d["w"])]
        for i, (target, text) in enumerate(zip(words, d["text"])):
            size = int(rng.uniform(0.55, 0.75) * STYLE_HEIGHT)
            neighbors = []
            if len(words) > 2 and rng.random() < 0.5:
                neighbors = [words[j] for j in rng.choice([j for j in range(len(words)) if j != i], 2, replace=False)]
            line, (t0, t1) = compose_line(target, neighbors, size, rng)
            photo, scale = photograph(line, size, rng)
            found = locate_word(photo)
            if found is None or not legible(found[0]):
                continue
            x, (_, top, _, bottom) = found
            t0, t1 = t0 * scale, t1 * scale
            if min(bottom, t1) - max(top, t0) < 0.7 * (t1 - t0) or bottom - top > 1.6 * (t1 - t0):
                continue  # the normalized word is not the target
            samples.append(x)
            labels.append(str(text).lower())
            style_ids.append(s)
    return samples, labels, style_ids



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("split_dir")
    p.add_argument("raw_dir")
    p.add_argument("seed", type=int)
    p.add_argument("--bip39", required=True)
    p.add_argument("--styles", type=int, default=500)
    p.add_argument("--per-style", type=int, default=60)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--lower", action="store_true", help="lowercase words only")
    p.add_argument("--pages", help="file listing the GNHK pages to take the styles from, one name per line")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    import torch
    torch.manual_seed(a.seed)
    words = [w.strip() for w in open(a.bip39) if w.strip()]
    pages = open(a.pages).read().split() if a.pages else None
    styles = pick_styles(a.split_dir, a.styles, rng, pages)
    print(len(styles), "styles", flush=True)
    generate(styles, words, a.raw_dir, a.per_style, a.batch, rng, ("lower",) if a.lower else ("lower", "upper"))
    if a.generate_only:
        return
    samples, labels, style_ids = compose(a.raw_dir, len(styles), np.random.default_rng(a.seed + 1))
    x = np.zeros((len(samples), INPUT_HEIGHT, MAX_INPUT_WIDTH), np.uint8)
    for i, s in enumerate(samples):
        x[i, :, :s.shape[1]] = s
    np.savez_compressed(a.out, x=x, w=np.array([s.shape[1] for s in samples], np.int16), y=np.array(labels),
                        style=np.array(style_ids, np.int16))
    print(len(samples), "samples")



if __name__ == "__main__":
    main()
