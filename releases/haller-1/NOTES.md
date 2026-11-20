# Haller-1

First release of Haller, the handwritten BIP39 word reader behind the
handwritten seed reader of the SeedSigner fork
[`seedsigner-haller`, branch `handwritten-seed`](https://github.com/haller-project/seedsigner-haller/tree/handwritten-seed).
The following versions will be Haller-2, Haller-3, and so on; Haller-N.M
marks a change without retraining (calibration, thresholds, quantization).

**Status: proof of concept.** The reader works and is measured; the SeedSigner
user experience is not final and the feature is not proposed upstream. Every
word read must be reviewed against the paper.

## Weights

Three networks, run offline in 16-bit integers by SeedSigner's C library.
These files are byte for byte the ones embedded in the SeedSigner fork; check
them with `sha256sum -c SHA256SUMS`.

| File | Role | Parameters | SHA-256 |
|---|---|---|---|
| `haller-1-capitals.npz` | word reader, capitals expert; reads every photo | 163,557 | `2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14` |
| `haller-1-lowercase.npz` | word reader, lowercase and cursive expert; second reading, alone and blended | 163,557 | `5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509` |
| `haller-1-detector.npz` | word detector; words, line numbers and card when the page rules find no valid seed | 222,864 | `bcda59a220e52979139f44a634a0946dd9d4fe72cd4b246ee23edc6437964aa4` |

## Results

Full photos read by the SeedSigner engine; see `EVALUATION.md` for the
protocol, the intervals and the results by writing and layout.

| World | Photos | Words read correctly | Valid and exact | False valid |
|---|---|---|---|---|
| SeedSigner camera | TBD | TBD | TBD | TBD |
| Phone photos (write-a-seed) | TBD | TBD | TBD | TBD |
| Simulated | TBD | TBD | TBD | TBD |

On a Raspberry Pi Zero: TBD s for a 12-word card, TBD s for a 12-word list in
capitals.

## Release files

| File | Content |
|---|---|
| `haller-1-*.npz` | the three quantized networks |
| `SHA256SUMS` | SHA-256 of every file of the release |
| `MODEL_CARD.md` | architecture, data, intended use, limitations, evaluation |
| `RECIPE.md` | exact commands that rebuild the weights from this tag |
| `EVALUATION.md` | evaluation report |
| `LICENSE`, `NOTICE.md` | MIT license; third-party data and attributions |

## Reproducing

From the `haller-1` tag of this repository, `RECIPE.md` rebuilds the data,
the quantization and the reference vectors byte for byte, and
the training to the same level (GPU training is not bit-exact; the float
models are published in `recipes/haller-1/data`).

## Name

Haller is named after Neil M. Haller of Bellcore, author of RFC 1760, "The
S/KEY One-Time Password System" (1995). S/KEY, designed with Phil Karn and
John Walden on an idea by Leslie Lamport, already wrote a secret as words from
a 2,048-word dictionary, 11 bits per word, with spare bits as a checksum: the
same layout BIP39 would use eighteen years later.

## License

Code and weights: MIT. The training data keep their own licenses; GNHK
(CC BY 4.0) and EMNIST are credited in `NOTICE.md`, which accompanies the
weights.
