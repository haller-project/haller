# Changelog

Versions of the Haller weights. `haller-N` is a new training of the weights;
`haller-N.M` is a change without retraining (calibration, thresholds,
corrected quantization). Each version is an annotated git tag and a release
carrying the weights, their SHA-256 digests, the model card, the recipe and
the evaluation report (see `RELEASING.md`).

## Unreleased

## haller-1 (TBD)

First release.

- Word reader, capitals expert (`haller-1-capitals.npz`, 163,557 parameters):
  synthetic words (287 handwriting fonts, EMNIST letters) and GNHK words.
- Word reader, lowercase and cursive expert (`haller-1-lowercase.npz`, same
  architecture): adds 8 shards of cursive-font words, filtered GNHK words and
  BIP39 words generated in GNHK hands by Emuru.
- Word detector (`haller-1-detector.npz`, 223,531 parameters, 11 output
  maps): words, line numbers, card area, corners and kernels, number-word
  links and partial words; trained on simulated cards and lists, simulated
  handwriting composed on photos of blank supports and cards, and GNHK pages
  (`recipes/haller-1-detector`); SeedSigner finds the words of every photo
  with it. It replaced an earlier 8-map detector (`bcda59a2…`), whose recipe
  is in the history of `recipes/haller-1-detector`.
- Fingerprint reader (`haller-1-fingerprint.npz`, architecture of the word
  readers): the capitals expert fine-tuned on synthetic lines of which half
  are master key fingerprints (`recipes/haller-1-fingerprint`); SeedSigner
  reports a seed as verified only when the fingerprint read matches the one
  of the words.
- Quantization to 16-bit weights and 12-bit activations; reference vectors
  checked bit for bit by SeedSigner's C library.
- Recipe of the exact weights in `recipes/haller-1`; the simulated evaluation
  strips, derived from EMNIST, are regenerated from the official archive
  (`recipes/haller-1/evalsim`) instead of being stored.
- Weights under the MIT License, with the third-party notices of `NOTICE.md`.
