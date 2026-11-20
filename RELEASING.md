# Releasing a Haller version

A version of the weights is an annotated git tag of this repository
(`haller-1`, `haller-1.1`, `haller-2`…) and a GitHub release with the same
name. The tag fixes the exact code that produced the weights; the release
carries the files that are not in the git history.

## Numbering

- `haller-N`: new training of the weights.
- `haller-N.M`: change without retraining (calibration, thresholds, corrected
  quantization). The weight files keep the `haller-N-` prefix when they are
  unchanged; a changed file gets a new name only with a new `N`.
- A published tag is never moved or deleted. A mistake is fixed by a new
  version.
- The published weights are always the best ones available; the recipe
  reproduces them, not the reverse.

## Release files

| File | Source |
|---|---|
| `haller-N-*.npz` | the weights, byte for byte those embedded in SeedSigner; listed in `releases/VERSION/weights.sha256` |
| `SHA256SUMS` | SHA-256 of every release file |
| `MODEL_CARD.md` | `MODEL_CARD.md` at the tagged commit |
| `RECIPE.md` | `releases/VERSION/RECIPE.md` if present, else `recipes/haller-N/README.md` |
| `EVALUATION.md` | `releases/VERSION/EVALUATION.md` |
| `NOTES.md` | `releases/VERSION/NOTES.md`, also the text of the GitHub release |
| `LICENSE`, `NOTICE.md` | license and third-party notices; `NOTICE.md` accompanies the weights |

Weights are never committed outside `recipes/` data, and never under
`releases/`.

## Steps

1. **Prepare** in a commit: `releases/VERSION/` (notes, evaluation report,
   `weights.sha256`), `MODEL_CARD.md`, the `## VERSION (date)` section of
   `CHANGELOG.md`, the recipe. No `TBD` may remain. In the SeedSigner fork,
   update `resources/handwriting/haller-weights.json` (version and digests)
   and run its tests.
2. **Push the release commit** to `master` of the repository with the usual
   branch push. Commits are published with author and committer dates in UTC,
   so the published commit can differ from the local one while having the
   same tree. Then `git fetch origin`.
3. **Tag locally**:

   ```bash
   scripts/release.sh VERSION --weights SEEDSIGNER/src/seedsigner/resources/handwriting --dry-run
   scripts/release.sh VERSION --weights SEEDSIGNER/src/seedsigner/resources/handwriting
   ```

   The script checks a clean working tree, the release documents, the absence
   of `TBD`, that the tag does not exist, that the weights match
   `weights.sha256` and are quoted in the model card and the notes, and that
   the commit to tag (the commit of `origin/master` with the tree of `HEAD`)
   and its history are under the project identity with UTC dates. It writes
   the release files to `dist/VERSION/` and creates the annotated tag. It
   publishes nothing.
4. **Publish, only with the maintainer's approval**:

   ```bash
   HALLER_PUBLISH_APPROVED=VERSION scripts/publish.sh VERSION
   ```

   It pushes the tag alone (`refs/tags/VERSION`) and creates the GitHub
   release from `dist/VERSION/`.
5. **After publication**: in the SeedSigner fork, set `published` to `true`
   in `haller-weights.json`; move the `Unreleased` entries of the changelog.

## Pushing branches

Branch pushes never carry tags: push explicit branch refspecs
(`git push origin BRANCH:master`), never `--tags`, `--follow-tags` or
`--mirror`, and keep `push.followTags` unset. A tag `haller-*` reaches a
remote only through `scripts/publish.sh`.
