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

## Conditions for haller-1

Haller-1 is released when all of the following hold, not on a fixed date:

1. The evaluation photos taken with a SeedSigner camera (`eval_v3`),
   including the pairs of the same page photographed with a phone and with
   the SeedSigner camera, are taken and evaluated.
2. Enough photos from participants of the write-a-seed campaign are collected
   for a solid figure: target 300 photos.
3. The exact recipe of the weights is complete: `recipes/haller-1` rebuilds
   the published weights from public inputs, and every step that cannot be
   reproduced bit for bit is stated with its published intermediate files.
4. The reading times are measured on a Raspberry Pi Zero.
5. The model card, `releases/haller-1/` and the changelog contain no `TBD`
   (`scripts/release.sh` refuses to tag otherwise).

## Making the repository public

The repository stays private until the release day; the release is then made
from the public repository. Before changing its visibility, check every
branch and tag of the remote, not only `master`, in a fresh mirror clone
(`git clone --mirror`). Branches that are not meant to be published are
deleted from the remote first.

1. **Identity.** Every commit has the project identity as author and
   committer, and no other name or e-mail address appears in commit messages,
   trailers (`Signed-off-by`, `Co-authored-by`) or files:

   ```bash
   git log --all --format='%an <%ae>%n%cn <%ce>' | sort -u
   git log --all --format='%B' | grep -iE 'signed-off|co-authored|@' | sort -u
   ```

2. **Dates.** Author and committer dates are in UTC (`+0000`) for every
   commit and the tagger date of every tag:

   ```bash
   git log --all --format='%ai%n%ci' | grep -v '+0000$'                   # must print nothing
   git for-each-ref refs/tags --format='%(taggerdate:iso)' | grep -v '+0000$'
   ```

3. **No machine paths or host names.** No home directory, user name or host
   name of the build machines in any file or message of the history. Keep the
   list of names to search for outside the repository:

   ```bash
   git grep -nIE -f NAMES.txt $(git rev-list --all) | head
   git log --all --format='%B' | grep -E -f NAMES.txt
   ```

4. **No secret.** No key, token, password or private configuration in any
   blob of the history (scan with a secret scanner such as `gitleaks detect
   --log-opts=--all`, and look at every file whose name suggests a
   configuration or credential).
5. **No photo of a person, no private photo.** List every image and archive
   blob of the history and account for each one: only the card templates of
   `data/cards/` and generated figures are expected. Photos of the evaluation
   sets and of users are never committed:

   ```bash
   git rev-list --all --objects | grep -iE '\.(jpe?g|png|gif|webp|heic|npz|npy|zip|tar|gz)$' | sort -k2 -u
   ```

6. **Only redistributable data.** Every data file of the history is listed in
   `NOTICE.md` with a license that allows its redistribution. In particular no
   file derived from EMNIST (`recipes/haller-1/data/haller-1-eval-simulated.npz`
   was committed before it was replaced by its regeneration script, and must
   be removed from the history), no font file, no raw GNHK page, no
   third-party seed card template other than SeedSigner's.
7. **Internal links.** Every relative link of the Markdown files resolves in
   the repository; no link points to a private repository, a private issue or
   a local file.
8. **README.** The README describes the public state: how to install, which
   data to download, which results are measured on which photos, the license,
   and the link to the release.

A history that fails 1, 2, 3, 5 or 6 is rewritten (for example with `git
filter-repo`) before the repository becomes public, which changes every
commit identifier after the first rewritten commit: the release tag is created
after this rewrite, never before.

## Pushing branches

Branch pushes never carry tags: push explicit branch refspecs
(`git push origin BRANCH:master`), never `--tags`, `--follow-tags` or
`--mirror`, and keep `push.followTags` unset. A tag `haller-*` reaches a
remote only through `scripts/publish.sh`.
