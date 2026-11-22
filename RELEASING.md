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
| `test-manifest.json` | `releases/VERSION/test-manifest.json`, manifest of the private test set, when present |

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

## Data publication policy

Decided on 20/11/2026, for Haller-1 and every later version.

| Split | Content | Published |
|---|---|---|
| Training | simulated data and their generators; photos of blank seed supports taken by the maintainer (hands and background included); photos of participants whose consent explicitly covers publication | yes, all of it |
| Validation | data used to choose epochs, temperature and thresholds | yes |
| Test | SeedSigner camera photos (`eval_v3`) and their phone pairs; photos of participants of the write-a-seed campaign; images found on the web | no, while the set is in use |

**Why the test set stays private.** Once a test set is public, it ends up,
directly or through copies, in the training data of later models, ours or
other people's, and stops measuring reading on handwriting the model has not
seen. Images found on the web are not ours to redistribute in any case.

**What makes the private set checkable.**

1. **Manifest.** Each release publishes the manifest of its test set, built
   by `python -m haller.test_manifest build` (see the module): the SHA-256 of
   every file, the number of photos and their composition by world, source,
   writing, support and number of words; no image, no seed, no file name.
   The SHA-256 of the manifest is quoted in the model card and in
   `releases/VERSION/EVALUATION.md`. The set cannot change after the results
   are published without the digest changing.
2. **Evaluation code.** The code that reads the test photos and computes the
   numbers of the model card is published with the release.
3. **Third-party models.** On request (issue of this repository), we evaluate
   another model on the private set with the same code and publish the
   numbers, with the manifest digest of the set used.
4. **Replaced sets are published.** When a version uses a new test set (for
   example at Haller-2), the previous set is published, and anyone can check
   it against its manifest (`python -m haller.test_manifest verify`). Only
   files that may be redistributed are published: our own photos (with the
   agreement of the people who wrote the seeds), participants' photos whose
   consent covers publication; photos sent with an evaluation consent only,
   or with a training consent that did not mention publication, are never
   published, and images found on the web are listed by their source address,
   not redistributed. The manifest says which files are withheld.

**Participants' consents.** The write-a-seed site records with each photo the
version of the consent wording accepted. Version 1 (photos received before
20/11/2026): training, no publication. Version 2: training and publication in
the open training dataset. Only version 2 or later lets a photo enter the
published training set; the export of the site applies this filter.

**One photo, one split.** A photo is never both in a published training set
and in a private test set. The participants' part of a test set is frozen at
a date when the set is built; photos received later may go to training. A
photo of a frozen test set leaves it only when the set is replaced and
published.

## Conditions for haller-1

Decision of 2026-11-22: Haller-1 is released as it stands, after the launch
video, without a verification retraining (a fix follows if a gap is
reported; Haller-2 follows). The conditions set on 2026-11-20 (photos of the
`eval_v3` set, 300 photos of participants) are left to Haller-2. Before the
tag:

1. The model card, `releases/haller-1/` and the changelog contain no `TBD`
   (`scripts/release.sh` refuses to tag otherwise).
2. The weights are the ones of the SeedSigner fork (`weights.sha256`,
   `haller-weights.json`), each with its recipe in `recipes/`.
3. The manifest of the test set is committed as
   `releases/haller-1/test-manifest.json` and its SHA-256 quoted in the
   model card and in `releases/haller-1/EVALUATION.md`.
4. The datasets of the release (blank supports, blank cards, gpt-image-2
   pilot and v2) carry `LICENSE` (MIT), a README and `SHA256SUMS`, and hold
   no seed of a test bench.

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
   `data/cards/`, generated figures, and training photos published under the
   data publication policy above are expected. Photos of a private test set
   and photos without a consent to publication are never committed:

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
