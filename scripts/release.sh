#!/usr/bin/env bash
# Prepares a Haller release locally: checks, release files, annotated tag.
# Publishes nothing: pushing the tag and creating the GitHub release are done
# by scripts/publish.sh, a separate step (see RELEASING.md).
#
#   scripts/release.sh VERSION --weights DIR [--commit REF] [--out DIR]
#                      [--dry-run] [--allow-tbd]
#
#   VERSION     haller-N or haller-N.M
#   --weights   directory holding the weights of the release (for example the
#               resources/handwriting directory of the SeedSigner checkout);
#               each file must match releases/VERSION/weights.sha256
#   --commit    commit to tag (default: the commit of the remote branch
#               origin/master whose tree is the tree of HEAD, see RELEASING.md)
#   --out       release files directory (default: dist/VERSION)
#   --dry-run   every check and the release files, in a temporary directory
#               unless --out is given; no tag. Placeholders and a missing
#               remote commit are reported as warnings.
#   --allow-tbd accept TBD placeholders (dry runs only)
set -euo pipefail

die() { echo "release: error: $*" >&2; exit 1; }
warn() { echo "release: warning: $*" >&2; }
info() { echo "release: $*"; }

IDENTITY_NAME="pizza-bitcoiner"
IDENTITY_EMAIL="344241609+pizza-bitcoiner@users.noreply.github.com"
REMOTE_BRANCH="origin/master"

VERSION="" WEIGHTS="" COMMIT="" OUT="" DRY_RUN=0 ALLOW_TBD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --weights) WEIGHTS="${2:?}"; shift 2 ;;
        --commit) COMMIT="${2:?}"; shift 2 ;;
        --out) OUT="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --allow-tbd) ALLOW_TBD=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        -*) die "unknown option $1" ;;
        *) [ -z "$VERSION" ] || die "one version only"; VERSION="$1"; shift ;;
    esac
done
[ -n "$VERSION" ] || die "missing VERSION"
[ -n "$WEIGHTS" ] || die "missing --weights DIR"
[[ "$VERSION" =~ ^haller-([0-9]+)(\.([0-9]+))?$ ]] || die "VERSION must be haller-N or haller-N.M"
MAJOR="haller-${BASH_REMATCH[1]}"
TITLE="Haller-${VERSION#haller-}"
[ "$ALLOW_TBD" = 0 ] || [ "$DRY_RUN" = 1 ] || die "--allow-tbd is for dry runs only"

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
REL="releases/$VERSION"

# 1. Working tree: clean, no untracked file
[ -z "$(git status --porcelain)" ] || die "the working tree is not clean (git status)"

# 2. Release documents
for f in "$REL/NOTES.md" "$REL/EVALUATION.md" "$REL/weights.sha256" MODEL_CARD.md CHANGELOG.md LICENSE NOTICE.md; do
    [ -f "$f" ] || die "missing $f"
done
if [ -f "$REL/RECIPE.md" ]; then RECIPE="$REL/RECIPE.md"; else RECIPE="recipes/$MAJOR/README.md"; fi
[ -f "$RECIPE" ] || die "missing recipe ($REL/RECIPE.md or recipes/$MAJOR/README.md)"
grep -q "^## $VERSION\b" CHANGELOG.md || die "CHANGELOG.md has no '## $VERSION' section"
grep -q "\`$VERSION\`" MODEL_CARD.md || die "MODEL_CARD.md does not name \`$VERSION\`"

TBD=$(grep -n -w "TBD" "$REL/NOTES.md" "$REL/EVALUATION.md" MODEL_CARD.md CHANGELOG.md "$RECIPE" || true)
if [ -n "$TBD" ]; then
    n=$(printf '%s\n' "$TBD" | wc -l)
    if [ "$ALLOW_TBD" = 1 ]; then warn "$n lines with TBD placeholders (accepted by --allow-tbd)"
    else head -20 <<< "$TBD" >&2; die "$n lines with TBD placeholders"; fi
fi

# 3. Tag
if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
    die "tag $VERSION already exists"
fi

# 4. Commit to tag: same tree as HEAD, history in UTC under the project identity
HEAD_TREE=$(git rev-parse 'HEAD^{tree}')
if [ -z "$COMMIT" ]; then
    git rev-parse -q --verify "$REMOTE_BRANCH" >/dev/null || die "no $REMOTE_BRANCH (git fetch origin)"
    COMMIT=$(git log --format='%H %T' "$REMOTE_BRANCH" | awk -v t="$HEAD_TREE" '$2 == t && !found {print $1; found = 1}')
    if [ -z "$COMMIT" ]; then
        if [ "$DRY_RUN" = 1 ]; then
            warn "no commit of $REMOTE_BRANCH has the tree of HEAD: the release commit is not pushed yet; HEAD used for this dry run"
            COMMIT=$(git rev-parse HEAD)
        else
            die "no commit of $REMOTE_BRANCH has the tree of HEAD: push the release commit first (RELEASING.md, step 2)"
        fi
    fi
fi
COMMIT=$(git rev-parse --verify "$COMMIT^{commit}")
[ "$(git rev-parse "$COMMIT^{tree}")" = "$HEAD_TREE" ] || die "commit $COMMIT does not have the tree of HEAD"
BAD=$(git log --format='%H %an <%ae> %cn <%ce> %ad %cd' --date=raw "$COMMIT" \
      | awk -v id="$IDENTITY_NAME <$IDENTITY_EMAIL>" '{
            line = $0; sub(/^[^ ]+ /, "", line)
            if (index(line, id " " id " ") != 1 || $(NF-2) != "+0000" || $NF != "+0000") print $1 }')
if [ -n "$BAD" ]; then
    if [ "$DRY_RUN" = 1 ]; then warn "commits outside the project identity or not in UTC (first: $(echo "$BAD" | head -1 | cut -c1-12)); a pushed commit is required"
    else die "commits outside the project identity or not in UTC, first: $(echo "$BAD" | head -1)"; fi
fi

# 5. Weights
[ -d "$WEIGHTS" ] || die "no directory $WEIGHTS"
mapfile -t NAMES < <(awk '{print $2}' "$REL/weights.sha256")
[ "${#NAMES[@]}" -gt 0 ] || die "$REL/weights.sha256 is empty"
for name in "${NAMES[@]}"; do
    [[ "$name" == "$MAJOR-"*.npz ]] || die "weight $name does not start with $MAJOR-"
    [ -f "$WEIGHTS/$name" ] || die "missing weight $WEIGHTS/$name"
done
(cd "$WEIGHTS" && sha256sum --quiet -c "$ROOT/$REL/weights.sha256") || die "weights differ from $REL/weights.sha256"
for name in "${NAMES[@]}"; do
    grep -q "$(awk -v n="$name" '$2 == n {print $1}' "$REL/weights.sha256")" MODEL_CARD.md "$REL/NOTES.md" \
        || die "the digest of $name is missing from MODEL_CARD.md or NOTES.md"
done
info "weights: ${#NAMES[@]} files match $REL/weights.sha256"

# 6. Release files, from the tagged commit (not from the working tree)
if [ -z "$OUT" ]; then
    if [ "$DRY_RUN" = 1 ]; then OUT=$(mktemp -d); else OUT="dist/$VERSION"; fi
fi
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then die "$OUT is not empty"; fi
mkdir -p "$OUT"
for name in "${NAMES[@]}"; do cp "$WEIGHTS/$name" "$OUT/$name"; done
git show "$COMMIT:MODEL_CARD.md" > "$OUT/MODEL_CARD.md"
git show "$COMMIT:$RECIPE" > "$OUT/RECIPE.md"
git show "$COMMIT:$REL/EVALUATION.md" > "$OUT/EVALUATION.md"
git show "$COMMIT:$REL/NOTES.md" > "$OUT/NOTES.md"
git show "$COMMIT:LICENSE" > "$OUT/LICENSE"
git show "$COMMIT:NOTICE.md" > "$OUT/NOTICE.md"
(cd "$OUT" && sha256sum -- * | grep -v ' SHA256SUMS$' > SHA256SUMS && sha256sum --quiet -c SHA256SUMS)
info "release files in $OUT:"
(cd "$OUT" && ls -l | tail -n +2 | awk '{printf "  %10s  %s\n", $5, $9}')

# 7. Annotated tag (local)
MESSAGE=$(mktemp)
{
    echo "$TITLE"
    echo
    echo "Weights (SHA-256):"
    awk '{printf "  %s  %s\n", $1, $2}' "$REL/weights.sha256"
    echo
    echo "Model card, recipe and evaluation report: release files of $VERSION."
} > "$MESSAGE"
if [ "$DRY_RUN" = 1 ]; then
    info "dry run: tag $VERSION not created; it would point to $COMMIT with the message:"
    sed 's/^/  | /' "$MESSAGE"
    rm -f "$MESSAGE"
    exit 0
fi
TZ=UTC GIT_COMMITTER_NAME="$IDENTITY_NAME" GIT_COMMITTER_EMAIL="$IDENTITY_EMAIL" \
    GIT_COMMITTER_DATE="$(date -u +%s) +0000" \
    git -c tag.gpgSign=false tag -a "$VERSION" "$COMMIT" -F "$MESSAGE"
rm -f "$MESSAGE"
info "tag $VERSION created locally on $COMMIT; nothing is published."
info "next step, only with the approval of the maintainer: scripts/publish.sh $VERSION (RELEASING.md, step 4)"
