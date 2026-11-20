#!/usr/bin/env bash
# Publishes a release prepared by scripts/release.sh: pushes its tag and
# creates the GitHub release with the release files. Run it only with the
# approval of the maintainer (RELEASING.md, step 4); it refuses to run unless
# HALLER_PUBLISH_APPROVED is set to the version.
#
#   HALLER_PUBLISH_APPROVED=VERSION scripts/publish.sh VERSION [--repo OWNER/NAME] [--dir DIR]
#
#   --repo   GitHub repository (default: haller-project/haller)
#   --dir    release files (default: dist/VERSION)
# The GitHub CLI must be logged in with the project account.
set -euo pipefail

die() { echo "publish: error: $*" >&2; exit 1; }

VERSION="" REPO="haller-project/haller" DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="${2:?}"; shift 2 ;;
        --dir) DIR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        -*) die "unknown option $1" ;;
        *) VERSION="$1"; shift ;;
    esac
done
[[ "$VERSION" =~ ^haller-[0-9]+(\.[0-9]+)?$ ]] || die "VERSION must be haller-N or haller-N.M"
[ "${HALLER_PUBLISH_APPROVED:-}" = "$VERSION" ] || die "set HALLER_PUBLISH_APPROVED=$VERSION once the maintainer has approved the publication"
cd "$(git rev-parse --show-toplevel)"
DIR="${DIR:-dist/$VERSION}"
TITLE="Haller-${VERSION#haller-}"

[ "$(git cat-file -t "refs/tags/$VERSION" 2>/dev/null)" = tag ] || die "no annotated tag $VERSION (scripts/release.sh)"
COMMIT=$(git rev-parse "$VERSION^{commit}")
git fetch -q origin master
git merge-base --is-ancestor "$COMMIT" origin/master || die "the commit of $VERSION is not on origin/master"
[ -f "$DIR/SHA256SUMS" ] && [ -f "$DIR/NOTES.md" ] || die "no release files in $DIR"
(cd "$DIR" && sha256sum --quiet -c SHA256SUMS) || die "release files differ from $DIR/SHA256SUMS"
! grep -q -w TBD "$DIR"/*.md || die "TBD placeholders in $DIR"
command -v gh >/dev/null || die "the GitHub CLI (gh) is required"

echo "publish: pushing tag $VERSION ($COMMIT) to origin"
git push origin "refs/tags/$VERSION:refs/tags/$VERSION"
echo "publish: creating the release $TITLE on $REPO"
gh release create "$VERSION" --repo "$REPO" --verify-tag --title "$TITLE" \
    --notes-file "$DIR/NOTES.md" "$DIR"/*
echo "publish: done; check the release page and the digests of the downloaded files"
