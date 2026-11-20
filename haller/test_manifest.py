"""
Manifest of a private test set: the SHA-256 digest of every file and the
composition of the set, without the images and without the seeds.

The manifest is published with a release whose test set stays private (see
RELEASING.md, "Data publication policy"). Its own SHA-256 digest is quoted in
the model card, so that the set cannot be changed after the results are
published; when the set is published (once replaced by a newer one), anyone
can check its files against the manifest.

Build, from a listing of the files (JSON Lines, one object per file):

    python -m haller.test_manifest build LISTING --name haller-1-test --out manifest.json

Each line of LISTING has `path` (file to hash) and the attributes `world`
(`seedsigner`, `phone`, `web`), `source` (where the photo comes from, for
example `eval_v3`, `participants`, `web`), `writing`, `support` and `words`
(number of words written). The path is read but not written to the manifest.
Writes `manifest.json` (sorted keys, files sorted by digest) and prints its
digest; `manifest.json.sha256` holds the same digest in `sha256sum` format.

Check a directory of files against a manifest (every listed file present, no
other file):

    python -m haller.test_manifest verify manifest.json DIR
"""
import argparse
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

ATTRIBUTES = ("world", "source", "writing", "support", "words")
FORMAT = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(entries: list[dict], name: str, date: str) -> dict:
    files = {}
    for entry in entries:
        missing = [key for key in ("path",) + ATTRIBUTES if entry.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{entry.get('path')}: missing {', '.join(missing)}")
        digest = sha256_file(Path(entry["path"]))
        if digest in files:
            raise ValueError(f"{entry['path']}: same content as another file of the set")
        files[digest] = {"sha256": digest, **{key: entry[key] for key in ATTRIBUTES}}
    rows = [files[d] for d in sorted(files)]
    composition = {key: dict(sorted(collections.Counter(str(r[key]) for r in rows).items()))
                   for key in ATTRIBUTES}
    by_world = {}
    for world in sorted({r["world"] for r in rows}):
        subset = [r for r in rows if r["world"] == world]
        by_world[world] = {key: dict(sorted(collections.Counter(str(r[key]) for r in subset).items()))
                           for key in ("source", "writing", "support", "words")}
    return {"format": FORMAT, "name": name, "date": date, "photos": len(rows),
            "composition": composition, "composition_by_world": by_world, "files": rows}


def dumps(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode()


def verify(manifest: dict, folder: Path) -> list[str]:
    """Problems found: files of the manifest missing from `folder`, files of `folder` not in the manifest."""
    expected = {f["sha256"] for f in manifest["files"]}
    found = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        found.setdefault(sha256_file(path), path)
    problems = [f"missing: {d}" for d in sorted(expected - set(found))]
    problems += [f"not in the manifest: {found[d]}" for d in sorted(set(found) - expected)]
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("listing", type=Path)
    b.add_argument("--name", required=True, help="name of the test set, for example haller-1-test")
    b.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    b.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest", type=Path)
    v.add_argument("folder", type=Path)
    args = parser.parse_args(argv)

    if args.command == "build":
        entries = [json.loads(line) for line in args.listing.read_text().splitlines() if line.strip()]
        data = dumps(build(entries, args.name, args.date))
        args.out.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        Path(f"{args.out}.sha256").write_text(f"{digest}  {args.out.name}\n")
        print(f"{len(entries)} files, manifest {args.out} sha256 {digest}")
        return 0

    manifest = json.loads(args.manifest.read_text())
    problems = verify(manifest, args.folder)
    for line in problems:
        print(line)
    print(f"{manifest['photos']} files in the manifest: {'OK' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
