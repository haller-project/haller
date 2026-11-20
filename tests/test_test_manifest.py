import hashlib
import json

from haller import test_manifest as tm


def listing(tmp_path, contents):
    entries = []
    for i, (content, world, writing) in enumerate(contents):
        path = tmp_path / "set" / f"{i}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": str(path), "world": world, "source": "eval", "writing": writing,
                        "support": "blank", "words": 12})
    out = tmp_path / "listing.jsonl"
    out.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return out


def test_build_and_verify(tmp_path):
    lst = listing(tmp_path, [(b"a", "phone", "capitals"), (b"b", "phone", "lowercase"),
                             (b"c", "seedsigner", "capitals")])
    out = tmp_path / "manifest.json"
    assert tm.main(["build", str(lst), "--name", "t", "--date", "2026-11-20", "--out", str(out)]) == 0
    data = out.read_bytes()
    manifest = json.loads(data)
    assert manifest["photos"] == 3
    assert manifest["composition"]["writing"] == {"capitals": 2, "lowercase": 1}
    assert manifest["composition_by_world"]["phone"]["writing"] == {"capitals": 1, "lowercase": 1}
    assert [f["sha256"] for f in manifest["files"]] == sorted(hashlib.sha256(x).hexdigest() for x in (b"a", b"b", b"c"))
    assert "path" not in manifest["files"][0] and str(tmp_path) not in data.decode()
    assert (tmp_path / "manifest.json.sha256").read_text().split()[0] == hashlib.sha256(data).hexdigest()
    # Same inputs, same bytes
    tm.main(["build", str(lst), "--name", "t", "--date", "2026-11-20", "--out", str(tmp_path / "again.json")])
    assert (tmp_path / "again.json").read_bytes() == data

    assert tm.main(["verify", str(out), str(tmp_path / "set")]) == 0
    (tmp_path / "set" / "extra.jpg").write_bytes(b"d")
    (tmp_path / "set" / "0.jpg").unlink()
    problems = tm.verify(manifest, tmp_path / "set")
    assert len(problems) == 2 and problems[0].startswith("missing")


def test_duplicate_and_missing_attribute_refused(tmp_path):
    import pytest
    lst = listing(tmp_path, [(b"a", "phone", "capitals"), (b"a", "phone", "capitals")])
    entries = [json.loads(line) for line in lst.read_text().splitlines()]
    with pytest.raises(ValueError, match="same content"):
        tm.build(entries, "t", "2026-11-20")
    with pytest.raises(ValueError, match="missing writing"):
        tm.build([{**entries[0], "writing": ""}], "t", "2026-11-20")
