"""
An interrupted Emuru run resumes only on the files written with the same
arguments (haller.emuru.run_key); the vision check warns before sending ink
maps to another machine (haller.filter.is_local).
"""
import numpy as np
import pytest
from PIL import Image

emuru = pytest.importorskip("haller.emuru")



def write_raw(path, key, page="p1"):
    np.savez_compressed(path, x=np.zeros((0, 64, 1), np.uint8), w=np.zeros(0, np.int16), text=np.array([]),
                        page=page, style_text="t", style=np.zeros((64, 160), np.uint8),
                        **({} if key is None else {"key": key}))



def test_run_key_depends_on_the_arguments():
    words = ["abandon", "ability"]
    key = emuru.run_key(7, words, 285, 60, ("lower", "upper"))
    assert key == emuru.run_key(7, list(words), 285, 60, ("lower", "upper"))
    for other in (emuru.run_key(8, words, 285, 60, ("lower", "upper")),
                  emuru.run_key(7, words[::-1], 285, 60, ("lower", "upper")),
                  emuru.run_key(7, words, 286, 60, ("lower", "upper")),
                  emuru.run_key(7, words, 285, 61, ("lower", "upper")),
                  emuru.run_key(7, words, 285, 60, ("lower",))):
        assert other != key



def test_resume_refuses_another_run(tmp_path):
    style = [("p1", Image.new("L", (160, 64), 255), "t")]
    rng = lambda: np.random.default_rng(0)
    key = emuru.run_key(7, ["abandon"], 1, 1, ("lower",))
    write_raw(tmp_path / "0.npz", key)
    # same run: the file is kept, nothing is generated (no model loaded)
    emuru.generate(style, ["abandon"], str(tmp_path), 1, 1, rng(), ("lower",), key=key)
    for found, page in (("another", "p1"), (None, "p1"), (key, "p2")):
        write_raw(tmp_path / "0.npz", found, page)
        with pytest.raises(ValueError, match="another run"):
            emuru.generate(style, ["abandon"], str(tmp_path), 1, 1, rng(), ("lower",), key=key)



def test_vision_check_url():
    from haller.filter import is_local
    assert is_local("http://127.0.0.1:8097/v1/vision") and is_local("http://localhost:8097/") and is_local("http://[::1]:80/")
    assert not is_local("https://api.example.com/v1/vision") and not is_local("http://10.0.0.5:8097/")
