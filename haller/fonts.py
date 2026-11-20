"""
Downloads the handwriting fonts of the synthetic data from the google/fonts
repository, at the commit they were taken from, and checks their SHA-256.

    python -m haller.fonts FONTS_DIR                                # 287 fonts (data/fonts.sha256)
    python -m haller.fonts CURSIVE_FONTS_DIR --list data/fonts_cursive.sha256   # 115 cursive fonts

The fonts are written side by side in the directory (the layout expected by
haller.generate --fonts), with LICENSES.json: for each file, its license
(SIL Open Font License 1.1 or Apache License 2.0, from its directory in
google/fonts) and the URL of the license text.
"""
import argparse
import hashlib
import json
import os
import urllib.request

COMMIT = "7af0d90b4d532ffa4b041937b560c6df4e7b1e95"
RAW = "https://raw.githubusercontent.com/google/fonts/{commit}/{path}"
LICENSES = {"ofl": ("OFL-1.1", "OFL.txt"), "apache": ("Apache-2.0", "LICENSE.txt")}



def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--list", default=os.path.join(os.path.dirname(__file__), "..", "data", "fonts.sha256"))
    p.add_argument("--commit", default=COMMIT)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    licenses = {}
    for line in open(a.list):
        digest, path = line.split()
        name = os.path.basename(path)
        target = os.path.join(a.out, name)
        if not (os.path.exists(target) and hashlib.sha256(open(target, "rb").read()).hexdigest() == digest):
            url = RAW.format(commit=a.commit, path=urllib.request.quote(path))
            data = urllib.request.urlopen(url, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise SystemExit(f"{path}: SHA-256 mismatch")
            with open(target, "wb") as f:
                f.write(data)
        spdx, text = LICENSES[path.split("/")[0]]
        licenses[name] = dict(license=spdx, source=path,
                              license_text=RAW.format(commit=a.commit, path=os.path.dirname(path) + "/" + text))
    json.dump(licenses, open(os.path.join(a.out, "LICENSES.json"), "w"), indent=1, sort_keys=True)
    print(len(licenses), "fonts in", a.out)



if __name__ == "__main__":
    main()
