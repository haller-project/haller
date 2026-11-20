"""
Content digest of .npz files, independent of the zip container.

The SHA-256 of an .npz file changes with every write (numpy stores the time
of writing in the archive), so a regenerated file never has the checksum of
the original. This digest covers only the arrays: for each array, in name
order, its name, dtype, shape and bytes (C order).

    python digest.py FILE.npz [FILE.npz ...]         # prints "digest  file"
    python digest.py --check SHA256SUMS.content      # checks such a list
"""
import hashlib
import os
import sys

import numpy as np



def digest(path: str) -> str:
    h = hashlib.sha256()
    with np.load(path) as d:
        for name in sorted(d.files):
            a = np.ascontiguousarray(d[name])
            h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()



def main():
    if sys.argv[1:2] == ["--check"]:
        listing = sys.argv[2]
        base = os.path.dirname(listing)
        bad = 0
        for line in open(listing):
            if not line.strip() or line.startswith("#"):
                continue
            expected, name = line.split(maxsplit=1)
            path = os.path.join(base, name.strip())
            if not os.path.exists(path):
                print(f"{name.strip()}: missing")
                continue
            ok = digest(path) == expected
            bad += not ok
            print(f"{name.strip()}: {'OK' if ok else 'DIFFERENT'}", flush=True)
        sys.exit(1 if bad else 0)
    for path in sys.argv[1:]:
        print(f"{digest(path)}  {os.path.basename(path)}", flush=True)



if __name__ == "__main__":
    main()
