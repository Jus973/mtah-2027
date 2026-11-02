"""make metrica-data: download Metrica Sports Sample Game 3 (EPTS tracking + JSON events).

Source: github.com/metrica-sports/sample-data at the pinned commit COMMIT (master as of
2026-11-01, "Added Sample Game 3", 2021-04-15). Only the three Sample_Game_3 files are fetched
(Games 1-2 are the CSV format and are not used). Each file is verified against the SHA256
recorded below (computed at first download) and skipped when already present and matching.

Licence: the repository has NO formal LICENSE file at the pinned commit - only README.md terms
("Legal stuff: Please be responsible with the use of this data. If you use it for anything
public, please aknowledge the source."). The data is the organisers' own public sample data
(same vendor as the competition), anonymised, published for the analytics community. It is used
here as validation data about other games (RULES.md-allowed), stays outside the repo, and the
provenance is disclosed in the technical document. A SOURCE.txt with repo, commit and terms is
written next to the files.
"""

import argparse
import hashlib
import os
import urllib.request

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/{}"
DEFAULT_DIR = os.path.expanduser(os.environ.get("MTAH_METRICA_DIR", "~/opendata/metrica"))

# repo path -> (size bytes per the git tree at COMMIT, sha256 computed at first download).
FILES = {
    "data/Sample_Game_3/Sample_Game_3_events.json": (
        1709486, "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0"),
    "data/Sample_Game_3/Sample_Game_3_metadata.xml": (
        81806, "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1"),
    "data/Sample_Game_3/Sample_Game_3_tracking.txt": (
        53443115, "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658"),
}

TERMS = """Metrica Sports sample data, Sample Game 3 only.
Repository: https://github.com/metrica-sports/sample-data
Pinned commit: {commit}
No formal LICENSE file at that commit. README.md terms ("Legal stuff"):
  - "Please be responsible with the use of this data."
  - "If you use it for anything public, please aknowledge the source."
Fetched by scripts/fetch_metrica.py; sha256 per file recorded in that script.
"""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for repo_path, (size, want) in sorted(FILES.items(), key=lambda kv: kv[1][0]):
        name = os.path.basename(repo_path)
        path = os.path.join(args.out, name)
        if os.path.isfile(path) and sha256(path) == want:
            print("ok      {}".format(name))
            continue
        print("fetch   {} ({:.1f} MB)".format(name, size / 1e6), flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(RAW.format(COMMIT, repo_path), tmp)
        got = sha256(tmp)
        if want is None:
            print("  sha256 {}  <- record this in scripts/fetch_metrica.py".format(got))
        elif got != want:
            os.remove(tmp)
            raise SystemExit("sha256 mismatch: {} (got {})".format(name, got))
        os.replace(tmp, path)
    with open(os.path.join(args.out, "SOURCE.txt"), "w") as handle:
        handle.write(TERMS.format(commit=COMMIT))
    print("done: {} files in {}".format(len(FILES), args.out))


if __name__ == "__main__":
    main()
