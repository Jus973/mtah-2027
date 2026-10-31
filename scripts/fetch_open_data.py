"""make open-data: download the DFL/IDSSE open dataset (Bassek et al., Sci Data 12, 195, 2025; CC BY).

7 Bundesliga 2022/23 matches: match information, raw events and 25 Hz positions with per-frame
ball status and possession (2.63 GB, figshare article 28196177). Files keep their published
names; each is checked against figshare's MD5 and skipped when already present. Open data is not
the competition bundle: it may be read freely, but it stays out of the repo (size).
"""

import argparse
import hashlib
import json
import os
import urllib.request

ARTICLE = 28196177
API = "https://api.figshare.com/v2/articles/{}".format(ARTICLE)
DEFAULT_DIR = os.path.expanduser(os.environ.get("MTAH_OPEN_DIR", "~/opendata/idsse"))


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        article = json.load(response)
    for f in sorted(article["files"], key=lambda f: f["size"]):
        path = os.path.join(args.out, f["name"])
        if os.path.isfile(path) and md5(path) == f["computed_md5"]:
            print("ok      {}".format(f["name"]))
            continue
        print("fetch   {} ({:.1f} MB)".format(f["name"], f["size"] / 1e6), flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(f["download_url"], tmp)
        if md5(tmp) != f["computed_md5"]:
            os.remove(tmp)
            raise SystemExit("md5 mismatch: {}".format(f["name"]))
        os.replace(tmp, path)
    with open(os.path.join(args.out, "LICENSE.txt"), "w") as handle:
        handle.write("{}\n{}\nhttps://doi.org/{}\n".format(article["title"], article["license"]["name"],
                                                          article["doi"]))
    print("done: {} files in {}".format(len(article["files"]), args.out))


if __name__ == "__main__":
    main()
