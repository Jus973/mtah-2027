"""make doc REPO=<public repo URL> [TEAM="<name>"]: the technical document for the Form.

Fills the placeholders in docs/technical.md, drops its HTML comments (notes for the team) and
writes docs/<team>.md (team: configs/base.yaml unless given); with pandoc also docs/<team>.pdf (PLAN.md
§8; the Form accepts either). Reads only docs/technical.md: no match data.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "docs", "technical.md")

parser = argparse.ArgumentParser()
parser.add_argument("--team", default=None, help="default: the team in configs/base.yaml")
parser.add_argument("--repo", default="")
args = parser.parse_args()
if not args.team:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from mtah import config
    args.team = config.load()["team"]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,59}", args.team):
    sys.exit("TEAM: letters, digits, spaces, _ and - only (it is also the file name)")

with open(SOURCE) as handle:
    text = handle.read()
text = re.sub(r"<!--.*?-->\n*", "", text, flags=re.S)
text = text.replace("<TEAM_NAME>", args.team)
if args.repo:
    text = text.replace("<REPO_URL>", args.repo)
left = sorted(set(re.findall(r"<[A-Z_]+>", text)))
out = os.path.join(ROOT, "docs", args.team + ".md")
with open(out, "w") as handle:
    handle.write(text)
print("wrote", os.path.relpath(out, ROOT), "({} words)".format(len(text.split())))
if left:
    print("placeholders still to fill:", ", ".join(left))
if shutil.which("pandoc"):
    pdf = out[:-3] + ".pdf"
    # 10 pt on A4 with 2 cm margins: the layout the <= 6 page estimate assumes (PLAN.md §8).
    subprocess.run(["pandoc", out, "-o", pdf, "-V", "papersize=a4", "-V", "fontsize=10pt",
                    "-V", "geometry:margin=2cm"], check=True)
    print("wrote", os.path.relpath(pdf, ROOT))
else:
    print("pandoc not installed: Markdown only (the Form accepts .md)")
