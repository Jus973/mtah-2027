"""make doc TEAM=<TEAM_NAME> REPO=<public repo URL>: the technical document for the Form.

Fills the placeholders in docs/technical.md, drops its HTML comments (notes for the team) and
writes docs/<TEAM_NAME>.md. If pandoc is installed it also writes docs/<TEAM_NAME>.pdf (PLAN.md
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
parser.add_argument("--team", required=True)
parser.add_argument("--repo", default="")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.team):
    sys.exit("TEAM: letters, digits, _ and - only (it is also the file name)")

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
    subprocess.run(["pandoc", out, "-o", pdf], check=True)
    print("wrote", os.path.relpath(pdf, ROOT))
else:
    print("pandoc not installed: Markdown only (the Form accepts .md)")
