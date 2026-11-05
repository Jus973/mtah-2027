"""make doc [REPO=<public repo URL>] [TEAM="<name>"]: the technical document for the Form.

Fills the placeholders in docs/technical.md, drops its HTML comments (notes for the team) and
writes docs/<team>.md and docs/<team>.pdf (team from configs/base.yaml unless given; PLAN.md
§8; the Form accepts either). Reads only docs/technical.md: no match data.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "docs", "technical.md")

parser = argparse.ArgumentParser()
parser.add_argument("--team", default=None, help="default: the team in configs/base.yaml")
parser.add_argument("--repo", default="https://github.com/Jus973/mtah-2027")
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
    sys.exit("Unresolved document placeholders: " + ", ".join(left))
missing = [tool for tool in ("pandoc", "xelatex", "pdfinfo") if not shutil.which(tool)]
if missing:
    sys.exit("PDF build requires {} on PATH (see README.md)".format(", ".join(missing)))
pdf = out[:-3] + ".pdf"
with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
    proof = os.path.join(tmp, "proof.pdf")
    subprocess.run(["pandoc", out, "-o", proof, "--pdf-engine=xelatex", "--fail-if-warnings",
                    "-V", "mainfont=STIXGeneral", "-V", "papersize=a4", "-V", "fontsize=10pt",
                    "-V", "geometry:margin=2cm"], check=True)
    info = subprocess.check_output(["pdfinfo", proof], text=True, env=dict(os.environ, LC_ALL="C"))
    pages = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if not pages:
        sys.exit("Could not verify the PDF page count")
    count = int(pages[1])
    if count > 6:
        sys.exit("Document is {} pages; submission limit is 6".format(count))
    os.replace(proof, pdf)
print("wrote {} ({} pages)".format(os.path.relpath(pdf, ROOT), count))
