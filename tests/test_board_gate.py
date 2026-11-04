import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from board_gate import links, verdict  # noqa: E402


def issue(state="open", *labels):
    return {"state": state, "labels": [{"name": n} for n in labels]}


def test_links_parsing():
    assert links("Closes Jus973/mtah-board#12\n\n## What") == [("closes", 12)]
    assert links("Refs Jus973/mtah-board#10 (follow-up)") == [("refs", 10)]
    assert links("fixes jus973/MTAH-board#3, resolves Jus973/mtah-board#4") == \
        [("closes", 3), ("closes", 4)]
    assert links("Closes Jus973/mtah-board#<!-- board issue number -->") == []
    assert links("<!-- Closes Jus973/mtah-board#5 -->") == []
    assert links("Closes #12") == [] and links(None) == []
    assert links("Closes Jus973/mtah-board#123x") == []


def test_verdict():
    assert verdict("closes", issue("open", "status:in-progress", "size:S")) is None
    assert verdict("closes", issue("open", "status:review")) is None
    assert "claim" in verdict("closes", issue("open", "status:todo"))
    assert verdict("closes", issue("closed", "status:review")) is not None
    assert verdict("refs", issue("closed")) is None
    assert verdict("refs", None) == "not found on the board"
    assert "pull request" in verdict("closes", dict(issue("open", "status:review"),
                                                    pull_request={}))
