import importlib.machinery
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def board():
    loader = importlib.machinery.SourceFileLoader("board_cli", os.path.join(ROOT, "scripts", "board"))
    spec = importlib.util.spec_from_loader("board_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_marker_kind_is_first_word():
    b = board()
    assert b.parse_marker(b.MARK + " RELEASE\nagent: a\n") == ("RELEASE", {"agent": "a"})
    kind, fields = b.parse_marker(b.MARK + " RELEASE (agent name changed with the network)\nagent: a")
    assert kind == "RELEASE" and fields == {"agent": "a"}
    assert b.parse_marker("plain comment") is None
