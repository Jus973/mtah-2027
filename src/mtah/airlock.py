"""The airlock: an agent can run real-data scripts and receive only validated aggregates.

Scripts report through Report.put (named values) and Report.check (assertions). In a terminal
these print as they go. Under scripts/airlock.py (OUT_ENV set) nothing is printed: the report is
written as JSON, and the runner releases it only if violations() finds nothing. The rules are the
PLAN.md §3 output contract made mechanical: numbers, booleans and short single-line strings, in
small lists and dicts. They catch accidents (a per-frame array, a raw ATD line, an XML fragment, a
crash message quoting the text it failed on), not an adversary; the deny rules and
.claude/hooks/guard_data.py cover the agent's other routes to the bundle.
"""

import atexit
import json
import math
import os
import re
from fractions import Fraction

OUT_ENV = "MTAH_AIRLOCK_OUT"
LIMITS = {"bytes": 16384, "list": 64, "depth": 6, "values": 1024, "str": 120, "key": 80}

_KEY = re.compile(r"^[A-Za-z0-9_ +\-/().,=<>%:\[\]]+$")
_RAW = (
    re.compile(r"-?\d+(\.\d+)?\s*[,;:]\s*-?\d+(\.\d+)?\s*[,;:]\s*-?\d+(\.\d+)?"),  # x,y,s / id:x,y
    re.compile(r"</?[A-Za-z]|/>|=\""),                                            # XML
)
_FRAME = re.compile(r'^\s*File "(.+)", line (\d+), in (\S+)\s*$')
_EXC = re.compile(r"^([A-Za-z_][\w.]*)(:|$)")


def plain(value):
    """JSON-ready copy: numpy scalars to Python, tuples to lists, floats to 6 significant digits,
    NaN/inf to None, dict keys to str. Arrays are refused: reports carry aggregates."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if getattr(value, "ndim", 0):
        raise TypeError("airlock reports take scalars, not arrays (shape {})".format(value.shape))
    if hasattr(value, "item") and not isinstance(value, str):  # numpy scalar
        value = value.item()
    if isinstance(value, Fraction):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float("{:.6g}".format(value)) if math.isfinite(value) else None
    raise TypeError("airlock reports take numbers, booleans, strings, lists and dicts, not "
                    + type(value).__name__)


def _fmt(value):
    if isinstance(value, dict):
        return "{" + ", ".join("{}: {}".format(k, _fmt(v)) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def format_value(key, value):
    return "{}: {}".format(key, _fmt(value))


def format_check(row):
    detail = ", ".join("{} {}".format(k, _fmt(v)) for k, v in row["detail"].items())
    return "  [{}] {}{}".format("PASS" if row["ok"] else "FAIL", row["name"],
                                "  " + detail if detail else "")


def render(report):
    lines = [format_value(k, v) for k, v in report["values"].items()]
    if report["checks"]:
        lines += ["", "checks"] + [format_check(row) for row in report["checks"]]
    return "\n".join(lines)


class Report:
    """Values and checks of one script run: printed live in a terminal, dumped as JSON in the airlock."""

    def __init__(self):
        self.out = os.environ.get(OUT_ENV)
        self.values, self.checks = {}, []
        if self.out:
            atexit.register(self.dump)

    def put(self, key, value):
        self.values[str(key)] = value = plain(value)
        if not self.out:
            print(format_value(key, value))

    def check(self, name, ok, **detail):
        row = {"name": str(name), "ok": bool(ok), "detail": plain(detail)}
        if not self.out:
            if not self.checks:
                print("\nchecks")
            print(format_check(row))
        self.checks.append(row)
        return bool(ok)

    def dump(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"values": self.values, "checks": self.checks}, handle)
        os.replace(tmp, self.out)


def _raw_like(text):
    return any(p.search(text) for p in _RAW)


def violations(report):
    """[(path, rule)] for everything in a report the release policy rejects; [] means releasable.

    Paths are positions ("/values/3/0"), not keys, so printing them reveals nothing a key holds.
    """
    if not (isinstance(report, dict) and set(report) == {"values", "checks"}
            and isinstance(report["values"], dict) and isinstance(report["checks"], list)
            and all(isinstance(r, dict) and set(r) == {"name", "ok", "detail"}
                    and isinstance(r["detail"], dict) for r in report["checks"])):
        return [("/", "not a report")]
    found, count = [], [0]

    def walk(value, path, depth):
        if depth > LIMITS["depth"]:
            found.append((path, "nested deeper than {}".format(LIMITS["depth"])))
        elif isinstance(value, dict):
            for i, (key, item) in enumerate(value.items()):
                here = "{}/{}".format(path, i)
                if not (0 < len(key) <= LIMITS["key"] and _KEY.match(key)) or _raw_like(key):
                    found.append((here, "key not allowed"))
                walk(item, here, depth + 1)
        elif isinstance(value, list):
            if len(value) > LIMITS["list"]:
                found.append((path, "list longer than {}".format(LIMITS["list"])))
                return
            for i, item in enumerate(value):
                walk(item, "{}/{}".format(path, i), depth + 1)
        elif isinstance(value, str):
            count[0] += 1
            if len(value) > LIMITS["str"] or not (value.isascii() and value.isprintable()) \
                    or _raw_like(value):
                found.append((path, "string not allowed"))
        elif value is None or isinstance(value, (bool, int)) \
                or (isinstance(value, float) and math.isfinite(value)):
            count[0] += 1
        else:
            found.append((path, "type not allowed"))

    walk(report, "", 0)
    if count[0] > LIMITS["values"]:
        found.append(("/", "more than {} values".format(LIMITS["values"])))
    if len(json.dumps(report)) > LIMITS["bytes"]:
        found.append(("/", "larger than {} bytes".format(LIMITS["bytes"])))
    return found


def crash_summary(log_text, root):
    """{"type", "frames": [[file, line, function]]} of the last traceback in a log, or None.

    Never the message: a float() error, for one, quotes the raw text it failed to parse. Files
    inside the repo are shown relative to it, others by basename under "lib/". Angle brackets
    ("<module>") are dropped so the summary itself passes the XML rule of violations().
    """
    start = log_text.rfind("Traceback (most recent call last):")
    if start < 0:
        return None
    frames, exc = [], "<unparsed>"
    for line in log_text[start:].splitlines()[1:]:
        m = _FRAME.match(line)
        if m:
            path = os.path.abspath(m.group(1))
            path = (os.path.relpath(path, root) if path.startswith(root + os.sep)
                    else "lib/" + os.path.basename(path))
            frames.append([path, int(m.group(2)), m.group(3).strip("<>")])
        elif line and not line[0].isspace():
            m = _EXC.match(line)
            exc = m.group(1) if m else exc
            break
    return {"type": exc, "frames": frames[-12:]}
