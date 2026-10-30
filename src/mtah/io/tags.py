"""Smart Tagging (SportsCode XML) -> list of Tag records on the tactical video timeline.

<start>/<end> are seconds from tactical frame 0, so frame = round(seconds * fps). Codes are upper
case and tag values title case in the delivered file; everything here compares case-insensitively.
Team tags name clubs, not sides: the club -> H/A map comes from configs (tag_team_map).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class Tag:
    code: str
    start: int            # first frame
    end: int              # frame of <end>, rounded (inclusive end of the interval)
    tags: dict = field(default_factory=dict)

    def get(self, group, default=None):
        for k, v in self.tags.items():
            if k.strip().lower() == group.lower():
                return v
        return default

    def is_code(self, code):
        return (self.code or "").strip().lower() == code.lower()

    def is_type(self, value):
        return (self.get("Type") or "").strip().lower() == value.lower()


def load_tags(path, fps=25):
    out = []
    for inst in ET.parse(path).iter("instance"):
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        out.append(Tag(
            code=inst.findtext("code"),
            start=int(round(start_s * fps)),
            end=int(round(end_s * fps)),
            tags={lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")},
        ))
    return out


def team_side(tag, team_map):
    """'H', 'A' or None (N/A, missing, unknown club) for a tag's Team value."""
    value = (tag.get("Team") or "").strip().lower()
    for club, side in team_map.items():
        if club.strip().lower() == value:
            return side
    return None


def half_number(tag):
    """1, 2 or None from the string-valued Half tag ('1st Half', '2nd Half')."""
    value = (tag.get("Half") or "").strip().lower()
    if value.startswith("1"):
        return 1
    if value.startswith("2"):
        return 2
    return None
