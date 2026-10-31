"""make probe: container/stream metadata of both videos via PyAV. Decodes no frames.

Prints codec, channels, sample rate and frame rate per stream, the same summary ffprobe would.
"""

import os
import sys

import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args  # noqa: E402

from mtah import registry  # noqa: E402

args = game_args()
for source in ("tactical", "broadcast"):
    path = registry.asset_paths(args.game, args.data_dir, source)["video"]
    if not os.path.isfile(path):
        print("{}: missing".format(source))
        continue
    with av.open(path) as container:
        dur = container.duration / 1e6 if container.duration else float("nan")
        print("{}: {} streams, duration {:.1f} s".format(source, len(container.streams), dur))
        for s in container.streams:
            ctx = s.codec_context
            if s.type == "audio":
                print("  audio  codec={} channels={} sample_rate={}".format(
                    ctx.name, ctx.channels, ctx.sample_rate))
            elif s.type == "video":
                print("  video  codec={} {}x{} rate={} frames={}".format(
                    ctx.name, ctx.width, ctx.height, s.average_rate, s.frames))
            else:
                print("  {}".format(s.type))
