"""make probe: container/stream metadata of both videos via PyAV. Decodes no frames.

Reports codec, channels, sample rate and frame rate per stream, the same summary ffprobe would.
"""

import os
import sys

import av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _guard import game_args, put  # noqa: E402

from mtah import registry  # noqa: E402

args = game_args()
for source in ("tactical", "broadcast"):
    path = registry.asset_paths(args.game, args.data_dir, source)["video"]
    if not os.path.isfile(path):
        put(source, "missing")
        continue
    with av.open(path) as container:
        streams = []
        for s in container.streams:
            ctx = s.codec_context
            if s.type == "audio":
                streams.append({"type": "audio", "codec": ctx.name, "channels": ctx.channels,
                                "sample_rate": ctx.sample_rate})
            elif s.type == "video":
                streams.append({"type": "video", "codec": ctx.name, "width": ctx.width,
                                "height": ctx.height, "rate": s.average_rate, "frames": s.frames})
            else:
                streams.append({"type": s.type})
        put(source, {"duration_s": container.duration / 1e6 if container.duration else None,
                     "streams": streams})
