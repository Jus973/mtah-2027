"""Write tracking in the bundle's ATD layout (EPTS metadata XML + raw text) for synthetic matches.

fakedata/generate.py and learn/degrade.py both write through here, so every synthetic match is
read back by the same loader (io/atd.py) and by kloppy exactly like the real bundle. Positions are
normalised 0-1 (NaN = missing); speeds are derived from consecutive positions in m/s.
"""

import numpy as np


def speeds(xy, pitch, fps):
    """(T, S) m/s from consecutive normalised positions, NaN where either side is missing."""
    v = np.full(xy.shape[:2], np.nan)
    v[1:] = np.linalg.norm(np.diff(xy * np.asarray(pitch), axis=0), axis=-1) * fps
    return v


def fmt_group(x, y, s):
    if np.isnan(x):
        return "NaN,NaN,NaN"
    return "{:.5f},{:.5f},{:.2f}".format(x, y, 0.0 if np.isnan(s) else s)


def write_raw(path, slots, ball, pitch, fps):
    """slots (T, S, 2) and ball (T, 2), normalised. Line t is "<t+1>:<46 groups>:<ball group>"."""
    v = speeds(slots, pitch, fps)
    bv = speeds(ball[:, None, :], pitch, fps)[:, 0]
    with open(path, "w") as handle:
        for t in range(slots.shape[0]):
            groups = ";".join(fmt_group(slots[t, s, 0], slots[t, s, 1], v[t, s])
                              for s in range(slots.shape[1]))
            handle.write("{}:{}:{}\n".format(t + 1, groups, fmt_group(ball[t, 0], ball[t, 1], bv[t])))


def write_meta(path, n_frames, fps, pitch, team_of_slot, team_names, params, home_dir1,
               provider="Metrica Sports", date="2026-10-30"):
    """EPTS metadata structured the way kloppy's metrica_epts reader expects; home team first.

    team_of_slot: (S,) 0 = home, 1 = away. params: [(name, value)] provider game parameters
    (half start/end frame ids). home_dir1: +1 if home attacks towards x = 1 in the first half.
    """
    players, pchan, refs = [], [], []
    for s, team in enumerate(team_of_slot):
        pid = "P{}".format(s + 1)
        players.append(
            '<Player id="{0}" teamId="{1}"><Name>Track {2}</Name><ShirtNumber>{2}</ShirtNumber>'
            '<ProviderPlayerParameters/></Player>'.format(pid, "T1" if team == 0 else "T2", s + 1))
        for ch in ("x", "y", "speed"):
            pchan.append('<PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(pid, ch))
        refs.append('<SplitRegister separator=","><PlayerChannelRef playerChannelId="{0}_x"/>'
                    '<PlayerChannelRef playerChannelId="{0}_y"/>'
                    '<PlayerChannelRef playerChannelId="{0}_speed"/></SplitRegister>'.format(pid))
    gp = "".join("<ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>".format(k, v)
                 for k, v in params)
    direction = "left_to_right" if home_dir1 > 0 else "right_to_left"
    xml = """<?xml version="1.0" encoding="utf-8"?>
<main>
<Metadata>
<GlobalConfig><FileDate>{date}</FileDate><TrackingType>Optical</TrackingType>
<ProviderName>{provider}</ProviderName><FrameRate>{fps}</FrameRate>
<ProviderGlobalParameters>{gp}</ProviderGlobalParameters></GlobalConfig>
<Sessions><Session id="S1"><SessionType>Match</SessionType><Start>{date}T00:00:00</Start>
<FieldSize><Width>{L}</Width><Height>{W}</Height></FieldSize>
<MatchParameters><Score idLocalTeam="T1" idVisitingTeam="T2"><LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore></Score></MatchParameters>
</Session></Sessions>
<Teams>
<Team id="T1"><Name>{home}</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name><Value>{dir}</Value></ProviderParameter></ProviderTeamsParameters></Team>
<Team id="T2"><Name>{away}</Name><ProviderTeamsParameters/></Team>
</Teams>
<Players>
{players}
</Players>
<Devices><Device id="D1"><Name>Device 1</Name><Sensors>
<Sensor id="position"><Name>Position</Name><Channels>
<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
</Channels></Sensor>
<Sensor id="distance"><Name>Distance</Name><Channels>
<Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>
</Channels></Sensor>
</Sensors></Device></Devices>
<PlayerChannels>
{pchan}
</PlayerChannels>
</Metadata>
<DataFormatSpecifications>
<DataFormatSpecification id="DFS1" startFrame="1" endFrame="{n}" separator=":">
<StringRegister id="frameCount" name="frameCount"/>
<SplitRegister separator=";">{refs}</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="speed"/></SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications>
</main>
""".format(date=date, provider=provider, fps=fps, gp=gp, L=int(pitch[0]), W=int(pitch[1]),
           home=team_names[0], away=team_names[1], dir=direction, players="\n".join(players),
           pchan="\n".join(pchan), n=n_frames, refs="".join(refs))
    with open(path, "w") as handle:
        handle.write(xml)


def write_gt(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
