"""CleanMatch: an open-data match before degradation (true positions, ball, ball status, possession).

One timeline at the tracking rate: [pad][half 1][pad][half 2][pad], so it maps 1:1 onto the frames
of the synthetic ATD that learn/degrade.py writes. Metres with the origin at a pitch corner (the
ATD convention once normalised), NaN where an object is not on the pitch.
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class CleanMatch:
    name: str
    fps: int
    pitch: tuple              # (length, width) metres
    players: np.ndarray       # (T, P, 2) float64 metres, NaN when not on the pitch
    team_of_player: np.ndarray  # (P,) int8, 0 = home, 1 = away
    ball: np.ndarray          # (T, 2) float64 metres
    alive: np.ndarray         # (T,) bool, ball in play
    owner: np.ndarray         # (T,) int8, 0 = home, 1 = away, -1 = unknown
    halves: np.ndarray        # (2, 2) int, inclusive [first, last] frame of each half
    team_names: tuple

    @property
    def n_frames(self):
        return self.ball.shape[0]

    def graded(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for first, last in self.halves:
            mask[first:last + 1] = True
        return mask

    def gt_labels(self):
        """Sim GT: X outside the halves, D where the ball is dead, else the owning team. Alive
        frames with unknown possession keep the last known owner (the GT never leaves a gap)."""
        labels = np.full(self.n_frames, "X", dtype="<U1")
        for first, last in self.halves:
            last_owner = -1
            for t in range(first, last + 1):
                if self.owner[t] >= 0:
                    last_owner = self.owner[t]
                if not self.alive[t]:
                    labels[t] = "D"
                elif last_owner >= 0:
                    labels[t] = "HA"[last_owner]
                else:
                    labels[t] = "D"
        return labels


def save(match, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, name=match.name, fps=match.fps, pitch=np.array(match.pitch),
                        players=match.players.astype(np.float32),
                        team_of_player=match.team_of_player, ball=match.ball.astype(np.float32),
                        alive=match.alive, owner=match.owner, halves=match.halves,
                        team_names=np.array(match.team_names))


def load(path):
    z = np.load(path, allow_pickle=False)
    return CleanMatch(name=str(z["name"]), fps=int(z["fps"]),
                      pitch=tuple(float(p) for p in z["pitch"]),
                      players=z["players"].astype(np.float64), team_of_player=z["team_of_player"],
                      ball=z["ball"].astype(np.float64), alive=z["alive"], owner=z["owner"],
                      halves=z["halves"], team_names=tuple(str(n) for n in z["team_names"]))
