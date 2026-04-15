import random
import pexpect
import multiprocessing
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerSpec:
    """Serialisable description of a player — safe to pickle across processes."""
    kind: str  # 'alpha' or 'stockfish'
    net_path: Optional[str] = None
    steps: Optional[int] = None
    c_puct: float = 1.0
    sf_hash: int = 256
    sf_depth: int = 5
    sf_skill: int = -1
    sf_elo: int = -1
    sf_endgames: bool = True

    def __str__(self):
        if self.kind == 'alpha':
            return f"Alpha steps={self.steps} c_puct={self.c_puct} {self.net_path}"
        return f"Stockfish h:{self.sf_hash} d:{self.sf_depth} sk:{self.sf_skill} elo:{self.sf_elo} end:{self.sf_endgames}"


def run_match(spec_a, spec_b):
    """Worker entry point.  Builds both players in the worker process and plays a game."""
    from AlphaZero_player import AlphaZero
    from Stockfish import Stockfish
    from match import match

    def build(spec):
        if spec.kind == 'alpha':
            return AlphaZero(spec.net_path, spec.steps, c_puct=spec.c_puct)
        return Stockfish(spec.sf_hash, spec.sf_depth, spec.sf_skill, spec.sf_elo, spec.sf_endgames)

    return match(build(spec_a), build(spec_b))


class Entry:
    def __init__(self, spec: PlayerSpec):
        self.spec = spec
        self.elo = 1000
        self.win = 0
        self.draw = 0
        self.lose = 0

    def __str__(self):
        return f"{self.spec} : {self.elo} {self.win}/{self.draw}/{self.lose}"


@dataclass
class MatchDetails:
    thread: object
    a: Entry
    b: Entry


def eloAB(ra, rb, result):
    qa = 10 ** (ra / 400)
    qb = 10 ** (rb / 400)
    ea = qa / (qa + qb)
    eb = qb / (qa + qb)
    sa = (result + 1) / 2
    sb = 1 - sa
    aElo = ra + 32 * (sa - ea)
    bElo = rb + 32 * (sb - eb)
    return aElo, bElo


if __name__ == '__main__':
    NET = '/workspace/chess/data/model_data/latest.gz'
    STEPS = 200

    players = []
    for c in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
        players.append(Entry(PlayerSpec('alpha', net_path=NET, steps=STEPS, c_puct=c)))
    for depth in [5, 10]:
        players.append(Entry(PlayerSpec('stockfish', sf_hash=256, sf_depth=depth)))

    # spawn gives each worker its own GIL and CUDA context — required for true parallelism
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=5) as pool:
        for i in range(10000):
            print(f" MATCH {i}")
            matches = []
            playera = None
            for player in sorted(players, key=lambda x: x.elo + random.randint(0, 200), reverse=True):
                if playera is None:
                    playera = player
                else:
                    if random.choice([True, False]):
                        x = pool.apply_async(run_match, (playera.spec, player.spec))
                        matches.append(MatchDetails(x, playera, player))
                    else:
                        x = pool.apply_async(run_match, (player.spec, playera.spec))
                        matches.append(MatchDetails(x, player, playera))
                    playera = None

            for matchEntry in matches:
                try:
                    result = matchEntry.thread.get()
                    (aElo, bElo) = eloAB(matchEntry.a.elo, matchEntry.b.elo, result)
                    matchEntry.a.elo = aElo
                    matchEntry.b.elo = bElo
                    if result == 0:
                        matchEntry.a.draw += 1
                        matchEntry.b.draw += 1
                    if result == 1:
                        matchEntry.a.win += 1
                        matchEntry.b.lose += 1
                    if result == -1:
                        matchEntry.a.lose += 1
                        matchEntry.b.win += 1
                except pexpect.exceptions.TIMEOUT as e:
                    print(e)

            for player in sorted(players, key=lambda x: x.elo, reverse=True):
                print(f"{player}")
