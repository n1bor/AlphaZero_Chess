import random
import multiprocessing
import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Optional


# Number of matches to keep running in parallel at all times.
# 12 is optimal for a 16-core CPU + RTX 4080 (see profiling notes).
NUM_WORKERS = 12


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
    """Worker entry point. Builds both players in the worker process and plays a game."""
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
        self.elo = 1000.0
        self.win = 0
        self.draw = 0
        self.lose = 0
        self.in_flight = 0  # matches currently running

    @property
    def games_played(self):
        return self.win + self.draw + self.lose

    @property
    def total_assigned(self):
        return self.games_played + self.in_flight

    def __str__(self):
        return (f"{str(self.spec):<58} "
                f"elo:{self.elo:>7.1f}  "
                f"W{self.win}/D{self.draw}/L{self.lose} "
                f"({self.games_played}g)")


@dataclass
class MatchDetails:
    a: Entry   # corresponds to first argument of run_match (spec_a)
    b: Entry   # corresponds to second argument of run_match (spec_b)


def eloAB(ra, rb, result):
    qa = 10 ** (ra / 400)
    qb = 10 ** (rb / 400)
    ea = qa / (qa + qb)
    eb = qb / (qa + qb)
    sa = (result + 1) / 2
    sb = 1 - sa
    return ra + 32 * (sa - ea), rb + 32 * (sb - eb)


def select_pairing(players):
    """
    Pick the player with the fewest total assigned games (completed + in-flight)
    as player A. Break ties randomly among equally under-played players.
    Pick player B by ELO-weighted random selection from the rest.
    """
    fewest = min(e.total_assigned for e in players)
    candidates = [e for e in players if e.total_assigned == fewest]
    player_a = random.choice(candidates)
    others = [p for p in players if p is not player_a]
    player_b = max(others, key=lambda x: x.elo + random.randint(0, 200))
    return player_a, player_b


def _submit(executor, in_flight, players):
    """Select a pairing, increment in-flight counters, and submit to the executor."""
    a, b = select_pairing(players)
    a.in_flight += 1
    b.in_flight += 1
    # Randomly assign sides so neither player always has white
    if random.choice([True, False]):
        f = executor.submit(run_match, a.spec, b.spec)
        in_flight[f] = MatchDetails(a, b)
    else:
        f = executor.submit(run_match, b.spec, a.spec)
        in_flight[f] = MatchDetails(b, a)  # b is spec_a, a is spec_b


STATE_FILE = '/workspace/chess/data/tournament_state.json'


def _save_state(players, match_count):
    state = {
        'match_count': match_count,
        'players': {str(e.spec): {'elo': e.elo, 'win': e.win, 'draw': e.draw, 'lose': e.lose}
                    for e in players},
    }
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)  # atomic on POSIX


def _load_state(players):
    if not os.path.exists(STATE_FILE):
        return 0
    with open(STATE_FILE) as f:
        state = json.load(f)
    saved = state.get('players', {})
    for e in players:
        key = str(e.spec)
        if key in saved:
            e.elo   = saved[key]['elo']
            e.win   = saved[key]['win']
            e.draw  = saved[key]['draw']
            e.lose  = saved[key]['lose']
    loaded = state.get('match_count', 0)
    if loaded:
        print(f"  Resumed from {STATE_FILE} — {loaded} matches already played.")
    return loaded


def _print_standings(players, match_count):
    W = 80
    print(f"\n  ── Standings after {match_count} matches {'─' * (W - 30)}")
    for p in sorted(players, key=lambda x: -x.elo):
        print(f"  {p}")


if __name__ == '__main__':
    NET    = '/workspace/chess/data/model_data/latest.gz'
    AR_NET = '/workspace/chess/data/model_data/model_1_loss2.53_2026-04-14-194300.gz'
    STEPS  = 200

    players = []
    for c in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]:
        players.append(Entry(PlayerSpec('alpha', net_path=NET, steps=STEPS, c_puct=c)))
    for c in [1, 1.5, 2, 2.5]:
        players.append(Entry(PlayerSpec('alpha', net_path=AR_NET, steps=STEPS, c_puct=c)))
    for depth in [5, 10]:
        players.append(Entry(PlayerSpec('stockfish', sf_hash=256, sf_depth=depth)))

    ctx = multiprocessing.get_context('spawn')
    in_flight: dict = {}   # future -> MatchDetails

    match_count = _load_state(players)
    print(f"Starting tournament: {NUM_WORKERS} concurrent matches, {len(players)} players")

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=NUM_WORKERS, mp_context=ctx) as executor:

        # Fill the pool to NUM_WORKERS
        for _ in range(NUM_WORKERS):
            _submit(executor, in_flight, players)

        try:
            while True:
                # Block until at least one match finishes
                done, _ = concurrent.futures.wait(
                    list(in_flight.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for f in done:
                    md = in_flight.pop(f)
                    md.a.in_flight -= 1
                    md.b.in_flight -= 1
                    match_count += 1

                    try:
                        result = f.result()
                    except Exception as e:
                        print(f"  Match error ({md.a.spec} vs {md.b.spec}): {e}")
                        # Re-submit a replacement without updating stats
                        _submit(executor, in_flight, players)
                        continue

                    # Update ELO and win/draw/loss counts
                    md.a.elo, md.b.elo = eloAB(md.a.elo, md.b.elo, result)
                    if result == 1:
                        md.a.win += 1;  md.b.lose += 1
                        outcome = f"{md.a.spec} beat {md.b.spec}"
                    elif result == -1:
                        md.a.lose += 1; md.b.win += 1
                        outcome = f"{md.b.spec} beat {md.a.spec}"
                    else:
                        md.a.draw += 1; md.b.draw += 1
                        outcome = f"{md.a.spec} drew {md.b.spec}"

                    print(f"\n  Match {match_count}: {outcome}")
                    _save_state(players, match_count)
                    _print_standings(players, match_count)

                    # Immediately submit a new match to keep the pool full
                    _submit(executor, in_flight, players)

        except KeyboardInterrupt:
            print(f"\nInterrupted — {match_count} matches completed, "
                  f"{len(in_flight)} in flight.")
            print("Cancelling in-flight matches and shutting down...\n")
            executor.shutdown(wait=False, cancel_futures=True)

    # Final standings printed after clean shutdown
    _print_standings(players, match_count)
