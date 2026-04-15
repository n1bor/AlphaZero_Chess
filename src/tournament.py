import random
import multiprocessing
import concurrent.futures
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# Number of matches to keep running in parallel at all times.
# 12 is optimal for a 16-core CPU + RTX 4080 (see profiling notes).
NUM_WORKERS = 12

# Maximum wall-clock seconds a single match may take before the worker is
# considered hung.  At 2000 MCTS steps a game rarely exceeds 30 minutes;
# 3600 s gives plenty of headroom while still recovering from genuine hangs.
MATCH_TIMEOUT = 3600


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
    import os, sys
    from AlphaZero_player import AlphaZero
    from Stockfish import Stockfish
    from match import match

    # Silence the per-player init prints and in-game move commentary that
    # workers emit — they would interleave unreadably with the main process
    # output.  Redirect to /dev/null for the lifetime of this worker call.
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    def build(spec):
        if spec.kind == 'alpha':
            return AlphaZero(spec.net_path, spec.steps, c_puct=spec.c_puct)
        return Stockfish(spec.sf_hash, spec.sf_depth, spec.sf_skill, spec.sf_elo, spec.sf_endgames)

    result = match(build(spec_a), build(spec_b))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    devnull.close()
    return result


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


def _k_factor(games_played: int) -> float:
    """Decaying K-factor: high early for fast convergence, floors at 32."""
    return max(32.0, 128.0 / (games_played + 1) ** 0.5)


def eloAB(ra, rb, result, games_a, games_b):
    qa = 10 ** (ra / 400)
    qb = 10 ** (rb / 400)
    ea = qa / (qa + qb)
    eb = qb / (qa + qb)
    sa = (result + 1) / 2
    sb = 1 - sa
    return (ra + _k_factor(games_a) * (sa - ea),
            rb + _k_factor(games_b) * (sb - eb))


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


def _save_state(players, match_count, h2h):
    state = {
        'match_count': match_count,
        'players': {str(e.spec): {'elo': e.elo, 'win': e.win, 'draw': e.draw, 'lose': e.lose}
                    for e in players},
        'h2h': {k: dict(v) for k, v in h2h.items()},
    }
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)  # atomic on POSIX


def _load_state(players, h2h):
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
    for k, v in state.get('h2h', {}).items():
        h2h[k].update(v)
    loaded = state.get('match_count', 0)
    if loaded:
        print(f"  Resumed from {STATE_FILE} — {loaded} matches already played.")
    return loaded


def _short(spec: PlayerSpec) -> str:
    """Short label used in the head-to-head table."""
    if spec.kind == 'alpha':
        model = os.path.basename(spec.net_path)
        return f"Az({model[:8]} c={spec.c_puct})"
    return f"SF(d={spec.sf_depth})"


def _print_standings(players, match_count, h2h):
    ranked = sorted(players, key=lambda x: -x.elo)
    W = 80
    print(f"\n  ── Standings after {match_count} matches {'─' * (W - 30)}")
    for p in ranked:
        print(f"  {p}")

    if not h2h:
        return
    labels = [_short(p.spec) for p in ranked]
    col = max(len(l) for l in labels)
    row = max(len(l) for l in labels)
    # header
    print(f"\n  Head-to-head  (row beat col  W/D/L)")
    header = ' ' * (row + 4) + '  '.join(f"{l:>{col}}" for l in labels)
    print(f"  {header}")
    for pa, la in zip(ranked, labels):
        cells = []
        for pb, lb in zip(ranked, labels):
            if pa is pb:
                cells.append(' ' * col)
                continue
            key = f"{la}|{lb}"
            rec = h2h.get(key, {})
            w, d, l = rec.get('w', 0), rec.get('d', 0), rec.get('l', 0)
            cells.append(f"{w}/{d}/{l}".rjust(col))
        print(f"  {la:>{row}}    {'  '.join(cells)}")


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
    in_flight: dict = {}        # future -> MatchDetails
    h2h: dict = defaultdict(dict)  # "labelA|labelB" -> {w, d, l}

    match_count = _load_state(players, h2h)
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
                        result = f.result(timeout=MATCH_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        print(f"  Match timed out after {MATCH_TIMEOUT}s "
                              f"({md.a.spec} vs {md.b.spec}) — resubmitting")
                        f.cancel()
                        _submit(executor, in_flight, players)
                        continue
                    except Exception as e:
                        print(f"  Match error ({md.a.spec} vs {md.b.spec}): {e}")
                        _submit(executor, in_flight, players)
                        continue

                    # Update ELO and win/draw/loss counts
                    md.a.elo, md.b.elo = eloAB(
                        md.a.elo, md.b.elo, result,
                        md.a.games_played, md.b.games_played)
                    la, lb = _short(md.a.spec), _short(md.b.spec)
                    ab_key, ba_key = f"{la}|{lb}", f"{lb}|{la}"
                    if result == 1:
                        md.a.win += 1;  md.b.lose += 1
                        h2h[ab_key]['w'] = h2h[ab_key].get('w', 0) + 1
                        h2h[ba_key]['l'] = h2h[ba_key].get('l', 0) + 1
                        outcome = f"{md.a.spec} beat {md.b.spec}"
                    elif result == -1:
                        md.a.lose += 1; md.b.win += 1
                        h2h[ab_key]['l'] = h2h[ab_key].get('l', 0) + 1
                        h2h[ba_key]['w'] = h2h[ba_key].get('w', 0) + 1
                        outcome = f"{md.b.spec} beat {md.a.spec}"
                    else:
                        md.a.draw += 1; md.b.draw += 1
                        h2h[ab_key]['d'] = h2h[ab_key].get('d', 0) + 1
                        h2h[ba_key]['d'] = h2h[ba_key].get('d', 0) + 1
                        outcome = f"{md.a.spec} drew {md.b.spec}"

                    print(f"\n  Match {match_count}: {outcome}")
                    _save_state(players, match_count, h2h)
                    _print_standings(players, match_count, h2h)

                    # Immediately submit a new match to keep the pool full
                    _submit(executor, in_flight, players)

        except KeyboardInterrupt:
            print(f"\nInterrupted — {match_count} matches completed, "
                  f"{len(in_flight)} in flight.")
            print("Cancelling in-flight matches and shutting down...\n")
            executor.shutdown(wait=False, cancel_futures=True)

    # Final standings printed after clean shutdown
    _print_standings(players, match_count, h2h)
