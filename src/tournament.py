import random
import multiprocessing
import concurrent.futures
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# Number of matches to keep running in parallel at all times.
# 12 is optimal for a 16-core CPU + RTX 4080 (see profiling notes).
NUM_WORKERS = 20

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

    log_path = f"/tmp/tournament_worker_{os.getpid()}.log"
    log = open(log_path, 'w', buffering=1)
    sys.stdout = log
    sys.stderr = log

    def build(spec):
        if spec.kind == 'alpha':
            return AlphaZero(spec.net_path, spec.steps, c_puct=spec.c_puct)
        return Stockfish(spec.sf_hash, spec.sf_depth, spec.sf_skill, spec.sf_elo, spec.sf_endgames)

    import time as _time
    t0 = _time.monotonic()
    result = match(build(spec_a), build(spec_b))
    duration = _time.monotonic() - t0

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log.close()
    os.remove(log_path)
    return result, duration


class Entry:
    def __init__(self, spec: PlayerSpec):
        self.spec = spec
        self.elo = 1000.0
        self.win = 0
        self.draw = 0
        self.lose = 0
        self.total_seconds = 0.0
        self.in_flight = 0  # matches currently running

    @property
    def games_played(self):
        return self.win + self.draw + self.lose

    @property
    def total_assigned(self):
        return self.games_played + self.in_flight

    @property
    def avg_seconds(self):
        return self.total_seconds / self.games_played if self.games_played else 0.0

    def __str__(self):
        avg = f"{self.avg_seconds:>6.0f}s/g" if self.games_played else "       n/a"
        return (f"{str(self.spec):<58} "
                f"elo:{self.elo:>7.1f}  "
                f"W{self.win}/D{self.draw}/L{self.lose} "
                f"({self.games_played}g)  {avg}")


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
    Pick player B as either the next-higher or next-lower ELO neighbour of
    player A (chosen at random).  This keeps matches competitive and prevents
    high-ELO outliers from monopolising the opponent slot.
    """
    fewest = min(e.total_assigned for e in players)
    candidates = [e for e in players if e.total_assigned == fewest]
    player_a = random.choice(candidates)
    others = sorted([p for p in players if p is not player_a], key=lambda x: x.elo)
    elo_a = player_a.elo
    idx = next((i for i, p in enumerate(others) if p.elo >= elo_a - 1e-9), len(others))
    above_elo = others[idx].elo if idx < len(others) else None
    below_elo = others[idx - 1].elo if idx > 0 else None
    # For each distinct neighbour ELO level, pick one player at random from that
    # level — this avoids always selecting the same index-0 player when many
    # players share the same ELO (e.g. at tournament start).
    neighbours = []
    for target in dict.fromkeys(e for e in [above_elo, below_elo] if e is not None):
        pool = [p for p in others if abs(p.elo - target) < 1e-9]
        neighbours.append(random.choice(pool))
    player_b = random.choice(neighbours)
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
        'players': {str(e.spec): {'elo': e.elo, 'win': e.win, 'draw': e.draw, 'lose': e.lose,
                                   'total_seconds': e.total_seconds}
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
            e.elo          = saved[key]['elo']
            e.win          = saved[key]['win']
            e.draw         = saved[key]['draw']
            e.lose         = saved[key]['lose']
            e.total_seconds = saved[key].get('total_seconds', 0.0)
    for k, v in state.get('h2h', {}).items():
        h2h[k].update(v)
    loaded = state.get('match_count', 0)
    if loaded:
        print(f"  Resumed from {STATE_FILE} — {loaded} matches already played.")
    return loaded


def _make_net_labels(players) -> dict:
    """Map each unique net_path to a short unique name by stripping the common filename prefix."""
    paths = list({e.spec.net_path for e in players if e.spec.kind == 'alpha'})
    stems = [os.path.basename(p).removesuffix('.gz') for p in paths]
    if len(stems) > 1:
        prefix_len = len(os.path.commonprefix(stems))
    else:
        prefix_len = max(0, len(stems[0]) - 12) if stems else 0
    labels = {}
    for path, stem in zip(paths, stems):
        unique = stem[prefix_len:].lstrip('_')
        labels[path] = unique[:12] if unique else stem[-12:]
    return labels


def _short(spec: PlayerSpec, net_labels: dict = None) -> str:
    """Short label used in the head-to-head table."""
    if spec.kind == 'alpha':
        name = (net_labels or {}).get(spec.net_path) or os.path.basename(spec.net_path)[:10]
        return f"Az({name} s={spec.steps} c={spec.c_puct})"
    return f"SF(d={spec.sf_depth})"


def _print_standings(players, match_count, h2h, start_time=None, matches_at_start=0, net_labels=None):
    ranked = sorted(players, key=lambda x: -x.elo)
    W = 80
    rate_str = ""
    if start_time is not None:
        elapsed = time.monotonic() - start_time
        completed_this_session = match_count - matches_at_start
        if elapsed > 0 and completed_this_session > 0:
            gph = completed_this_session / elapsed * 3600
            avg_min = elapsed / completed_this_session / 60
            rate_str = f"  {gph:.1f} games/hr  (~{avg_min:.1f} min/game)"
    print(f"\n  ── Standings after {match_count} matches {'─' * (W - 30)}{rate_str}")
    for p in ranked:
        print(f"  {p}")

    if not h2h:
        return
    labels = [_short(p.spec, net_labels) for p in ranked]

    def _split(label):
        mid = len(label) // 2
        left = label.rfind(' ', 0, mid + 1)
        right = label.find(' ', mid)
        if left == -1 and right == -1:
            return label, ''
        if left == -1:
            split = right
        elif right == -1:
            split = left
        else:
            split = left if (mid - left) <= (right - mid) else right
        return label[:split], label[split + 1:]

    splits = [_split(l) for l in labels]
    col = max(max(len(s[0]), len(s[1])) for s in splits)
    row = max(len(l) for l in labels)
    indent = ' ' * (row + 4)
    print(f"\n  Head-to-head  (row beat col  W/D/L)")
    print(f"  {indent}{'  '.join(s[0].rjust(col) for s in splits)}")
    print(f"  {indent}{'  '.join(s[1].rjust(col) for s in splits)}")
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
    NET    = '/workspace/chess/data/model_data/small_model_20hrs_2.08.gz'
    # AR_NET = '/workspace/chess/data/model_data/model_1_loss2.53_2026-04-14-194300.gz'
    AR_NET = '/workspace/chess/data/model_data/small_model_10hrs_2.12.gz'
    STEPS  = 200

    players = []
    for c in [1, 2, 2.5, 3, 3.5, 4] :
        players.append(Entry(PlayerSpec('alpha', net_path=NET, steps=STEPS, c_puct=c)))
        players.append(Entry(PlayerSpec('alpha', net_path=NET, steps=400, c_puct=c)))
        players.append(Entry(PlayerSpec('alpha', net_path=NET, steps=800, c_puct=c)))
    for depth in [1, 5, 10, 15]:
        players.append(Entry(PlayerSpec('stockfish', sf_hash=256, sf_depth=depth)))

    net_labels = _make_net_labels(players)
    ctx = multiprocessing.get_context('spawn')
    in_flight: dict = {}        # future -> MatchDetails
    h2h: dict = defaultdict(dict)  # "labelA|labelB" -> {w, d, l}

    match_count = _load_state(players, h2h)
    matches_at_start = match_count
    start_time = time.monotonic()
    print(f"Starting tournament: {NUM_WORKERS} concurrent matches, {len(players)} players")

    try:
      while True:  # outer loop: recreate pool if a worker is killed
        # Reset in-flight counters for any futures that died with the old pool
        for md in in_flight.values():
            md.a.in_flight -= 1
            md.b.in_flight -= 1
        in_flight.clear()

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=NUM_WORKERS, mp_context=ctx,
                max_tasks_per_child=1) as executor:

            # Fill the pool to NUM_WORKERS
            for _ in range(NUM_WORKERS):
                _submit(executor, in_flight, players)

            pool_broken = False
            while not pool_broken:
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
                        result, duration = f.result(timeout=MATCH_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        print(f"  Match timed out after {MATCH_TIMEOUT}s "
                              f"({md.a.spec} vs {md.b.spec}) — resubmitting")
                        f.cancel()
                    except concurrent.futures.process.BrokenProcessPool as e:
                        print(f"  Process pool broken ({md.a.spec} vs {md.b.spec}): {e}"
                              f" — recreating pool")
                        pool_broken = True
                        break
                    except Exception as e:
                        print(f"  Match error ({md.a.spec} vs {md.b.spec}): {e}")
                    else:
                        md.a.total_seconds += duration
                        md.b.total_seconds += duration
                        # Update ELO and win/draw/loss counts
                        md.a.elo, md.b.elo = eloAB(
                            md.a.elo, md.b.elo, result,
                            md.a.games_played, md.b.games_played)
                        la, lb = _short(md.a.spec, net_labels), _short(md.b.spec, net_labels)
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
                        _print_standings(players, match_count, h2h,
                                         start_time, matches_at_start, net_labels)

                    if pool_broken:
                        break

                    # Immediately submit a new match to keep the pool full
                    try:
                        _submit(executor, in_flight, players)
                    except concurrent.futures.process.BrokenProcessPool as e:
                        print(f"  Process pool broken on submit: {e} — recreating pool")
                        pool_broken = True
                        break

    except KeyboardInterrupt:
        print(f"\nInterrupted — {match_count} matches completed, "
              f"{len(in_flight)} in flight.")
        print("Cancelling in-flight matches and shutting down...\n")

    # Final standings printed after clean shutdown
    _print_standings(players, match_count, h2h, start_time, matches_at_start, net_labels)
