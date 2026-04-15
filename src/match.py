import argparse
import sys
import time
import cProfile
import pstats
import io
from dataclasses import dataclass
from Stockfish import Stockfish
from Player import Player
from AlphaZero_player import AlphaZero


def match(a, b):
    moves = []
    lastmove = ""
    checkmate = False
    result = 0
    while not checkmate:
        move = a.getMove(moves)
        if move == "onlykings":
            print(f"draw - only kings {' '.join(moves)}")
            break
        if move is None:
            print(f"Player B {b} won {' '.join(moves)}")
            checkmate = True
            result = -1
            break
        if move == lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove = move
        moves.append(move)
        move = b.getMove(moves)
        if move == "onlykings":
            print(f"draw - only kings {' '.join(moves)}")
            break
        if move is None:
            print(f"Player A {a} won {' '.join(moves)}")
            result = 1
            checkmate = True
            break
        if move == lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove = move
        moves.append(move)
        if len(moves) > 250:
            print(f"draw 250 {' '.join(moves)}")
            break

    aBoard = a.getBoard(moves)
    bBoard = b.getBoard(moves)
    for index, item_list in enumerate(aBoard):
        if item_list != bBoard[index]:
            print(f"{aBoard} != {bBoard} for {' '.join(moves)}")
            raise Exception(f"{aBoard} != {bBoard} for {' '.join(moves)}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Profiling helpers
# ─────────────────────────────────────────────────────────────────────────────

class _PhaseClock:
    """Accumulates wall-clock time per labelled phase."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.uct_calls: int = 0        # number of full UCT_search() invocations
        self.move_times: list[float] = []  # seconds per AlphaZero move

    def record(self, phase: str, elapsed: float) -> None:
        self.totals[phase] = self.totals.get(phase, 0.0) + elapsed
        self.counts[phase] = self.counts.get(phase, 0) + 1

    def report(self) -> None:
        wall = sum(self.totals.values())
        sims = self.counts.get('net_forward', 0)
        moves = self.uct_calls

        W = 68
        print(f"\n{'═' * W}")
        print(f"  AlphaZero Performance Profile")
        print(f"{'═' * W}")
        print(f"  AlphaZero moves made        : {moves}")
        print(f"  Total MCTS simulations      : {sims:,}")
        if moves:
            print(f"  Avg simulations / move      : {sims / moves:.1f}")
            avg_move = sum(self.move_times) / len(self.move_times)
            print(f"  Avg wall-time / move        : {avg_move:.3f}s")
        print()
        print(f"  {'Phase':<32} {'Total(s)':>10} {'Calls':>8} {'Avg(ms)':>10} {'%':>7}")
        print(f"  {'─' * (W - 2)}")
        for phase, t in sorted(self.totals.items(), key=lambda x: -x[1]):
            n = self.counts[phase]
            pct = 100.0 * t / wall if wall else 0.0
            avg = 1000.0 * t / n if n else 0.0
            print(f"  {phase:<32} {t:>10.3f} {n:>8,} {avg:>10.3f} {pct:>6.1f}%")
        print(f"  {'─' * (W - 2)}")
        print(f"  {'TOTAL':<32} {wall:>10.3f}")
        print(f"{'═' * W}")
        print()
        print("  Notes:")
        print("  • select_leaf    – includes copy.deepcopy per new tree node")
        print("  • expand         – includes board.actions() (legal move gen)")
        print("  • net_forward    – measured after torch.cuda.synchronize()")
        print("  • detach_to_cpu  – GPU→CPU tensor transfer")
        print()


_profile_clock: _PhaseClock | None = None


def _build_profiled_uct_search():
    """Return an instrumented drop-in for MCTS_chess.UCT_search."""
    import numpy as np
    import torch
    import encoder_decoder as ed
    import MCTS_chess

    def profiled_uct_search(game_state, num_reads, net, c_puct=1):
        clk = _profile_clock
        clk.uct_calls += 1
        device = next(net.parameters()).device
        is_cuda = device.type == 'cuda'

        root = MCTS_chess.UCTNode(game_state, move=None,
                                  parent=MCTS_chess.DummyNode(), c_puct=c_puct)

        for _ in range(num_reads):
            # ── tree traversal (deepcopy happens here for new nodes) ──────────
            t = time.perf_counter()
            leaf = root.select_leaf()
            clk.record('select_leaf (+ deepcopy)', time.perf_counter() - t)

            # ── board → numpy tensor ──────────────────────────────────────────
            t = time.perf_counter()
            encoded = ed.encode_board(leaf.game).transpose(2, 0, 1)
            clk.record('encode_board', time.perf_counter() - t)

            # ── numpy → torch + host-to-device transfer ───────────────────────
            t = time.perf_counter()
            tensor = torch.from_numpy(encoded).float().to(device)
            clk.record('tensor_to_device', time.perf_counter() - t)

            # ── neural network forward pass ───────────────────────────────────
            t = time.perf_counter()
            child_priors_t, value_t = net(tensor)
            if is_cuda:
                torch.cuda.synchronize()   # wait for GPU work to finish
            clk.record('net_forward', time.perf_counter() - t)

            # ── GPU → CPU / detach ────────────────────────────────────────────
            t = time.perf_counter()
            child_priors = child_priors_t.detach().cpu().numpy().reshape(-1)
            value_estimate = value_t.item()
            clk.record('detach_to_cpu', time.perf_counter() - t)

            # ── expand + backup (or just backup if checkmate) ─────────────────
            if (leaf.game.check_status()
                    and leaf.game.in_check_possible_moves() == []):
                t = time.perf_counter()
                leaf.backup(value_estimate)
                clk.record('backup', time.perf_counter() - t)
                continue

            t = time.perf_counter()
            leaf.expand(child_priors)   # includes board.actions() inside
            clk.record('expand (+ legal moves)', time.perf_counter() - t)

            t = time.perf_counter()
            leaf.backup(value_estimate)
            clk.record('backup', time.perf_counter() - t)

        return np.argmax(root.child_number_visits), root

    return profiled_uct_search


def _patch_alphazero_timing(player):
    """Wrap getMove on an AlphaZero instance to record per-move wall time."""
    if not isinstance(player, AlphaZero):
        return
    original = player.getMove

    def timed_getMove(moves):
        t = time.perf_counter()
        result = original(moves)
        if _profile_clock is not None:
            _profile_clock.move_times.append(time.perf_counter() - t)
        return result

    player.getMove = timed_getMove


def _print_cprofile_summary(pr: cProfile.Profile, top_n: int = 25) -> None:
    """Print the top functions by cumulative time, filtered to relevant code."""
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
    ps.print_stats(top_n)
    raw = buf.getvalue()

    W = 68
    print(f"{'═' * W}")
    print(f"  cProfile: top {top_n} functions by cumulative time")
    print(f"{'═' * W}")
    # print the header lines then the function rows
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # skip boilerplate pstats lines
        if stripped.startswith(('ncalls', 'Ordered', 'List', 'function')):
            print(f"  {line}")
            continue
        if stripped[0].isdigit() or '/' in stripped[:6]:
            print(f"  {line}")
        elif 'tottime' in stripped or 'cumtime' in stripped:
            print(f"  {line}")
    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='match.py',
        description='Play a game between two players (AlphaZero and/or Stockfish).')

    parser.add_argument('--aType', required=True)
    parser.add_argument('--aSFHash', type=int)
    parser.add_argument('--aSFDepth', type=int)
    parser.add_argument('--aAnetwork')
    parser.add_argument('--aAsteps', type=int)
    parser.add_argument('--bType', required=True)
    parser.add_argument('--bSFHash', type=int)
    parser.add_argument('--bSFDepth', type=int)
    parser.add_argument('--bAnetwork')
    parser.add_argument('--bAsteps', type=int)
    parser.add_argument('-p', '--printBoard', action='store_true',
                        help='display board after each move')
    parser.add_argument('--profile', action='store_true',
                        help='print a detailed per-phase performance breakdown '
                             'after the game')

    args = parser.parse_args()

    if args.aType == "alpha":
        playerA = AlphaZero(args.aAnetwork, args.aAsteps)
    elif args.aType == "stockfish":
        playerA = Stockfish(args.aSFHash, args.aSFDepth)
    else:
        sys.exit('need to pass alpha or stockfish as aType')

    if args.bType == "alpha":
        playerB = AlphaZero(args.bAnetwork, args.bAsteps)
    elif args.bType == "stockfish":
        playerB = Stockfish(args.bSFHash, args.bSFDepth)
    else:
        sys.exit('need to pass alpha or stockfish as bType')

    if args.profile:
        import AlphaZero_player as _az_module
        _profile_clock = _PhaseClock()
        # AlphaZero_player does `from MCTS_chess import UCT_search`, so we must
        # patch the name in that module's namespace, not in MCTS_chess itself.
        _az_module.UCT_search = _build_profiled_uct_search()
        _patch_alphazero_timing(playerA)
        _patch_alphazero_timing(playerB)

        pr = cProfile.Profile()
        pr.enable()
        match(playerA, playerB)
        pr.disable()

        _profile_clock.report()
        _print_cprofile_summary(pr, top_n=25)
    else:
        match(playerA, playerB)
