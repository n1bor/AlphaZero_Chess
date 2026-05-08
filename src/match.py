import argparse
import sys
import time
import cProfile
import pstats
import io
from dataclasses import dataclass
import numpy as np
from chess_board import board as c_board
from Stockfish import Stockfish
from Player import Player
from AlphaZero_player import AlphaZero

MOVE_LIMIT = 500


def _apply_move_str(board, move_str):
    """Apply a UCI move string to a chess_board, including castling rook."""
    i_x = ord(move_str[0].lower()) - 97
    i_y = 8 - int(move_str[1])
    f_x = ord(move_str[2].lower()) - 97
    f_y = 8 - int(move_str[3])
    prom = move_str[4:5].lower() if len(move_str) == 5 else None
    board.move_piece((i_y, i_x), (f_y, f_x), prom)
    iy, ix = i_y, i_x
    fy, fx = f_y, f_x
    if board.current_board[fy, fx] in ["K", "k"] and abs(fx - ix) == 2:
        if iy == 7 and fx - ix > 0:
            board.player = 0; board.move_piece((7, 7), (7, 5), None)
        if iy == 7 and fx - ix < 0:
            board.player = 0; board.move_piece((7, 0), (7, 3), None)
        if iy == 0 and fx - ix > 0:
            board.player = 1; board.move_piece((0, 7), (0, 5), None)
        if iy == 0 and fx - ix < 0:
            board.player = 1; board.move_piece((0, 0), (0, 3), None)


def _is_insufficient(white, black, board):
    """Return True if the position is a theoretical draw by insufficient material."""
    if not white and not black:
        return True
    if not white and len(black) == 1 and black[0].lower() in ('n', 'b'):
        return True
    if not black and len(white) == 1 and white[0].upper() in ('N', 'B'):
        return True
    if (len(white) == 1 and white[0] == 'B' and
            len(black) == 1 and black[0] == 'b'):
        def bishop_colour(piece_char):
            positions = list(zip(*np.where(board == piece_char)))
            return (positions[0][0] + positions[0][1]) % 2 if positions else None
        if bishop_colour('B') == bishop_colour('b'):
            return True
    return False


def _draw_check(board, moves):
    """Return a draw description string if the position is drawn, else None."""
    pieces = [p for p in board.current_board.flatten()
              if p != ' ' and p.upper() != 'K']
    if not pieces:
        return f"draw - only kings after {len(moves)} moves"
    if board.no_progress_count >= 100:
        return f"draw - 50-move rule after {len(moves)} moves"
    white = [str(p) for p in pieces if p.isupper()]
    black = [str(p) for p in pieces if p.islower()]
    if _is_insufficient(white, black, board.current_board):
        return f"draw - insufficient material after {len(moves)} moves"
    if len(moves) >= MOVE_LIMIT:
        return f"draw - {MOVE_LIMIT} move limit"
    return None


def match(a, b):
    moves = []
    lastmove = ""
    result = 0
    board = c_board()

    while True:
        draw_msg = _draw_check(board, moves)
        if draw_msg:
            print(draw_msg)
            break

        # ── Player A moves ────────────────────────────────────────────────────
        move = a.getMove(moves)
        if move == "onlykings":
            print(f"draw - only kings after {len(moves)} moves")
            break
        if move is None:
            if board.check_status():
                print(f"Player B {b} won - checkmate after {len(moves)} moves")
                result = -1
            else:
                print(f"draw - stalemate after {len(moves)} moves")
            break
        if move == lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove = move
        moves.append(move)
        _apply_move_str(board, move)

        draw_msg = _draw_check(board, moves)
        if draw_msg:
            print(draw_msg)
            break

        # ── Player B moves ────────────────────────────────────────────────────
        move = b.getMove(moves)
        if move == "onlykings":
            print(f"draw - only kings after {len(moves)} moves")
            break
        if move is None:
            if board.check_status():
                print(f"Player A {a} won - checkmate after {len(moves)} moves")
                result = 1
            else:
                print(f"draw - stalemate after {len(moves)} moves")
            break
        if move == lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove = move
        moves.append(move)
        _apply_move_str(board, move)

    aBoard = a.getBoard(moves)
    bBoard = b.getBoard(moves)
    for index, item_list in enumerate(aBoard):
        if item_list != bBoard[index]:
            print(f"{aBoard} != {bBoard} for {' '.join(moves)}")
            raise Exception(f"{aBoard} != {bBoard} for {' '.join(moves)}")

    return result, moves


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
        result, moves = match(playerA, playerB)
        pr.disable()

        _profile_clock.report()
        _print_cprofile_summary(pr, top_n=25)
    else:
        result, moves = match(playerA, playerB)
