#!/usr/bin/env python3
"""
game_analyzer.py
Plays Stockfish depth-10 vs Stockfish depth-10, records all positions, then
lets the user step through the game (f=forward, b=back) showing at each
position:
  - Board display
  - Stockfish top-10 moves with scores and win probabilities
  - AlphaZero evaluation for steps in {200,400,800,1600} × c_puct in {1,2,4,8,16}

Usage:
    cd src
    python3 game_analyzer.py --network /home/owensr/chess/data/model_data/best.gz
    # Save/load game to skip re-playing
    python3 game_analyzer.py --network ... --save-moves game.txt
    python3 game_analyzer.py --network ... --load-moves game.txt
"""

import sys
import os
import argparse
import curses
import pexpect
import re
import copy
import time
import threading
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_board import board as c_board
from MCTS_chess import UCT_search, DummyNode, UCTNode
import encoder_decoder as ed
from alpha_net import ChessNet
import config

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SF_PATH = os.path.join(config.rootDir, "stockfish", "stockfish-ubuntu-x86-64-avx2")
SF_HASH_MB = 256
SF_DEPTH = 10
AZ_STEPS  = [200, 400, 800, 1600]
AZ_CPUCT  = [1, 2, 4, 8, 16]

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def cp_to_winprob(cp: float) -> float:
    """Centipawn score → probability that the moving side wins."""
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def action_to_uci(board: c_board, action_idx: int) -> str:
    """Decode a 0-4671 action index to UCI move string (e.g. 'e2e4')."""
    i_pos, f_pos, prom = ed.decode_action(board, action_idx)
    iy, ix = int(i_pos[0][0]), int(i_pos[0][1])
    fy, fx = int(f_pos[0][0]), int(f_pos[0][1])
    move = chr(ix + ord('a')) + str(8 - iy) + chr(fx + ord('a')) + str(8 - fy)
    if prom[0] is not None:
        move += prom[0][:1].lower()
    return move


def apply_move(board: c_board, move: str) -> c_board:
    """Apply one UCI move to board in-place and return it."""
    ix = ord(move[0].lower()) - 97
    iy = 8 - int(move[1])
    fx = ord(move[2].lower()) - 97
    fy = 8 - int(move[3])
    prom = move[4:5].lower() if len(move) == 5 else 'q'
    board.move_piece((iy, ix), (fy, fx), prom)
    if board.current_board[fy, fx] in ['K', 'k'] and abs(fx - ix) == 2:
        if iy == 7 and fx - ix > 0:   # white kingside
            board.player = 0; board.move_piece((7, 7), (7, 5), None)
        elif iy == 7 and fx - ix < 0:  # white queenside
            board.player = 0; board.move_piece((7, 0), (7, 3), None)
        elif iy == 0 and fx - ix > 0:  # black kingside
            board.player = 1; board.move_piece((0, 7), (0, 5), None)
        elif iy == 0 and fx - ix < 0:  # black queenside
            board.player = 1; board.move_piece((0, 0), (0, 3), None)
    return board


def build_board(moves: list) -> c_board:
    """Reconstruct board state from a list of UCI move strings."""
    board = c_board()
    for m in moves:
        apply_move(board, m)
    return board


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish game player
# ─────────────────────────────────────────────────────────────────────────────

class StockfishPlayer:
    def __init__(self, depth=10, hash_mb=256):
        self.depth = depth
        self.proc = pexpect.spawn(SF_PATH, encoding='utf-8', timeout=120)
        self.proc.expect('Stockfish')
        self.proc.sendline(f'setoption name Hash value {hash_mb}')
        self.proc.sendline('uci')
        self.proc.expect('uciok')

    def get_move(self, moves):
        move_str = ' '.join(moves)
        self.proc.sendline(f'position startpos moves {move_str}')
        cmd = 'go movetime 100' if len(moves) < 10 else f'go depth {self.depth}'
        self.proc.sendline(cmd)
        self.proc.expect([r'bestmove (\S+) ponder', r'bestmove (\S+)\r'], timeout=120)
        m = re.search(r'bestmove (\S+)', self.proc.after)
        bm = m.group(1) if m else None
        return None if (not bm or bm == '(none)') else bm

    def close(self):
        try:
            self.proc.sendline('quit')
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish multi-PV analyzer
# ─────────────────────────────────────────────────────────────────────────────

class StockfishAnalyzer:
    """Uses MultiPV to return the top N moves with scores."""

    def __init__(self, depth=10, num_pv=10, hash_mb=256):
        self.depth = depth
        self.num_pv = num_pv
        self.proc = pexpect.spawn(SF_PATH, encoding='utf-8', timeout=120)
        self.proc.expect('Stockfish')
        self.proc.sendline(f'setoption name Hash value {hash_mb}')
        self.proc.sendline(f'setoption name MultiPV value {num_pv}')
        self.proc.sendline('uci')
        self.proc.expect('uciok')

    def analyze(self, moves: list, player: int):
        """
        Returns (pv_list, white_win_prob) where:
          pv_list = [(move, cp, white_win_prob, rel_prob), ...]  sorted by rank
          white_win_prob = estimated probability of white winning this position
        player: 0=white to move, 1=black to move
        """
        move_str = ' '.join(moves) if moves else ''
        self.proc.sendline(f'position startpos moves {move_str}')
        self.proc.sendline(f'go depth {self.depth}')
        self.proc.expect(r'bestmove \S+', timeout=120)
        output = self.proc.before or ''

        # Keep only the deepest info line per multipv rank
        pv_data = {}
        for line in output.splitlines():
            m = re.search(
                r'multipv (\d+) score (cp|mate) (-?\d+).*? pv (\S+)', line)
            if m:
                pv_num = int(m.group(1))
                score_type = m.group(2)
                score_val = int(m.group(3))
                move = m.group(4)
                cp = score_val if score_type == 'cp' else (
                    30000 if score_val > 0 else -30000)
                pv_data[pv_num] = (move, cp)

        if not pv_data:
            return [], 0.5

        pv_list_raw = [pv_data[k] for k in sorted(pv_data)]

        # Scores are from the moving side's perspective; flip for black
        sign = 1 if player == 0 else -1
        scores_cp = np.array([cp for _, cp in pv_list_raw], dtype=float)
        white_win_probs = np.array(
            [cp_to_winprob(sign * cp) for cp in scores_cp])

        # Relative probabilities via softmax over raw scores
        t = scores_cp / 100.0
        t -= t.max()
        exp_t = np.exp(t)
        rel_probs = exp_t / exp_t.sum()

        pv_list = [
            (move, int(cp), float(white_win_probs[i]), float(rel_probs[i]))
            for i, (move, cp) in enumerate(pv_list_raw)
        ]
        return pv_list, float(white_win_probs[0])

    def close(self):
        try:
            self.proc.sendline('quit')
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# AlphaZero analyzer
# ─────────────────────────────────────────────────────────────────────────────

class AlphaZeroAnalyzer:
    def __init__(self, network_path: str):
        checkpoint = torch.load(network_path, weights_only=True,
                                map_location='cpu')
        prefix = '_orig_mod.'
        state_dict = {
            k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in checkpoint['state_dict'].items()
        }
        num_res = sum(1 for k in state_dict
                      if k.startswith('res_') and k.endswith('.conv1.weight'))
        policy_filters = state_dict['outblock.conv1.weight'].shape[0]
        self.net = ChessNet(num_res_blocks=num_res,
                            policy_filters=policy_filters)
        self.net.load_state_dict(state_dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.net.eval()

    def net_value_and_top3(self, board: c_board):
        """
        Single forward pass.
        Returns (value, top3) where:
          value  – float in [-1,+1] (+1 = white winning)
          top3   – list of up to 3 (move_uci, pct) from the raw policy,
                   masked to legal moves and normalised to 100 %
        """
        enc = ed.encode_board(board).transpose(2, 0, 1)
        tensor = torch.from_numpy(enc).float().to(self.device)
        with torch.no_grad():
            policy_t, v = self.net(tensor)
        value = float(v.item())
        policy = policy_t.cpu().numpy().reshape(-1)

        # Mask to legal moves only
        legal_idxs = []
        for action in board.actions():
            if action:
                i_pos, f_pos, underpromote = action
                try:
                    legal_idxs.append(
                        ed.encode_action(board, i_pos, f_pos, underpromote))
                except Exception:
                    pass

        if not legal_idxs:
            return value, []

        masked = np.zeros_like(policy)
        masked[legal_idxs] = policy[legal_idxs]
        total = masked.sum()
        if total > 0:
            masked /= total

        top3_idxs = np.argsort(masked)[::-1][:3]
        top3 = []
        for idx in top3_idxs:
            if masked[idx] <= 0:
                break
            try:
                move = action_to_uci(board, int(idx))
            except Exception:
                move = '????'
            top3.append((move, float(masked[idx] * 100)))
        return value, top3

    def analyze(self, board: c_board, steps: int, c_puct: float,
                stop_event: threading.Event = None):
        """
        Run MCTS; returns top-3 list [(move_uci, visit_pct), ...] or None if cancelled.
        Only legal moves are considered — restricted to root.action_idxes, which
        is populated by expand() from board.actions() during the search.
        """
        if stop_event and stop_event.is_set():
            return None
        board_copy = copy.deepcopy(board)
        _, root = UCT_search(board_copy, steps, self.net, c_puct)
        if stop_event and stop_event.is_set():
            return None

        legal_idxs = root.action_idxes   # set by expand(); contains only legal moves
        if not legal_idxs:
            return []

        visits = root.child_number_visits
        # Restrict to legal indices only
        legal_visits = [(idx, visits[idx]) for idx in legal_idxs if visits[idx] > 0]
        legal_visits.sort(key=lambda x: -x[1])
        total = sum(v for _, v in legal_visits)

        top3 = []
        for idx, v in legal_visits[:3]:
            try:
                move = action_to_uci(board_copy, int(idx))
            except Exception:
                move = '????'
            pct = float(v / total * 100) if total > 0 else 0.0
            top3.append((move, pct))
        return top3


# ─────────────────────────────────────────────────────────────────────────────
# Game play
# ─────────────────────────────────────────────────────────────────────────────

def play_game(depth: int = SF_DEPTH) -> list:
    """Play Stockfish depth vs depth, return list of UCI moves."""
    print(f"Playing Stockfish d{depth} vs d{depth}...")
    playerA = StockfishPlayer(depth=depth)
    playerB = StockfishPlayer(depth=depth)
    moves = []
    last_move = None
    while True:
        m = playerA.get_move(moves)
        if not m or m == last_move:
            break
        last_move = m; moves.append(m)

        m = playerB.get_move(moves)
        if not m or m == last_move:
            break
        last_move = m; moves.append(m)

        print(f"  {len(moves):3d}: {' '.join(moves[-2:])}    ", end='\r')
        if len(moves) >= 250:
            break

    print(f"\nGame done: {len(moves)} moves")
    playerA.close()
    playerB.close()
    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Shared state for background analysis threads
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sf_result   = None   # (pv_list, white_win_prob) or None
        self.az_results  = {}     # (steps, c_puct) -> [(move,pct),...] top3 | None
        self.az_net_val  = None   # raw NN value float in [-1,+1]
        self.az_net_top3 = None   # [(move,pct),...] top3 from raw NN policy
        self.pos_idx     = -1
        self.computing   = False


def _sf_worker(sf: StockfishAnalyzer, moves: list, player: int,
               state: AnalysisState, pos_idx: int, stop: threading.Event):
    result = sf.analyze(moves, player)
    with state.lock:
        if state.pos_idx == pos_idx and not stop.is_set():
            state.sf_result = result


def _az_worker(az: AlphaZeroAnalyzer, board: c_board,
               state: AnalysisState, pos_idx: int, stop: threading.Event):
    with state.lock:
        state.computing = True

    # Net value + policy top3 (fast single forward pass, do first)
    try:
        net_val, net_top3 = az.net_value_and_top3(board)
    except Exception:
        net_val, net_top3 = None, []
    with state.lock:
        if state.pos_idx != pos_idx:
            return
        state.az_net_val  = net_val
        state.az_net_top3 = net_top3

    # MCTS evaluations for every (steps, c_puct) combo
    for steps in AZ_STEPS:
        for c_puct in AZ_CPUCT:
            if stop.is_set():
                break
            result = az.analyze(board, steps, c_puct, stop)
            with state.lock:
                if state.pos_idx != pos_idx:
                    return
                state.az_results[(steps, c_puct)] = result
        if stop.is_set():
            break

    with state.lock:
        if state.pos_idx == pos_idx:
            state.computing = False


# ─────────────────────────────────────────────────────────────────────────────
# Curses display helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colour pair IDs
CP_WHITE_PIECE = 1
CP_BLACK_PIECE = 2
CP_HEADER      = 3
CP_LIGHT_SQ    = 4
CP_DARK_SQ     = 5
CP_COMPUTING   = 6


def _safe_addstr(win, row, col, text, attr=curses.A_NORMAL):
    h, w = win.getmaxyx()
    if row < 0 or row >= h:
        return
    available = w - col
    if available <= 0:
        return
    try:
        win.addstr(row, col, text[:available], attr)
    except curses.error:
        pass


def draw_board(win, board_arr: list, r0: int, c0: int):
    """Draw the 8×8 board starting at (r0, c0)."""
    file_labels = '     a  b  c  d  e  f  g  h'
    _safe_addstr(win, r0, c0, file_labels)
    _safe_addstr(win, r0 + 9, c0, file_labels)

    for row in range(8):
        rank = 8 - row
        _safe_addstr(win, r0 + 1 + row, c0, f'{rank} │')
        for col in range(8):
            piece = board_arr[row][col]
            is_light = (row + col) % 2 == 0
            sq_attr = curses.color_pair(CP_LIGHT_SQ if is_light else CP_DARK_SQ)
            # pad the cell
            _safe_addstr(win, r0 + 1 + row, c0 + 4 + col * 3, '   ', sq_attr)
            if piece == ' ':
                _safe_addstr(win, r0 + 1 + row, c0 + 5 + col * 3, '·', sq_attr)
            elif piece.isupper():
                _safe_addstr(win, r0 + 1 + row, c0 + 5 + col * 3, piece,
                             sq_attr | curses.color_pair(CP_WHITE_PIECE) | curses.A_BOLD)
            else:
                _safe_addstr(win, r0 + 1 + row, c0 + 5 + col * 3, piece,
                             sq_attr | curses.color_pair(CP_BLACK_PIECE) | curses.A_BOLD)
        _safe_addstr(win, r0 + 1 + row, c0 + 28, f'│ {rank}')


def draw_sf_panel(win, sf_result, player: int, r0: int, c0: int):
    """Stockfish analysis panel: top 10 moves."""
    if sf_result is None:
        _safe_addstr(win, r0, c0,
                     f'Stockfish d{SF_DEPTH}  [ computing... ]',
                     curses.color_pair(CP_COMPUTING))
        return
    pv_list, white_win = sf_result
    side = 'White' if player == 0 else 'Black'
    _safe_addstr(win, r0, c0,
                 f'Stockfish d{SF_DEPTH}  {side} to move  |  '
                 f'White win prob: {white_win*100:.1f}%',
                 curses.color_pair(CP_HEADER) | curses.A_BOLD)
    _safe_addstr(win, r0 + 1, c0,
                 f'  {"#":>2}  {"Move":<7} {"Score":>7}  '
                 f'{"W-win%":>7}  {"Rel%":>6}')
    _safe_addstr(win, r0 + 2, c0, '  ' + '─' * 40)
    for i, (move, cp, wwin, rel) in enumerate(pv_list[:10]):
        score_str = f'+{cp/100:.2f}' if cp >= 0 else f'{cp/100:.2f}'
        _safe_addstr(win, r0 + 3 + i, c0,
                     f'  {i+1:>2}  {move:<7} {score_str:>7}  '
                     f'{wwin*100:>6.1f}%  {rel*100:>5.1f}%')


def draw_az_panel(win, az_results: dict, az_net_val, az_net_top3,
                  computing: bool, player: int, r0: int, c0: int):
    """
    AlphaZero panel.

    Layout (rows relative to r0):
      0   Header: NN value + computing status
      1   NN policy: top-3 moves with raw policy probabilities
      2   Shared column-header row (c=1 … c=16)
      3   ─── Best move (#1) ───
      4   Net row (raw NN policy #1, same across all c_puct)
      5-8 MCTS steps 200/400/800/1600 for rank #1
      9   ─── 2nd move (#2) ───
      10  Net row for #2
      11-14 MCTS steps for rank #2
      15  ─── 3rd move (#3) ───
      16  Net row for #3
      17-20 MCTS steps for rank #3
    Total: 21 rows
    """
    CELL = 11   # chars per c_puct column

    # ── Header ──────────────────────────────────────────────────────────────
    # The NN is trained with values from the CURRENT PLAYER's perspective
    # (+1 = current player winning).  Flip sign for black so the display
    # always shows "white win probability".
    if az_net_val is not None:
        sign = 1 if player == 0 else -1
        white_pct = (sign * az_net_val + 1) / 2 * 100
        val_str = f'{az_net_val:+.3f}  →  {white_pct:.1f}% white'
    else:
        val_str = '...'
    status = '  [computing...]' if computing else ''
    _safe_addstr(win, r0, c0,
                 f'AlphaZero  |  NN value: {val_str}{status}',
                 curses.color_pair(CP_HEADER) | curses.A_BOLD)

    # ── NN policy top-3 ─────────────────────────────────────────────────────
    if az_net_top3:
        pol_parts = [f'#{i+1} {m} {p:.1f}%' for i, (m, p) in enumerate(az_net_top3)]
        pol_str = '    '.join(pol_parts)
    else:
        pol_str = '...'
    _safe_addstr(win, r0 + 1, c0,
                 f'NN policy:  {pol_str}',
                 curses.color_pair(CP_HEADER))

    # ── Shared column-header ─────────────────────────────────────────────────
    hdr = f'  {"steps":>6}  '
    for c in AZ_CPUCT:
        hdr += f'{"c="+str(c):^{CELL}}'
    _safe_addstr(win, r0 + 2, c0, hdr)

    # ── Three sub-tables (rank 0=best, 1=2nd, 2=3rd) ─────────────────────────
    rank_labels = ['Best move  (#1)', '2nd move  (#2)', '3rd move  (#3)']
    base = r0 + 3
    for rank in range(3):
        # Section separator / label
        _safe_addstr(win, base, c0,
                     f'  ─── {rank_labels[rank]} ' + '─' * 44,
                     curses.A_DIM)
        base += 1

        # Net row: raw NN policy for this rank (c_puct-independent → repeat)
        net_cell = (az_net_top3[rank] if az_net_top3 and rank < len(az_net_top3)
                    else None)
        net_row = f'  {"Net":>6}  '
        for _ in AZ_CPUCT:
            if net_cell:
                move, pct = net_cell
                net_row += f'{move+" "+f"{pct:.1f}%":^{CELL}}'
            else:
                net_row += f'{"...":^{CELL}}'
        _safe_addstr(win, base, c0, net_row,
                     curses.color_pair(CP_COMPUTING) if not net_cell else curses.A_NORMAL)
        base += 1

        # MCTS rows — one per step count
        for steps in AZ_STEPS:
            row_str = f'  {steps:>6}  '
            for c_puct in AZ_CPUCT:
                key = (steps, c_puct)
                if key not in az_results:
                    cell = '...'
                elif az_results[key] is None:
                    cell = 'cancel'
                elif rank < len(az_results[key]):
                    move, pct = az_results[key][rank]
                    cell = f'{move} {pct:.0f}%'
                else:
                    cell = 'N/A'
                row_str += f'{cell:^{CELL}}'
            _safe_addstr(win, base, c0, row_str)
            base += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main curses viewer loop
# ─────────────────────────────────────────────────────────────────────────────

def run_viewer(stdscr, moves: list,
               sf_analyzer: StockfishAnalyzer,
               az_analyzer: AlphaZeroAnalyzer):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_WHITE_PIECE, curses.COLOR_WHITE,   -1)
    curses.init_pair(CP_BLACK_PIECE, curses.COLOR_CYAN,    -1)
    curses.init_pair(CP_HEADER,      curses.COLOR_YELLOW,  -1)
    curses.init_pair(CP_LIGHT_SQ,    curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(CP_DARK_SQ,     curses.COLOR_WHITE,   curses.COLOR_BLACK)
    curses.init_pair(CP_COMPUTING,   curses.COLOR_MAGENTA, -1)
    stdscr.timeout(300)   # ms; allows background thread updates

    total = len(moves)    # positions 0 … total (inclusive)
    pos   = 0

    state = AnalysisState()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    def launch(idx: int):
        nonlocal threads
        # Cancel existing work
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
        stop_event.clear()
        threads = []

        board = build_board(moves[:idx])
        player = board.player

        with state.lock:
            state.sf_result   = None
            state.az_results  = {}
            state.az_net_val  = None
            state.az_net_top3 = None
            state.pos_idx     = idx
            state.computing   = True

        # Stockfish analysis thread
        t_sf = threading.Thread(
            target=_sf_worker,
            args=(sf_analyzer, list(moves[:idx]), player, state, idx, stop_event),
            daemon=True)
        t_sf.start()
        threads.append(t_sf)

        # AlphaZero thread (net forward pass + all MCTS combos)
        t_az = threading.Thread(
            target=_az_worker,
            args=(az_analyzer, copy.deepcopy(board), state, idx, stop_event),
            daemon=True)
        t_az.start()
        threads.append(t_az)

    launch(pos)

    while True:
        stdscr.erase()

        board = build_board(moves[:pos])
        board_arr = board.current_board.tolist()
        player = board.player
        side = 'White' if player == 0 else 'Black'

        # ── Header ────────────────────────────────────────────────────────────
        if pos < total:
            nxt = f'next: {moves[pos]}'
        else:
            nxt = 'end of game'
        _safe_addstr(stdscr, 0, 0,
                     f'Move {pos}/{total}  {side} to move  ({nxt})'
                     f'   [f]forward  [b]back  [q]quit',
                     curses.color_pair(CP_HEADER) | curses.A_BOLD)

        # ── Board ─────────────────────────────────────────────────────────────
        draw_board(stdscr, board_arr, 1, 0)

        # ── Analysis panels (read shared state under lock) ────────────────────
        with state.lock:
            sf_result    = state.sf_result
            az_results   = dict(state.az_results)
            az_net_val   = state.az_net_val
            az_net_top3  = list(state.az_net_top3) if state.az_net_top3 else None
            computing    = state.computing

        draw_sf_panel(stdscr, sf_result, player, 12, 0)
        draw_az_panel(stdscr, az_results, az_net_val, az_net_top3, computing, player, 26, 0)

        # ── Status line ───────────────────────────────────────────────────────
        done = len(az_results)
        total_az = len(AZ_STEPS) * len(AZ_CPUCT)
        _safe_addstr(stdscr, 50, 0,
                     f'AZ progress: {done}/{total_az} MCTS evals'
                     + ('  [working...]' if computing else '  [done]'),
                     curses.color_pair(CP_COMPUTING) if computing else curses.A_DIM)

        stdscr.refresh()

        # ── Input ─────────────────────────────────────────────────────────────
        key = stdscr.getch()
        if key in (ord('q'), ord('Q'), 27):          # q / ESC
            break
        elif key in (ord('f'), curses.KEY_RIGHT, ord(' ')):
            if pos < total:
                pos += 1
                launch(pos)
        elif key in (ord('b'), curses.KEY_LEFT):
            if pos > 0:
                pos -= 1
                launch(pos)

    # Shutdown
    stop_event.set()
    for t in threads:
        t.join(timeout=3)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stockfish d10 vs d10 — interactive game analyzer')
    parser.add_argument('--network', required=True,
                        help='Path to AlphaZero network .gz file')
    parser.add_argument('--sf-depth', type=int, default=SF_DEPTH,
                        help=f'Stockfish depth for the game (default {SF_DEPTH})')
    parser.add_argument('--save-moves', default=None,
                        help='Save recorded game moves to this file')
    parser.add_argument('--load-moves', default=None,
                        help='Load moves from file instead of playing a new game')
    args = parser.parse_args()

    if args.load_moves:
        with open(args.load_moves) as f:
            moves = f.read().split()
        print(f"Loaded {len(moves)} moves from {args.load_moves}")
    else:
        moves = play_game(depth=args.sf_depth)
        if args.save_moves:
            with open(args.save_moves, 'w') as f:
                f.write(' '.join(moves))
            print(f"Saved moves to {args.save_moves}")

    print("Loading AlphaZero network...")
    az = AlphaZeroAnalyzer(args.network)

    print("Starting Stockfish analyzer...")
    sf = StockfishAnalyzer(depth=SF_DEPTH, num_pv=10, hash_mb=SF_HASH_MB)

    print("Entering viewer (terminal must be ≥80 wide × 52 tall)...")
    print("Controls: [f] / Right-arrow = forward,  [b] / Left-arrow = back,  [q] = quit")
    print("Note: AlphaZero evaluations for 1600 steps may take 1-2 min each on CPU.")
    time.sleep(1.5)

    try:
        curses.wrapper(run_viewer, moves, sf, az)
    finally:
        sf.close()

    print("Done.")


if __name__ == '__main__':
    main()
