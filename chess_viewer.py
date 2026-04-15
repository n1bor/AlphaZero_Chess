#!/usr/bin/env python3
"""
Chess game viewer — step through a game in coordinate notation.

Usage:
    python chess_viewer.py "e2e4 d7d5 e4d5 ..."
    python chess_viewer.py          # uses built-in example game

Controls:
    f / right arrow  — forward one move
    b / left arrow   — back one move
    q / Ctrl-C       — quit
"""

import copy
import select
import sys
import termios
import tty

UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
}

EXAMPLE_GAME = (
    "e2e4 h7h6 d2d4 g7g5 b1c3 b8c6 h2h4 h6h5 h4g5 g8f6 "
    "g5f6 c6d4 g1f3 f8g7 f6g7 e7e6 g7h8q e8e7 h8d8 e7d8 "
    "c1g5 f7f6 g5f6 d8e8 f3d4 c7c5 d4b5 e8f8 d1d6 f8e8 d6e7"
)


# ---------------------------------------------------------------------------
# Board representation
# ---------------------------------------------------------------------------

def initial_board():
    """Return the starting position as an 8×8 list of strings."""
    board = [['.'] * 8 for _ in range(8)]
    back = list('RNBQKBNR')
    board[0] = [p.lower() for p in back]   # black back rank  (row 0 = rank 8)
    board[1] = ['p'] * 8                    # black pawns
    board[6] = ['P'] * 8                    # white pawns
    board[7] = back[:]                      # white back rank  (row 7 = rank 1)
    return board


def sq(s):
    """'e4' → (row, col) where row 0 = rank 8, row 7 = rank 1."""
    col = ord(s[0]) - ord('a')
    row = 8 - int(s[1])
    return row, col


def apply_move(board, from_sq, to_sq, promo=None):
    """Return (new_board, captured_piece) with one move applied."""
    b = copy.deepcopy(board)
    fr, fc = sq(from_sq)
    tr, tc = sq(to_sq)
    piece = b[fr][fc]
    captured = b[tr][tc]

    # Castling — move the rook automatically
    if piece == 'K' and from_sq == 'e1':
        if to_sq == 'g1':
            b[7][5], b[7][7] = b[7][7], '.'
        elif to_sq == 'c1':
            b[7][3], b[7][0] = b[7][0], '.'
    elif piece == 'k' and from_sq == 'e8':
        if to_sq == 'g8':
            b[0][5], b[0][7] = b[0][7], '.'
        elif to_sq == 'c8':
            b[0][3], b[0][0] = b[0][0], '.'

    # En passant — pawn moves diagonally onto an empty square
    if piece in ('P', 'p') and fc != tc and b[tr][tc] == '.':
        captured = b[fr][tc]
        b[fr][tc] = '.'

    b[tr][tc] = piece
    b[fr][fc] = '.'

    # Promotion
    if promo:
        b[tr][tc] = promo.upper() if piece == 'P' else promo.lower()

    return b, captured


# ---------------------------------------------------------------------------
# Move parsing
# ---------------------------------------------------------------------------

def is_square(s):
    """Return True if s is a valid algebraic square like 'e4'."""
    return len(s) == 2 and s[0] in 'abcdefgh' and s[1] in '12345678'


def parse_moves(game_str):
    """Parse space-separated coordinate tokens like 'e2e4' or 'e7e8q'.
    Non-move tokens (annotations, comments, etc.) are silently skipped."""
    moves = []
    for token in game_str.strip().split():
        if len(token) in (4, 5) and is_square(token[:2]) and is_square(token[2:4]):
            promo = token[4] if len(token) == 5 else None
            moves.append((token[:2], token[2:4], promo))
        # else: silently skip annotations like "draw", "250", etc.
    return moves


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(board, move_idx, total, label, warning=None):
    lines = []
    lines.append("")
    lines.append(f"   Move {move_idx}/{total}   {label}")
    if warning:
        lines.append(f"   *** {warning} ***")
    lines.append("")
    lines.append("    a  b  c  d  e  f  g  h")
    lines.append("   ┌──┬──┬──┬──┬──┬──┬──┬──┐")
    for row in range(8):
        rank = 8 - row
        cells = [UNICODE_PIECES.get(board[row][col], ' ') for col in range(8)]
        lines.append(f" {rank} │" + "│".join(f" {c}" for c in cells) + "│")
        if row < 7:
            lines.append("   ├──┼──┼──┼──┼──┼──┼──┼──┤")
    lines.append("   └──┴──┴──┴──┴──┴──┴──┴──┘")
    lines.append("    a  b  c  d  e  f  g  h")
    lines.append("")
    lines.append("   [f / →] forward    [b / ←] backward    [q] quit")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terminal input
# ---------------------------------------------------------------------------

def get_key():
    """Read one keypress; translate arrow-key escape sequences."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            # Peek for CSI sequence  (ESC [ A/B/C/D)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == '[' and select.select([sys.stdin], [], [], 0.05)[0]:
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'C':
                        return 'RIGHT'
                    if ch3 == 'D':
                        return 'LEFT'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    game_str = sys.argv[1] if len(sys.argv) > 1 else EXAMPLE_GAME

    moves = parse_moves(game_str)
    if not moves:
        print("No valid moves found.")
        sys.exit(1)

    # Pre-compute every position
    boards = [initial_board()]
    labels = ["Starting position"]
    warnings = [None]
    for from_sq, to_sq, promo in moves:
        new_board, captured = apply_move(boards[-1], from_sq, to_sq, promo)
        boards.append(new_board)
        label = f"{from_sq}{to_sq}{promo or ''}"
        if captured in ('K', 'k'):
            color = "White" if captured == 'K' else "Black"
            warnings.append(f"ILLEGAL: {color} king captured by {label}!")
        else:
            warnings.append(None)
        labels.append(label)

    pos = 0
    total = len(moves)

    while True:
        print('\033[2J\033[H', end='', flush=True)   # clear screen
        print(render(boards[pos], pos, total, labels[pos], warnings[pos]))

        key = get_key()

        if key in ('q', 'Q', '\x03', '\x04'):        # q / Ctrl-C / Ctrl-D
            break
        elif key in ('f', 'F', 'RIGHT') and pos < total:
            pos += 1
        elif key in ('b', 'B', 'LEFT') and pos > 0:
            pos -= 1

    print()  # leave terminal in a clean state


if __name__ == '__main__':
    main()
