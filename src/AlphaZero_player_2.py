"""
AlphaZero_player_2.py — optimised AlphaZero player for tournament_2.py

Changes versus AlphaZero_player.py:
  #2 MCTS tree reuse: after each move the child subtree is kept and used as
     the starting root for the next search, so prior simulations are not wasted.
  #4 Incremental board state: the board is updated move-by-move instead of
     replaying all moves from the start on every getMove call (O(n) vs O(n²)).
  #5 torch.compile: reduces GPU kernel-launch overhead on repeated fixed-size
     forward passes (most effective when a worker plays multiple games).
"""

import numpy as np
import torch

from Player import Player
from alpha_net import ChessNet
from chess_board import board as c_board
from MCTS_chess_2 import UCT_search_batched, detach_as_root
import encoder_decoder as ed


def _text_to_action_idx(board, move_text: str):
    """Convert a UCI move string to a policy action index for the given board.

    Returns None if encoding fails (tree reuse is then skipped for this move).
    """
    i_x = ord(move_text[0].lower()) - 97
    i_y = 8 - int(move_text[1])
    f_x = ord(move_text[2].lower()) - 97
    f_y = 8 - int(move_text[3])
    prom_char = move_text[4:5].lower() if len(move_text) == 5 else None
    prom_map = {'q': 'queen', 'r': 'rook', 'n': 'knight', 'b': 'bishop'}
    underpromote = prom_map.get(prom_char)  # None for normal moves
    try:
        return ed.encode_action(board, (i_y, i_x), (f_y, f_x), underpromote)
    except Exception:
        return None


class AlphaZero2(Player):
    def __init__(self, parameterFile, steps, c_puct=1):
        self.parameterFile = parameterFile
        self.steps = steps
        self.c_puct = c_puct

        checkpoint = torch.load(parameterFile, weights_only=True, map_location='cpu')
        remove_prefix = '_orig_mod.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v
                      for k, v in checkpoint['state_dict'].items()}

        num_res_blocks = sum(1 for k in state_dict
                             if k.startswith('res_') and k.endswith('.conv1.weight'))
        policy_filters = state_dict['outblock.conv1.weight'].shape[0]

        self.net = ChessNet(num_res_blocks=num_res_blocks, policy_filters=policy_filters)
        self.net.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

        # torch.compile is intentionally omitted here.  mode="reduce-overhead"
        # uses CUDA graphs, which require fixed tensor shapes.  We send shape
        # [1, 22, 8, 8] for the bootstrap pass and [BATCH_SIZE, 22, 8, 8] for
        # every subsequent batch — two shapes means two graphs captured
        # simultaneously across 12 workers, exhausting VRAM and killing workers.

        print(f"{self}")

        # Incremental board (#4): avoid replaying all moves on each call.
        self._board: c_board | None = None
        self._n_applied: int = 0  # moves from the list already applied to _board

        # Tree-reuse state (#2): subtree rooted at our last move's child.
        # On the next call we advance it by the opponent's reply before searching.
        self._pending_root = None

    # ── move helpers (identical to AlphaZero_player) ──────────────────────────

    @staticmethod
    def _letter_to_pos(letter: str) -> int:
        return ord(letter.lower()) - 97

    @staticmethod
    def _pos_to_letter(pos: int) -> str:
        return chr(pos + 97)

    def _do_move(self, board: c_board, move: str) -> c_board:
        i_x = self._letter_to_pos(move[0])
        i_y = 8 - int(move[1])
        f_x = self._letter_to_pos(move[2])
        f_y = 8 - int(move[3])
        prom = move[4:5].lower() if len(move) == 5 else 'q'
        board.move_piece((i_y, i_x), (f_y, f_x), prom)
        a, b = i_y, i_x
        c, d = f_y, f_x
        if board.current_board[c, d] in ["K", "k"] and abs(d - b) == 2:
            if a == 7 and d - b > 0:
                board.player = 0; board.move_piece((7, 7), (7, 5), None)
            if a == 7 and d - b < 0:
                board.player = 0; board.move_piece((7, 0), (7, 3), None)
            if a == 0 and d - b > 0:
                board.player = 1; board.move_piece((0, 7), (0, 5), None)
            if a == 0 and d - b < 0:
                board.player = 1; board.move_piece((0, 0), (0, 3), None)
        return board

    def _get_move_text(self, i_pos, f_pos, prom) -> str:
        iy, ix = i_pos[0]
        fy, fx = f_pos[0]
        l5 = prom[0][:1].lower() if prom[0] is not None else ''
        return (self._pos_to_letter(ix.item()) + str(8 - iy.item()) +
                self._pos_to_letter(fx.item()) + str(8 - fy.item()) + l5)

    # ── main interface ─────────────────────────────────────────────────────────

    def getMove(self, moves):
        # Initialise on first call
        if self._board is None:
            self._board = c_board()
            self._n_applied = 0
            self._pending_root = None

        # Apply every move that has been played since our last getMove call.
        # _text_to_action_idx is called BEFORE applying the move so the board
        # matches the pending root's game state.
        while self._n_applied < len(moves):
            new_move = moves[self._n_applied]
            if self._pending_root is not None:
                idx = _text_to_action_idx(self._board, new_move)
                if idx is not None and idx in self._pending_root.children:
                    self._pending_root = detach_as_root(
                        self._pending_root.children[idx])
                else:
                    self._pending_root = None  # cache miss — start fresh
            self._do_move(self._board, new_move)
            self._n_applied += 1

        # Search, optionally continuing from the reused subtree.
        best_move_idx, root = UCT_search_batched(
            self._board, self.steps, self.net, self.c_puct,
            root=self._pending_root,
        )
        self._pending_root = None

        i_pos, f_pos, prom = ed.decode_action(self._board, best_move_idx)
        best_move_txt = self._get_move_text(i_pos, f_pos, prom)

        if best_move_txt == 'a8a15':
            return None

        if np.count_nonzero(self._board.current_board != ' ') == 2:
            print("onlykings")
            return "onlykings"

        # Apply our move to the incremental board and save the child subtree.
        self._do_move(self._board, best_move_txt)
        self._n_applied += 1

        if best_move_idx in root.children:
            self._pending_root = detach_as_root(root.children[best_move_idx])

        return best_move_txt

    def getBoard(self, moves):
        board = c_board()
        for move in moves:
            self._do_move(board, move)
        return board.current_board.tolist()

    def __str__(self):
        return f"Alpha2 steps={self.steps} c_puct={self.c_puct} {self.parameterFile}"
