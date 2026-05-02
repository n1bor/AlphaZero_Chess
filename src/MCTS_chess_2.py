"""
MCTS_chess_2.py — optimised MCTS for tournament_2.py

Changes versus MCTS_chess.py:
  #1 Batched leaf evaluation: collect BATCH_SIZE leaves per iteration, run one
     GPU forward pass instead of one per simulation step.
  #3 Fast board copy: maybe_add_child uses copy.copy (triggers board.__copy__)
     instead of copy.deepcopy — 5-10× faster per node creation.

Virtual loss is applied during batch leaf selection so parallel selects are
steered to different paths.
"""

import collections
import copy
import math

import numpy as np
import torch

import encoder_decoder as ed

# Leaves collected per GPU forward pass.  8 is a good default for a single
# RTX 4080 with 12 parallel worker processes; increase if GPU utilisation is low.
BATCH_SIZE = 8

# ── action-index LRU cache (shared with MCTS_chess.py logic) ─────────────────

_ACTIONS_CACHE_MAXSIZE = 50_000
_actions_cache: collections.OrderedDict = collections.OrderedDict()


def _cached_action_idxs(game) -> list:
    key = (game.current_board.tobytes(), game.player,
           game.en_passant,
           game.r1_move_count, game.r2_move_count, game.k_move_count,
           game.R1_move_count, game.R2_move_count, game.K_move_count)
    if key in _actions_cache:
        _actions_cache.move_to_end(key)
        return _actions_cache[key]
    idxs = []
    for action in game.actions():
        if action != []:
            initial_pos, final_pos, underpromote = action
            idxs.append(ed.encode_action(game, initial_pos, final_pos, underpromote))
    _actions_cache[key] = idxs
    if len(_actions_cache) > _ACTIONS_CACHE_MAXSIZE:
        _actions_cache.popitem(last=False)
    return idxs


# ── tree nodes ────────────────────────────────────────────────────────────────

class UCTNode:
    def __init__(self, game, move, parent=None, c_puct=1):
        self.game = game
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.action_idxes = []
        self.c_puct = c_puct

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return self.c_puct * math.sqrt(self.number_visits) * (
            abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes:
            scores = self.child_Q() + self.child_U()
            return self.action_idxes[np.argmax(scores[self.action_idxes])]
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        """Traverse to an unexpanded leaf, returning (leaf, path).

        path is the list of nodes visited FROM root's first child down to leaf
        (inclusive).  It is used to apply and revert virtual loss.
        """
        current = self
        path = []
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
            path.append(current)
        return current, path

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid = child_priors[action_idxs]
        valid = 0.75 * valid + 0.25 * np.random.dirichlet(
            np.zeros([len(valid)], dtype=np.float32) + 0.3)
        child_priors[action_idxs] = valid
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = _cached_action_idxs(self.game)
        if not action_idxs:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p = np.zeros_like(child_priors)
        if action_idxs:
            c_p[action_idxs] = child_priors[action_idxs]
        if self.parent.parent is None:  # root node gets Dirichlet noise
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def _apply_move(self, board, move):
        """Apply a decoded MCTS action to a board (including castling rook)."""
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.player = self.game.player
            board.move_piece(i, f, p)
            a, b = i; c, d = f
            if board.current_board[c, d] in ["K", "k"] and abs(d - b) == 2:
                if a == 7 and d - b > 0:
                    board.player = self.game.player; board.move_piece((7, 7), (7, 5), None)
                if a == 7 and d - b < 0:
                    board.player = self.game.player; board.move_piece((7, 0), (7, 3), None)
                if a == 0 and d - b > 0:
                    board.player = self.game.player; board.move_piece((0, 7), (0, 5), None)
                if a == 0 and d - b < 0:
                    board.player = self.game.player; board.move_piece((0, 0), (0, 3), None)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            # copy.copy triggers board.__copy__ — far faster than deepcopy
            new_board = copy.copy(self.game)
            new_board = self._apply_move(new_board, move)
            self.children[move] = UCTNode(new_board, move, parent=self, c_puct=self.c_puct)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            value_estimate = -value_estimate
            current.total_value += value_estimate
            current = current.parent


class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


# ── virtual loss helpers ──────────────────────────────────────────────────────

def _apply_virtual_loss(path, vl: int = 1):
    """Temporarily penalise path nodes so parallel selects prefer other paths."""
    for node in path:
        node.number_visits += vl
        node.total_value -= vl


def _revert_virtual_loss(path, vl: int = 1):
    for node in path:
        node.number_visits -= vl
        node.total_value += vl


# ── tree-reuse helpers ────────────────────────────────────────────────────────

def detach_as_root(node: UCTNode) -> UCTNode:
    """Promote a child node to search root, preserving its visit statistics."""
    dummy = DummyNode()
    # Read from the old parent before re-parenting
    dummy.child_number_visits[node.move] = node.number_visits
    dummy.child_total_value[node.move] = node.total_value
    node.parent = dummy
    return node


def _drop_subtree(node: UCTNode) -> None:
    """Recursively null out parent references and clear children.

    UCTNode forms reference cycles (parent ↔ child via .parent and
    .children[]).  Python's reference counter cannot break these, so
    without explicit help the cycle GC must do it — often long after
    the subtree is logically dead.  Setting .parent = None on every
    node in the subtree removes the child→parent leg of each cycle,
    reducing every node's refcount to zero as soon as its parent's
    .children dict is cleared, so the reference counter frees the
    whole subtree immediately.
    """
    node.parent = None
    for child in node.children.values():
        _drop_subtree(child)
    node.children.clear()


# Minimum visit count a child must have to be kept in the reused subtree.
# Children below this threshold are pruned after each move to stop the
# pending root growing without bound.  10 ≈ 0.3% of 3200 steps; well-
# visited paths are retained, rarely-explored branches are discarded.
PRUNE_MIN_VISITS = 10


def prune_tree(node: UCTNode, min_visits: int = PRUNE_MIN_VISITS) -> None:
    """Drop low-visit children to keep the reused tree memory-bounded.

    Traverses the subtree rooted at *node* and drops any child whose
    visit count (in the parent's child_number_visits array) is below
    *min_visits*.  Uses _drop_subtree so dropped nodes are freed
    immediately by the reference counter rather than waiting for the
    cycle GC.
    """
    to_drop = [m for m in list(node.children)
               if node.child_number_visits[m] < min_visits]
    for m in to_drop:
        _drop_subtree(node.children.pop(m))
    for child in node.children.values():
        prune_tree(child, min_visits)


# ── main search ───────────────────────────────────────────────────────────────

def UCT_search_batched(game_state, num_reads: int, net, c_puct: float = 1,
                       batch_size: int = BATCH_SIZE, root: UCTNode = None):
    """
    Run MCTS with batched neural-network evaluation.

    If `root` is supplied (tree reuse), `num_reads` additional simulations are
    run on top of whatever statistics the reused tree already contains.
    """
    device = next(net.parameters()).device

    if root is None:
        root = UCTNode(game_state, move=None, parent=DummyNode(), c_puct=c_puct)

    # Bootstrap: if root has never been evaluated, expand it with a single forward
    # pass first.  Without this, all batch-size leaves in the first iteration
    # would be the root itself (select_leaf returns immediately on an unexpanded
    # node, and virtual loss cannot redirect it), wasting batch_size-1 evaluations
    # on an identical board state.
    reads_done = 0
    if not root.is_expanded:
        enc = ed.encode_board(root.game).transpose(2, 0, 1)
        t = torch.from_numpy(enc[np.newaxis]).float().to(device)  # [1, 22, 8, 8]
        with torch.no_grad():
            cp, v = net(t)
        cp_np = cp.cpu().numpy().reshape(-1)
        v_scalar = float(v.item())
        if not (root.game.check_status() and root.game.in_check_possible_moves() == []):
            root.expand(cp_np)
        root.backup(v_scalar)
        reads_done = 1

    while reads_done < num_reads:
        this_batch = min(batch_size, num_reads - reads_done)
        leaves, paths = [], []

        for _ in range(this_batch):
            leaf, path = root.select_leaf()
            _apply_virtual_loss(path)
            leaves.append(leaf)
            paths.append(path)

        # One batched GPU call for the whole batch
        encoded = np.stack([
            ed.encode_board(leaf.game).transpose(2, 0, 1) for leaf in leaves
        ])  # [N, 22, 8, 8] — ConvBlock.view(-1, 22, 8, 8) handles any N
        tensor = torch.from_numpy(encoded).float().to(device)

        with torch.no_grad():
            child_priors_batch, value_batch = net(tensor)

        child_priors_np = child_priors_batch.cpu().numpy()   # [N, 4672]
        values_np = value_batch.cpu().numpy().flatten()       # [N]

        for leaf, path, child_priors, value in zip(
                leaves, paths, child_priors_np, values_np):
            _revert_virtual_loss(path)
            if leaf.game.check_status() and leaf.game.in_check_possible_moves() == []:
                leaf.backup(float(value))
            else:
                leaf.expand(child_priors.reshape(-1))
                leaf.backup(float(value))

        reads_done += this_batch

    return int(np.argmax(root.child_number_visits)), root
