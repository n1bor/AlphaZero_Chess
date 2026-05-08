import numpy as np
from Player import Player
import pexpect
import re

class Stockfish(Player):
    def __init__(self, hash: int, depth: int, skill=-1, elo=-1, endgames=True, random_plies=0):
        self.hash = hash
        self.depth = depth
        self.skill = skill
        self.elo = elo
        self.endgames = endgames
        self.random_plies = random_plies  # randomise SF's choice for this many plies

        self.code="/workspace/chess/stockfish/stockfish-ubuntu-x86-64-avx2"

        self.child=pexpect.spawn(self.code)
        self.child.expect('Stockfish')
        self.child.sendline("setoption name Hash value "+str(self.hash))
        if self.endgames:
            self.child.sendline("setoption name SyzygyPath value /home/owensr/chess/stockfish/syzygy")
        if self.skill!=-1:
            self.child.sendline("setoption name Skill Level value "+str(self.skill))
        if self.elo!=-1:
            self.child.sendline("setoption name UCI_LimitStrength value true")
            self.child.sendline("setoption name UCI_Elo value "+str(self.elo))
        if self.random_plies > 0:
            self.child.sendline("setoption name MultiPV value 5")
        self.child.sendline("uci")
        self.child.expect("uciok")
        print(f"{self}")

    def getMove(self, moves):
        self.child.sendline("position startpos moves "+' '.join(moves))
        self.child.sendline("go depth "+str(self.depth))

        i=self.child.expect([r"bestmove (.*) ponder",r"bestmove (.*)\r"])

        if i==0:
            out=self.child.after.decode()
            b=re.match(r"bestmove (.*) ponder",out)
        else:
            out=self.child.after.decode()
            b=re.match(r"bestmove (.*)\r",out)
        bestmove=b.group(1)

        if bestmove=="(none)":
            print(f"checkmate moves:{len(moves)}")
            return None

        # Random opening: for the first random_plies half-moves, pick a move
        # from the top-5 MultiPV list weighted by a softmax over the scores.
        # Temperature = 50 cp: a 1-pawn difference makes the best move ~7× more
        # likely than the alternative, so quality is preserved but games vary.
        if self.random_plies > 0 and len(moves) < self.random_plies:
            score_move = []
            for line in self.child.before.decode().splitlines():
                m = re.search(
                    rf"depth {self.depth} \S+ \d+ multipv \d+ score (cp|mate) ([-]?\d+).*? pv (\S+)",
                    line,
                )
                if m:
                    mate, score_str, move = m.groups()
                    score = int(score_str)
                    if mate == "mate":
                        score = 10000 if score > 0 else -10000
                    score_move.append((score, move))
            if len(score_move) > 1:
                scores = np.array([s for s, _ in score_move], dtype=float)
                scores = (scores - scores.max()) / 50.0
                weights = np.exp(scores)
                weights /= weights.sum()
                return score_move[np.random.choice(len(score_move), p=weights)][1]

        return bestmove
    def getBoard(self,moves):
        self.child.sendline("position startpos moves "+' '.join(moves))
        self.child.sendline("d")
        self.child.expect("  a   b   c   d   e   f   g   h")
        
        board=[None]*8

        for row in re.findall(r".*\| (.) \| (.) \| (.) \| (.) \| (.) \| (.) \| (.) \| (.) \| (.)",
                              self.child.before.decode()):
            board[8-int(row[8])]=list(row[:8])
        return board
    
    def close(self):
        try:
            self.child.sendline("quit")
            self.child.close(force=True)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __str__(self):
        return f"Stockfish h:{self.hash} d:{self.depth} sk:{self.skill} elo:{self.elo} end:{self.endgames}"

        
