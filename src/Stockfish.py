from Player import Player
import pexpect
import re 

class Stockfish(Player):
    def __init__(self,hash: int,depth: int, skill=-1,elo=-1,endgames=True):
        self.hash=hash
        self.depth=depth
        self.skill=skill
        self.elo=elo
        self.endgames=endgames

        self.code="/home/owensr/chess/stockfish/stockfish-ubuntu-x86-64-avx2"

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
        self.child.sendline("uci")
        self.child.expect("uciok")
        print(f"{self}")
    def getMove(self,moves):

        self.child.sendline("position startpos moves "+' '.join(moves))
        if len(moves)<10:
            self.child.sendline("go movetime 100")
        else:
            self.child.sendline("go depth "+str(self.depth))
        
        i=self.child.expect([r"bestmove (.*) ponder",r"bestmove (.*)\r"])
        
        #print(self.child.before.decode())
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
    
    def __str__(self):
        return f"Stockfish h:{self.hash} d:{self.depth} sk:{self.skill} elo:{self.elo} end:{self.endgames}"

        