from Player import Player
import pexpect
import re 

class Stockfish(Player):
    def __init__(self,hash: int,depth: int):
        self.hash=hash
        self.depth=depth

        self.code="/home/owensr/chess/stockfish/stockfish-ubuntu-x86-64-avx2"

        self.child=pexpect.spawn(self.code)
        self.child.expect('Stockfish')
        self.child.sendline("setoption name Hash value "+str(self.hash))
        self.child.sendline("setoption name SyzygyPath value setoption name SyzygyPath value")
        self.child.sendline("uci")
        self.child.expect("uciok")
        print(f"Stockfish starter {self.hash}, {self.depth}")
    def getMove(self,moves):

        self.child.sendline("position startpos moves "+' '.join(moves))
        if len(moves)<10:
            self.child.sendline("go movetime "+str(self.depth*10))
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