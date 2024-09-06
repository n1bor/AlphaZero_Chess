from Stockfish import Stockfish
from match import match
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass
from Player import Player
from AlphaZero_player import AlphaZero
import random
import pexpect
class Entry():
    def __init__(self,player):
        self.player=player
        self.elo=1000
        self.win=0
        self.draw=0
        self.lose=0
    def __str__(self) -> str:
        return f"{self.player} : {self.elo} {self.win}/{self.draw}/{self.lose}"

def eloAB(ra,rb,result):
    qa=10**(ra/400)
    qb=10**(rb/400)
    ea=qa/(qa+qb)
    eb=qb/(qa+qb)
       
    sa=(result+1)/2
    sb=1-sa

    aElo=ra+32*(sa-ea)
    bElo=rb+32*(sb-eb)
    return aElo,bElo

@dataclass
class MatchDetails:
    thread: object
    a: Player
    b: Player

if __name__=="__main__":
    players=[]

    players.append(Entry(Stockfish(100,1,elo=1320)))
    players.append(Entry(AlphaZero('/home/owensr/chess/data/model_data/model_28_train_3.62_2024-09-06-184211.gz',2)))
    players.append(Entry(AlphaZero('/home/owensr/chess/data/model_data/model_28_train_3.62_2024-09-06-184211.gz',7)))
    players.append(Entry(AlphaZero('/home/owensr/chess/data/model_data/model_28_train_3.62_2024-09-06-184211.gz',15)))
    players.append(Entry(AlphaZero('/home/owensr/chess/data/model_data/model_13_train_4.44_2024-09-06-094323.gz',2)))
    players.append(Entry(AlphaZero('/home/owensr/chess/data/model_data/random.gz',2)))
                
    pool = ThreadPool(processes=1)
    for i in range(10000):
        print(f" MATCH {i}")
        matches=[]
        playera=None
        for player in sorted(players,key=lambda x: x.elo+random.randint(0,200),reverse=True):
            if playera==None:
                playera=player
            else:
                if random.choice([True, False]): # switch black white randomly.
                    x=pool.apply_async(match,(playera.player,player.player))
                    matches.append(MatchDetails(x,playera,player))
                else:
                    x=pool.apply_async(match,(player.player,playera.player))
                    matches.append(MatchDetails(x,player,playera))
                playera=None
        for matchEntry in matches:
            try:
                result=matchEntry.thread.get()
                (aElo,bElo)=eloAB(matchEntry.a.elo,matchEntry.b.elo,result)
                matchEntry.a.elo=aElo
                matchEntry.b.elo=bElo
                if result==0:
                    matchEntry.a.draw +=1
                    matchEntry.b.draw +=1
                if result==1:
                    matchEntry.a.win +=1
                    matchEntry.b.lose +=1
                if result==-1:
                    matchEntry.a.lose +=1
                    matchEntry.b.win +=1
            except pexpect.exceptions.TIMEOUT as e:
                print(e)
                pass

        for player in sorted(players,key=lambda x: x.elo,reverse=True):
            print(f"{player}")


            

