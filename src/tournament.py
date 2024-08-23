from Stockfish import Stockfish
from match import match
import random
class Entry():
    def __init__(self,player):
        self.player=player
        self.elo=1000

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

if __name__=="__xmain__":

    print(f"{eloAB(1000,1000,1)}")
    print(f"{eloAB(1000,1000,0)}")
    print(f"{eloAB(1000,1000,-1)}")

    print(f"{eloAB(1100,900,1)}")
    print(f"{eloAB(1100,900,0)}")
    print(f"{eloAB(1100,900,-1)}")



if __name__=="__main__":
    players=[]
    
    players.append(Entry(Stockfish(100,1)))
    players.append(Entry(Stockfish(100,2)))
    players.append(Entry(Stockfish(100,4)))
    players.append(Entry(Stockfish(100,8)))
    players.append(Entry(Stockfish(100,10)))
    players.append(Entry(Stockfish(100,12)))
    players.append(Entry(Stockfish(100,15)))

    players.append(Entry(Stockfish(1,8)))
    players.append(Entry(Stockfish(10,8)))
    players.append(Entry(Stockfish(200,8)))
    players.append(Entry(Stockfish(400,8)))
    players.append(Entry(Stockfish(800,8)))

    for i in range(10000):
        print(f" MATCH {i}")
        playera=random.choice(players)
        playerb=random.choice(players)
        if playera !=playerb and abs(playera.elo-playerb.elo)<400:
            result=match(playera.player,playerb.player)
            (aElo,bElo)=eloAB(playera.elo,playerb.elo,result)
            playera.elo=aElo
            playerb.elo=bElo
    
        for player in sorted(players,key=lambda x: x.elo,reverse=True):
            print(f"{player.player.hash} {player.player.depth} {player.elo}")


            

