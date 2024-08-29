import argparse
import sys
from dataclasses import dataclass
from Stockfish import Stockfish
from Player import Player
from AlphaZero_player import AlphaZero



def match(a,b):
    moves=[]
    lastmove=""
    checkmate=False
    result=0
    while not checkmate:
        move=a.getMove(moves)
        #print(f"PlayerA {move} {len(moves)}")
        if move=="onlykings":
            print(f"draw - only kings {' '.join(moves)}")
            break
        if move==None:
            print(f"Player B {b} won {' '.join(moves)}")
            checkmate=True
            result=-1
            break
        if move==lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove=move
        moves.append(move)
        move=b.getMove(moves)
        if move=="onlykings":
            print("draw - only kings {moves}")
            break
        #print(f"PlayerB {move} {len(moves)}")
        if move==None:
            print(f"Player A {a} won {' '.join(moves)}")
            result=1
            checkmate=True
            break

        if move==lastmove:
            print(f"error {' '.join(moves)}")
            break
        lastmove=move

        moves.append(move)
        if len(moves)>250:
            print(f"draw 250 {' '.join(moves)}")
            break
     
    aBoard=a.getBoard(moves) 
    bBoard=b.getBoard(moves)
    for index, item_list in enumerate(aBoard):
        if item_list != bBoard[index]:
            print(f"{aBoard} != {bBoard} for {' '.join(moves)}")
            raise Exception(f"{aBoard} != {bBoard} for {' '.join(moves)}")

    return result



if __name__=="__main__":
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('--aType')
    parser.add_argument('--aSFHash', type=int)
    parser.add_argument('--aSFDepth', type=int)
    #parser.add_argument('--aSFELO', type=int)
    parser.add_argument('--aAnetwork')
    parser.add_argument('--aAsteps', type=int)
    parser.add_argument('--bType')
    parser.add_argument('--bSFHash', type=int)
    parser.add_argument('--bSFDepth', type=int)
    #parser.add_argument('--bSFELO', type=int)
    parser.add_argument('--bAnetwork')
    parser.add_argument('--bAsteps', type=int)
    parser.add_argument("-p", "--printBoard", action="store_true",
                        help="display board after each move")

    args = parser.parse_args()

    if args.aType=="alpha":
        playerA=AlphaZero(args.aAnetwork, args.aAsteps)
    elif args.aType=="stockfish" :
        playerA=Stockfish(args.aSFHash,args.aSFDepth)
    else :
        sys.exit('need to pass alpha or stockfish as aType')

    if args.bType=="alpha":
        playerB=AlphaZero(args.bAnetwork, args.bAsteps)
    elif args.aType=="stockfish" :
        playerB=Stockfish(args.bSFHash,args.bSFDepth)
    else :
        sys.exit('need to pass alpha or stockfish as bType')
    
    match(playerA,playerB)
