#!/usr/bin/env python3
from compress_pickle import dump
import copy
import pexpect
import re 
import numpy as np
import sys
import time
#sys.path.insert(0, '../AlphaZero_Chess/src')
from chess_board import board as c_board
import encoder_decoder as ed

import config

if len(sys.argv) < 2:
    print("need to pass run id and runtime")
    quit()


runId=sys.argv[1]
runtime=int(sys.argv[2])
print(f"RunID {runId} runtime (seconds) {runtime}")
startTime=time.time()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x=np.array(x)
    x=x/20
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def alg_to_xy(move):
    bx=ord(move[:1].lower())-97
    by=8-int(move[1:2])
    tx=ord(move[2:3].lower())-97
    ty=8-int(move[3:4])
    promote=move[4:5].upper() or None
    return ((by,bx),(ty,tx),promote)

def do_decode_n_move_pieces(board,i,f,p):
    board.move_piece(i,f,p) # move piece to get next board state s
    a,b = i; c,d = f
    if board.current_board[c,d] in ["K","k"] and abs(d-b) == 2: # if king moves 2 squares, then move rook too for castling
        if a == 7 and d-b > 0: # castle kingside for white
            board.player = 0
            board.move_piece((7,7),(7,5),None)
        if a == 7 and d-b < 0: # castle queenside for white
            board.player = 0
            board.move_piece((7,0),(7,3),None)
        if a == 0 and d-b > 0: # castle kingside for black
            board.player = 1
            board.move_piece((0,7),(0,5),None)
        if a == 0 and d-b < 0: # castle queenside for black
            board.player = 1
            board.move_piece((0,0),(0,3),None)
    return board

code=config.rootDir+"/stockfish/stockfish-ubuntu-x86-64-avx2"

child=pexpect.spawn(code)
child.expect('Stockfish')
child.sendline("setoption name MultiPV value 10")
child.sendline("setoption name Hash value 256")
child.sendline("setoption name UCI_ShowWDL value true")
child.sendline("setoption name SyzygyPath value setoption name SyzygyPath value")

child.sendline("uci")
child.expect("uciok")

dataset_p = []

for gameId in range(10000):
    if time.time()>startTime+runtime:
        print("exceeded runtime - exiting")
        break

    print(f"RunId {runId} game {gameId} remaining time {startTime+runtime-time.time()}")
    current_board = c_board()
    moveNumber=0
    moves=""
    
    continueGame=True
    while continueGame:
        board_state = copy.deepcopy(ed.encode_board(current_board))

        child.sendline("position startpos moves "+moves)
        child.sendline("go depth 10")
        i=child.expect([r"bestmove (.*) ponder",r"bestmove (.*)\r"])
        bestScore=-999999
        
        scoreList=[]
        moveList=[]
        for line in child.before.decode().splitlines():
            #print("l:"+line)
            m=re.match("info depth 10 seldepth (\d+) multipv (\d+) score (cp|mate) ([-]*\d+) wdl (\d+) (\d+) (\d+) nodes (\d+) nps (\d+) hashfull (\d+) tbhits (\d+) time (\d+) pv ([a-zA-Z0-9_]*)",line)
            if m:
                (_,_,mate,score,win,draw,lose,_,_,_,_,_,proposedMove)=m.groups()
                score=int(score)
                
                if mate=="mate":
                    if score>0:
                        score=1000+(10-score)*100
                    else:
                        score-=1000
                if score>bestScore:
                    bestScore=score
                    bestDraw=int(draw)
                    if int(win)>=int(lose):
                        bestValue=int(win)/1000.0
                    else:
                        bestValue=-int(lose)/1000.0

                if score>(bestScore-200):
                    scoreList.append(score)
                    moveList.append(proposedMove)
                

        if i==0:
            out=child.after.decode()    
            b=re.match(r"bestmove (.*) ponder",out)
        else:
            out=child.after.decode()
            b=re.match(r"bestmove (.*)\r",out)
        bestmove=b.group(1)
        
        if bestmove=="(none)":
            print(f"checkmate moves:{moveNumber}")
            break

        max=softmax(scoreList)
        newmove=np.random.choice(moveList,1,True,max)
        #print(f"N:{newmove} B:{bestmove} {moveNumber}")

        policy=np.zeros(4672)
        for (move,odds) in zip(moveList,max):
            (initial,final,promote)=alg_to_xy(move)
            match promote:
                case "Q":
                    promote="queen"
                case "R":
                    promote="rook"
                case "N":
                    promote="knight"
                case "B":
                    promote="bishop"
            try: 
                enc_index=ed.encode_action(current_board,initial,final,promote)
            except:
                continueGame=False
                print(f"moves: {moves} newmove {newmove}")
                break
            policy[enc_index]=odds
        dataset_p.append([copy.deepcopy(ed.encode_board(current_board)),
                        policy,bestValue])
        (initial,final,promote)=alg_to_xy(newmove[0])
        current_board=do_decode_n_move_pieces(current_board,initial,final,promote)

        #moves=moves+" "+bestmove
        moves=moves+" "+newmove[0]
        moveNumber+=1
        if moveNumber>500 or (bestDraw>997 and moveNumber>100):
            print(f"draw moves:{moveNumber}")
            break

print("saving games")
with open(f"{config.rootDir}/data/games/data_{runId}.gz", 'wb') as output:
         dump(dataset_p, output)
