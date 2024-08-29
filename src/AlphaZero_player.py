from Player import Player
import torch
from alpha_net import ChessNet
from chess_board import board as c_board
from MCTS_chess import UCT_search,do_decode_n_move_pieces
import encoder_decoder as ed

class AlphaZero(Player):
    def __init__(self,parameterFile,steps):
        self.parameterFile=parameterFile
        self.steps=steps
        #self.current_board=None
        checkpoint = torch.load(self.parameterFile,weights_only=True)
                
        self.net = ChessNet()
        self.net.load_state_dict(checkpoint['state_dict'])
        cuda = torch.cuda.is_available()
        if cuda:
            self.net.cuda()
        print(f"{self}")
    #def newGame(self):
    #    self.current_board = c_board()
    
    def letterToPos(self,letter):
        return ord(letter.lower())-97

    def doMove(self,board,move):
        i_x=self.letterToPos(move[:1])
        i_y=8-int(move[1:2])
        f_x=self.letterToPos(move[2:3])
        f_y=8-int(move[3:4])
        if len(move)==5:
            prom=move[4:5].lower()
        else:
            prom='q'

        board.move_piece((i_y,i_x),(f_y,f_x),prom) # move piece to get next board state s
        a,b = i_y,i_x
        c,d = f_y,f_x
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
    
    def posToLetter(self,pos):
        return chr(pos+97)

    def getMoveText(self,i_pos,f_pos,prom):
        (iy,ix)=i_pos[0]
        (fy,fx)=f_pos[0]
        l1=self.posToLetter(ix.item())
        l2=str(8-iy.item())
        l3=self.posToLetter(fx.item())
        l4=str(8-fy.item())
        l5=''
        if prom[0] != None:
            l5=prom[0][:1].lower()
        return l1+l2+l3+l4+l5

    def getMove(self,moves):
        current_board = c_board()
        for move in moves:
            current_board = self.doMove(current_board,move)
        
        best_move, _ = UCT_search(current_board,self.steps,self.net)
        i_pos, f_pos, prom = ed.decode_action(current_board,best_move)
        #print(f"MOVE {i_pos} {f_pos} {prom}")
        best_move_txt=self.getMoveText(i_pos,f_pos,prom)
        #current_board=do_decode_n_move_pieces(current_board,best_move)
        if best_move_txt=='a8a15':
            return None

        pieceCount=0 
        for ib in range(8):
            for jb in range(8):
                if current_board.current_board[ib,jb]!=' ':
                    pieceCount+=1
        if pieceCount==2:
            print("onlykings")
            return "onlykings"
        return best_move_txt
    
    def getBoard(self,moves):
        current_board = c_board()
        for move in moves:
            current_board = self.doMove(current_board,move)
            #print(f"{move}")
            #for i in current_board.current_board.tolist():
            #    print(", ".join([str(l).rjust(3) for l in i]))
        #return current_board   
        return current_board.current_board.tolist()
    
    def __str__(self):
        return f"Alpha {self.steps} {self.parameterFile}"

        