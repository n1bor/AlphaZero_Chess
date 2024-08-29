from AlphaZero_player import AlphaZero

a=AlphaZero('/home/owensr/chess/data/model_data/random.gz',8)
moves=['c2c4','c7c5','e2e3','g7g6','f1e2','f8g7','h2h3','b7b6','b1c3','e7e6','d1a4','c8b7','a4a5','b6a5','e2d1','b7g2','d2d4','d8c7','a2a4','b8c6','g1e2','a8b8']

board=a.getBoard(moves)
#for i in board:
#    print(", ".join([str(l).rjust(3) for l in i]))
print(f"{board.actions()}")
