#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import compress_pickle as pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import datetime

if __name__=="__main__":
    for iteration in range(1):
        # Runs MCTS
        net_to_play="current_net_trained8_iter1.pth.tar"
        mp.set_start_method("spawn",force=True)
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        #net.eval()
        print("hi")
        #current_net_filename = os.path.join("./model_data/",\
        #                                net_to_play)
        #checkpoint = torch.load(current_net_filename)
        #net.load_state_dict(checkpoint['state_dict'])
        #processes1 = []
        #for i in range(6):
        #    p1 = mp.Process(target=MCTS_self_play,args=(net,2,i))
        #    p1.start()
        #    processes1.append(p1)
        #for p1 in processes1:
        #    p1.join()
            
        # Runs Net training
        net_to_train="start_net.gz"; 
        save_as="trained_net_%s.gz" % datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S")
        # gather data
        data_path = "/home/owensr/chess/data/games3"
        datasets = []
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            print(f"{idx} {filename}")
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo))
        print(len(datasets))
        print(len(datasets[0]))
        datasets = np.array(datasets,dtype="object")
        
        mp.set_start_method("spawn",force=True)
        net = ChessNet()
        
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.train()
        print("hi")
        current_net_filename = os.path.join("/home/owensr/chess/data/model_data/",\
                                        net_to_train)
        checkpoint = torch.load(current_net_filename, weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])
        
        

        processes2 = []
        for i in range(1):
            p2 = mp.Process(target=train,args=(net,datasets,0,10,i))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()
        # save results
        torch.save({'state_dict': net.state_dict()}, os.path.join("/home/owensr/chess/data/model_data/",\
                                        save_as))