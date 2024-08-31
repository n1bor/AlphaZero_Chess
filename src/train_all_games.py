#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import compress_pickle as pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import datetime
from config import rootDir

if __name__=="__main__":

    # Runs MCTS
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    #net.eval()
    print("hi")
                
    # Runs Net training
    net_to_train="start_net.gz"; 
    
    # gather data
    data_path = rootDir+"/data/train"
            
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.train()
    print("hi")
    current_net_filename = os.path.join(rootDir+"/data/model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename, weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])
    
    

    processes2 = []
    for i in range(1):
        p2 = mp.Process(target=train,args=(net,data_path,0,1,i))
        p2.start()
        processes2.append(p2)
    for p2 in processes2:
        p2.join()

    # save results
    save_as="trained_net_%s.gz" % datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S")
    torch.save({'state_dict': net.state_dict()}, os.path.join(rootDir+"/data/model_data/",\
                                    save_as))