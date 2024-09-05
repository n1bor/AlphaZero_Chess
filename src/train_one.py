import os
import compress_pickle as pickle
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet
from config import rootDir
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from alpha_net import AlphaLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import random
import sys
import shutil
import config
import time
class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].transpose(2,0,1), self.y_p[idx], self.y_v[idx]

class board_data_all(IterableDataset):
    def __init__(self,directory): # dataset = np.array of (s, p, v)
        super().__init__()
        self.runtime=int(sys.argv[3])
        self.starttime=time.time()
        self.files = []
        for idx,file in enumerate(os.listdir(directory)):
            filename = os.path.join(directory,file)
            self.files.append(filename)
        self.loader=None
    def generate(self):
        random.shuffle(self.files)
        for file in self.files:
            if time.time()> self.starttime+self.runtime:
                break
            print(f"new file {file}")
            with open(file, 'rb') as fo:
                try:
                    data=pickle.load(fo)
                except EOFError:
                    print(f"EOFError in file {file}")
                    data=[]
            data = np.array(data,dtype="object")
            file_data=board_data(data)

            loader=iter(DataLoader(file_data, shuffle=True, pin_memory=False))
            while loader !=None and time.time()< self.starttime+self.runtime:
                #print(f"time left: {time.time()-self.starttime-self.runtime}")
                data_item=next(loader,None)
                if data_item==None:
                    loader=None
                else:
                    yield (torch.squeeze(data_item[0]),
                            torch.squeeze(data_item[1]),
                            torch.squeeze(data_item[2]))

    def __iter__(self):
        return iter(self.generate())


def train(net,train_path,lr,batch_size,cpu,run):
    #torch.manual_seed(cpu+run*123)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    roll_99=7
    roll_9=7
    
    train_data=board_data_all(train_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1, pin_memory=False)
    total_loss = 0.0
    losses_per_batch = []
    losses_99=[]
    losses_9=[]
    for i,data in enumerate(train_loader,0):
        #print(f"batch {i}")
        state, policy, value = data
        if cuda:
            state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
        else:
            state, policy, value = state.float(), policy.float(), value.float()
        optimizer.zero_grad()
        policy_pred, value_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
        loss = criterion(value_pred[:,0], value, policy_pred, policy)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        roll_99=roll_99*0.99+loss.item()*0.01
        roll_9=roll_9*0.9+loss.item()*0.1
        print(f"{loss.item()} {roll_9} {roll_99}")
        losses_per_batch.append(loss.item())
        losses_99.append(roll_99)
        losses_9.append(roll_9)
        
        if i % 100 == 99: 
            print('Process ID: %d [%5d points] total loss per batch: %.3f' %
                    (os.getpid(),(i + 1)*batch_size, total_loss/(i+1)))
            print("Policy     :",policy[0].argmax().item(),policy_pred[0].argmax().item())
            print("Policy Odds:",policy[0][policy[0].argmax().item()],policy[0][policy[0].argmax().item()])
            print("Value      :",value[0].item(),value_pred[0,0].item())
            print(f"Run       : {run}  LR: {lr}")
    return roll_99

#test_data=board_data_all(train_path,testCount)
if __name__=="__main__":
    batch_size=config.batch_size
    run=int(sys.argv[1])
    trainDir=sys.argv[2]

    mp.set_start_method("spawn",force=True)
    lr=config.lr
    # gather data
    train_path = rootDir+"/data/"+trainDir
    
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    #net.eval()
    # Runs Net training
    net_to_train=f"latest.gz"; 
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    current_net_filename = os.path.join(rootDir+"/data/model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename, weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])

    loss_99=train(net,train_path,lr,batch_size,0,run)
    #train(net,test_path,testCount,lr,1,batch_size,0,'test')
        #processes2 = []
        #for i in range(1):
        #    p2 = mp.Process(target=train,args=(net,train_path,trainingCount,lr,epochs,i))
        #    p2.start()
        #    processes2.append(p2)
        #for p2 in processes2:
        #    p2.join()
    filename=f"{rootDir}/data/model_data/model_{run}_{trainDir}_{loss_99:.2f}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M%S')}.gz"
    torch.save({'state_dict': net.state_dict()}, filename)
    shutil.copy(filename,f"{rootDir}/data/model_data/latest.gz")
