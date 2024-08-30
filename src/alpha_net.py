#!/usr/bin/env python
import compress_pickle as pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

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
        self.files = []
        for idx,file in enumerate(os.listdir(directory)):
            filename = os.path.join(directory,file)
            self.files.append(filename)
        
        self.loaders=[]        
    def files_left(self):
        return len(self.files)+len(self.loaders)
    
    def generate(self):
        print("a")
        while len(self.files)>0 or len(self.loaders)>0:
            data_item=None
            while data_item==None and not (len(self.files)==0 and len(self.loaders)==0):
                if len(self.loaders)<5 and len(self.files)>0:
                    file=random.choice(self.files)
                    self.files.remove(file)
                    print(f"new file {file}")
                    with open(file, 'rb') as fo:
                        try:
                            data=pickle.load(fo)
                        except EOFError:
                            print(f"EOFError in file {file}")
                            data=[]
                    data = np.array(data,dtype="object")
                    new_file_data=board_data(data)

                    newLoader=DataLoader(new_file_data,  shuffle=True, pin_memory=False)
                    self.loaders.append(iter(newLoader))
                loader=random.choice(self.loaders)
                data_item=next(loader,None)
                if data_item==None:
                    self.loaders.remove(loader)
            if data_item!=None:
                yield (torch.squeeze(data_item[0]),
                        torch.squeeze(data_item[1]),
                        torch.squeeze(data_item[2]))
    
    def __iter__(self):
        return iter(self.generate())
        

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
def train(net, data_directory, epoch_start=0, epoch_stop=20, cpu=0):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200], gamma=0.2)
    

    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        print(f"epoch {epoch} LR:{scheduler.get_last_lr()}")
        train_set = board_data_all(data_directory)
        train_loader = DataLoader(train_set, batch_size=30, shuffle=False, num_workers=0, pin_memory=False)
        total_loss = 0.0
        losses_per_batch = []
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
            if i % 10 == 9:    # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d points %d files] total loss per batch: %.3f' %
                        (os.getpid(), epoch + 1, (i + 1)*30,train_set.files_left(),  total_loss/10))
                print("Policy     :",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Policy Odds:",policy[0][policy[0].argmax().item()],policy_pred[0][policy_pred[0].argmax().item()])
                print("Value      :",value[0].item(),value_pred[0,0].item())
                print("Policy top :",policy[0].nonzero())
                print("Policy odds:",policy[0][policy[0].nonzero()])
                print("Pol Prd odd:",policy_pred[0][policy[0].nonzero()])

                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break
        
        scheduler.step()

    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1,epoch_stop+1,1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')
    plt.savefig(os.path.join("/home/owensr/chess/data/", "Loss_vs_Epoch_%s.png" % datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S")))

