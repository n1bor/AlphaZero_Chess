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



class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].transpose(2,0,1), self.y_p[idx], self.y_v[idx]

class board_data_all(IterableDataset):
    def __init__(self,directory,count): # dataset = np.array of (s, p, v)
        super().__init__()
        self.files = []
        for idx,file in enumerate(os.listdir(directory)):
            filename = os.path.join(directory,file)
            self.files.append(filename)
        self.count=count
        self.loader=None
    def generate(self):
        print("a")
        while self.count>0 and len(self.files) > 0:
            file=self.files.pop()
            print(f"new file {file}")
            with open(file, 'rb') as fo:
                try:
                    data=pickle.load(fo)
                except EOFError:
                    print(f"EOFError in file {file}")
                    data=[]
            data = np.array(data,dtype="object")
            file_data=board_data(data)

            loader=iter(DataLoader(file_data,  shuffle=True, pin_memory=False))
            while loader !=None and self.count>0:
                data_item=next(loader,None)
                if data_item==None:
                    loader=None
                else:
                    self.count-=1
                    yield (torch.squeeze(data_item[0]),
                            torch.squeeze(data_item[1]),
                            torch.squeeze(data_item[2]))

    def __iter__(self):
        return iter(self.generate())


def train(net,train_path,trainingCount,lr,epochs,batch_size,cpu,test_train):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    losses_per_epoch=[]
    for epoch in range(epochs):
        train_data=board_data_all(train_path,trainingCount)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
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
            if test_train=='train':
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            #print(loss.item())
            losses_per_batch.append(loss.item())
            if i % 40 == 39: 
                print('Process ID: %d [Epoch: %d, %5d points] total loss per batch: %.3f' %
                        (os.getpid(), epoch + 1, (i + 1)*batch_size, total_loss/(i)))
                print("Policy     :",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Policy Odds:",policy[0][policy[0].argmax().item()],policy_pred[0][policy[0].argmax().item()])
                print("Value      :",value[0].item(),value_pred[0,0].item())
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter([e for e in range(1,int(trainingCount/batch_size)+1)], losses_per_batch)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss per batch")
        ax.set_title("Loss vs Batch")
        plt.savefig(os.path.join(rootDir+"/data/graphs/", "Loss_per_batch_%s_lr%s_e%s_c%s_%s.png" % 
                                 (test_train,str(lr),str(epoch),str(cpu),datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S"))))
        losses_per_epoch.append((total_loss*batch_size )/trainingCount )
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter([e for e in range(1,epochs+1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per Epoch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join(rootDir+"/data/graphs/", "Loss_per_Epoch_%s_lr%s_c%s_%s.png" % (test_train,str(lr),str(cpu),datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S"))))

def train2(net,train_path,trainingCount,lr,batch_size,cpu):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,12,18], gamma=0.2)
    roll_99=8.5
    roll_9=8.5
    
    train_data=board_data_all(train_path,trainingCount)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
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
        scheduler.step()
        total_loss += loss.item()
        roll_99=roll_99*0.99+loss.item()*0.01
        roll_9=roll_9*0.9+loss.item()*0.1
        print(f"{loss.item()} {roll_9} {roll_99}")
        losses_per_batch.append(loss.item())
        losses_99.append(roll_99)
        losses_9.append(roll_9)
        
        if i % 4 == 3: 
            print('Process ID: %d [%5d points] total loss per batch: %.3f' %
                    (os.getpid(),(i + 1)*batch_size, total_loss/(i+1)))
            print("Policy     :",policy[0].argmax().item(),policy_pred[0].argmax().item())
            print("Policy Odds:",policy[0][policy[0].argmax().item()],policy_pred[0][policy[0].argmax().item()])
            print("Value      :",value[0].item(),value_pred[0,0].item())
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter([e for e in range(1,i+2)], losses_per_batch)
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss per batch")
            ax.set_title("Loss vs Batch")
            plt.savefig(os.path.join(rootDir+"/data/graphs/", "Loss_per_batch_lr%s_c%s_%s.png" % 
                                    (str(lr),str(cpu),datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S"))))
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter([e for e in range(1,i+2)], losses_99)
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss per batch")
            ax.set_title("Loss vs Batch")
            plt.savefig(os.path.join(rootDir+"/data/graphs/", "Loss_99_lr%s_c%s_%s.png" % 
                                    (str(lr),str(cpu),datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S"))))
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter([e for e in range(1,i+2)], losses_9)
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss per batch")
            ax.set_title("Loss vs Batch")
            plt.savefig(os.path.join(rootDir+"/data/graphs/", "Loss_9_lr%s_c%s_%s.png" % 
                                    (str(lr),str(cpu),datetime.datetime.today().strftime("%Y-%m-%d-%H%M%S"))))
            plt.close()

#test_data=board_data_all(train_path,testCount)
if __name__=="__main__":
    trainingCount=1000000
    testCount=10000
    epochs=10
    lrList=[0.01]
    batch_size=2500

    mp.set_start_method("spawn",force=True)
    
    # gather data
    train_path = rootDir+"/data/train"
    test_path = rootDir+"/data/test"
    
    for lr in lrList:
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        #net.eval()
        # Runs Net training
        net_to_train="random.gz"; 
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        current_net_filename = os.path.join(rootDir+"/data/model_data/",\
                                        net_to_train)
        checkpoint = torch.load(current_net_filename, weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])

        train2(net,train_path,trainingCount,lr,batch_size,0,)
        #train(net,test_path,testCount,lr,1,batch_size,0,'test')
        #processes2 = []
        #for i in range(1):
        #    p2 = mp.Process(target=train,args=(net,train_path,trainingCount,lr,epochs,i))
        #    p2.start()
        #    processes2.append(p2)
        #for p2 in processes2:
        #    p2.join()