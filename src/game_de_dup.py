import os
import compress_pickle as pickle
import numpy as np
from alpha_net import board_data
import hashlib
import dbm
import random
# gather data
data_path_in = "/home/owensr/chess/data/games"
data_path_train = "/home/owensr/chess/data/train"
data_path_test = "/home/owensr/chess/data/test"
data_path_validate = "/home/owensr/chess/data/validate"

data_path_db = "/home/owensr/chess/data/db.dbm"
duplicate=0
positions=0
file_count={}
file_count[data_path_train]=0
file_count[data_path_test]=0
file_count[data_path_validate]=0
runId=1

def dumpFile(path,data,force=False):
     if len(data)>10000 or force:
        with open(f"{path}/data_{runId}_{file_count[path]}.gz", 'wb') as output:
            pickle.dump(data, output)
        file_count[path]+=1
        data.clear()

train=[]
test=[]
validate=[]

for idx,file in enumerate(os.listdir(data_path_in)):
            filename = os.path.join(data_path_in,file)
            print(f"{idx} {filename}")

            with open(filename, 'rb') as fo:
                with dbm.open(data_path_db, 'c') as db:
                    data=pickle.load(fo)
                    data = np.array(data,dtype="object")
                    
                    for (s,p,v) in board_data(data):
                        r=s.flatten()
                        r="".join(map(str,r))
                        hash=hashlib.sha256(str.encode(r)).hexdigest()
                        if db.get(hash,False)==False:
                            positions+=1
                            number = random.randint(1, 10)
                            if number<9:
                                train.append((s,p,v))
                                db[hash]="t"
                            if number==9:
                                test.append((s,p,v))
                                db[hash]="s"
                            if number==10:
                                validate.append((s,p,v))
                                db[hash]="v"
                        else:
                             duplicate+=1
            print(f"{idx} p:{positions} d:{duplicate} t:{len(train)} tst:{len(test)} v:{len(validate)}")
            dumpFile(data_path_train,train)
            dumpFile(data_path_test,test)
            dumpFile(data_path_validate,validate)
                                 
print(f"pos:{positions} dup:{duplicate}")
dumpFile(data_path_train,train,True)
dumpFile(data_path_test,test,True)
dumpFile(data_path_validate,validate,True)