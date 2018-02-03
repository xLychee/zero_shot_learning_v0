import numpy as np
import model_imagenet


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

file = open('/home/xunluan/zero_shot/datasets/imageNet/wnids.txt')
netIDs = []
for id in file.readlines():
    netIDs.append(id.split('\n')[0])

w2v = np.load('/home/xunluan/zero_shot/datasets/imageNet/w2v.npy')
assert len(netIDs) == w2v.shape[0]

class_embedding_table = {}

for i in range(len(netIDs)):
    class_embedding_table[netIDs[i]] = w2v[i]

input_dim = 2048
num_planes = 5
num_models = 10

model = model_imagenet.LogRegLshModel(input_dim, num_planes, num_models, class_embedding_table)

# build training set
'''
home_dir = r'/home/tharun/zmach/ResNet-50/'
top1000 = netIDs[:1000]
train_X = []
train_y = []
for id in top1000:
    print(id, len(train_X))
    f = open(home_dir+id+'.txt')
    for line in f.readlines():
        tx = [float(i) for i in line.split()]
        train_X.append(tx)
        train_y.append(id)
train_X = np.array(train_X)
'''
train_X = np.load('/home/xunluan/zero_shot/datasets/imageNet/train_X.npy')
train_y = np.load('/home/xunluan/zero_shot/datasets/imageNet/train_y.npy')
model.train(train_X,train_y)

# begin prediction:
home_dir = r'/home/tharun/zmach/2hop/'
hop2 = netIDs[1000:2549]



hop3 = netIDs[2549:8860]
rest = netIDs[8860:]
