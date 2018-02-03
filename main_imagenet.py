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
num_models = 2
embedding_dim = 500

model = model_imagenet.LogRegLshModel(input_dim, embedding_dim, num_planes, num_models, class_embedding_table)

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
print('Training Completed')
# begin prediction:
home_dir = r'/home/tharun/zmach/2hop/'
hop2 = netIDs[1000:2549]


total_samples = 0
total_1_hit = 0
total_5_hit = 0
for id in hop2:
    test_X = []
    test_y = []
    try:
        f = open(home_dir + id + '.txt')
    except IOError:
        print('file {}.txt not exist'.format(id))
        continue
    for line in f.readlines():
        tx = [float(i) for i in line.split()]
        test_X.append(tx)
        test_y.append(id)
    print('File Readed {}'.format(id))
    test_X = np.array(test_X)
    print(test_X.shape)
    top_5_hit = 0
    top_1_hit = 0
    num_samples = test_X.shape[0]
    predictions = model.predict_top_K(test_X, 5)
    print('Prediction completed')
    for i in range(num_samples):
        if test_y[i] == predictions[i][0]:
            top_1_hit+=1
        if test_y[i] in predictions[i]:
            top_5_hit+=1
    print('Class id: {}; 1 hit: {}/{}={}; 5 hit: {}/{}={}'.format(id, top_1_hit, num_samples,
                                                                  top_1_hit/num_samples,
                                                                  top_5_hit, num_samples,
                                                                  top_5_hit/num_samples
                                                                  ))
    total_samples+=num_samples
    total_1_hit+=top_1_hit
    total_5_hit+=top_5_hit
print('2 hop accuracy. top 1 hit: {}/{}={}; top 5 hit {}/{}={}'.format(total_1_hit, total_samples,
                                                                       total_1_hit/total_samples,
                                                                       top_5_hit,total_samples,
                                                                       top_5_hit/total_samples))



train_X = np.array(train_X)


hop3 = netIDs[2549:8860]
rest = netIDs[8860:]
