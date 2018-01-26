import mymodel
import mylsh
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train_X = np.load('/home/xunluan/zero_shot/datasets/awa/train_X.npy')
train_y = np.load('/home/xunluan/zero_shot/datasets/awa/train_y.npy')
test_X = np.load('/home/xunluan/zero_shot/datasets/awa/test_X.npy')
test_y = np.load('/home/xunluan/zero_shot/datasets/awa/test_y.npy')
unseen_X = np.load('/home/xunluan/zero_shot/datasets/awa/unseen_X.npy')
unseen_y = np.load('/home/xunluan/zero_shot/datasets/awa/unseen_y.npy')

class_embedding_table = {}
file = open('/home/xunluan/zero_shot/datasets/awa/Animals_with_Attributes2/predicate-matrix-binary.txt')
c = 1
for line in file.readlines():
    embedding = []
    for i in line.split():
        embedding.append(float(i))
    embedding = np.array(embedding)
    class_embedding_table[c] = embedding
    c+=1
assert c == 51

input_shape = (128,128,3)
embedding_dim = 85
output_dim = 4
num_models = 20
top_K = 5

model = mymodel.CNN_model(input_shape, embedding_dim, output_dim, num_models, class_embedding_table)
model.train(train_X, train_y)
# Predict on test set
prediction = model.predict_top_K(test_X, top_K)
num_hit = 0
for i in range(test_X.shape[0]):
    if test_y[i] in set(prediction[i]):
        num_hit+=1
print("Test set. Top {} hit rate: {}/{} = {}".format(top_K, num_hit, test_X.shape[0], num_hit/test_X.shape[0]))
# Predict on unseen set
prediction = model.predict_top_K(unseen_X, top_K)
num_hit = 0
for i in range(unseen_X.shape[0]):
    if unseen_y[i] in set(prediction[i]):
        num_hit+=1
print("Unseen set. Top {} hit rate: {}/{} = {}".format(top_K, num_hit, unseen_X.shape[0], num_hit/unseen_X.shape[0]))



