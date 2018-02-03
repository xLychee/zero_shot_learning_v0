# input :2048 d
# output : 2^5 = 32 d
# tables: 20

import numpy as np
import mylsh
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation
from keras.utils import np_utils
from keras.regularizers import L1L2
from sklearn.model_selection import train_test_split
import concurrent.futures
import copy

class LogRegLshModel:
    def __init__(self, input_dim, embedding_dim, num_planes, num_models, class_embedding_table):
        assert input_dim == 2048
        self.num_models = num_models
        self.output_dim = 2**num_planes
        self.class_embedding_table = class_embedding_table
        self.models = []
        self.lshs = []
        for _ in range(num_models):
            self.lshs.append(mylsh.LSH(embedding_dim,num_planes))
            self.models.append(self._build_model(input_dim, 2**num_planes))

    def train(self, X, y):
        num_samples = X.shape[0]
        for i in range(self.num_models):
            lsh = self.lshs[i]
            model = self.models[i]
            y_index = []
            for j in range(num_samples):
                y_label = str(y[j])
                embedding = self.class_embedding_table[y_label]
                index = lsh.indexing(embedding)
                y_index.append(index)
            y_index = np.array(y_index)
            dummy_y = np_utils.to_categorical(y_index, num_classes=self.output_dim)
            print('Training for model {}/{}'.format(i+1, self.num_models))
            train_X, vali_X, train_y, vali_y = train_test_split(X, dummy_y, test_size=0.05)
            model.fit(train_X, train_y, batch_size=1024, epochs=1, validation_data = (vali_X, vali_y))

    def _process_individual_sample(self, i, outputs, K):
        print(i)
        lshs = copy.deepcopy(self.lshs)
        class_value_table = {}
        class_embedding_table = copy.deepcopy(self.class_embedding_table)
        for c in class_embedding_table.keys():
            embedding = class_embedding_table[c]
            value = 0
            for j in range(self.num_models):
                index = lshs[j].indexing(embedding)
                value += outputs[j][i][index]
            class_value_table[c] = value
        predict_y = []
        for c in sorted(class_value_table, key=class_value_table.get, reverse=True):
            predict_y.append(c)
            if len(predict_y) == K:
                break
        return predict_y

    def predict_top_K(self, test_X, K):
        num_samples = test_X.shape[0]
        outputs = []
        for i in range(self.num_models):
            model = self.models[i]
            outputs.append(model.predict(test_X))
            print('model {} prediction'.format(i))
        #predict_Y = []

        predict_Y = [None for i in range(num_samples)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(self._process_individual_sample, i, copy.deepcopy(outputs), K): i for i in range(num_samples)}
            for future in concurrent.futures.as_completed(future_to_index):
                ind = future_to_index[future]
                predict_Y[ind] = future.result()
                print(ind)
        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, predict_i in zip(range(num_samples), executor.map(self.predict_top_K(), range(num_samples))):
                predict_Y.append(predict_i)
        
        predict_Y = list(map(process_sample, range(num_samples)))
        
        predict_Y = []
        for i in range(num_samples):
            class_value_table = {}
            def cal_value_for_class(c):
                embedding = self.class_embedding_table[c]
                value = 0
                for j in range(self.num_models):
                    index = self.lshs[j].indexing(embedding)
                    value += outputs[j][i][index]
                return (c, value)
            value_tuples = [cal_value_for_class(c) for c in self.class_embedding_table.keys()]
            
            for c in self.class_embedding_table.keys():
                embedding = self.class_embedding_table[c]
                value = 0
                for j in range(self.num_models):
                    index = self.lshs[j].indexing(embedding)
                    value += outputs[j][i][index]
                class_value_table[c]=value
            for c in sorted(class_value_table, key=class_value_table.get, reverse=True):
                predict_y.append(c)
                if len(predict_y) == K:
                    break
            
            predict_y = []
            for t in sorted(value_tuples, key = lambda x:x[1]):
                predict_y.append(t[0])
                if len(predict_y) == K:
                    break
            print(i)
            predict_Y.append(predict_y)
        '''
        return predict_Y


    def _build_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(500,
                        activation='relu',
                        kernel_regularizer=L1L2(l1=0.0, l2=0.002),
                        input_dim=input_dim))
        model.add(Dense(output_dim,
                        activation='softmax',
                        kernel_regularizer=L1L2(l1=0.0, l2=0.002)))
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
'''
    def _build_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(output_dim,
                        activation='softmax',
                        kernel_regularizer=L1L2(l1=0.0, l2=0.01),
                        input_dim=input_dim))  # input dimension = number of features your data has
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
'''
