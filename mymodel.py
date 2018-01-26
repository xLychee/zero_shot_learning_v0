import numpy as np
import mylsh
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation
from keras.utils import np_utils

class CNN_model:
    def __init__(self, input_shape, embedding_dim, output_dim, num_models, class_embedding_table):
        self.models = []
        self.lshs = []
        self.num_models = num_models
        self.output_dim = output_dim
        self.class_embedding_table = class_embedding_table
        for _ in range(num_models):
            model = self._build_model(input_shape, output_dim**2)
            lsh = mylsh.LSH(embedding_dim, output_dim)
            self.models.append(model)
            self.lshs.append(lsh)

    def train(self, train_X, train_y):
        for i in range(self.num_models):
            print("Training for model {} / {}".format(i+1,self.num_models))
            lsh = self.lshs[i]
            model = self.models[i]
            encoded_y = []
            for j in range(train_X.shape[0]):
                embedding = self.class_embedding_table[train_y[j]]
                encoded_y.append(lsh.indexing(embedding))
            dummy_y = np_utils.to_categorical(encoded_y, num_classes= self.output_dim**2)
            model.fit(train_X, dummy_y, batch_size=32, epochs=5)

    def predict_top_K(self, test_X, K):
        predictions = []
        for i in range(self.num_models):
            model = self.models[i]
            predictions.append(model.predict(test_X))
        predict_Y = []
        for i in range(test_X.shape[0]):
            p = []
            for j in range(self.num_models):
                p.append(predictions[j][i])
            class_value_table = {}
            for c in self.class_embedding_table.keys():
                embedding = self.class_embedding_table[c]
                value = 0
                for j in range(self.num_models):
                    index = self.lshs[j].indexing(embedding)
                    value += p[j][index]
                class_value_table[c]=value
            predict_y = []
            for c in sorted(class_value_table, key = class_value_table.get, reverse=True):
                predict_y.append(c)
                if len(predict_y) == K:
                    break
            predict_Y.append(predict_y)
        return predict_Y

    def _build_model(self, input_shape, output_length):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, init='uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_length, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model._make_predict_function()
        #model._make_train_function()
        return model


