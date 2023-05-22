"""
This is a modified version of the Keras mnist example.
https://keras.io/examples/mnist_cnn/

Instead of using a fixed number of epochs this version continues to train until a stop criteria is reached.

A siamese neural network is used to pre-train an embedding for the network. The resulting embedding is then extended
with a softmax output layer for categorical predictions.

Model performance should be around 99.84% after training. The resulting model is identical in structure to the one in
the example yet shows considerable improvement in relative error confirming that the embedding learned by the siamese
network is useful.
"""

from __future__ import print_function
import warnings

warnings.filterwarnings("ignore")

import os, csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate, LSTM
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split
from siamese import SiameseNetwork

def ReadMyCsv2(SaveList, fileName, mode=0):
    csv_reader = csv.reader(open(fileName, encoding="utf-8-sig"))
    for row in csv_reader:
        if not mode:
            row = [float(x) for x in row]
        else:
            row = int(row[0]) - 1
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def storFile2(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x:[x],data))
    return
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding="utf-8-sig"))
    for row in csv_reader:
        SaveList.append(row)
    return

label = []
ReadMyCsv2(label, "label//biase.csv",1)
SMat = []
ReadMyCsv(SMat, "data//biase.csv")

Ze = np.zeros((56, 0))
data = np.hstack((SMat, Ze))
mult = 100

batch_size = 3
epochs = 10
# img_rows, img_cols = 80, 36 #
img_rows, img_cols = 70, 80

x, y = [], []
x = data
y = label
x, y = np.array(x).astype(np.float), np.array(y)
y = y - y.min()

num_classes = y.max() - y.min() + 1

x = np.concatenate([x] * mult, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = np.concatenate([x_train] * mult, axis=0)
y_train = np.concatenate([y_train] * mult, axis=0)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0],  img_rows, img_cols)
    x = x.reshape(x.shape[0],  img_rows, img_cols)
    input_shape = (img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    x = x.reshape(x.shape[0], img_rows, img_cols)
    input_shape = (img_rows, img_cols)

def create_base_model(input_shape):
    model_input = Input(shape=input_shape)

    embedding = Conv2D(16, kernel_size=(3, 3), input_shape=input_shape)(model_input)

    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(8, kernel_size=(3, 3))(embedding)
    # embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    # embedding = Conv2D(8, kernel_size=(3, 3))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    # embedding = MaxPooling2D(pool_size=(1, 1))(embedding)


    embedding = Flatten()(embedding)

    # embedding = LSTM(128)(embedding)
    embedding = Dense(32, name="dense-32")(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape)
    embedding_b = Input(shape=embedding_shape)

    head = Concatenate(name='result')([embedding_a, embedding_b])
    head = Dense(8, name='Dense')(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    head = Dense(1)(head)
    head = BatchNormalization()(head)
    head = Activation(activation='sigmoid')(head)

    return Model([embedding_a, embedding_b], head)


base_model = Sequential()
base_model.add(LSTM(units=256, input_shape=(img_rows, img_cols)))
base_model.add(Dense(32, activation='softmax', name="dense-32"))

# base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)


base_model.summary()
head_model.summary()

# for i in range(1):
siamese_checkpoint_path = "./siamese_checkpoint.hdf5"

siamese_network = SiameseNetwork(base_model, head_model)
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])

siamese_callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=siamese_callbacks)

# siamese_network.load_weights(siamese_checkpoint_path)
embedding = base_model.outputs[-1]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Add softmax layer to the pre-trained embedding network
embedding = Dense(num_classes)(embedding)
embedding = BatchNormalization()(embedding)
embedding = Activation(activation='sigmoid')(embedding)

model = Model(base_model.inputs[0], embedding)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])

model_checkpoint_path = "./model_checkpoint.hdf5"

model__callbacks = [
    EarlyStopping(monitor='val_acc', patience=10, verbose=0),
    ModelCheckpoint(model_checkpoint_path, monitor='val_acc', save_best_only=True, verbose=0)
]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=model__callbacks,
          validation_data=(x_test, y_test),
          verbose=1)


# model.load_weights(model_checkpoint_path)
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



dense2_layer_model = Model(inputs=model.input,outputs=model.get_layer('dense-32').output)
MatrixDense = []
MatrixDense = dense2_layer_model.predict(x)

storFile(MatrixDense, 'biaseSiaLSTM.csv')



from sklearn.cluster import KMeans
clf = KMeans(n_clusters= num_classes)
clustering = clf.fit(MatrixDense)
label_pred = clustering.predict(MatrixDense)

from ACC import acc
from sklearn import metrics
print('ARI:', metrics.adjusted_rand_score(y, label_pred))
from sklearn.metrics.cluster import normalized_mutual_info_score
print('NMI:', normalized_mutual_info_score(y, label_pred))
print("ACC:", acc(y, label_pred))