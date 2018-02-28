import numpy as np
from numpy import random
from sklearn.externals import joblib
import os
import dataset

CREATE_NEW_DATA = False

if CREATE_NEW_DATA:
    X_train, X_test, y_train, y_test, char_mapping = dataset.format_data('data/train.csv')
else:
    try:
        X_train = np.load('formatted_data/x_train.npy')
        X_test = np.load('formatted_data/x_test.npy')
        y_train = np.load('formatted_data/y_train.npy')
        y_test = np.load('formatted_data/y_test.npy')
        char_mapping = joblib.load('char_mapping.sav')
    except:
        X_train, X_test, y_train, y_test, char_mapping = dataset.format_data('data/train.csv')

# Encode a letter into a vector
def encode(char_mapping, letter):
    assert type(char_mapping) == dict
    assert letter in char_mapping
    arr = np.zeros((len(char_mapping)), dtype=np.float32)
    arr[char_mapping[letter]] = 1
    return arr

def generator(batch_size, X, y, char_mapping, fn_encode):
    n_vocab = len(char_mapping)
    x_batch = np.empty((batch_size,len(X[0]),n_vocab-1),dtype=np.float32)
    y_batch = np.empty((batch_size,len(y[0])),dtype=np.float32)
    while True:
        for i in range(batch_size):
            index = random.randint(len(X))
            encoded = np.array([fn_encode(char_mapping,letter)[:-1] for letter in X[index]],dtype=np.float32)
            x_batch[i] = encoded
            y_batch[i] = y[index]
        yield x_batch, y_batch

output_size = len(y_train[0])
max_len = len(X_train[0])

# Create the model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
model = Sequential()
model.add(LSTM(units=300, input_shape=(max_len,len(char_mapping)-1)))
model.add(Dropout(0.2))
model.add(Dense(units=output_size, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print('Built model!')
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
from keras.callbacks import ModelCheckpoint
filepath = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

batch_size = 128
model.fit_generator(generator=generator(batch_size,X_train,y_train,char_mapping,encode),
                    steps_per_epoch=len(X_train)//batch_size,
                    validation_data=generator(batch_size,X_test,y_test,char_mapping,encode),
                    validation_steps=len(X_test)//batch_size,
                    epochs=20, callbacks=callbacks_list)
model.save('model.h5')