import pandas as pd
from sklearn.externals import joblib
import numpy as np

train = pd.read_csv('data/train.csv')

# Load data from disk and sanitize strings
import re
X = train.iloc[:,1]
X = [re.sub(r'\s+',' ',comment.strip().lower()) for comment in X]
X = [re.sub(r'[^\sa-zA-Z]','',comment) for comment in X]

max_len = max([len(comment) for comment in X])
X = [comment.rjust(max_len) for comment in X]
print('Formatted x data!')

train.iloc[:,1] = X

# Extract output data
y = np.array(train.iloc[:,2:])
print('Extracted y data!')

# Create a set of unique characters in input data
if(__import__('os').path.isfile('char_mapping.sav')):
    char_mapping = joblib.load('char_mapping.sav')
else:
    letters = set()
    for comment in X:
        for letter in comment:
            letters.add(letter)
    letters = sorted(list(letters))
    char_mapping = {letters[i]:i for i in range(len(letters))}
    joblib.dump(char_mapping,'char_mapping.sav')
    
print('Created char mappings!')

# Encode a letter into a vector
def encode(char_mapping, letter):
    assert type(char_mapping) == dict
    assert letter in char_mapping
    arr = np.zeros((len(char_mapping)), dtype=np.float32)
    arr[char_mapping[letter]] = 1
    return arr

# Decode a vector into a letter
def decode(char_mapping, arr):
    assert type(char_mapping) == dict
    assert sum(arr) == 1
    reverse_mapping = {char_mapping[x]:x for x in char_mapping}
    return reverse_mapping[np.argmax(arr)]

def generator(batch_size, X, y, char_mapping, fn_encode):
    n_vocab = len(char_mapping)
    x_train = np.empty((batch_size,len(X[0]),n_vocab),dtype=np.float32)
    y_train = np.empty((batch_size,len(y[0])),dtype=np.float32)
    while True:
        for i in range(batch_size):
            index = np.random.randint(len(X))
            encoded = np.array([fn_encode(char_mapping,letter) for letter in X[index]],dtype=np.float32)
            x_train[i] = encoded
            y_train[i] = y[index]
        yield x_train, y_train

input_size = max_len
output_size = len(y[0])

# Create the model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
model = Sequential()
model.add(LSTM(units=300, input_shape=(max_len,len(char_mapping))))
model.add(Dropout(0.2))
model.add(Dense(units=output_size, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

from keras.callbacks import ModelCheckpoint
filepath = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

batch_size = 128
model.fit_generator(generator=generator(batch_size,X,y,char_mapping,encode),
                    steps_per_epoch=len(X)//batch_size, epochs=20, callbacks=callbacks_list)
model.save('model.h5')