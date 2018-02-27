import pandas as pd
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

train_set = pd.read_csv('data/train.csv')
train_set = shuffle(train_set, random_state=42)

# Load data from disk and sanitize strings
import re
X = train_set.iloc[:,1]
X = [re.sub(r'\s+',' ',comment.strip().lower()) for comment in X]
X = [re.sub(r'[^\sa-zA-Z]','',comment) for comment in X]

max_len = 1000
X = [comment.rjust(max_len) if len(comment) <= max_len else comment[:max_len] for comment in X]
print('Formatted x data!')

# Extract output data
y = np.array(train_set.iloc[:,2:], dtype=np.float32)
print('Extracted y data!')

# Create a set of unique characters in input data
import os
if(os.path.isfile('char_mapping.sav')):
    char_mapping = joblib.load('char_mapping.sav')
else:
    letters = set()
    for comment in X:
        for letter in comment:
            letters.add(letter)
    letters = sorted(list(letters))
    char_mapping = {letters[i]:i for i in range(len(letters))}
    joblib.dump(char_mapping,'char_mapping.sav')
del os
print('Created char mappings!')

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
np.save('x_train.npy',X_train)
np.save('x_test.npy',X_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)
print('Created train/test split!')

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
            index = np.random.randint(len(X))
            encoded = np.array([fn_encode(char_mapping,letter)[:-1] for letter in X[index]],dtype=np.float32)
            x_batch[i] = encoded
            y_batch[i] = y[index]
        yield x_batch, y_batch

output_size = len(y[0])

# Create the model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
model = Sequential()
model.add(LSTM(units=300, input_shape=(max_len,len(char_mapping)-1)))
model.add(Dropout(0.2))
model.add(Dense(units=output_size, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print('Built model!')

import os
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
del os
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