from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

test_set = pd.read_csv('data/test.csv')

# Load data from disk and sanitize strings
import re
X = test_set.iloc[:,1]
X = [re.sub(r'\s+',' ',comment.strip().lower()) for comment in X]
X = [re.sub(r'[^\sa-zA-Z]','',comment) for comment in X]

max_len = 1000
X = [comment.rjust(max_len) if len(comment) <= max_len else comment[:max_len] for comment in X]
print('Formatted x data!')

y_true = np.array(test_set.iloc[:,2:], dtype=np.float32)
print('Extracted y data!')

model_path = 'model.h5'

model = load_model('model_path')
char_mapping = joblib.load('char_mapping.sav')

# Encode a letter into a vector
def encode(char_mapping, letter):
    assert type(char_mapping) == dict
    assert letter in char_mapping
    arr = np.zeros((len(char_mapping)), dtype=np.float32)
    arr[char_mapping[letter]] = 1
    return arr

y_pred = np.empty(y_true.shape, dtype=y_true.dtype)
batch_size=128
index = 0
while index < len(X):
    x_raw = X[index:index+batch_size]
    
    x_batch = np.empty((len(x_raw),len(X[0]),len(char_mapping)-1))
    for i,letter in enumerate(x_raw):
        x_batch[i] = np.array([encode(char_mapping,letter)[:-1] for letter in X[index]],dtype=np.float32)
    
    y = model.predict(x_batch)
    stop = min(index + batch_size, len(X))
    y_pred[index:stop] = y
    index += batch_size

overall_confusion_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())
confusion_matrices = [confusion_matrix(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]