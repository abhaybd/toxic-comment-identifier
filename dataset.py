import pandas as pd
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import os
from data_reducer import reduce

DEBUG = True

def _debug(s):
    if DEBUG:
        print(s)

def format_data(path, max_len=1000, test_size=0.3):
    train_set = pd.read_csv(path)
    train_set = shuffle(train_set, random_state=42)
    
    # Load data from disk and sanitize strings
    import re
    X = train_set.iloc[:,1]
    X = [re.sub(r'\s+',' ',comment.strip().lower()) for comment in X]
    X = [re.sub(r'[^\sa-zA-Z]','',comment) for comment in X]
    
    X = [comment.rjust(max_len) if len(comment) <= max_len else comment[:max_len] for comment in X]
    _debug('Formatted x data!')
    
    # Extract output data
    y = np.array(train_set.iloc[:,2:], dtype=np.float32)
    _debug('Extracted y data!')
    
    # Create a set of unique characters in input data
    letters = set()
    for comment in X:
        for letter in comment:
            letters.add(letter)
    letters = sorted(list(letters))
    char_mapping = {letters[i]:i for i in range(len(letters))}
    joblib.dump(char_mapping,'char_mapping.sav')
    _debug('Created char mappings!')
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    
    # Create data folder if not present and save full data
    if not os.path.isdir('formatted_data'):
        os.mkdir('formatted_data')
    
    X_train, y_train = reduce(X_train, y_train, 'formatted_data/x_train.npy',
                              'formatted_data/x_train_full.npy', 
                              'formatted_data/y_train.npy', 
                              'formatted_data/y_train_full.npy')
    
    X_test, y_test = reduce(X_test, y_test, 'formatted_data/x_test.npy',
                            'formatted_data/x_test_full.npy',
                            'formatted_data/y_test.npy',
                            'formatted_data/y_test_full.npy')
    
    return X_train, X_test, y_train, y_test, char_mapping