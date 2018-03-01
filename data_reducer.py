import numpy as np
from numpy import random
from sklearn.utils import shuffle

def reduce(x, y, x_path, x_full_path, y_path, y_full_path, random_state=42):
    clean_indexes = [i for i in range(len(y)) if sum(y[i]) == 0]
    toxic_indexes = [i for i in range(len(y)) if sum(y[i]) != 0]
    
    # Fill the reduced arrays with entries from the full datasets
    reduced_x = []
    reduced_y = []
    while len(reduced_x) < len(toxic_indexes):
        rand_i = random.randint(0,len(clean_indexes))
        index = clean_indexes[rand_i]
        reduced_x.append(x[index])
        reduced_y.append(y[index])
        del clean_indexes[rand_i], index, rand_i
    
    # Add all entries from the toxic dataset
    reduced_x.extend([x[i] for i in toxic_indexes])
    reduced_y.extend([y[i] for i in toxic_indexes])
    
    reduced_x = np.array(shuffle(reduced_x))
    reduced_y = np.array(shuffle(reduced_y))
    
    np.save('formatted_data/x_test.npy',shuffle(reduced_x, random_state=random_state))
    np.save('formatted_data/y_test.npy',shuffle(reduced_y, random_state=random_state))
    
    np.save('formatted_data/x_test_full.npy', x)
    np.save('formatted_data/y_test_full.npy', y)
    
    return reduced_x, reduced_y
    
if __name__ == '__main__':
    x_test = np.load('formatted_data/x_test.npy')
    y_test = np.load('formatted_data/y_test.npy')
    reduce(x_test, y_test)