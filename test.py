from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy as np

X = np.load('x_test.npy')
y_true = np.load('y_test.npy')

model_path = 'checkpoints/weights-improvement-11-0.0538.h5'

model = load_model(model_path)
char_mapping = joblib.load('char_mapping.sav')

# Encode a letter into a vector
def encode(char_mapping, letter):
    assert type(char_mapping) == dict
    assert letter in char_mapping
    arr = np.zeros((len(char_mapping)), dtype=np.float32)
    arr[char_mapping[letter]] = 1
    return arr

y_pred = np.empty(y_true.shape, dtype=np.float32)
batch_size=1024
index = 0
while index < len(X):
    x_raw = X[index:index+batch_size]
    
    x_batch = np.empty((len(x_raw),len(X[0]),len(char_mapping)-1))
    for i,comment in enumerate(x_raw):
        x_batch[i] = np.array([encode(char_mapping,letter)[:-1] for letter in comment],dtype=np.float32)
    
    y = model.predict(x_batch)
    stop = min(index + batch_size, len(X))
    y_pred[index:stop] = y
    index += batch_size
    print('{:.2f}% complete'.format(100.*index/len(X)))

y_forced = np.reshape([round(x) for x in y_pred.flatten()],y_pred.shape)

overall_confusion_matrix = confusion_matrix(y_true.flatten(), y_forced.flatten())
confusion_matrices = [confusion_matrix(y_true[:,i], y_forced[:,i]) for i in range(y_true.shape[1])]
joblib.dump(overall_confusion_matrix,'cm.sav')

tp,fp,fn,tn = overall_confusion_matrix.flatten()

ppv = tp/(tp+fp)
tpr = tp/(tp+fn)
f_one = 2.0 * (ppv*tpr)/(ppv+tpr)
f_two = 5.0 * (ppv*tpr)/(4.0*ppv+tpr)

print('f1={:.2f}, f2={:.2f}'.format(f_one,f_two))