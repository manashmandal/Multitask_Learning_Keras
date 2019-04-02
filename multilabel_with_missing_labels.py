
# coding: utf-8

# In[10]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import keras.backend as K
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from keras.optimizers import SGD

if not 'workbookDir' in globals():
    BASE_DIR = os.path.abspath(os.getcwd())
print('BASE_DIR: ' + BASE_DIR)
os.chdir(BASE_DIR)  # If you changed the current working dir, this will take you back to the workbook dir.

DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILEPATH = os.path.join(DATA_DIR, 'dataset.h5')
MODEL_FILEPATH = os.path.join(DATA_DIR, 'multitask_model.h5')
print('DATA_FILEPATH: ' + DATA_FILEPATH)
assert(os.path.isfile(DATA_FILEPATH))

def load():
    f = h5py.File(os.path.join(BASE_DIR, 'data', 'dataset.h5'))
    x = f['x'].value
    y = f['y'].value
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    x_train = np.rollaxis(x_train, 1, 4)
    x_test = np.rollaxis(x_test, 1, 4)
    x_train = x_train  / 255.0
    x_test = x_test / 255.0
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load()


# In[11]:


MISSING_LABEL_FLAG = -1
MISSING_LABEL_PROB = 0

y_train_missing = np.random.rand(*y_train.shape) < MISSING_LABEL_PROB
y_train[y_train_missing] = MISSING_LABEL_FLAG

def build_masked_loss(loss_function, mask_value=MISSING_LABEL_FLAG):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        dtype = K.floatx()
        mask = K.cast(K.not_equal(y_true, mask_value), dtype)
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function

def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.cast(K.sum(K.cast(K.not_equal(y_true, MASK_VALUE), dtype)), dtype)
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


# In[13]:


batch_size = 50
num_classes = 5
epochs = 4


# input image dimensions
img_rows, img_cols = 100, 100
channels = 3

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=(img_rows, img_cols, channels)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

## This is what Francois suggests for merging model outputs and using an mask on the input
## https://github.com/keras-team/keras/issues/3206#issuecomment-232446030
## uses the functional Keras API:
# from keras.layers import Masking, Merge
# Merge([network_outputs, Masking(mask_value=MISSING_LABEL_FLAG)(mask_input)], mode=lambda xs: xs[0], output_mask=lambda xs: xs[1])


model.compile(loss=build_masked_loss(K.binary_crossentropy),
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# In[25]:


CLASSES = np.array(['desert', 'mountain', 'sea', 'sunset', 'trees'])

def infer(input_data, model=model):
    labels = []
    y_pred = model.predict(input_data)
    
    # Performing masking
    y_pred = (y_pred > 0.5) * 1.0
    
    for i in range(y_pred.shape[0]):
        # select the indices
        indices = np.where(y_pred[i] == 1.0)[0]
        # Adding the results 
        labels.append(CLASSES[indices].tolist())
        
    return labels


# In[26]:


model.save(MODEL_FILEPATH)


# In[27]:


infer(x_test, model=model)


# In[30]:


df_test = pd.DataFrame(model.predict(x_test), columns=CLASSES)
df_true = pd.DataFrame(y_test, columns=CLASSES)


# In[32]:


label_acc = (df_test - df_true).abs().sum() / len(df_test)


# In[38]:


name = '_'.join([''.join((label, str(int(acc*100)))) for (label, acc) in zip(label_acc.index, label_acc.values)])
name


# In[40]:


MODEL_FILEPATH = os.path.join(DATA_DIR, f"model_{name}.h5")
print(f"MODEL_FILEPATH: {MODEL_FILEPATH}")
model.save(MODEL_FILEPATH)

