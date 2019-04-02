import numpy as np
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import h5py
# from sklearn.model_selection import train_test_split
import keras.backend as K
import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
# from keras.optimizers import SGD

MASK_VALUE = -1
n = 25 # # datapoints
n_tasks = 19  # tasks / # binary classes
input_dim = 2048 # vector size

# generate random X vectors and random 
# Y labels (binary labels [0,1] or -1 for missing value
x = np.random.rand(n, input_dim)
x_test = np.random.rand(5, input_dim)
y = np.random.randint(3, size=(n, n_tasks))-1


def build_masked_loss(loss_function, mask_value=MASK_VALUE):
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

# create model
model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=input_dim))
model.add(Dense(n_tasks, activation='sigmoid'))
model.compile(loss=build_masked_loss(K.binary_crossentropy), optimizer='adam', metrics=[masked_accuracy])
model.fit(x, y)


