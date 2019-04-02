""" Confirmation that skipping some missing labels by flagging them with -1 successfully blocks backpropagaion.

(or any other value) if they are ignored in the loss function.

Test set accuracy is only mildly reduced for a simple CNN image recognition, multi-label problem. 
But it's only ~80% accurate with no missing labels, and 5 mulihot labels, which is piss poor.

Francois Chollet suggests using the Merge and Masking layers to mask the input.[1]
    ## 
    ## uses the functional Keras API:
    # from keras.layers import Masking, Merge
    # Merge([network_outputs, Masking(mask_value=MISSING_LABEL_FLAG)(mask_input)], mode=lambda xs: xs[0], output_mask=lambda xs: xs[1])

Reference:
   [1] Masked loss function approach by @tivaro: https://github.com/keras-team/keras/issues/3893
   [2] Merge and Mask layers per Francois Chollet (@fchollet): https://github.com/keras-team/keras/issues/3206 
   [3] https://github.com/keras-team/keras/issues/3206#issuecomment-232446030
"""
import os

import numpy as np
import pandas as pd
import h5py
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# try training CNN on a completely different task in parallel?
# import matplotlib.pyplot as plt
# from keras.datasets import mnist

if not 'workbookDir' in globals():
    BASE_DIR = os.path.abspath(os.getcwd())
print('BASE_DIR: ' + BASE_DIR)
os.chdir(BASE_DIR)  # If you changed the current working dir, this will take you back to the workbook dir.

DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILEPATH = os.path.join(DATA_DIR, 'dataset.h5')
MODEL_FILEPATH = os.path.join(DATA_DIR, 'multitask_model.h5')
print('DATA_FILEPATH: ' + DATA_FILEPATH)
assert(os.path.isfile(DATA_FILEPATH))

MISSING_LABEL_PROBS = [0.75, 0.50, 0.25, 0.00]
CLASSES = np.array(['desert', 'mountain', 'sea', 'sunset', 'trees'])

batch_size = 50
num_classes = len(CLASSES)
epochs = 4

# input image dimensions
img_rows, img_cols = 100, 100
channels = 3


def load(test_size=.2, random_state=100):
    f = h5py.File(os.path.join(BASE_DIR, 'data', 'dataset.h5'))
    x = f['x'].value
    y = f['y'].value
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    x_train = np.rollaxis(x_train, 1, 4)
    x_test = np.rollaxis(x_test, 1, 4)
    x_train = x_train  / 255.0
    x_test = x_test / 255.0
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train_orig, y_test = load()


MISSING_LABEL_FLAG = -1


def build_masked_loss(loss_function=K.binary_crossentropy, mask_value=MISSING_LABEL_FLAG):
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
    total = K.cast(K.sum(K.cast(K.not_equal(y_true, MISSING_LABEL_FLAG), dtype)), dtype)
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype)) #  - K.cast(K.sum(K.cast(K.equal(y_true, MISSING_LABEL_FLAG), dtype)), dtype)
    return correct / total


def compile_model(   dropouts=[   .25,    .25,  .5],
                  num_neurons=[32, 32, 64, 64, 512, num_classes],
                  activations=['relu'] * 5 +       ['sigmoid'] ):
    model = Sequential()
    model.add(Conv2D(num_neurons[0], kernel_size=(3, 3),
        padding='same', input_shape=(img_rows, img_cols, channels)))
    model.add(Activation(activations[0]))
    model.add(Conv2D(num_neurons[1], (3, 3)))
    model.add(Activation(activations[1]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropouts[0]))

    model.add(Conv2D(num_neurons[2],(3, 3), padding='same'))
    model.add(Activation(activations[2]))
    model.add(Conv2D(num_neurons[3], (3, 3)))
    model.add(Activation(activations[3]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropouts[1]))

    model.add(Flatten())
    model.add(Dense(num_neurons[-2]))
    model.add(Activation(activations[-2]))
    model.add(Dropout(dropouts[-1]))
    model.add(Dense(num_neurons[-1]))
    model.add(Activation(activations[-1]))

    model.compile(loss=build_masked_loss(),
                optimizer='adam',
                metrics=[masked_accuracy])
    return model


confusions = []
testset_preds = []

for missing_label_prob in MISSING_LABEL_PROBS:
    print(f'Setting {int(missing_label_prob * 100)}% of the labels to {MISSING_LABEL_FLAG} (flag them as missing).')
    y_train = y_train_orig.copy()
    mask_labels_to_remove = np.random.rand(*y_train.shape) < missing_label_prob
    y_train[mask_labels_to_remove] = MISSING_LABEL_FLAG

    model = compile_model()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))


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


    infer(x_test, model=model)
    df_pred = pd.DataFrame(model.predict(x_test), columns=['pred_' + c for c in CLASSES])
    df_true = pd.DataFrame(y_test, columns=['true_' + c for c in CLASSES])
    df = pd.concat([df_pred, df_true], axis=1)
    print(df.head())
    testset_preds.append(df)
    confusions.append([])
    for c in CLASSES:
        labels = (f'not_{c}', f'{c}')
        confusion = pd.DataFrame(
            confusion_matrix(df['true_' + c], df['pred_' + c]),
            columns=[f'pred_{labels[0]}', f'{labels[0]}'],
            index=[f'true_{labels[0]}', f'{labels[0]}'])
        confusions[-1].append(confusion)
        print(confusion)
        print(classification_report(df_true[c], df_pred[c], labels=labels))
    label_acc = 1.0 - np.sum(np.abs((df_pred.values - df_true.values)), axis=0) / len(df_test)
    name = '_'.join([f'{label}{int(acc*100):02}' for (label, acc) in zip(CLASSES, label_acc)])
    filename = f"{int(missing_label_prob * 100):02}pct-missing-labels_{name}"

    filepath = os.path.join(DATA_DIR, filename)
    print(f"filepath: {filepath}")
    model.save(filepath + ".h5")
    df.to_csv(filepath + ".csv")


# should save results in one big h5:
# print(confusions)
# print([df.head() for df in testset_preds])


