# Multitask_Learning_Keras

```bash
python  multilabel_with_missing_labels.py [num_epochs] [batch_size]
```

Both batch size and num_epochs are optional.

Here's an overfitting example:

```bash
python multilabel_with_missing_labels.py 30 20
```

Which outputs:

```text
Using TensorFlow backend.
BASE_DIR: /midata/manceps/Multitask_Learning_Keras
DATA_FILEPATH: /midata/manceps/Multitask_Learning_Keras/data/dataset.h5
Starting 30 epochs of training, with batch_size=20
Setting 75% of the labels to -1 (flag them as missing).
Train on 1600 samples, validate on 400 samples
Epoch 1/30
2019-04-02 16:14:00.920357: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-04-02 16:14:02.223488: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 980 Ti major: 5 minor: 2 memoryClockRate(GHz): 1.19
pciBusID: 0000:01:00.0
totalMemory: 5.94GiB freeMemory: 5.83GiB
2019-04-02 16:14:02.223522: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-04-02 16:14:07.104842: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-04-02 16:14:07.104896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-04-02 16:14:07.104909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-04-02 16:14:07.112474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5607 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980 Ti, pci bus id: 0000:01:00.0, compute capability: 5.2)
1600/1600 [==============================] - 16s 10ms/step - loss: 0.1447 - masked_accuracy: 0.7469 - val_loss: 0.4879 - val_masked_accuracy: 0.7785
Epoch 2/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1240 - masked_accuracy: 0.7750 - val_loss: 0.4800 - val_masked_accuracy: 0.7820
Epoch 3/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1138 - masked_accuracy: 0.7951 - val_loss: 0.4276 - val_masked_accuracy: 0.8095
Epoch 4/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1060 - masked_accuracy: 0.8123 - val_loss: 0.3808 - val_masked_accuracy: 0.8355
Epoch 5/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0980 - masked_accuracy: 0.8340 - val_loss: 0.4067 - val_masked_accuracy: 0.8265
Epoch 6/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0888 - masked_accuracy: 0.8488 - val_loss: 0.4173 - val_masked_accuracy: 0.8160
Epoch 7/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0825 - masked_accuracy: 0.8639 - val_loss: 0.4228 - val_masked_accuracy: 0.8150
Epoch 8/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0742 - masked_accuracy: 0.8699 - val_loss: 0.4436 - val_masked_accuracy: 0.8220
Epoch 9/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0632 - masked_accuracy: 0.8953 - val_loss: 0.4909 - val_masked_accuracy: 0.8390
Epoch 10/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0519 - masked_accuracy: 0.9189 - val_loss: 0.5678 - val_masked_accuracy: 0.8115
Epoch 11/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0510 - masked_accuracy: 0.9185 - val_loss: 0.5236 - val_masked_accuracy: 0.8250
Epoch 12/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0387 - masked_accuracy: 0.9423 - val_loss: 0.7700 - val_masked_accuracy: 0.8250
Epoch 13/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0334 - masked_accuracy: 0.9550 - val_loss: 0.6401 - val_masked_accuracy: 0.8350
Epoch 14/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0300 - masked_accuracy: 0.9552 - val_loss: 0.7985 - val_masked_accuracy: 0.8130
Epoch 15/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0261 - masked_accuracy: 0.9635 - val_loss: 0.9562 - val_masked_accuracy: 0.8150
Epoch 16/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0196 - masked_accuracy: 0.9743 - val_loss: 0.8508 - val_masked_accuracy: 0.8345
Epoch 17/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0091 - masked_accuracy: 0.9878 - val_loss: 1.0294 - val_masked_accuracy: 0.8175
Epoch 18/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0157 - masked_accuracy: 0.9801 - val_loss: 1.0422 - val_masked_accuracy: 0.8190
Epoch 19/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0158 - masked_accuracy: 0.9781 - val_loss: 1.0176 - val_masked_accuracy: 0.8190
Epoch 20/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0108 - masked_accuracy: 0.9862 - val_loss: 0.9089 - val_masked_accuracy: 0.8215
Epoch 21/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0100 - masked_accuracy: 0.9851 - val_loss: 1.0028 - val_masked_accuracy: 0.8230
Epoch 22/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0082 - masked_accuracy: 0.9884 - val_loss: 1.2085 - val_masked_accuracy: 0.8165
Epoch 23/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0064 - masked_accuracy: 0.9909 - val_loss: 1.1890 - val_masked_accuracy: 0.8285
Epoch 24/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0031 - masked_accuracy: 0.9984 - val_loss: 1.2149 - val_masked_accuracy: 0.8345
Epoch 25/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0096 - masked_accuracy: 0.9904 - val_loss: 1.0375 - val_masked_accuracy: 0.8305
Epoch 26/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0129 - masked_accuracy: 0.9873 - val_loss: 0.9981 - val_masked_accuracy: 0.8175
Epoch 27/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0074 - masked_accuracy: 0.9912 - val_loss: 1.1144 - val_masked_accuracy: 0.8215
Epoch 28/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0103 - masked_accuracy: 0.9865 - val_loss: 1.1518 - val_masked_accuracy: 0.8115
Epoch 29/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0085 - masked_accuracy: 0.9933 - val_loss: 1.1429 - val_masked_accuracy: 0.8215
Epoch 30/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0020 - masked_accuracy: 0.9983 - val_loss: 1.1594 - val_masked_accuracy: 0.8245
    pred_desert  pred_mountain      pred_sea   pred_sunset    pred_trees  true_desert  true_mountain  true_sea  true_sunset  true_trees
0  3.292350e-10   1.340053e-01  1.605090e-01  4.493526e-07  1.153393e-01            0              0         1            0           0
1  8.249252e-05   1.015770e-09  2.430120e-07  9.973046e-01  3.352832e-01            0              0         0            1           0
2  1.924371e-04   9.572676e-01  1.765816e-01  3.920957e-01  1.115145e-07            0              0         1            0           0
3  4.689917e-06   1.469896e-03  1.138463e-03  1.073311e-10  2.240388e-01            1              0         0            0           0
4  9.711512e-05   5.061292e-03  3.569540e-02  4.637474e-08  6.418815e-08            0              0         1            0           0
                 pred_not_desert  pred_desert
true_not_desert              292           24
true_desert                   40           44
             precision    recall  f1-score   support

 not_desert       0.88      0.92      0.90       316
     desert       0.65      0.52      0.58        84

avg / total       0.83      0.84      0.83       400

                   pred_not_mountain  pred_mountain
true_not_mountain                282             23
true_mountain                     57             38
              precision    recall  f1-score   support

not_mountain       0.83      0.92      0.88       305
    mountain       0.62      0.40      0.49        95

 avg / total       0.78      0.80      0.78       400

              pred_not_sea  pred_sea
true_not_sea           271        26
true_sea                64        39
             precision    recall  f1-score   support

    not_sea       0.81      0.91      0.86       297
        sea       0.60      0.38      0.46       103

avg / total       0.76      0.78      0.76       400

                 pred_not_sunset  pred_sunset
true_not_sunset              304            9
true_sunset                   33           54
             precision    recall  f1-score   support

 not_sunset       0.90      0.97      0.94       313
     sunset       0.86      0.62      0.72        87

avg / total       0.89      0.90      0.89       400

                pred_not_trees  pred_trees
true_not_trees             245          29
true_trees                  46          80
             precision    recall  f1-score   support

  not_trees       0.84      0.89      0.87       274
      trees       0.73      0.63      0.68       126

avg / total       0.81      0.81      0.81       400

filepath: /midata/manceps/Multitask_Learning_Keras/data/75pct-missing-labels_desert83_mountain79_sea77_sunset88_trees81
Setting 50% of the labels to -1 (flag them as missing).
Train on 1600 samples, validate on 400 samples
Epoch 1/30
1600/1600 [==============================] - 4s 2ms/step - loss: 0.2528 - masked_accuracy: 0.7676 - val_loss: 0.4532 - val_masked_accuracy: 0.8020
Epoch 2/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2048 - masked_accuracy: 0.8129 - val_loss: 0.4216 - val_masked_accuracy: 0.8270
Epoch 3/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2025 - masked_accuracy: 0.8183 - val_loss: 0.3986 - val_masked_accuracy: 0.8260
Epoch 4/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1859 - masked_accuracy: 0.8372 - val_loss: 0.3559 - val_masked_accuracy: 0.8465
Epoch 5/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1714 - masked_accuracy: 0.8521 - val_loss: 0.3533 - val_masked_accuracy: 0.8540
Epoch 6/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1683 - masked_accuracy: 0.8547 - val_loss: 0.3543 - val_masked_accuracy: 0.8530
Epoch 7/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1523 - masked_accuracy: 0.8683 - val_loss: 0.3589 - val_masked_accuracy: 0.8490
Epoch 8/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1331 - masked_accuracy: 0.8881 - val_loss: 0.3884 - val_masked_accuracy: 0.8380
Epoch 9/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1147 - masked_accuracy: 0.9023 - val_loss: 0.3823 - val_masked_accuracy: 0.8525
Epoch 10/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0933 - masked_accuracy: 0.9241 - val_loss: 0.4743 - val_masked_accuracy: 0.8335
Epoch 11/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0773 - masked_accuracy: 0.9364 - val_loss: 0.4855 - val_masked_accuracy: 0.8405
Epoch 12/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0557 - masked_accuracy: 0.9589 - val_loss: 0.6027 - val_masked_accuracy: 0.8350
Epoch 13/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0450 - masked_accuracy: 0.9711 - val_loss: 0.5488 - val_masked_accuracy: 0.8430
Epoch 14/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0454 - masked_accuracy: 0.9657 - val_loss: 0.6524 - val_masked_accuracy: 0.8385
Epoch 15/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0381 - masked_accuracy: 0.9709 - val_loss: 0.6986 - val_masked_accuracy: 0.8310
Epoch 16/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0361 - masked_accuracy: 0.9749 - val_loss: 0.8100 - val_masked_accuracy: 0.8370
Epoch 17/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0320 - masked_accuracy: 0.9764 - val_loss: 0.8209 - val_masked_accuracy: 0.8110
Epoch 18/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0327 - masked_accuracy: 0.9777 - val_loss: 0.7443 - val_masked_accuracy: 0.8325
Epoch 19/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0268 - masked_accuracy: 0.9850 - val_loss: 0.8105 - val_masked_accuracy: 0.8345
Epoch 20/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0201 - masked_accuracy: 0.9853 - val_loss: 0.9363 - val_masked_accuracy: 0.8305
Epoch 21/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0246 - masked_accuracy: 0.9838 - val_loss: 0.9512 - val_masked_accuracy: 0.8205
Epoch 22/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0181 - masked_accuracy: 0.9874 - val_loss: 0.9597 - val_masked_accuracy: 0.8320
Epoch 23/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0148 - masked_accuracy: 0.9904 - val_loss: 0.9692 - val_masked_accuracy: 0.8340
Epoch 24/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0114 - masked_accuracy: 0.9941 - val_loss: 1.0193 - val_masked_accuracy: 0.8340
Epoch 25/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0141 - masked_accuracy: 0.9908 - val_loss: 1.0051 - val_masked_accuracy: 0.8275
Epoch 26/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0112 - masked_accuracy: 0.9937 - val_loss: 1.2279 - val_masked_accuracy: 0.8200
Epoch 27/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0109 - masked_accuracy: 0.9912 - val_loss: 1.2385 - val_masked_accuracy: 0.8340
Epoch 28/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0119 - masked_accuracy: 0.9929 - val_loss: 1.0205 - val_masked_accuracy: 0.8395
Epoch 29/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0100 - masked_accuracy: 0.9938 - val_loss: 1.0486 - val_masked_accuracy: 0.8360
Epoch 30/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0105 - masked_accuracy: 0.9934 - val_loss: 0.9836 - val_masked_accuracy: 0.8370
   pred_desert  pred_mountain      pred_sea   pred_sunset    pred_trees  true_desert  true_mountain  true_sea  true_sunset  true_trees
0     0.006304   4.303808e-01  1.938690e-01  1.528974e-03  2.188808e-02            0              0         1            0           0
1     0.000417   1.564696e-10  6.450834e-01  9.999877e-01  7.698569e-08            0              0         0            1           0
2     0.806282   8.805794e-10  2.056117e-08  8.866864e-01  8.817954e-08            0              0         1            0           0
3     0.002179   1.263367e-02  3.140691e-09  2.402591e-13  9.434300e-04            1              0         0            0           0
4     0.000099   4.023485e-01  3.370391e-02  3.924646e-01  8.451742e-06            0              0         1            0           0
                 pred_not_desert  pred_desert
true_not_desert              288           28
true_desert                   31           53
             precision    recall  f1-score   support

 not_desert       0.90      0.91      0.91       316
     desert       0.65      0.63      0.64        84

avg / total       0.85      0.85      0.85       400

                   pred_not_mountain  pred_mountain
true_not_mountain                272             33
true_mountain                     45             50
              precision    recall  f1-score   support

not_mountain       0.86      0.89      0.87       305
    mountain       0.60      0.53      0.56        95

 avg / total       0.80      0.81      0.80       400

              pred_not_sea  pred_sea
true_not_sea           273        24
true_sea                61        42
             precision    recall  f1-score   support

    not_sea       0.82      0.92      0.87       297
        sea       0.64      0.41      0.50       103

avg / total       0.77      0.79      0.77       400

                 pred_not_sunset  pred_sunset
true_not_sunset              305            8
true_sunset                   27           60
             precision    recall  f1-score   support

 not_sunset       0.92      0.97      0.95       313
     sunset       0.88      0.69      0.77        87

avg / total       0.91      0.91      0.91       400

                pred_not_trees  pred_trees
true_not_trees             245          29
true_trees                  40          86
             precision    recall  f1-score   support

  not_trees       0.86      0.89      0.88       274
      trees       0.75      0.68      0.71       126

avg / total       0.82      0.83      0.83       400

filepath: /midata/manceps/Multitask_Learning_Keras/data/50pct-missing-labels_desert85_mountain80_sea77_sunset90_trees82
Setting 25% of the labels to -1 (flag them as missing).
Train on 1600 samples, validate on 400 samples
Epoch 1/30
1600/1600 [==============================] - 4s 2ms/step - loss: 0.3698 - masked_accuracy: 0.7737 - val_loss: 0.4368 - val_masked_accuracy: 0.8025
Epoch 2/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2842 - masked_accuracy: 0.8339 - val_loss: 0.3864 - val_masked_accuracy: 0.8405
Epoch 3/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2611 - masked_accuracy: 0.8500 - val_loss: 0.3530 - val_masked_accuracy: 0.8500
Epoch 4/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2443 - masked_accuracy: 0.8546 - val_loss: 0.3592 - val_masked_accuracy: 0.8445
Epoch 5/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2314 - masked_accuracy: 0.8662 - val_loss: 0.3435 - val_masked_accuracy: 0.8585
Epoch 6/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2044 - masked_accuracy: 0.8816 - val_loss: 0.3599 - val_masked_accuracy: 0.8490
Epoch 7/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1822 - masked_accuracy: 0.8969 - val_loss: 0.3407 - val_masked_accuracy: 0.8585
Epoch 8/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1582 - masked_accuracy: 0.9126 - val_loss: 0.4102 - val_masked_accuracy: 0.8415
Epoch 9/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1310 - masked_accuracy: 0.9289 - val_loss: 0.3980 - val_masked_accuracy: 0.8515
Epoch 10/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1029 - masked_accuracy: 0.9437 - val_loss: 0.4362 - val_masked_accuracy: 0.8595
Epoch 11/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0723 - masked_accuracy: 0.9617 - val_loss: 0.5577 - val_masked_accuracy: 0.8565
Epoch 12/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0708 - masked_accuracy: 0.9676 - val_loss: 0.5698 - val_masked_accuracy: 0.8565
Epoch 13/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0492 - masked_accuracy: 0.9765 - val_loss: 0.6771 - val_masked_accuracy: 0.8545
Epoch 14/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0492 - masked_accuracy: 0.9771 - val_loss: 0.6442 - val_masked_accuracy: 0.8575
Epoch 15/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0467 - masked_accuracy: 0.9779 - val_loss: 0.6966 - val_masked_accuracy: 0.8565
Epoch 16/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0383 - masked_accuracy: 0.9821 - val_loss: 0.6460 - val_masked_accuracy: 0.8620
Epoch 17/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0224 - masked_accuracy: 0.9914 - val_loss: 0.8470 - val_masked_accuracy: 0.8550
Epoch 18/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0263 - masked_accuracy: 0.9884 - val_loss: 0.8686 - val_masked_accuracy: 0.8610
Epoch 19/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0212 - masked_accuracy: 0.9886 - val_loss: 0.7884 - val_masked_accuracy: 0.8550
Epoch 20/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0187 - masked_accuracy: 0.9918 - val_loss: 0.8481 - val_masked_accuracy: 0.8590
Epoch 21/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0317 - masked_accuracy: 0.9866 - val_loss: 0.7973 - val_masked_accuracy: 0.8570
Epoch 22/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0181 - masked_accuracy: 0.9923 - val_loss: 0.9214 - val_masked_accuracy: 0.8495
Epoch 23/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0202 - masked_accuracy: 0.9914 - val_loss: 0.9000 - val_masked_accuracy: 0.8540
Epoch 24/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0190 - masked_accuracy: 0.9917 - val_loss: 0.7713 - val_masked_accuracy: 0.8565
Epoch 25/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0123 - masked_accuracy: 0.9946 - val_loss: 0.8774 - val_masked_accuracy: 0.8635
Epoch 26/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0167 - masked_accuracy: 0.9920 - val_loss: 0.7712 - val_masked_accuracy: 0.8580
Epoch 27/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0167 - masked_accuracy: 0.9921 - val_loss: 0.9571 - val_masked_accuracy: 0.8535
Epoch 28/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0144 - masked_accuracy: 0.9934 - val_loss: 0.9271 - val_masked_accuracy: 0.8450
Epoch 29/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0102 - masked_accuracy: 0.9963 - val_loss: 1.0734 - val_masked_accuracy: 0.8435
Epoch 30/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0148 - masked_accuracy: 0.9949 - val_loss: 0.9495 - val_masked_accuracy: 0.8465
    pred_desert  pred_mountain      pred_sea   pred_sunset    pred_trees  true_desert  true_mountain  true_sea  true_sunset  true_trees
0  1.192957e-08   1.189864e-01  9.958159e-01  2.933643e-06  1.288675e-03            0              0         1            0           0
1  2.576223e-06   2.469365e-08  5.246953e-01  9.999957e-01  2.445184e-06            0              0         0            1           0
2  7.791089e-02   3.258799e-06  2.683018e-07  9.990827e-01  2.900005e-08            0              0         1            0           0
3  9.836031e-01   1.667133e-03  5.846175e-14  7.937005e-17  3.397266e-11            1              0         0            0           0
4  2.320096e-01   1.535056e-02  1.633725e-01  5.128523e-05  9.025378e-04            0              0         1            0           0
                 pred_not_desert  pred_desert
true_not_desert              303           13
true_desert                   37           47
             precision    recall  f1-score   support

 not_desert       0.89      0.96      0.92       316
     desert       0.78      0.56      0.65        84

avg / total       0.87      0.88      0.87       400

                   pred_not_mountain  pred_mountain
true_not_mountain                284             21
true_mountain                     51             44
              precision    recall  f1-score   support

not_mountain       0.85      0.93      0.89       305
    mountain       0.68      0.46      0.55        95

 avg / total       0.81      0.82      0.81       400

              pred_not_sea  pred_sea
true_not_sea           244        53
true_sea                37        66
             precision    recall  f1-score   support

    not_sea       0.87      0.82      0.84       297
        sea       0.55      0.64      0.59       103

avg / total       0.79      0.78      0.78       400

                 pred_not_sunset  pred_sunset
true_not_sunset              304            9
true_sunset                   22           65
             precision    recall  f1-score   support

 not_sunset       0.93      0.97      0.95       313
     sunset       0.88      0.75      0.81        87

avg / total       0.92      0.92      0.92       400

                pred_not_trees  pred_trees
true_not_trees             248          26
true_trees                  38          88
             precision    recall  f1-score   support

  not_trees       0.87      0.91      0.89       274
      trees       0.77      0.70      0.73       126

avg / total       0.84      0.84      0.84       400

filepath: /midata/manceps/Multitask_Learning_Keras/data/25pct-missing-labels_desert86_mountain81_sea78_sunset91_trees84
Setting 0% of the labels to -1 (flag them as missing).
Train on 1600 samples, validate on 400 samples
Epoch 1/30
1600/1600 [==============================] - 4s 2ms/step - loss: 0.5130 - masked_accuracy: 0.7661 - val_loss: 0.4142 - val_masked_accuracy: 0.8070
Epoch 2/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.3949 - masked_accuracy: 0.8260 - val_loss: 0.3665 - val_masked_accuracy: 0.8420
Epoch 3/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.3505 - masked_accuracy: 0.8485 - val_loss: 0.3401 - val_masked_accuracy: 0.8475
Epoch 4/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.3287 - masked_accuracy: 0.8562 - val_loss: 0.3308 - val_masked_accuracy: 0.8555
Epoch 5/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.3153 - masked_accuracy: 0.8654 - val_loss: 0.3338 - val_masked_accuracy: 0.8595
Epoch 6/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2943 - masked_accuracy: 0.8759 - val_loss: 0.3297 - val_masked_accuracy: 0.8650
Epoch 7/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2597 - masked_accuracy: 0.8897 - val_loss: 0.3603 - val_masked_accuracy: 0.8480
Epoch 8/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2289 - masked_accuracy: 0.9010 - val_loss: 0.3474 - val_masked_accuracy: 0.8590
Epoch 9/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.2042 - masked_accuracy: 0.9130 - val_loss: 0.3904 - val_masked_accuracy: 0.8515
Epoch 10/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1723 - masked_accuracy: 0.9279 - val_loss: 0.4297 - val_masked_accuracy: 0.8510
Epoch 11/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1471 - masked_accuracy: 0.9435 - val_loss: 0.3931 - val_masked_accuracy: 0.8555
Epoch 12/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1193 - masked_accuracy: 0.9554 - val_loss: 0.4638 - val_masked_accuracy: 0.8530
Epoch 13/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0953 - masked_accuracy: 0.9629 - val_loss: 0.4784 - val_masked_accuracy: 0.8540
Epoch 14/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.1009 - masked_accuracy: 0.9623 - val_loss: 0.5218 - val_masked_accuracy: 0.8540
Epoch 15/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0809 - masked_accuracy: 0.9700 - val_loss: 0.4942 - val_masked_accuracy: 0.8515
Epoch 16/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0782 - masked_accuracy: 0.9709 - val_loss: 0.5988 - val_masked_accuracy: 0.8520
Epoch 17/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0593 - masked_accuracy: 0.9789 - val_loss: 0.5647 - val_masked_accuracy: 0.8535
Epoch 18/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0450 - masked_accuracy: 0.9839 - val_loss: 0.6071 - val_masked_accuracy: 0.8570
Epoch 19/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0544 - masked_accuracy: 0.9799 - val_loss: 0.5846 - val_masked_accuracy: 0.8585
Epoch 20/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0462 - masked_accuracy: 0.9808 - val_loss: 0.6732 - val_masked_accuracy: 0.8490
Epoch 21/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0370 - masked_accuracy: 0.9874 - val_loss: 0.7838 - val_masked_accuracy: 0.8535
Epoch 22/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0425 - masked_accuracy: 0.9836 - val_loss: 0.7474 - val_masked_accuracy: 0.8490
Epoch 23/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0361 - masked_accuracy: 0.9870 - val_loss: 0.6916 - val_masked_accuracy: 0.8510
Epoch 24/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0329 - masked_accuracy: 0.9880 - val_loss: 0.7082 - val_masked_accuracy: 0.8505
Epoch 25/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0477 - masked_accuracy: 0.9858 - val_loss: 0.7186 - val_masked_accuracy: 0.8470
Epoch 26/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0421 - masked_accuracy: 0.9838 - val_loss: 0.6955 - val_masked_accuracy: 0.8485
Epoch 27/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0299 - masked_accuracy: 0.9911 - val_loss: 0.8340 - val_masked_accuracy: 0.8440
Epoch 28/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0287 - masked_accuracy: 0.9893 - val_loss: 0.7307 - val_masked_accuracy: 0.8565
Epoch 29/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0249 - masked_accuracy: 0.9921 - val_loss: 0.7451 - val_masked_accuracy: 0.8605
Epoch 30/30
1600/1600 [==============================] - 3s 2ms/step - loss: 0.0281 - masked_accuracy: 0.9911 - val_loss: 0.7226 - val_masked_accuracy: 0.8485
   pred_desert  pred_mountain      pred_sea   pred_sunset  pred_trees  true_desert  true_mountain  true_sea  true_sunset  true_trees
0     0.000616   3.790570e-01  8.411935e-01  8.725114e-04    0.009372            0              0         1            0           0
1     0.748184   7.385359e-09  7.081182e-07  8.477726e-01    0.061890            0              0         0            1           0
2     0.898512   2.442272e-03  2.931880e-04  8.062791e-01    0.000018            0              0         1            0           0
3     0.999542   4.645074e-05  2.518172e-07  3.892232e-07    0.000018            1              0         0            0           0
4     0.999941   4.006170e-09  9.399549e-03  1.992321e-06    0.000008            0              0         1            0           0
                 pred_not_desert  pred_desert
true_not_desert              286           30
true_desert                   27           57
             precision    recall  f1-score   support

 not_desert       0.91      0.91      0.91       316
     desert       0.66      0.68      0.67        84

avg / total       0.86      0.86      0.86       400

                   pred_not_mountain  pred_mountain
true_not_mountain                286             19
true_mountain                     45             50
              precision    recall  f1-score   support

not_mountain       0.86      0.94      0.90       305
    mountain       0.72      0.53      0.61        95

 avg / total       0.83      0.84      0.83       400

              pred_not_sea  pred_sea
true_not_sea           249        48
true_sea                40        63
             precision    recall  f1-score   support

    not_sea       0.86      0.84      0.85       297
        sea       0.57      0.61      0.59       103

avg / total       0.79      0.78      0.78       400

                 pred_not_sunset  pred_sunset
true_not_sunset              298           15
true_sunset                   24           63
             precision    recall  f1-score   support

 not_sunset       0.93      0.95      0.94       313
     sunset       0.81      0.72      0.76        87

avg / total       0.90      0.90      0.90       400

                pred_not_trees  pred_trees
true_not_trees             264          10
true_trees                  45          81
             precision    recall  f1-score   support

  not_trees       0.85      0.96      0.91       274
      trees       0.89      0.64      0.75       126

avg / total       0.87      0.86      0.86       400

filepath: /midata/manceps/Multitask_Learning_Keras/data/00pct-missing-labels_desert84_mountain83_sea77_sunset90_trees85
```

