import numpy as np
import math, os, time, sys, re, datetime
from datetime import timedelta
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import data_process

transform_start = time.time()
data = data_process.getdata_onehot(datafile="A0101__")
transform_end = time.time()
transform_time = transform_end - transform_start

#optional: shuffle the training data and its labels
shuffle_ = np.arange(len(data['Y_train']))
np.random.shuffle(shuffle_)
data['Y_train']=data['Y_train'][shuffle_]
data['X_train']=data['X_train'][shuffle_]

print("X_Train size ", data['X_train'].shape)
print("Y_Train size ", data['Y_train'].shape)
print("Train data value=1 ", np.sum(data['Y_train']==1))
print("X_Test size " , data['X_test'].shape)
print("Y_Test size " , data['Y_test'].shape)
print("Test data value=1 ", np.sum(data['Y_test']==1))


Y_train_labels = data_process.binary2onehot(data['Y_train']) # binary output converted into two classes
Y_test_labels = data_process.binary2onehot(data['Y_test'])
X_train_data = data['X_train']  #already one hot encoded
X_test_data = data['X_test']


# simple model
model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(11, 21, 1)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))


'''
model = models.Sequential()
model.add(layers.Conv2D(128, (2, 2), strides=(1, 1), activation='relu', input_shape=(11, 21, 1)))
model.add(layers.Conv2D(128, (2, 2), strides=(1, 1), activation='relu')) 
model.add(layers.Conv2D(256, (2, 2), strides=(1, 1), activation='relu')) 
model.add(layers.Conv2D(256, (2, 2), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(256, (2, 2), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(512, (1, 2), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(512, (1, 1), strides=(1, 1), activation='relu'))
model.add(layers.Conv2D(256, (1, 1), strides=(1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2))'''

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train_data, Y_train_labels, epochs=20, validation_split = 0.1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()