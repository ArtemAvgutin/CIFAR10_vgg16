# -*- coding: utf-8 -*-

import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras import applications
import tensorflow as tf
import tensorflow.keras.preprocessing.image
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

n = 5
plt.imshow(x_train[n])
plt.show()
print("Номер класса:", y_train[n])
print("Тип объекта:", classes[y_train[n][0]])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from tensorflow.keras.layers.experimental.preprocessing import Resizing

vgg16 = applications.VGG16(weights='imagenet',
                    include_top=False,
                    classes=10,
                    input_shape=(None,None,3)# input: 32x32 images with 3 channels
                   )
VGG16_model = Sequential()

VGG16_model.add(Resizing(256,256))

for layer in vgg16.layers:
    VGG16_model.add(layer)

VGG16_model.add(Flatten())
VGG16_model.add(Dense(512, activation='relu', name='hidden1'))
VGG16_model.add(BatchNormalization())
VGG16_model.add(Dropout(0.5))
VGG16_model.add(Dense(256, activation='relu', name='hidden2'))
VGG16_model.add(BatchNormalization())
VGG16_model.add(Dropout(0.5))
VGG16_model.add(Dense(10, activation='softmax', name='predictions'))

VGG16_model.build((None,None,None,3))
VGG16_model.summary()

for layer in VGG16_model.layers[:18]:
    layer.trainable = False

VGG16_model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(lr=0.0001),
    metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath='cifar10-VGG16.h5', verbose=1, save_best_only=True)

history=VGG16_model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=15,
    verbose=1,
    callbacks=[checkpointer],
    validation_data=(x_test, y_test),
    shuffle=True)

#Потери
plt.figure(figsize=[15,15])
plt.subplot(211)
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Testing Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
#Точность
plt.subplot(212)
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'green', linewidth=2.0)
plt.legend(['Training Accuracy', 'Testing Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Acc', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

test_loss, test_acc = VGG16_model.evaluate(x_test, y_test)
print('Test loss: {}\nTest accuracy: {}'.format(test_loss, test_acc))

from tensorflow.keras.layers.experimental.preprocessing import Resizing

vgg16_2 = applications.VGG16(weights='imagenet',
                    include_top=False,
                    classes=10,
                    input_shape=(None,None,3)
                   )
VGG16_model_2 = Sequential()
VGG16_model_2.add(Resizing(256,256))
for layer in vgg16.layers:
    VGG16_model_2.add(layer)
VGG16_model_2.add(Flatten())
VGG16_model_2.add(Dense(512, activation='relu', name='hidden1'))
VGG16_model_2.add(BatchNormalization())
VGG16_model_2.add(Dropout(0.5))
VGG16_model_2.add(Dense(256, activation='relu', name='hidden2'))
VGG16_model_2.add(BatchNormalization())
VGG16_model_2.add(Dropout(0.5))
VGG16_model_2.add(Dense(10, activation='softmax', name='predictions'))

VGG16_model_2.build((None,None,None,3))

for layer in VGG16_model_2.layers[:14]:
    layer.trainable = False

VGG16_model_2.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(lr=0.0001),
    metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath='cifar10-VGG16.h5', verbose=1, save_best_only=True)

history_2=VGG16_model_2.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=20,
    verbose=1,
    callbacks=[checkpointer],
    validation_data=(x_test, y_test),
    shuffle=True)

#Потери
plt.figure(figsize=[15,15])
plt.subplot(211)
plt.plot(history_2.history['loss'], 'black', linewidth=2.0)
plt.plot(history_2.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Testing Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
#Точность
plt.subplot(212)
plt.plot(history_2.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history_2.history['val_accuracy'], 'green', linewidth=2.0)
plt.legend(['Training Accuracy', 'Testing Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Acc', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

test_loss, test_acc = VGG16_model_2.evaluate(x_test, y_test)
print('Test loss: {}\nTest accuracy: {}'.format(test_loss, test_acc))
