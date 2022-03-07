import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import cv2
from google.colab import drive

drive.mount('/content/drive/')

DATADIR = '/content/drive/MyDrive/Datasets/Dogs_vs_cats(full)/'
CATEGORIES = ['dogs', 'cats']
IMG_SIZE = 200
training_data = []
x = []
y = []
print(os.listdir(DATADIR))

k = 0

def create_training_data():
  global k
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    num_class = CATEGORIES.index(category)
    for img in os.listdir(path):
      img_read = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
      img_array = cv2.resize(img_read, (IMG_SIZE, IMG_SIZE))
      training_data.append([img_array, num_class])
      k = k + 1
      if k == 6000:
        break
    k = 0

create_training_data()

random.shuffle(training_data)

for features, label in training_data:
  x.append(features)
  y.append(label)

x = np.array(x)
y = np.array(y)

x = x / 255

model = keras.Sequential([
                          Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                          BatchNormalization(),
                          MaxPooling2D((2,2)),
                          Conv2D(128, (3,3), activation='relu'),
                          BatchNormalization(),
                          MaxPooling2D((2,2)),
                          Conv2D(256, (3,3), activation='relu'),
                          BatchNormalization(),
                          MaxPooling2D((2,2)),
                          Conv2D(512, (3,3), activation='relu'),
                          BatchNormalization(),
                          MaxPooling2D((2,2)),
                          Conv2D(1024, (3,3), activation='relu'),
                          BatchNormalization(),
                          MaxPooling2D((2,2)),
                          Flatten(),
                          Dense(64, activation='relu'),
                          #Dropout(0.7),
                          BatchNormalization(),
                          Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)

test_data = []

CATEGORIES = ['test']

k = 0

size = 1000

def create_test_data():
  global k
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
      img_read = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      img_array = cv2.resize(img_read, (IMG_SIZE, IMG_SIZE))
      test_data.append(img_array)
      k = k + 1
      if k == size:
        break

create_test_data()

for i in range(10):
  k = np.random.randint(0, size)
  n = np.expand_dims(test_data[k], axis=0)
  res = model.predict(n)
  if res == 0:
    print('Dog')
  else:
    print('Cat')
  plt.imshow(test_data[k], cmap='gray')
  plt.show()