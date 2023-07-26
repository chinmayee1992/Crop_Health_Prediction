import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

IMG_SIZE = 256
LR =0.005
MODEL_NAME = 'apples'

listfile=[]
listfile.append('/home/normal-userd/Biswadeep/train/0')
listfile.append('/home/normal-userd/Biswadeep/train/1')
listfile.append('/home/normal-userd/Biswadeep/train/2')
listfile.append('/home/normal-userd/Biswadeep/train/3')
listfile.append('/home/normal-userd/Biswadeep/train/4')
listfile.append('/home/normal-userd/Biswadeep/train/5')
listfile.append('/home/normal-userd/Biswadeep/train/6')
listfile.append('/home/normal-userd/Biswadeep/train/7')
listfile.append('/home/normal-userd/Biswadeep/train/8')
listfile.append('/home/normal-userd/Biswadeep/train/9')
listfile.append('/home/normal-userd/Biswadeep/train/10')
listfile.append('/home/normal-userd/Biswadeep/train/11')
listfile.append('/home/normal-userd/Biswadeep/train/12')
listfile.append('/home/normal-userd/Biswadeep/train/13')
listfile.append('/home/normal-userd/Biswadeep/train/14')
listfile.append('/home/normal-userd/Biswadeep/train/15')
listfile.append('/home/normal-userd/Biswadeep/train/16')
listfile.append('/home/normal-userd/Biswadeep/train/17')
listfile.append('/home/normal-userd/Biswadeep/train/18')
listfile.append('/home/normal-userd/Biswadeep/train/19')
listfile.append('/home/normal-userd/Biswadeep/train/20')
listfile.append('/home/normal-userd/Biswadeep/train/21')
listfile.append('/home/normal-userd/Biswadeep/train/22')
listfile.append('/home/normal-userd/Biswadeep/train/23')
listfile.append('/home/normal-userd/Biswadeep/train/24')
listfile.append('/home/normal-userd/Biswadeep/train/25')
listfile.append('/home/normal-userd/Biswadeep/train/26')
listfile.append('/home/normal-userd/Biswadeep/train/27')
listfile.append('/home/normal-userd/Biswadeep/train/28')
listfile.append('/home/normal-userd/Biswadeep/train/29')
listfile.append('/home/normal-userd/Biswadeep/train/30')
listfile.append('/home/normal-userd/Biswadeep/train/31')
listfile.append('/home/normal-userd/Biswadeep/train/32')
listfile.append('/home/normal-userd/Biswadeep/train/33')
listfile.append('/home/normal-userd/Biswadeep/train/34')
listfile.append('/home/normal-userd/Biswadeep/train/35')
listfile.append('/home/normal-userd/Biswadeep/train/36')
listfile.append('/home/normal-userd/Biswadeep/train/37')

listfile1=[]
listfile1.append('/home/normal-userd/Biswadeep/test/0')
listfile1.append('/home/normal-userd/Biswadeep/test/1')
listfile1.append('/home/normal-userd/Biswadeep/test/2')
listfile1.append('/home/normal-userd/Biswadeep/test/3')
listfile1.append('/home/normal-userd/Biswadeep/test/4')
listfile1.append('/home/normal-userd/Biswadeep/test/5')
listfile1.append('/home/normal-userd/Biswadeep/test/6')
listfile1.append('/home/normal-userd/Biswadeep/test/7')
listfile1.append('/home/normal-userd/Biswadeep/test/8')
listfile1.append('/home/normal-userd/Biswadeep/test/9')
listfile1.append('/home/normal-userd/Biswadeep/test/10')
listfile1.append('/home/normal-userd/Biswadeep/test/11')
listfile1.append('/home/normal-userd/Biswadeep/test/12')
listfile1.append('/home/normal-userd/Biswadeep/test/13')
listfile1.append('/home/normal-userd/Biswadeep/test/14')
listfile1.append('/home/normal-userd/Biswadeep/test/15')
listfile1.append('/home/normal-userd/Biswadeep/test/16')
listfile1.append('/home/normal-userd/Biswadeep/test/17')
listfile1.append('/home/normal-userd/Biswadeep/test/18')
listfile1.append('/home/normal-userd/Biswadeep/test/19')
listfile1.append('/home/normal-userd/Biswadeep/test/20')
listfile1.append('/home/normal-userd/Biswadeep/test/21')
listfile1.append('/home/normal-userd/Biswadeep/test/22')
listfile1.append('/home/normal-userd/Biswadeep/test/23')
listfile1.append('/home/normal-userd/Biswadeep/test/24')
listfile1.append('/home/normal-userd/Biswadeep/test/25')
listfile1.append('/home/normal-userd/Biswadeep/test/26')
listfile1.append('/home/normal-userd/Biswadeep/test/27')
listfile1.append('/home/normal-userd/Biswadeep/test/28')
listfile1.append('/home/normal-userd/Biswadeep/test/29')
listfile1.append('/home/normal-userd/Biswadeep/test/30')
listfile1.append('/home/normal-userd/Biswadeep/test/31')
listfile1.append('/home/normal-userd/Biswadeep/test/32')
listfile1.append('/home/normal-userd/Biswadeep/test/33')
listfile1.append('/home/normal-userd/Biswadeep/test/34')
listfile1.append('/home/normal-userd/Biswadeep/test/35')
listfile1.append('/home/normal-userd/Biswadeep/test/36')
listfile1.append('/home/normal-userd/Biswadeep/test/37')

arr2=[]
def create_train_data():
    training_data = []

    for i in xrange(len(listfile)):
     for img in tqdm(os.listdir(listfile[i])):

        path = os.path.join(listfile[i], img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        global arr2
        arr2.append([i])

        training_data.append([np.array(img_data)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


testing_labels=[]
arr1=[]
def create_test_data():
    testing_data = []

    for i in xrange(len(listfile1)):
      for img in tqdm(os.listdir(listfile1[i])):
        path = os.path.join(listfile1[i], img)

        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.resize(img_data,(IMG_SIZE, IMG_SIZE))
        global arr1
        arr1.append([i])
        testing_data.append([np.array(img_data)])




    np.save('test_data.npy', testing_data)
    return testing_data



train_data = create_train_data()
arr3=np.array(arr2)


test_data = create_test_data()
arr4=np.array(arr1)

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)



input = keras.utils.to_categorical(arr3,num_classes=38)



X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


output=keras.utils.to_categorical(arr4,num_classes=38)



model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))




model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(38, activation='softmax'))

sgd = SGD(lr=0.005, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


print model.summary()
model.fit(X_train, input, batch_size=24, epochs=5)
score = model.evaluate(X_test, output, batch_size=24)

print score



# tf.reset_default_graph()
# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
#
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# convnet = fully_connected(convnet, 38, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#
# model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
#
# model.fit({'input': X_train}, {'targets': y_train}, n_epoch=50,
#           validation_set=({'input': X_test}, {'targets': y_test}),
#           snapshot_step=50, show_metric=True, run_id=MODEL_NAME)

