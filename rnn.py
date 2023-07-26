import numpy as np
import csv
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import keras.utils
from sklearn.cross_validation import train_test_split

x1 = csv.reader(open('/home/normal-userd/Biswadeep/Trainig2.csv'))
mat_features=[]
list_label=[]

for rows1 in x1:

         list_label.append(rows1[20])
         mat_features.append(rows1)

featuremat_train=np.array(mat_features)
featuremat_train=np.delete(featuremat_train, 20, axis=1)

labeltrain=np.array(list_label)

x2 = csv.reader(open('/home/normal-userd/Biswadeep/Testing2.csv'))
mat_features1=[]
list_label1=[]

for rows2 in x2:

         list_label1.append(rows2[20])
         mat_features1.append(rows2)

featuremat_test=np.array(mat_features1)
featuremat_test=np.delete(featuremat_test, 20, axis=1)

labeltest=np.array(list_label1)


data_dim=20
timesteps=1
num_classes=6




featuremat_train = np.reshape(featuremat_train, (featuremat_train.shape[0], timesteps, featuremat_train.shape[1]))
featuremat_test = np.reshape(featuremat_test, (featuremat_test.shape[0], timesteps, featuremat_test.shape[1]))

input_labels = keras.utils.to_categorical(labeltrain, num_classes)
output_labels = keras.utils.to_categorical(labeltest, num_classes)

model = Sequential()
model.add(LSTM(32, return_sequences=True,input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(featuremat_train, input_labels,batch_size=140, epochs=100)



scores = model.evaluate(featuremat_test, output_labels,verbose=0)

print scores








