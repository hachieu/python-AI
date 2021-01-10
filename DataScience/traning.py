import sys, os
import pandas as pd
import numpy as np
import cv2
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

basedir = r"D:/Data/dataset"
Image_data = []
Label = []

#target_name = D:/Data/dataset/1 =>1

target_name = os.listdir(basedir)
for target in target_name:
    list_image_name = os.listdir(basedir+"/"+target)#D:/Data/dataset/1/User.1.1.jpg => User.1.1.jpg
    for each_image in list_image_name:
        try:
            image = cv2.imread(basedir+"/"+target+"/"+each_image)
            image = cv2.resize(image,(128,128))
            Image_data.append(image)
            Label.append(target)
        except:
            os.remove(basedir+"/"+target+"/"+each_image)

Image_data = np.array(Image_data)
Label = np.array(Label)
#Gan nhan label
for target in target_name:
    list_image_name = os.listdir(basedir+"/"+target)
    for each_image in list_image_name:
        imgPath = basedir+"/"+target+"/"+each_image
        ID = int(os.path.split(imgPath)[-1].split('.')[1])
        Label[Label==target]=ID

len = len(Label)
n = np.arange(len) #n chua du lieu bi xao tron
random.shuffle(n)
maxindex = int(len-len/4)

Training_data = Image_data[n[0:maxindex],:,:,:]
Training_label = Label[n[0:maxindex]]

Testing_data = Image_data[n[maxindex:len],:,:,:]
Testing_label = Label[n[maxindex:len]]

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48

X_train = np.array(Training_data,'float32')
train_y = np.array(Training_label,'float32')
X_test = np.array(Testing_data,'float32')
test_y = np.array(Testing_label,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

X_train -= np.mean(X_train, axis=0) #normalize dữ liệu giữa 0 và 1
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

# X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
#
# X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,  optimizer=Adam(),  metrics=['accuracy'])

model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, test_y), shuffle=True)

fer_json = model.to_json()
with open("D:/DataScience/training/traningData.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("D:/DataScience/training/trainingData.h5")

# print(my_model.build)
# print(my_model.summary())
#
# print(my_model.predict(Testing_data))
#
# print(my_model.evaluate(Testing_data, Testing_label))
