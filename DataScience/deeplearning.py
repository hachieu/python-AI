import tensorflow
import keras
import numpy as np
from keras import layers, losses
from keras.utils import to_categorical
import cv2
import os
import random
from keras.layers import Conv2D, Activation,MaxPooling2D, Flatten, Dense, Dropout

basedir = r"D:/Data/dataset"
Image_data = []
Label = []

#target_name = D:/Data/dataset/1 =>1

target_name = os.listdir(basedir)
for target in target_name:
    list_image_name = os.listdir(basedir+"/"+target) #D:/Data/dataset/1/User.1.1.jpg => User.1.1.jpg
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
print(maxindex)
#chia du lieu ngau nhien de lam training va testing
Training_data = Image_data[n[0:maxindex],:,:,:]
Training_label = Label[n[0:maxindex]]

Testing_data = Image_data[n[maxindex:len],:,:,:]
Testing_label = Label[n[maxindex:len]]

#chuann hoa du lieu giup mang hoi tu nhan hon , thoi gian chay nhanh hon
Training_data = Training_data/255.0
Testing_data = Testing_data/255.0

#Xay dung mang deeplearning

my_model = keras.Sequential()

my_model.add(Conv2D(64,(3,3), input_shape=(128,128,3)))# convolution layer, 64 neurals 3x3
my_model.add(Activation("relu"))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Conv2D(64,(3,3)))
my_model.add(Activation("relu"))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.1))

my_model.add(Conv2D(32,(3,3)))
my_model.add(Activation("relu"))
my_model.add(MaxPooling2D(pool_size=(2,2)))
my_model.add(Dropout(0.1))

my_model.add(Flatten())

my_model.add(Dense(10))
my_model.add(Activation('softmax'))

my_model.compile(optimizer='rmsprop',loss="sparse_categorical_crossentropy",metrics=['accuracy'])

Training_label = Training_label.astype(float)
Testing_label = Testing_label.astype(float)

#training the model
my_model.fit(Training_data, Training_label, epochs=25)

fer_json = my_model.to_json()
with open("D:/DataScience/training/trainingData.json", "w") as json_file:
    json_file.write(fer_json)
my_model.save_weights("D:/DataScience/training/trainingData.dat")

# print(my_model.build)
# print(my_model.summary())
#
# print(my_model.predict(Testing_data))
#
# print(my_model.evaluate(Testing_data, Testing_label))
