import sys
import os
import dlib
import glob
import cv2
import pickle
import numpy as np
from sklearn.svm import SVC

predictor_path = r"D:\Data\recognizer\shape_predictor_5_face_landmarks.dat"  # xac dinh 68 diem tren khuon mat
face_rec_model_path = r"D:\Data\recognizer\dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = r"D:\Data\dataSet"

detector = dlib.get_frontal_face_detector()  # gọi hàm phát hiện gương mặt trong thư viện dlib
sp = dlib.shape_predictor(predictor_path)  # goi file shape_predictor_5_face_landmarks.dat
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

Image_data = []
Label = []
count=0
ID = []
Target_name = []
target_name = os.listdir(faces_folder_path)
for target in target_name:
    ID.append(target)
    list_image_name = os.listdir(faces_folder_path + "\\" + target)
    for each_image in list_image_name:
        image = cv2.imread(faces_folder_path + "\\" + target + "\\" + each_image)
        image = cv2.resize(image, (128, 128))
        win.clear_overlay()
        win.set_image(image)
        imagePath = faces_folder_path + "\\" + target + "\\" + each_image

        dets = detector(image, 1)

        for k, d in enumerate(dets):
            shape = sp(image, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            face_descriptor = facerec.compute_face_descriptor(image, shape)
            Image_data.append(face_descriptor)
            Target_name.append(target)

            Label.append(count) # dang so float, tang 0--het
    count = count + 1

Face_emb_vectors = np.array(Image_data)
Label = np.array(Label)

#training SVm de phan loai
from sklearn.svm import SVC

my_model = SVC(kernel="linear", probability=True)
my_model.fit(Face_emb_vectors, Label)
with open(r"D:\Data\recognizer\SVM_model_linear.pkl", "wb") as output_file: # wb: write
      pickle.dump([my_model, ID], output_file)

