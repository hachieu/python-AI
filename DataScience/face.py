import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("D:/DataScience/training/trainingData.dat")
id = 0
# set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203, 23, 252)


# get data from sqlite by ID
def getProfile(id):
    conn = sqlite3.connect("D:/Data/FaceBase.db")
    cmd = "SELECT * FROM People WHERE id =" + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


while (True):
    # camera read
    ret, img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255 , 0), 1)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        print(conf)
        if(conf<65):
            profile = getProfile(id)
        # set text to window
            if (profile != None):
                cv2.putText(img, str(profile[1]), (x, y + h + 30), fontface, fontscale, fontcolor, 2)
        else:
            cv2.putText(img, "Unknown", (x, y + h + 30), fontface, fontscale, fontcolor, 2)

        cv2.imshow('Face', img)
    if cv2.waitKey(1) == ord('q'):
        break;
cam.release()
cv2.destroyAllWindows()