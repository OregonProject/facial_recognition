import os
import sqlite3
import numpy as np
import cv2 as cv
from PIL import Image


def insertOrUpdate(Id, Name, Age, Gen):
    conn = sqlite3.connect(
        "/Users/cameron.karagitz/Documents/code/python/openCV/FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+"'{}'".format(str(Id))
    cursor = conn.execute(cmd)

    if cursor:
        cmd = "UPDATE People SET Name="+"'{}'".format(str(Name))+" WHERE ID="+"'{}'".format(str(Id))
        cmd2 = "UPDATE People SET Age="+"'{}'".format(str(Age))+" WHERE ID="+"'{}'".format(str(Id))
        cmd3 = "UPDATE People SET Gender="+"'{}'".format(str(Gen))+" WHERE ID="+"'{}'".format(str(Id))
    else:
        cmd = "INSERT INTO People(ID,Name,Age,Gender) Values("+"'{}'".format(str(
            Id))+","+"'{}'".format(str(Name))+","+"'{}'".format(str(Age))+","+"'{}'".format(str(Gen))+")"
        cmd2 = ""
        cmd3 = ""

    conn.execute(cmd)
    conn.execute(cmd2)
    conn.execute(cmd3)
    conn.commit()
    conn.close()

def getProfile(Id):
    conn = sqlite3.connect(
        "/Users/cameron.karagitz/Documents/code/python/openCV/FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+"'{}'".format(str(Id))
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row

    conn.close()

    return profile

def detectAndSave(Id):
    face_cascade = cv.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    capture = cv.VideoCapture(0)
    sampleNum = 0
    while True:
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1
            cv.imwrite("/Users/cameron.karagitz/Documents/code/python/openCV/dataSet/User-" +
                       str(Id)+"." + str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
            cv.rectangle(frame, (x, y), (x+w, y+h), (250, 200, 0), 4)
            cv.waitKey(100)

        cv.imshow('Face', frame)
        cv.waitKey(1)
        if(sampleNum > 20):
            break

    capture.release()
    cv.destroyAllWindows()
    train_faces(Id)

def getImagesWithID(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagepath in imagepaths:
        faceImg = Image.open(imagepath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv.imshow("Training", faceNp)
        cv.waitKey(10)

    return np.array(IDs), faces

def trainFaces(Id):
    recognizer = cv.face.LBPHFaceRecognizer_create()
    path = '/Users/cameron.karagitz/Documents/code/python/openCV/dataSet'
    IDs, faces = getImagesWithID(path)
    recognizer.train(faces, IDs)
    recognizer.save(
        '/Users/cameron.karagitz/Documents/code/python/openCV/recognizer/trainingData.yml')
    cv.destroyAllWindows()
    recognize(Id)

def recognize(Id):
    faceDetect = cv.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    cam = cv.VideoCapture(0)
    rec = cv.face.LBPHFaceRecognizer_create()
    rec.read(
        "/Users/cameron.karagitz/Documents/code/python/openCV/recognizer/trainingData.yml")
    font = cv.FONT_HERSHEY_COMPLEX

    while(True):
        ret, frame = cam.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (250, 200, 0), 2)
            Id, conf = rec.predict(gray[y:y+h, x:x+w])
            profile = getProfile(Id)
            if(profile != None):
                cv.putText(frame, "Name : " +
                            str(profile[1]), (x, y+h+20), font, (0, 255, 0)) ###### HOW TO FIX THIS - Error: must be real number, not tuple #####
                cv.putText(frame, "Age : " +
                            str(profile[2]), (x, y+h+45), font, (0, 255, 0))
                cv.putText(frame, "Gender : " +
                            str(profile[3]), (x, y+h+70), font, (0, 255, 0))
        cv.imshow("Face", frame)
        if(cv.waitKey(1) == ord('q')):
            break

    cam.release()
    cv.destroyAllWindows()

def inputPersonDetails():
    new_or_existing = input('Is this a NEW or EXISTING user?: ')
    if new_or_existing.lower() == "new":
        Id = input('Enter User ID: ')
        name = input('Enter User Name: ')
        age = input('Enter User Age: ')
        gen = input('Enter User Gender: ')
        insertOrUpdate(Id, name, age, gen)
        detect_and_save(Id)
    else:
        Id = input('Enter User ID: ')
        recognize(Id)


if __name__ == "__main__":
    inputPersonDetails()
    
