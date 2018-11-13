import cv2 as cv

face_cascade = cv.CascadeClassifier('/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (250, 200, 0), 4)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, 'Name: Cameron Karagitz', (15, 550), font, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, 'Age: 22', (15, 620), font, 2, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(frame, 'Gender: Male', (15, 690), font, 2, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow('Facial Recognition', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
