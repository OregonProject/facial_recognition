import cv2 as cv

face_cascade = cv.CascadeClassifier('/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
img = cv.imread('/Users/cameron.karagitz/Documents/code/python/openCV/test.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (250, 200, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv.imshow('Facial Recognition', img)
k = cv.waitKey(0) & 0xFF
if k == ord('q'):
    cv.destroyAllWindows()
