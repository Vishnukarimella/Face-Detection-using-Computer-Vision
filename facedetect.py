import cv2
import imutils
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

