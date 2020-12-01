import cv2
import os

faceDetect = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifiers/haarcascade_lbph_face_recognition.yml")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
w_face, h_face = 220, 220  # width and height of a face

def getNameByID(id):
    for dir in os.listdir('faceDatasets'):
        if not os.path.isdir('faceDatasets/' + dir):
            continue
        person = dir.split('.')
        personID = int(person[0])
        if personID == id:
            return person[1]
    return 'unknown'

frames = cv2.VideoCapture(0)

while (True):
    conect, image_color = frames.read()

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    imageFace = faceDetect.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(100, 100))

    for (x, y, w, h) in imageFace:
        faceDetected = cv2.resize(image_gray[y:y + h, x:x + w], (w_face, h_face))
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faceID, conf = recognizer.predict(faceDetected)
        print('id =', faceID, 'conf =', conf)
        if conf > 20:
            name = getNameByID(faceID)
            cv2.putText(image_color, name, (x, y + (h + 40)), font, 2, (0, 255, 0))

    cv2.imshow("Face Recognizer - LBPH.", image_color)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

frames.release()
cv2.destroyAllWindows()
