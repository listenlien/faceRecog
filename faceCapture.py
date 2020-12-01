import cv2
import os

face_detect = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")

cv2.namedWindow("image")
cap = cv2.VideoCapture(0)

time = 0
time_interval = 10
count_photo = 1
total_photos = 50

w_face, h_face = 250, 250  # width and height of a face

def getNewPersonID():
    persons = []
    for dir in os.listdir('faceDatasets'):
        if not os.path.isdir('faceDatasets/' + dir):
            continue
        person = dir.split('.')
        person[0] = int(person[0])
        persons.append(person)

    if len(persons) > 0:
        persons = sorted(persons, key=lambda p:p[0], reverse=True)
        return persons[0][0]+1
    else:
        return 1

# create a directory if necessary
if not os.path.exists('faceDatasets'):
    os.makedirs('faceDatasets')

name = input("input your name: ")
if (name):
    print("Starting to capture face images...")
personID = getNewPersonID()

# read a image from video
conect, image_color = cap.read()

while True:
    if image_color is not None:
        cv2.imshow("image", image_color)

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        faces = face_detect.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(100, 100))

        # if only one face on image
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('image', image_color)

            if (time % time_interval) == 0:
                image_face = cv2.resize(image_gray[y:y + h, x:x + w], (w_face, h_face))

                # save the face image to a file
                name_path = 'faceDatasets/{}.{}'.format(personID, name)
                if not os.path.exists(name_path):
                    os.mkdir(name_path)
                filename = "{}/{}.{}".format(name_path, count_photo, 'jpg')
                print(filename)
                cv2.imwrite(filename, image_face)

                count_photo += 1

    conect, image_color = cap.read()

    time += 1
    cv2.waitKey(1)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or (count_photo >= total_photos + 1):
        break

print('time = ', time)
cap.release()
cv2.destroyAllWindows()
