import cv2
import os
import numpy as np

faceImgs = []
faceIDs = []

# loop through each directory in the /faceDatasets
for person in os.listdir('faceDatasets'):
    if not os.path.isdir('faceDatasets/'+person):
        continue

    p = person.split('.')
    id = int(p[0])

    # loop through each training image for the current person
    for file in os.listdir('faceDatasets/'+person):
        imgFile = 'faceDatasets/{}/{}'.format(person, file)

        face = cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2GRAY)
        faceIDs.append(id)
        faceImgs.append(face)

faceIDs = np.array(faceIDs)
print("load training images, ok!")

# print("training: eigenface ...")
# eigenface = cv2.face.EigenFaceRecognizer_create()
# eigenface.train(faceImgs, faceIDs)
# eigenface.write('classifiers/haarcascade_eigenface_face_recognition.yml')
# print("Finished: training: eigenface")

# This needs at least 2 persons to work
# print("training: fisherface ...")
# fisherface = cv2.face.FisherFaceRecognizer_create()
# fisherface.train(faceImgs, faceIDs)
# fisherface.write('classifiers/haarcascade_fisherface_face_recognition.yml')
# print("Finished: training: fisherface")

print("training: lbphface ...")
lbphface = cv2.face.LBPHFaceRecognizer_create()
lbphface.train(faceImgs, faceIDs)
lbphface.write('classifiers/haarcascade_lbph_face_recognition.yml')
print("Finished: training: lbphface")

print("Finished: the training process!")
