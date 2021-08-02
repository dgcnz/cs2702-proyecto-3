import os
import numpy as np
import face_recognition

PATH = './lfw'

dataset = []
for dirpath, dirnames, filenames in os.walk(PATH):
    for dirname in dirnames:
        for file in os.listdir(PATH + '/' + dirname):
            image = face_recognition.load_image_file(PATH + '/' + dirname + '/' + file)
            features_vector = face_recognition.api.face_encodings(image)

            person = [dirname]
            person.append(features_vector[0].tolist())
            dataset.append(person)
            

            print (person)
                
