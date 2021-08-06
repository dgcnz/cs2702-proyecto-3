import os
import pickle
import face_recognition
from tqdm import tqdm
from lib.rtree import RTree

PATH = './lfw'

answers = []
vectors = []
for dirpath, dirnames, filenames in os.walk(PATH):
    for dirname in tqdm(dirnames):
        for file in os.listdir(PATH + '/' + dirname):
            ipath = PATH + '/' + dirname + '/' + file
            image = face_recognition.load_image_file(ipath)
            features_vector = face_recognition.api.face_encodings(image)
            if len(features_vector) > 0:
                answers.append(ipath)
                vectors.append(features_vector[0].tolist())

assert (len(answers) == len(vectors))
D = len(vectors[0])
rtree = RTree(D, True)
rtree.build(vectors)
with open('answers.txt', 'w+') as f:
    f.write("\n".join(answers))
