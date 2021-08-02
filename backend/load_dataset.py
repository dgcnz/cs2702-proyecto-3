import os

print(os.getcwd())
import pickle
import face_recognition
from tqdm import tqdm
from lib.rtree import RTree

PATH = './lfw'

people = []
vectors = []
for dirpath, dirnames, filenames in os.walk(PATH):
    for dirname in tqdm(dirnames):
        for file in os.listdir(PATH + '/' + dirname):
            image = face_recognition.load_image_file(PATH + '/' + dirname +
                                                     '/' + file)
            features_vector = face_recognition.api.face_encodings(image)
            if len(features_vector) > 0:
                people.append(dirname)
                vectors.append(features_vector[0].tolist())

assert (len(people) == len(vectors))
D = len(vectors[0])
rtree = RTree(D, True)
rtree.build(vectors)
with open('people.txt', 'w') as f:
    f.write("\n".join(people))
