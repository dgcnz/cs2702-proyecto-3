import os
import pickle
import face_recognition

PATH = './lfw'

def load_dataset (): 
    dataset = []
    for dirpath, dirnames, filenames in os.walk(PATH):
        for dirname in dirnames:
            for file in os.listdir(PATH + '/' + dirname):
                image = face_recognition.load_image_file(PATH + '/' + dirname + '/' + file)
                features_vector = face_recognition.api.face_encodings(image)

                if (len(features_vector) > 0):
                    person = [dirname]
                    person.append(features_vector[0].tolist())
                    dataset.append(person)
                    
                    print ([dirname])

    with open('features_vectors', 'wb') as file:
        pickle.dump(dataset, file)

def load_image (img_path: string) -> List[int]:
    image = face_recognition.load_image_file(img_path)
    
    return face_recognition.api.face_encodings.api.face_encodings(image)


