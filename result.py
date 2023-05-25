import pickle
from keras.models import model_from_json

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

INPUT_SIZE = 224

def diagnosis(file):
    image = Image.open(file)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))

    # Show image
    '''
    plt.figure(figsize=(5, 5))
    plt.gray()
    plt.imshow(image)
    plt.show()
    '''
    # Load model
    # Normalize the data
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    img = np.expand_dims(image, axis=0)
    p = model.predict(img)
    print("Predict: ", np.argmax(p))

    # Find the name of the diagnosis
    res = ''
    if np.argmax(p) == 0:
        res = "Miocardial Infarction"
    elif np.argmax(p) == 1:
        res = "History Miocardial Infarction"
    elif np.argmax(p) == 2:
        res = "Abnormal hartbeat"
    elif np.argmax(p) == 3:
        res = "Normal"

    return res

#print ("Diagnosis is:", diagnosis("D:/СШІ/3 курс/Нейроні мережі/CourseProject/test/Normal/test (1).jpg"))