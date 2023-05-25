import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.models import Model
from keras import Sequential, layers
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras import utils

import tensorflow as tf
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

import pickle
from keras.models import model_from_json

INPUT_SIZE = 224

def ReadData(path, name, y):
  dataset = []
  label = []
  for i in os.listdir(path + name):
    if(i.split('.')[1]=='jpg'):
        #image=cv2.imread(path_train +'/Myocardial_Infarction/' + i)
        #image=Image.fromarray(image,'RGB')
        image = Image.open(path + name + i)
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        #print(image)
        dataset.append(np.array(image))
        label.append(y)
        #MI.append(np.array(image))
  return dataset, label

path_train = "D:/СШІ/3 курс/Нейроні мережі/CourseProject/train"
path_test = "D:/СШІ/3 курс/Нейроні мережі/CourseProject/test"

MI, MI_label = ReadData(path_train, '/Myocardial_Infarction/', 0)
PMI, PMI_label = ReadData(path_train, '/PMI/', 1)
HB, HB_label = ReadData(path_train, '/abnormal_heartbeat/', 2)
NORMAL, NORMAL_label = ReadData(path_train, '/Normal/', 3)


MI_test, MI_label_test = ReadData(path_test, '/Myocardial_Infarction/', 0)
PMI_test, PMI_label_test = ReadData(path_test, '/PMI/', 1)
HB_test, HB_label_test = ReadData(path_test, '/abnormal_heartbeat/', 2)
NORMAL_test, NORMAL_label_test = ReadData(path_test, '/Normal/', 3)


dataset = []
label = []
def setDataset(arr, y):
  for i in arr:
    dataset.append(i)
  for i in y:
    label.append(i)
setDataset(MI, MI_label)
setDataset(PMI, PMI_label)
setDataset(HB, HB_label)
setDataset(NORMAL, NORMAL_label)
print(len(dataset), len(label))


dataset_test = []
label_test = []
def setDatasetTest(arr, y):
  for i in arr:
    dataset_test.append(i)
  for i in y:
    label_test.append(i)
setDatasetTest(MI_test, MI_label_test)
setDatasetTest(PMI_test, PMI_label_test)
setDatasetTest(HB_test, HB_label_test)
setDatasetTest(NORMAL_test, NORMAL_label_test)
print(len(dataset_test), len(label_test))

dataset = np.array(dataset)
dataset = dataset / 255.0


dataset_test = np.array(dataset_test)
dataset_test = dataset_test / 255.0


train_label = utils.to_categorical(label, 4)

test_label = utils.to_categorical(label_test, 4)

x_train, x_val, y_train, y_val = train_test_split(dataset, train_label, test_size=0.3, random_state=42)

def CreateCNN():
  return Sequential([
    Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=x_train[0].shape),
    MaxPooling2D((2, 2)),
    #Dropout(0.2),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),
    #Dropout(0.1),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(4, activation='softmax')
])

model = CreateCNN()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

fitting = False
fitting_save = False

#history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))

if fitting:
    history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))
    if fitting_save:
    # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
        with open('history.pickle', 'wb') as f:
            pickle.dump(history.history, f)

# load model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into a new model
model.load_weights("model.h5")
with open('history.pickle', 'rb') as f:
    history = pickle.load(f)
print("Loaded model from disk")


def plot_loss(history):
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Training loss', 'Validation loss'], loc='upper right')
  plt.show()

def plot_accuracy(history):
  plt.plot(history['accuracy'])
  plt.plot(history['val_accuracy'])
  plt.ylabel('Acc')
  plt.xlabel('Epoch')
  plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
  plt.show()

print(history)
plot_loss(history)
plot_accuracy(history)

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
test_score = model.evaluate(dataset_test, test_label)
print("Accuracy on Test data: ", test_score[1])

pred = model.predict(dataset_test)

res = []
for i in pred:
  res.append(np.argmax(i))


y_test = []
for i in test_label:
  y_test.append(np.argmax(i))

confusion_matrix = metrics.confusion_matrix(y_test, res)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels = ["MI", "HMI", "AHB", "Normal"])

cm_display.plot()
plt.show()

model.summary()

img = np.expand_dims(dataset_test[11], axis=0)
p = model.predict(img)
print("Predict: ", np.argmax(p))

with tf.GradientTape() as tape:
    last_conv_layer = model.get_layer('conv2d_1')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(img)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((112, 112))
plt.matshow(heatmap)
plt.show()
print("========================")


img = dataset_test[11]
INTENSITY = 0.5

heatmap = cv2.resize(heatmap, (224, 224))

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

img = heatmap * INTENSITY + img

#plt.imshow(img)