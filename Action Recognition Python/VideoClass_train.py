##Video classification with keras and Deep learning

##Import
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle


##Chargement et prétraitement des données
PATH_DATA=r"C:/STOCK/Etude/Ensea3A/Projet3A/"
df = pd.read_excel(PATH_DATA+"dataSmall.xlsx",engine='openpyxl')
train_df,test_df = train_test_split(df,test_size=0.2, shuffle=True)
IMG_SIZE = 224
NBFRAME=29

def crop_center_square(frame): #redimensionne les videos (crop)
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=IMG_SIZE):
    cap = cv2.VideoCapture(path) #prend une vidéo
    frames = [] #liste des frames
    try:
        while True: #pour une vidéo
            ret, frame = cap.read()
            if not ret: #condition pour quitter la boucle while
                break
            frame = crop_center_square(frame) #on crop de dim 240x240
            frame = frame[:, :, [2, 1, 0]] #change les couleurs de la frame (pas sur)
            frames.append(frame) #on ajoute la frame à la frame

            if len(frames) == max_frames: #condition d'arret sur la boucle while
                break
    finally:
        cap.release()
    return np.array(frames) #return une vidéo sous forme d'un array de frame

def prepare_all_videos(df, root_dir): #préparation des vidéos
    num_samples = len(df) #taille de notre base
    video_paths = df["video_name"].values.tolist() #path vers la vidéo
    data=[]
    labels2=df["tag"].values.tolist() #pour le path vers la vidéo
    for idx,path in enumerate(video_paths):
        frames=load_video(os.path.join(root_dir,labels2[idx],path))
        data.append(frames)
    return data,labels2

train_data,train_labels=prepare_all_videos(train_df, PATH_DATA+"DatasetSMALL")
test_data,test_labels= prepare_all_videos(test_df, PATH_DATA+"DatasetSMALL")

LABELS=[]
for lb in test_labels:
    if lb not in LABELS:LABELS.append(lb)

def reduceFrame(data,size): #fixe une longueur maximale pour les videos
    Z=[]
    for i in range(len(data)):
       Z.append(data[i][:size])
    return np.array(Z)

train_data=reduceFrame(train_data,NBFRAME) #réduit la longueur des vidéos
test_data=reduceFrame(test_data,NBFRAME)

trainAug = ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
shear_range=0.15,horizontal_flip=True,fill_mode="nearest") #data augmentation
valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

def extendLabels(labelList,nbframepervideo): #
    L=[]
    L2=['BodyWeightSquats', 'JumpingJack', 'PushUps']
    n=len(labelList)
    for i in range(n):
        L+=[L2.index(labelList[i]) for j in range(nbframepervideo)]
    return L
lb=LabelBinarizer()
A=extendLabels(train_labels,29) #transforme les labels en numéro de longueur 29
B=extendLabels(test_labels,29)
labels=lb.fit_transform(np.array(A+B))

##Architecture du modèle
baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

opt=Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

train_dataR=train_data.reshape(len(train_data)*29,240,240,3)
train_labelsR=to_categorical(np.array(A))
test_dataR=test_data.reshape(len(test_data)*29,240,240,3)
batch=29
EPOCHS=8
test_labelsR=to_categorical(np.array(B))
##Entrainement
H = model.fit(trainAug.flow(train_dataR,train_labelsR,batch_size=batch),
    validation_data=valAug.flow(test_dataR,test_labelsR),shuffle=False,
    epochs=EPOCHS)

##Tracer de la loss et accuracy

# plot the training loss and accuracy
N = EPOCHS
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend("lower left")
plt.show()
##Sauvegarde du modèle et des labels
model.save(PATH_DATA+"model5.h5")
f = open(PATH_DATA+"lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()