##Importation
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

#color in BGR
purple=(191,64,191)
red=(0,0,255)
## DATA
"""
Collecte des coordonnées des points clés pour chaque frame de chaque actions
Lorsque on affiche starting collection ligne 62, on exécute l'action correspondant à l'action en cours d'enregistrement
"""
actions7=np.array(['pushups','bodyweightsquat','lunges','jumping jack','nothing']) #les actions que l'on souahite reconnaitre
DATA_PATH=os.path.join('MP_Data7') #Dossier de stockage
actions=actions7 #choose action list
no_sequences=50 #nombre de vidéos par action
sequence_length=30 #nombre d'images pour chaque vidéo

def mediapipe_detection(image,model): #extraction
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results): #draw landmarks
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=red,circle_radius=2),mp_drawing.DrawingSpec(purple))

def Getcoord(results): #renvoie les coordonnées des points clés si détecté sinon on renvoie une position à l'origine
    PC=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) #le fait de mettre les points non détecté à zéros permet d'éviter des erreurs de dimension dans la suite du programme
    return PC

def main():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH,action,str(sequence))) #On créer des dossiers pour chaque actions
            except:
                pass

    cap=cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame=cap.read() #capture frames
                    image,results=mediapipe_detection(frame,pose) #mediapipe
                    draw_landmarks(image,results)
                    #text to start data collection for one action
                    if frame_num==0:
                        cv2.putText(image,'STARTING COLLECTION',(120,300),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4,cv2.LINE_AA)
                        cv2.putText(image,'{} video number {}'.format(action,sequence),(12,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        #Collecting frames for XXX
                        cv2.putText(image,'{} video number {}'.format(action,sequence),
    (15,22),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)

                    keypoints=Getcoord(results) #extraction des coordonnées
                    npy_path=os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                    np.save(npy_path,keypoints) #store
                    cv2.imshow('camera',image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

##step 1 execution du programme de collecte de données
main() #récolte des données

##step 2 Entrainement
#En 2 étapes
##Step 2.1 Chargement des données
label_map={label:num for num,label in enumerate(actions)}
sequences,labels=[],[]
#for each key points we're going to label it to understand to which action it belongs to.
for action in actions:
    for sequence in range(no_sequences):
        window=[]
        for frame_num in range(sequence_length):
            res=np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num))) #enregistrement des données pour chaque frame sous forme de fichier npy
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

#separation of training data and test data
X=np.array(sequences)
y=to_categorical(labels).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05) #séparation des données
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
log_dir=os.path.join('Logs') #dossier où on enregistre les données d'entrainement
tb_callback=TensorBoard(log_dir=log_dir)

##step 2.2 Entrainement du modèle
#Archtecture du modèle
model=Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,33*4))) #take parameters display by X.shape (XXX,nb frames,nb of points)
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
H=model.fit(X_train,y_train,epochs=100,callbacks=[tb_callback])
model.summary()

##
res=model.predict(X_test)
model.save('action8.h5') #on enregistre le modèle

##Test du modèle avec données test
model.load_weights('action8.h5')
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
yhat=model.predict(X_train)
ytrue=np.argmax(y_train,axis=1).tolist()
yhat=np.argmax(yhat,axis=1).tolist()
multilabel_confusion_matrix(ytrue,yhat)
accuracy_score(ytrue,yhat)

##Tracé
N = 150 #nombre d'epochs
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"])
plt.plot(np.arange(0, N), H.history["categorical_accuracy"])
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend("lower left")
plt.show()


##step 4 TEST
#fonction pour tester le modèle entrainé
def test():
    sequence=[]
    threshold=0.4

    cap=cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame=cap.read()
            image,results=mediapipe_detection(frame,pose)
            draw_landmarks(image,results)
            lr=Getcoord(results)
            sequence.insert(0,lr)
            sequence=sequence[:30] #prédiction par paquets de 30 frames
            if len(sequence)==30:
                res=model.predict(np.expand_dims(sequence,axis=0))[0] #prediction
                print(res)
                prediction=actions[np.argmax(res)] #on prend l'action correspondant à la probabilité maximal
                print(prediction)
                cv2.rectangle(image, (0,0), (200,40), (0,0,0), -1)
                cv2.putText(image,prediction,(12,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA) #affichage de la prédiction
            cv2.imshow('camera',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

test()
