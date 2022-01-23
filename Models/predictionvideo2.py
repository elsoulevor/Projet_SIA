##Prediction video 2
""""
Programme pour tester le modèle7.h5
"""

#imporatation
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils

#color in BGR
purple=(191,64,191)
red=(0,0,255)

actions=np.array(['pushups','bodyweightsquat','lunges','jumping jack','nothing']) #choose action list
no_sequences=50 #number of videos
sequence_length=30 #number of frames for each videos


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

def coord(results): #extract right hand coordinates or body coordinates
    PC=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return PC

def test():
    sequence=[]
    threshold=0.4
    model=load_model('action7.h5')
    cap=cv2.VideoCapture(0)
    count=0
    lastpred='nothing' #dernière prédiction
    with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame=cap.read()
            image,results=mediapipe_detection(frame,pose)
            draw_landmarks(image,results)
            lr=coord(results)
            sequence.insert(0,lr)
            sequence=sequence[:30] #prédiction par paquets de 30 images
            if len(sequence)==30:
                res=model.predict(np.expand_dims(sequence,axis=0))[0]
                print(res)
                prediction=actions[np.argmax(res)] #prediciton de l'action on prend l'action ayant la probabilité la plus élevé
                print(prediction)
                if prediction==lastpred:
                    count+=1
                else:
                    count=0
                cv2.rectangle(image, (0,0), (300,40), (0,0,0), -1)
                cv2.putText(image,prediction,(12,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA) #affichage de la prédiction
                lastpred=prediction
            cv2.imshow('camera',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

test()