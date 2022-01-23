## Imports

import numpy as np
import mediapipe as mp
import cv2
mp_pose = mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils

## USEFUL FUNCTIONS

def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

# Affiche le squelette
def draw_landmarks(image,results):
    purple=(191,64,191)
    red=(0,0,255)
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=red,circle_radius=2),mp_drawing.DrawingSpec(purple))

def main_mediapipe(frame):
    with mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.7) as pose:
        image,results=mediapipe_detection(frame,pose)
        draw_landmarks(image,results)
    return image,results

# Obtention des coordonnées utiles (uniquement celles que l'on va utiliser)
def getcoordinates(landmarks,part):
    if part=='left shoulder': return([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    elif part=='right shoulder': return([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    elif part=='left elbow': return([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
    elif part=='right elbow': return([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

    elif part=='left wrist': return([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
    elif part=='right wrist': return([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

    elif part=='nose': return([landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y])

    elif part=='left hip': return([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    elif part=='right hip': return([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    elif part=='left knee': return([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    elif part=='right knee': return([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])


# Calcul de l'angle AOB
def calcAngle(A,O,B):
    #O est le centre là ou il y a l'angle A et B doivent etre des vecteurs
    OA=(A[0]-O[0],A[1]-O[1])
    OB=(B[0]-O[0],B[1]-O[1])
    normeA=np.sqrt(OA[0]**2+OA[1]**2)
    normeB=np.sqrt(OB[0]**2+OB[1]**2)
    scalaire=OA[0]*OB[0]+OA[1]*OB[1]
    theta=(scalaire/(normeA*normeB))
    return np.round(theta*93)#angle varie entre -90 et 88

# Visualisation d'angle pour tests
def visualize_angle(frame,value,position):
    cv2.putText(frame, str(value),tuple(np.multiply(position,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

# Determine l'état du pratiquant : 1 = position d'effort, 0 = position de repos
def bodyStatusSide(angleUP,angleDOWN,status): #renvoie 1 si le corps est en bas 0 sinon
    if angleUP>85 and angleDOWN<-70: return(1) #status=down
    elif angleUP<65 and angleDOWN<-70: return(0) #status=up
    else :return(status)

def onePushUps(status,laststatus): #renvoie si une pompe a été fait 0 sinon
    if status==0 and laststatus==1:
        return 1
    else: return 0

# Affiche un nombre au dessus de la tête lors de la détection (ici, le nombre de répétitions)
def visualize_count(frame,value,localisation):
    cv2.putText(frame, str(value),tuple(np.multiply(localisation,[640,480-200]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)

def bodySide(nose,width): #renvoie 0 si la tete à gauche de l'écran et 1 sinon
    if nose[0]*width<=int(width/2): return 0
    else: return 1

## MAIN FUNCTIONS

def main():

    count=0 #nombre de pompes effectués
    status=0 #normal en position haute
    laststatus=0
    vs = cv2.VideoCapture(0)

    while vs.isOpened():
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        width,height=vs.get(3),vs.get(4)
        frame=cv2.flip(frame,1) #on flip horizontalement effet miroir
        frame,results=main_mediapipe(frame)
        position=0
        try:
            # Affectation des keypoints utiles à des variables
            landmarks=results.pose_landmarks.landmark
            nose=getcoordinates(landmarks,'nose')
            position=bodySide(nose,width)

            # Acquisition des coordonnées
            shoulder=[getcoordinates(landmarks,'left shoulder'),getcoordinates(landmarks,'right shoulder')]
            elbow=[getcoordinates(landmarks,'left elbow'),getcoordinates(landmarks,'right elbow')]
            wrist=[getcoordinates(landmarks,'left wrist'),getcoordinates(landmarks,'right wrist')]
            hip=[getcoordinates(landmarks,'left hip'),getcoordinates(landmarks,'right hip')]
            knee=[getcoordinates(landmarks,'left knee'),getcoordinates(landmarks,'right knee')]

            # Liste des coordonnées utiles en fonction de l'orientation du corps
            if position==0:
                listCoord={'shoulder':shoulder[0],'elbow':elbow[0],'wrist':wrist[0],'hip':hip[0],'knee':knee[0]}
            else:
                listCoord={'shoulder':shoulder[1],'elbow':elbow[1],'wrist':wrist[1],'hip':hip[1],'knee':knee[1]}

            # Calcul des angles pertinents
            angleUP=calcAngle(listCoord['elbow'],listCoord['shoulder'],listCoord['hip'])
            angleDOWN=calcAngle(listCoord['shoulder'],listCoord['hip'],listCoord['knee'])

            # Affichage du nombre de répétitions
            visualize_count(frame,count,nose)

            # Visualisation des angles pour tests
            #visualize_angle(frame,angleUP,listCoord['shoulder'])
            #visualize_angle(frame,angleDOWN,listCoord['hip'])

            # Détermination du status à l'instant t
            status=bodyStatusSide(angleUP,angleDOWN,status)

            # Mise à jour du compteur
            count+=onePushUps(status,laststatus)

            # Mise à jour du status
            laststatus=status
        except:
            pass
        cv2.imshow("Camera", frame)

        # Termine le programme en appuyant sur la touche "q" du clavier
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()
main()