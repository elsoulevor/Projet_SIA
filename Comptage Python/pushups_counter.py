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
    cv2.putText(frame, str(value),tuple(np.multiply(position,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

# Determine l'état du pratiquant : 1 = position d'effort, 0 = position de repos
def bodyStatus(angle,status): #renvoie 1 si le corps est en bas 0 sinon
    if angle[0]>-30 and angle[1]>-30: return(1) #status=down
    elif angle[0]<-60 and angle[1]<-60: return(0) #status=up
    else :return(status)

def onePushUps(status,laststatus): #renvoie si une pompe a été fait 0 sinon
    if status==0 and laststatus==1:
        return 1
    else: return 0

# Affiche un nombre au dessus de la tête lors de la détection (ici, le nombre de répétitions)
def visualize_count(frame,value,position):
    cv2.putText(frame, str(value),tuple(np.multiply(position,[640,480-200]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)

## MAIN FUNCTION

def main():
    count=0 #nombre de pompes effectués
    status=0 #normal en position haute
    laststatus=0
    vs = cv2.VideoCapture(0)

    while vs.isOpened():
        (grabbed, frame) = vs.read()

        if not grabbed:
            break
        frame=cv2.flip(frame,1)
        frame,results=main_mediapipe(frame)
        try:
            # Affectation des keypoints utiles à des variables
            landmarks=results.pose_landmarks.landmark

            # Acquisition des coordonnées
            shoulder=[getcoordinates(landmarks,'left shoulder'),getcoordinates(landmarks,'right shoulder')]
            elbow=[getcoordinates(landmarks,'left elbow'),getcoordinates(landmarks,'right elbow')]
            wrist=[getcoordinates(landmarks,'left wrist'),getcoordinates(landmarks,'right wrist')]
            nose=getcoordinates(landmarks,'nose')

            # Calcul des angles pertinents
            angle=[calcAngle(shoulder[0],elbow[0],wrist[0]),calcAngle(shoulder[1],elbow[1],wrist[1])]

            # Mise à jour du status
            status=bodyStatus(angle,status)

            # Mise à jour du compteur
            count+=onePushUps(status,laststatus)

            # Affichage du nombre de répétitions
            visualize_count(frame,count,nose)

            # Affichage des angles pour tests
            #visualize_angle(frame,angle[0],elbow[0])
            #visualize_angle(frame,angle[1],elbow[1])

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