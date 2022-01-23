##jumping jack final

## Imports
import numpy as np
import mediapipe as mp
import cv2
mp_pose = mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils

HEIGHT = 480
WIDTH = 640

## USEFUL FUNCTIONS

def mediapipe_detection(image,model): #application de mediapipe sur la frame renvoie les coordonnées des points clés dans results
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results): #dessine les points clés et les relies
    lines=(255,255,255) # White
    dots=(0,0,255) # Red
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=dots,circle_radius=2),mp_drawing.DrawingSpec(lines))

def main_mediapipe(frame): #appelle les 2 fonctions précédentes
    with mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.7) as pose:
        image,results=mediapipe_detection(frame,pose)
        draw_landmarks(image,results)
    return image,results

def getcoordinates(landmarks,part): #renvoie les coordonnées d'un point clé

    if part=='left elbow': return([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
    elif part=='right elbow': return([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

    elif part=='left shoulder': return([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    elif part=='right shoulder': return([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    elif part=='nose': return([landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y])

    elif part=='left hip': return([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    elif part=='right hip': return([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    elif part=='left knee': return([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    elif part=='right knee': return([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])


def calcAngle(A,O,B): #calcul de l'angle AOB
    #O est le centre là ou il y a l'angle A et B doivent etre des vecteurs
    OA=(A[0]-O[0],A[1]-O[1])
    OB=(B[0]-O[0],B[1]-O[1])
    normeA=np.sqrt(OA[0]**2+OA[1]**2)
    normeB=np.sqrt(OB[0]**2+OB[1]**2)
    scalaire=OA[0]*OB[0]+OA[1]*OB[1]
    theta=(scalaire/(normeA*normeB))
    return abs(np.round(theta*90 -90)) # angle varie entre -90 et 88

def visualize_angle_red(frame,value,position): #affichage de l'angle gauche
    cv2.putText(frame, str(value),tuple(np.multiply(position,[WIDTH, HEIGHT]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2,cv2.LINE_AA)

def visualize_angle_white(frame,value,position): #affichage de langle droit
    cv2.putText(frame, str(value),tuple(np.multiply(position,[WIDTH, HEIGHT]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2,cv2.LINE_AA)

def bodyStatus (angle_left_shoulder, angle_right_shoulder, angle_left_hip, angle_right_hip, status): #conditions sur les états haut ou bas. 0 position haute, 1 position basse
    if (angle_left_shoulder, angle_right_shoulder) > (150, 150) and (angle_left_hip, angle_right_hip) != (180, 180): return 1 # 1 if arms up
    elif (angle_left_shoulder, angle_right_shoulder) < (40, 40) and (angle_left_hip, angle_right_hip) >= (178, 178): return 0 # 0 if arms down, repetition starts when arms are down
    else : return status

def oneRep(status, laststatus): #renvoie si une répétition a été faite 0 sinon
    if status==0 and laststatus==1:
        return 1
    else: return 0

def visualize_count(frame,value,localisation): #affiche le nombre de répétition réalisé au dessus de la tete
    cv2.putText(frame, str(value),tuple(np.multiply(localisation,[WIDTH,HEIGHT-400]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)


### MAIN FUNCTION

def main():
    count=0 # squats counter
    status=0 # 0 when UP position
    laststatus=0
    vs = cv2.VideoCapture(0)

    # Max resolution : 1280, 720
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while vs.isOpened():
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        width,height=vs.get(3),vs.get(4)
        frame=cv2.flip(frame,1) #on flip horizontalement effet miroir
        frame,results=main_mediapipe(frame)
        try:

            landmarks = results.pose_landmarks.landmark
            nose=getcoordinates(landmarks,'nose') #coordonnée du nez
            visualize_count(frame, count, nose)

            shoulder=[getcoordinates(landmarks,'left shoulder'),getcoordinates(landmarks,'right shoulder')]
            elbow=[getcoordinates(landmarks,'left elbow'),getcoordinates(landmarks,'right elbow')]
            hip=[getcoordinates(landmarks,'left hip'),getcoordinates(landmarks,'right hip')]
            knee=[getcoordinates(landmarks,'left knee'),getcoordinates(landmarks,'right knee')]

            listCoord = {'shoulder' : shoulder, 'elbow' : elbow, 'hip' : hip, 'knee' : knee}

            #calcul des angles
            angle_left_shoulder = calcAngle(listCoord['knee'][0], listCoord['shoulder'][0], listCoord['elbow'][0])
            angle_right_shoulder = calcAngle(listCoord['knee'][1], listCoord['shoulder'][1] ,listCoord['elbow'][1])
            angle_left_hip = calcAngle(listCoord['knee'][0], listCoord['hip'][0], listCoord['shoulder'][0])
            angle_right_hip = calcAngle(listCoord['knee'][1], listCoord['hip'][1], listCoord['shoulder'][1])

            # visualize_angle_red(frame, angle_left_shoulder, listCoord['shoulder'][0])
            # visualize_angle_white(frame, angle_right_shoulder, listCoord['shoulder'][1])
            # visualize_angle_red(frame, angle_left_hip, listCoord['hip'][0])
            # visualize_angle_white(frame, angle_right_hip, listCoord['hip'][1])
            # visualize_angle_white(frame, angle_crotch, crotch)

            #estimation du nouvel état (état haut ou bas)
            status = bodyStatus(angle_left_shoulder, angle_right_shoulder, angle_left_hip, angle_right_hip, status)

            #mise à jour du nombre de répétition
            count += oneRep(status,laststatus)
            laststatus=status

        except:
            pass
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()
#main()

def countJumpingJack(frame,laststatus,count,nbrepetition): #fonction utilisé par mainvirtualcoach.py
    countOut=count
    laststatusOut=laststatus
    frame,results=main_mediapipe(frame)

    landmarks = results.pose_landmarks.landmark

    nose=getcoordinates(landmarks,'nose')
    shoulder=[getcoordinates(landmarks,'left shoulder'),getcoordinates(landmarks,'right shoulder')]
    elbow=[getcoordinates(landmarks,'left elbow'),getcoordinates(landmarks,'right elbow')]
    hip=[getcoordinates(landmarks,'left hip'),getcoordinates(landmarks,'right hip')]
    knee=[getcoordinates(landmarks,'left knee'),getcoordinates(landmarks,'right knee')]

    listCoord = {'shoulder' : shoulder, 'elbow' : elbow, 'hip' : hip, 'knee' : knee}

    angle_left_shoulder = calcAngle(listCoord['knee'][0], listCoord['shoulder'][0], listCoord['elbow'][0])
    angle_right_shoulder = calcAngle(listCoord['knee'][1], listCoord['shoulder'][1] ,listCoord['elbow'][1])
    angle_left_hip = calcAngle(listCoord['knee'][0], listCoord['hip'][0], listCoord['shoulder'][0])
    angle_right_hip = calcAngle(listCoord['knee'][1], listCoord['hip'][1], listCoord['shoulder'][1])

    status = bodyStatus(angle_left_shoulder, angle_right_shoulder, angle_left_hip, angle_right_hip, laststatus)

    countOut += oneRep(status,laststatus)
    laststatusOut=status

    return laststatusOut,countOut #renvoie l'état et le nombre de répétition après analyse de l'image