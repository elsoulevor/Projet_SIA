## Imports

import numpy as np
import mediapipe as mp
import cv2
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

HEIGHT = 720
WIDTH = 1280

## USEFUL FUNCTIONS

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

# Affiche le squelette
def draw_landmarks(image,results):
    lines = (255,255,255) # White
    dots = (0,0,255) # Red
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=dots,circle_radius=2),mp_drawing.DrawingSpec(lines))


def main_mediapipe(frame):
    with mp_pose.Pose(min_detection_confidence = 0.6,min_tracking_confidence = 0.7) as pose:
        image,results = mediapipe_detection(frame,pose)
        draw_landmarks(image,results)
    return image,results

# Obtention des coordonnées utiles (uniquement celles que l'on va utiliser)
def getcoordinates(landmarks,part):
    if part == 'left shoulder': return([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    elif part == 'right shoulder': return([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    elif part == 'nose': return([landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y])

    elif part == 'left ear': return([landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y])

    elif part == 'left hip': return([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    elif part == 'right hip': return([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    elif part == 'left knee': return([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    elif part == 'right knee': return([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])

    elif part == 'left ankle': return([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    elif part == 'right ankle': return([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

# Calcul de l'angle AOB
def calcAngle(A,O,B):
    OA = (A[0]-O[0],A[1]-O[1])
    OB = (B[0]-O[0],B[1]-O[1])
    normeA = np.sqrt(OA[0]**2+OA[1]**2)
    normeB = np.sqrt(OB[0]**2+OB[1]**2)
    scalaire = OA[0]*OB[0]+OA[1]*OB[1]
    theta = (scalaire/(normeA*normeB))
    return abs(np.round(theta*90 -90))

# Visualisation d'angle pour tests
def visualize_angle_red(frame,value,position):
    cv2.putText(frame, str(value),tuple(np.multiply(position,[WIDTH, HEIGHT]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2,cv2.LINE_AA)

# Idem
def visualize_angle_blue(frame,value,position):
    cv2.putText(frame, str(value),tuple(np.multiply(position,[WIDTH, HEIGHT]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),2,cv2.LINE_AA)

# Determine l'état du pratiquant : 1 = position d'effort, 0 = position de repos
def bodyStatusSide (angle_hip ,angle_knee, status):
    if angle_hip < 100 and angle_knee < 100 : return 1 # Position basse si les muscles sont contractés (angles des hanches + genous respectés)
    elif angle_hip > 170 and angle_knee > 170 : return 0 # Position de repos
    else : return status

def oneRep(status,laststatus): #renvoie si une répétition a été faite 0 sinon
    if status == 0 and laststatus == 1:
        return 1
    else: return 0

# Affiche un nombre au dessus de la tête lors de la détection (ici, le nombre de répétitions)
def visualize_count(frame,value,localisation):
    cv2.putText(frame, str(value),tuple(np.multiply(localisation,[WIDTH,HEIGHT-400]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)

def bodySide(nose, left_ear): #renvoie 0 si la tete à gauche de l'écran et 1 sinon
    if nose[0] - left_ear[0] >= 0: return 0
    else: return 1

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
        width,height = vs.get(3),vs.get(4)
        frame = cv2.flip(frame,1) #on flip horizontalement effet miroir
        frame,results = main_mediapipe(frame)
        try:
            # Affectation des keypoints utiles à des variables
            landmarks = results.pose_landmarks.landmark
            nose = getcoordinates(landmarks,'nose')
            left_ear = getcoordinates(landmarks, 'left ear')
            position = bodySide(nose, left_ear)

            # Acquisition des coordonnées
            shoulder = [getcoordinates(landmarks,'left shoulder'),getcoordinates(landmarks,'right shoulder')]
            hip = [getcoordinates(landmarks,'left hip'),getcoordinates(landmarks,'right hip')]
            knee = [getcoordinates(landmarks,'left knee'),getcoordinates(landmarks,'right knee')]
            ankle = [getcoordinates(landmarks,'left ankle'),getcoordinates(landmarks,'right ankle')]

            # Liste des coordonnées utiles en fonction de l'orientation du corps
            if position == 0:
                listCoord = {'shoulder' : shoulder[0], 'hip' : hip[0], 'knee' : knee[0], 'ankle' : hip[0], 'ankle' : ankle[0]}
            else:
                listCoord = {'shoulder' : shoulder[1], 'hip' : hip[1], 'knee' : knee[1], 'ankle' : hip[1], 'ankle' : ankle[1]}

            # Affichage du nombre de répétitions
            visualize_count(frame, count, nose)

            # Calcul des angles pertinents
            angle_hip = calcAngle(listCoord['knee'],listCoord['hip'],listCoord['shoulder'])
            angle_knee = calcAngle(listCoord['ankle'],listCoord['knee'],listCoord['hip'])

            # Visualisation des angles pour tests
            # visualize_angle_red(frame, angle_hip, listCoord['hip'])
            # visualize_angle_blue(frame, angle_knee, listCoord['knee'])

            # Détermination du status à l'instant t
            status = bodyStatusSide(angle_hip, angle_knee, status)

            #Mise à jour du compteur
            count += oneRep(status,laststatus)

            #Mise à jour du status
            laststatus = status

        except:
            pass
        cv2.imshow("Camera", frame)

        # Termine le programme en appuyant sur la touche "q" du clavier
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    vs.release()
    cv2.destroyAllWindows()
main()