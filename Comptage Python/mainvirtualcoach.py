##Main file virtual coach
"""
Programme exemple d'une séance avec 1 exercice jumping jack
"""
##Importation
import cv2
import numpy as np
import time
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
import sys
sys.path.insert(0,r"C:/STOCK/Etude/Ensea3A/Projet3A/ProgramFinal")
import jumpingjacks_counter_final

##PATH
spiderman_hello_path=r"C:/STOCK/Etude/Ensea3A/Projet3A/Spidey_datas/helloSpiderman.avi"
spiderman_jumpingjacks_path=r"C:/STOCK/Etude/Ensea3A/Projet3A/Spidey_datas/jumingjackSpiderman.avi"
##CONSTANTS
WIDTH=640
HEIGHT=480
WHITE=(255,255,255)
RED=(0,0,255)
delay=3 #time in seconds to wait before the animation starts
CAMERA='camera' #Nom de la fenetre où l'utilisateur apparait
SPIDERCAM='spiderman' #Nom de la fenetre où spiderman apparait
cv2.namedWindow(CAMERA)
cv2.moveWindow(CAMERA, 40,30) #positionnement de la fenetre
cv2.namedWindow(SPIDERCAM)
cv2.moveWindow(SPIDERCAM, 40+WIDTH,30) #positionnement de la fenetre
exerciceNameDict={1:'Jumping Jacks'}
spidermanVideoDict={0:spiderman_hello_path,1:spiderman_jumpingjacks_path}
##PROGRAM

def mediapipe_detection(image,model): #application de mediapipe sur la frame renvoie les coordonnées des points clés dans results
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results): #dessine les points clés et les relies
    lines=WHITE #couleur
    dots=RED
    mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=dots,circle_radius=2),mp_drawing.DrawingSpec(lines))

def main_mediapipe(frame): #appelle les 2 fonctions précédentes
    with mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.7) as pose:
        image,results=mediapipe_detection(frame,pose)
        draw_landmarks(image,results)
    return image,results

def spiderVideo(path,image): #permet de charger les vidéos du coach et changer le fond
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        if not ret: break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        u_green = np.array([150,255,150])
        l_green = np.array([5,110,5])

        mask = cv2.inRange(frame, l_green, u_green)
        res = cv2.bitwise_and(frame, frame, mask = mask)

        f = frame - res
        f = np.where(f == 0, image, f)

        cv2.imshow(SPIDERCAM, f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()

##MAIN
def main(): #début de la séance
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH) #resize de la fenetre
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)
    firstframe=True
    """
    Programme de la séance P
    (x,y) avec x l'index de l'exercice et y le nombre de répétitions à faire
    1 <=> jumping jacks
    """
    P=[(1,5)] #programme de la séance jumping jacks 5 répétitions
    indexP=0 #indice de l'avancer de la séance
    count=0 #compteur pour les répétitions
    laststatus=0 #dernier état enregistrer
    firsthello=False #définit si l'animation d'introduction a été déjà lancé
    shown=False #défnit si l'animation démonstration de l'exeercice a été lancé
    while cap.isOpened():
        ret, frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        frameCopy=frame.copy()
        if firstframe: #capture du décor
            frame0=frame
            firstframe=False
        try:
            frame,results=main_mediapipe(frame)
            landmarks = results.pose_landmarks.landmark
            nose=(landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y) #position de la tete de l'utilisateur dans l'écran

            if nose!=(0,0) and not firsthello: #diffuse la vidéo spiderman hello si un utilisateur est détecté
                time.sleep(delay) #temps de delay avant lancement
                spiderVideo(spiderman_hello_path,frame0)
                firsthello=True #l'introduction a été réalisé
            if nose!=(0,0) and not shown:
                spiderVideo(spidermanVideoDict[P[indexP][0]],frame0) #démonstration de l'exercice à réaliser
                shown=True #l'exercice a été montré
            if shown and P[indexP][0]==1: #début de l'entrainement
                nbrepetition=P[indexP][1] #nombre de répétition à réaliser
                cv2.rectangle(frame, (0,0), (640,40), (0,0,0), -1)
                cv2.putText(frame,exerciceNameDict[1]+" "+str(count)+'/'+str(nbrepetition),(12,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA) #affichage de l'exercice, du nombre de répétition réalisé et le nombre de répétition à faire.
                laststatus,count=countJumpingJack(frameCopy,laststatus,count,nbrepetition)#vérification en faisant appel au fichier python correspondant à l'exercice
                if count==nbrepetition: #fin de l'exercice
                    indexP+=1 #passage à l'exercice suivant
                    shown=False #on reset l'animation de démonstration
            if indexP==len(P): #Le programme est terminé fin de l'entrainement
                cv2.rectangle(frame, (0,0), (200,40), (0,0,0), -1)
                cv2.putText(frame,"THE END WELL DONE",(12,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
                break
        except:
            pass
        cv2.imshow(CAMERA,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
main()