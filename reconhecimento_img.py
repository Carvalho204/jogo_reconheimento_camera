import cv2
import mediapipe as mp

#VINCULAR WEBCAM AO PYTHON

#cria conexão com a webcam

webcam = cv2.VideoCapture(0)

reconhecimento_maos = mp.solutions.hands
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()

#confirmar conexão e ler webcam
if webcam.isOpened():
    validacao, frame = webcam.read()
    #fazer vários frames 
    while validacao == True:
        validacao, frame = webcam.read()
        #converte BGR em RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        #desenhar a mão
        lista_maos = maos.process(frameRGB)
        if lista_maos.multi_hand_landmarks:
            for mao in lista_maos.multi_hand_landmarks:
                desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
        
        #mostrar frame
        cv2.imshow('video da webcam', frame)
        #tempo de espera
        tecla = cv2.waitKey(2)
        #mandar ele parar se eu clicar no Esc(na ordem da tabela ASCII)
        if tecla == 27:
            break
    
#desconectar da webcam
webcam.release()
cv2.destroyAllWindows()