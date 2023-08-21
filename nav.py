import cv2
import mediapipe as mp
import pyautogui

# Inicializar a detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializar o desenho de mãos
mp_drawing = mp.solutions.drawing_utils

# Inicializar a webcam
cap = cv2.VideoCapture(0)

# Fator de aumento para o deslocamento do mouse
fator_deslocamento = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Inverter horizontalmente a imagem para corrigir a orientação
    frame = cv2.flip(frame, 1)
    
    # Converter imagem para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detectar mãos no frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Verificar se a mão direita está aberta
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
                # Obter as coordenadas do dedo indicador
                x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                
                # Mover o mouse para as coordenadas com deslocamento aumentado
                pyautogui.moveTo(x * fator_deslocamento, y * fator_deslocamento)
            
            # Desenhar o esqueleto da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()