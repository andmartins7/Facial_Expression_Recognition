import cv2
import numpy as np
from keras.models import load_model

file_path = 'model\haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(file_path) # Carrega o modelo de detecção de faces

emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'} # Dicionário com as emoções correspondentes aos índices

file_path_emotion = 'model\emotion_detection_model.h5'

model = load_model(file_path_emotion) # Carrega o modelo de detecção de emoções

cap = cv2.VideoCapture(0) # Acessa a câmera de vídeo
while True:
    ret, frame = cap.read() # Captura um quadro de vídeo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Detecta faces no quadro
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] # Extrai a região de interesse correspondente ao rosto
        roi_gray = cv2.resize(roi_gray, (64, 64)) # Redimensiona a imagem para o tamanho esperado pelo modelo de detecção de emoções
        roi_gray = roi_gray.astype('float')/255.0 # Normaliza os valores de pixel para o intervalo [0, 1]
        roi_gray = np.expand_dims(roi_gray, axis=0) # Adiciona uma dimensão ao início do array para torná-lo compatível com o modelo
        roi_gray = np.expand_dims(roi_gray, axis=-1) # Adiciona uma dimensão ao final do array para torná-lo compatível com o modelo
        preds = model.predict(roi_gray)[0] # Executa a predição de emoção
        label = emotion_dict[np.argmax(preds)] # Determina a emoção correspondente ao índice com maior probabilidade
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Exibe a emoção detectada acima do retângulo que envolve o rosto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Desenha um retângulo ao redor do rosto detectado
    cv2.imshow('frame', frame) # Exibe o quadro de vídeo
    if cv2.waitKey(1) & 0xFF == ord('q'): # Sai do loop quando a tecla 'q' é pressionada
        break
cap.release() # Libera a câmera de vídeo
cv2.destroyAllWindows() # Fecha as janelas abertas