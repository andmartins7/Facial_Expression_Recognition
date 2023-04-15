import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

# Carrega o modelo de detecção de faces
face_cascade = cv2.CascadeClassifier(
    'model/haarcascade_frontalface_default.xml')
emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy', 4: 'Neutral',
                5: 'Sad', 6: 'Surprised'}  # Dicionário com as emoções correspondentes aos índices
# Carrega o modelo de detecção de emoções
model = load_model('model/emotion_detection_model.h5')


class EmotionDetector:
    def __init__(self, window):
        self.window = window
        self.window.title("Detector de Emoções")

        # Cria um widget para exibir o vídeo da câmera
        self.video_frame = tk.Label(window)
        self.video_frame.pack()

        # Cria widgets para exibir a emoção detectada e a probabilidade correspondente
        self.emotion_label = tk.Label(window, text="")
        self.emotion_label.pack()
        self.probability_label = tk.Label(window, text="")
        self.probability_label.pack()

        # Cria um botão para fechar o programa
        self.quit_button = tk.Button(window, text="Sair", command=window.quit)
        self.quit_button.pack()

        # Inicia a captura de vídeo
        self.cap = cv2.VideoCapture(0)
        self.update()

    def update(self):
        ret, frame = self.cap.read()  # Captura um quadro de vídeo
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detecta faces no quadro
        label = "Desconhecido"  # Adiciona um valor inicial à variável label
        probability = 0.0  # Adiciona um valor inicial à variável probability
        for (x, y, w, h) in faces:
            # Extrai a região de interesse correspondente ao rosto
            roi_gray = gray[y:y+h, x:x+w]
            # Redimensiona a imagem para o tamanho esperado pelo modelo de detecção de emoções
            roi_gray = cv2.resize(roi_gray, (64, 64))
            # Normaliza os valores de pixel para o intervalo [0, 1]
            roi_gray = roi_gray.astype('float')/255.0
            # Adiciona uma dimensão ao início do array para torná-lo compatível com o modelo
            roi_gray = np.expand_dims(roi_gray, axis=0)
            # Adiciona uma dimensão ao final do array para torná-lo compatível com o modelo
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            preds = model.predict(roi_gray)[0]  # Executa a predição de emoção
            if len(faces) > 0:
                label = emotion_dict[np.argmax(preds)]
                probability = np.max(preds)
            else:
                label = "Desconhecido"
                probability = 0.0
            # Determina a emoção correspondente ao índice com maior probabilidade
            label = emotion_dict[np.argmax(preds)]
            # Determina a probabilidade correspondente à emoção detectada e atualiza os widgets correspondentes
            probability = np.max(preds)
        # Atualiza o widget de vídeo
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.video_frame.configure(image=img)
        self.video_frame.image = img

        # Atualiza os widgets de emoção e probabilidade
        self.emotion_label.configure(text=f"Emoção: {label}")
        self.probability_label.configure(text=f"Probabilidade: {probability:.2f}")

        # Chama novamente esta função após 10 milissegundos
        self.window.after(10, self.update)
        
# Cria a janela principal
root = tk.Tk()
app = EmotionDetector(root)
root.mainloop()