import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))

# carrega o modelo e seus pesos ja treinados
modelo_carregado = load_model(os.path.join(BASE, "modelo", "modelo_01.h5"))

# carrega o classificador Haar Cascade para deteccao de faces
face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE, "modelo", "haarcascade_frontalface_default.xml")
)

# labels do dataset, na mesma ordem usada no treino
expressoes = ["Surpresa", "Medo", "Nojo", "Feliz", "Triste", "Raiva", "Neutro"]

cap = cv2.VideoCapture(0)
# abre a captura de video da webcam
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # tira a espelhagem do cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # transforma a imagem em escala de cinza
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # detecta as faces na imagem

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        # redimensiona o ROI para o tamanho do modelo (64,64)
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float32") / 255.0
        # normaliza a imagem para 0-1
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        # adiciona as dimensoes que o modelo espera

        predictions = modelo_carregado.predict(roi_gray)
        # faz a predicao da emocao
        max_index = int(np.argmax(predictions))
        # pega a maior probabilidade de predicao
        predicted_emotion = expressoes[max_index]
        # escreve a emocao com base na maior probabilidade

        cv2.putText(frame, predicted_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)

    cv2.imshow("Reconhecimento de emocoes", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
