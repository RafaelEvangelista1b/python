import cv2
import numpy as np

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Define um limite de largura e altura para a janela
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Inicializa variáveis para armazenar o quadro anterior
previous_frame = None

while True:
    # Captura um quadro do vídeo
    ret, frame = cap.read()

    if not ret:
        break

    # Converte o quadro para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Se for o primeiro quadro, inicialize o quadro anterior
    if previous_frame is None:
        previous_frame = gray_frame
        continue

    # Calcula a diferença entre o quadro atual e o anterior
    frame_delta = cv2.absdiff(previous_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilata a imagem para preencher buracos
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encontra contornos no frame threshold
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop através dos contornos
    for contour in contours:
        # Apenas considera contornos com uma área significativa
        if cv2.contourArea(contour) < 500:
            continue

        # Desenha um retângulo ao redor do movimento detectado
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o resultado
    cv2.imshow("Frame", frame)
    cv2.imshow("Threshold", thresh)

    # Atualiza o quadro anterior
    previous_frame = gray_frame

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()