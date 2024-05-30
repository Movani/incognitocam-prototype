import cv2
import mediapipe as mp
from pyzbar.pyzbar import decode
import numpy as np

# Frase predefinida
HIDE_ME = 'Hide me!'

def process_face_blur(img):
    mp_face_detection = mp.solutions.face_detection # método de detecção de faces da biblioteca opencv

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

        H, W, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # normaliza cor para aumentar grau de precisão da detecção de faces
        out = face_detection.process(img_rgb) # detecta faces

        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                # calcula tamanho da espaço a ser borrado
                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W) 
                y1 = int((y1 * 0.3) * H)
                w = int(w * W)
                h = int((h * H) * 1.5) 

                # borra faces
                try:
                    img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (80, 80))
                except:
                    print('erro')

        return img


# captura a partir de um dispositivo de entrada de vídeos
cap = cv2.VideoCapture(0) 

ret, frame = cap.read()

while ret:
    qr_info = decode(frame)

    if len(qr_info) > 0:
        qr = qr_info[0]

        data = qr.data
        rect = qr.rect
        polygon = qr.polygon

        # decodifica qrcode    
        qr_code_data = data.decode() 
        
        # escreve conteúdo do qrcode
        cv2.putText(frame, qr_code_data, (rect.left, rect.top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3) 

        # desenha retangulos do qrcode
        frame = cv2.rectangle(frame, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (0, 255, 0), 5)
        frame = cv2.polylines(frame, [np.array(polygon)], True, (255, 0, 0), 5)

        if qr_code_data == HIDE_ME:
            frame = process_face_blur(frame) # função de borras faces


    cv2.imshow('frame', frame) # mostra janela com o resultado do processo
    cv2.waitKey(25) # tecla parada

    ret, frame = cap.read()

cap.release()