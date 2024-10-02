import cv2
from deepface import DeepFace
import os

# TensorFlow oneDNN optimizasyonlarını kapatma (isteğe bağlı)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Kamerayı başlat (DirectShow kullanarak)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera çerçevesi alınamadı. Lütfen kameranın düzgün çalıştığından emin olun.")
        break

    # Yüzü tespit et ve analiz et
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion', 'age'], enforce_detection=False)

        for result in analysis:
            emotion = result['dominant_emotion']
            age = result['age']
            cv2.putText(frame, f"Emotion: {emotion}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Age: {age}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    except Exception as e:
        print("Error:", e)

    # Sonucu ekranda göster
    cv2.imshow('Yüz Duygu ve Yaş Analizi', frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
