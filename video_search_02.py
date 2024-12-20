import cv2
import torch
from ultralytics import YOLO
import random

# Загрузка обученной модели
model = YOLO('runs/detect/train18/weights/best.pt')

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация словаря цветов
colors = {}

# Порог уверенности (можно экспериментировать с этим значением)
confidence_threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for *xyxy, conf, cls in results[0].boxes.data.tolist():
        # Способ 1: Проверка уверенности для каждого объекта
        """if conf < confidence_threshold:
           label = "unknown"
           color = (0, 0, 0)
        else:
           label = f'{model.names[int(cls)]} {conf:.2f}'
           if model.names[int(cls)] not in colors:
               colors[model.names[int(cls)]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
           color = colors[model.names[int(cls)]]"""

        # Способ 3: Проверка максимальной уверенности для всех объектов в кадре

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            max_conf = max(results[0].boxes.conf.tolist())  # Преобразуем в список для получения максимального значения
        else:
            max_conf = 0

        if max_conf < confidence_threshold:
            label = "unknown"
            color = (0, 0, 0)
        else:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            if model.names[int(cls)] not in colors:
                colors[model.names[int(cls)]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = colors[model.names[int(cls)]]


        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()