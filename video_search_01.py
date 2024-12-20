import cv2
import torch
from ultralytics import YOLO
import random

# Загрузка обученной модели
model = YOLO('runs/detect/train18/weights/best.pt')

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация словаря цветов (ОБЯЗАТЕЛЬНО!)
colors = {}


# Цикл обработки кадров
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предсказание
    results = model(frame)

    # Рисование bounding boxes и имен классов на изображении
    for *xyxy, conf, cls in results[0].boxes.data.tolist():
        label = f'{model.names[int(cls)]} {conf:.2f}'
        #Генерируем цвет только если его нет
        if model.names[int(cls)] not in colors:
            colors[model.names[int(cls)]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = colors[model.names[int(cls)]]
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Показ изображения
    cv2.imshow('Object Detection', frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()