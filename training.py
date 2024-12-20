from ultralytics import YOLO
import os

# Относительные пути
data_yaml_path = "./data/data.yaml"
model_path = "yolov8n.pt" #Можно указать другую модель


try:
    # Загрузка модели (если она не существует, будет загружена из сети)
    model = YOLO(model_path)

    # Обучение модели
    results = model.train(data=data_yaml_path, epochs=100) # или другое количество эпох

    # Вывод результатов (может потребоваться доработка в зависимости от нужной информации)
    print(results)

except Exception as e:
    print(f"Произошла ошибка во время обучения: {e}")