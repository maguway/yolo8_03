from ultralytics import YOLO

model_path = "runs/detect/train18/weights/best.pt" # Путь к вашей лучшей модели
model = YOLO(model_path)

results = model.predict(source='data/images/val', conf=0.5) #Измените source на путь к валидационным изображениям,  conf - на порог уверенности
print(results)