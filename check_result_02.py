from ultralytics import YOLO

model_path = "runs/detect/train18/weights/best.pt"
model = YOLO(model_path)

metrics = model.val(data='data/data.yaml') # используем val для получения метрик

print("Метрики валидации:")
print(metrics)