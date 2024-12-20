from ultralytics import YOLO
import os
import shutil

# Путь к файлу с классами
classes_file = 'classes.txt'

# Пути к папкам с данными
data_dir = 'data'  # Создайте папку 'data' со структурой: images/train/, labels/train/
os.makedirs(os.path.join(data_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "labels", "train"), exist_ok=True)

# Копируем файлы — используйте shutil для копирования файлов.
#   Это гарантирует, что исходные файлы не изменятся и у вас будет резервная копия.
img_dir = "data_img"
label_dir = "data_labels"
for filename in os.listdir(label_dir):
  if filename.endswith("_yolo.txt"):
      label_path = os.path.join(label_dir, filename)
      img_filename = filename[:-8] + ".jpg" # Убираем "_yolo" из имени файла
      img_path = os.path.join(img_dir, img_filename)
      shutil.copy2(img_path, os.path.join(data_dir, "images", "train")) # Копирует и сохраняет метаданные
      shutil.copy2(label_path, os.path.join(data_dir, "labels", "train")) # Копирует и сохраняет метаданные



# Обучение модели
model = YOLO('yolov8n.yaml')  # Выберите подходящую модель (n, s, m, l, x)
results = model.train(data=data_dir, epochs=100, imgsz=640, batch=16, device='0') #epochs нужно подбирать, batch - размер батча, device - номер вашей видеокарты