from ultralytics import YOLO
import os
import shutil
import yaml

# Путь к файлу с классами
classes_file = 'classes.txt'

# Создаем необходимую структуру проекта
project_dir = 'yolo_project'
os.makedirs(os.path.join(project_dir, 'data', 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(project_dir, 'data', 'labels', 'train'), exist_ok=True)
shutil.copy2('classes.txt', project_dir)

print(os.path.join(project_dir, 'data', 'data.yaml'))
# Копируем файлы из старых папок в новые
img_dir = "data_img"
label_dir = "data_labels"
for filename in os.listdir(img_dir):
    if filename.endswith(('.jpg', '.png')):  # обработка jpg и png
        img_path = os.path.join(img_dir, filename)
        label_filename = filename[:-4] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        shutil.copy2(img_path, os.path.join(project_dir, 'data', 'images', 'train'))
        shutil.copy2(label_path, os.path.join(project_dir, 'data', 'labels', 'train'))

# Создаем data.yaml
data_yaml = {
    "train": {
        "path": "images/train",
        "img_size": 640,
        "include": "*.jpg",
        "labels": "labels/train",
        "names": "../classes.txt"
    }
}

with open(os.path.join(project_dir, 'data', 'data.yaml'), 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

# Обучение модели
model = YOLO('yolov8n.yaml')
results = model.train(data=os.path.join(project_dir, 'data', 'data.yaml'), epochs=100, imgsz=640, batch=16, device='0')
