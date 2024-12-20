import shutil
import os

project_dir = 'yolo_project'
img_dir = "data_img"
label_dir = "data_labels"

#создание директорий
os.makedirs(os.path.join(project_dir, 'data', 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(project_dir, 'data', 'labels', 'train'), exist_ok=True)


for filename in os.listdir(img_dir):
    if filename.endswith(('.jpg', '.png')):
        img_path = os.path.join(img_dir, filename)
        label_filename = filename[:-4] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        shutil.copy2(img_path, os.path.join(project_dir, 'data', 'images', 'train'))
        shutil.copy2(label_path, os.path.join(project_dir, 'data', 'labels', 'train'))
