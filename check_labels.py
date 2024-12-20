import cv2
import os
import numpy as np

def visualize_bboxes(image_dir, label_dir, output_dir):
    """Визуализирует bounding boxes на изображениях и сохраняет их."""
    os.makedirs(output_dir, exist_ok=True)  # Создаем выходную папку, если ее нет

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            label_path = os.path.join(label_dir, filename)
            img_filename = filename[:-4] + ".jpg"  # Предполагаем расширение .jpg
            img_path = os.path.join(image_dir, img_filename)
            output_path = os.path.join(output_dir, img_filename)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Ошибка: Невозможно загрузить изображение {img_path}")
                    continue

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])

                    x_min = int((x_center - width / 2) * img.shape[1])
                    y_min = int((y_center - height / 2) * img.shape[0])
                    x_max = int((x_center + width / 2) * img.shape[1])
                    y_max = int((y_center + height / 2) * img.shape[0])

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv2.imwrite(output_path, img)  # Сохраняем изображение
                print(f"Изображение с bounding boxes сохранено: {output_path}")

            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")


# Пример использования
image_dir = "data_img"
label_dir = "data_labels"
output_dir = "data_images_with_boxes"  # Папка для сохранения результатов
visualize_bboxes(image_dir, label_dir, output_dir)