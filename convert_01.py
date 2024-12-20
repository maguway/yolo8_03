import os
import json

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = f.read().strip().split(',')
    return {name.strip(): idx for idx, name in enumerate(classes)}

def convert_json_to_yolo(input_folder, output_folder, class_file):
    os.makedirs(output_folder, exist_ok=True) # Более компактное создание папки
    classes = load_classes(class_file)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_width = data['imageWidth']
            image_height = data['imageHeight']
            output_lines = []

            for shape in data['shapes']:
                class_name = shape['label']
                if class_name in classes:
                    class_id = classes[class_name]
                    points = shape['points']
                    x_min = min(point[0] for point in points)
                    x_max = max(point[0] for point in points)
                    y_min = min(point[1] for point in points)
                    y_max = max(point[1] for point in points)

                    x_center = (x_min + x_max) / (2 * image_width)
                    y_center = (y_min + y_max) / (2 * image_height)
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    output_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}") #добавил форматирование

            output_file = os.path.join(output_folder, filename[:-5] + ".txt")
            with open(output_file, 'w') as f:
                f.write("\n".join(output_lines))

# Пример использования
input_folder = 'data_json'
output_folder = 'data_labels'
class_file = 'classes.txt'
convert_json_to_yolo(input_folder, output_folder, class_file)