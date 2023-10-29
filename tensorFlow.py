import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Загрузка модели с TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1"
model = hub.load(model_url)

# Загрузка изображения
image_path = "test-photos/photo1.jpg"
image = cv2.imread(image_path)

# Подготовка изображения для обработки моделью
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = tf.image.resize(image_rgb, (640, 640))
image_resized = tf.expand_dims(image_resized, axis=0)
image_resized = tf.image.convert_image_dtype(image_resized, tf.uint8)

# Получение результатов обнаружения объектов
results = model(image_resized)

# Извлечение данных о боксах и классах объектов
detection_boxes = results["detection_boxes"][0].numpy()
detection_classes = results["detection_classes"][0].numpy()

# Отображение результатов на изображении с помощью Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)

for i in range(len(detection_boxes)):
    confidence = results["detection_scores"][0].numpy()[i]

    if confidence >= 0.02:
        ymin, xmin, ymax, xmax = detection_boxes[i]
        class_id = int(detection_classes[i])

        if class_id == 1:
            # Отрисовка ограничивающего прямоугольника
            image_h, image_w, _ = image_rgb.shape
            x, y, w, h = int(xmin * image_w), int(ymin * image_h), int((xmax - xmin) * image_w), int((ymax - ymin) * image_h)
            rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor='green')
            plt.gca().add_patch(rect)
            plt.text(x, y, s=f"Class: {class_id}", color='green', verticalalignment='top')
    
plt.axis('off')
# plt.savefig("final.jpg", bbox_inches='tight')
plt.show()