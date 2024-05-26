from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

app = FastAPI()

# Загрузка модели с TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1"
model = hub.load(model_url)

def detect_objects(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, (640, 640))
    image_resized = tf.expand_dims(image_resized, axis=0)
    image_resized = tf.image.convert_image_dtype(image_resized, tf.uint8)

    results = model(image_resized)

    detection_boxes = results["detection_boxes"][0].numpy()
    detection_classes = results["detection_classes"][0].numpy()
    detection_scores = results["detection_scores"][0].numpy()

    objects = []
    for i in range(len(detection_boxes)):
        objects.append({
            "box": detection_boxes[i].tolist(),
            "class": int(detection_classes[i]),
            "score": float(detection_scores[i])
        })

    return objects

@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Файл изображения не предоставлен")

    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    objects = detect_objects(image)
    
    return JSONResponse(content={"objects": objects})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)