import os
import shutil
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

PROJECT_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'HW/images')

OUT_DIR = os.path.join(PROJECT_DIR, 'HW/out')
PEOPLE_DIR = os.path.join(OUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUT_DIR, 'no_people')

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

prototxt_path = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.prototxt.txt')
model_path = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

PERSON_CLASS_ID = CLASSES.index('person')
CONFIRM_PERSON = 0.6

def people_detection(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), mean=(127.5, 127.5, 127.5))

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONFIRM_PERSON:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)

    return boxes, confidences

def run():
    answer = input("Очистити папки PEOPLE та NO_PEOPLE? (1 - так, 0 - ні): ").strip()

    if answer == "1":
        for folder in (PEOPLE_DIR, NO_PEOPLE_DIR):
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isfile(path):
                    os.remove(path)

    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        in_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(in_path)
        if img is None:
            continue

        boxes, confidences = people_detection(img)

        if len(boxes) > 0:
            shutil.copyfile(in_path, os.path.join(PEOPLE_DIR, filename))

            boxed = img.copy()

            for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(boxed, f"{conf:.2f}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imwrite(os.path.join(PEOPLE_DIR, f"boxed_{filename}"), boxed)
        else:
            shutil.copyfile(in_path, os.path.join(NO_PEOPLE_DIR, filename))

if __name__ == "__main__":
    run()
