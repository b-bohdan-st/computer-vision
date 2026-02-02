import os
import shutil
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

PROJECT_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'CW/images')

OUT_DIR = os.path.join(PROJECT_DIR, 'CW/out')
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

    best_conf = 0.0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == PERSON_CLASS_ID and confidence > CONFIRM_PERSON:
            box = detections[0, 0, i, 3:7]

            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            if confidence > best_conf:
                best_conf = confidence
                best_box = (x1, y1, x2, y2)

    return best_box is not None, best_box, best_conf

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

        found, best_box, best_conf = people_detection(img)

        if found:
            shutil.copyfile(in_path, os.path.join(PEOPLE_DIR, filename))

            boxed = img.copy()
            x1, y1, x2, y2 = best_box

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1] - 1, x2)
            y2 = min(img.shape[0] - 1, y2)

            cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(boxed, f"{best_conf:.2f}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imwrite(os.path.join(PEOPLE_DIR, f"boxed_{filename}"), boxed)
        else:
            shutil.copyfile(in_path, os.path.join(NO_PEOPLE_DIR, filename))

if __name__ == "__main__":
    run()
