import os
import cv2
import time

from numpy.testing.print_coercion_tables import print_coercion_table
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')
OUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUT_DIR, exist_ok=True)

def main(choice, filename):
    USER_CHOICE = choice  # 0 - web, 1 - vid
    if USER_CHOICE == 0:
        cap = cv2.VideoCapture(0)
    elif USER_CHOICE == 1:
        try:
            VIDEO_PATH = os.path.join(VIDEO_DIR, filename)
        except FileNotFoundError:
            print("Video file not found.")
    else:
        print("Invalid choice.")

    model = YOLO('yolov8n.pt')
    CONF_TRESHHOLD = 0.4

    RESIZE_WIDTH = 960 #None

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        if RESIZE_WIDTH is not None:
            h, w = frame.shape[:2]

            scale = RESIZE_WIDTH / w

            new_w = int(scale * w)
            new_h = int(scale * h)

            frame = cv2.resize(frame, (new_w, new_h))

            result = model(frame, conf = CONF_TRESHHOLD, verbose=False)

            people_count = 0
            psevdo_id = 0

            PERSON_CLASS_ID = 0

            for r in result:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls == PERSON_CLASS_ID:
                        people_count += 1
                        psevdo_id += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, f"ID: {psevdo_id} conf: {conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    now = time.time()
                    dt = now - prev_time
                    prev_time = now

                    if dt > 0:
                        fps = 1.0 / dt

                    cv2.putText(frame, f"People count: {people_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, f"FPS: {fps}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow('YOLO', frame)

if __name__ == '__main__':
    choice = int(input("Enter your choice (0 - webcam; 1 - video): "))
    if choice == 1:
        path = str(input("Enter path to video: "))
        main(choice, path)
    else:
        main(choice, None)