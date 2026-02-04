import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, 'output')
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')

OUT_DIR_PEOPLE = os.path.join(OUT_DIR, 'people')
OUT_DIR_ANIMALS = os.path.join(OUT_DIR, 'animals')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_DIR_PEOPLE, exist_ok=True)
os.makedirs(OUT_DIR_ANIMALS, exist_ok=True)

PERSON_CLASS_ID = 0
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

def main(choice, filename=None):
    if choice == 0:
        cap = cv2.VideoCapture(0)
        mode = "people"
        out_fps = 25
    elif choice == 1:
        video_path = os.path.join(VIDEO_DIR, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video file not found")
            return
        mode = "animals"
        out_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    else:
        print("Invalid choice")
        return

    model = YOLO("yolov8n.pt")
    CONF_THRESHOLD = 0.4
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 480

    prev_time = time.time()
    fps = 0.0

    writer = None
    output_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if RESIZE_WIDTH is not None and RESIZE_HEIGHT is not None:
            h, w = frame.shape[:2]
            scale = min(RESIZE_WIDTH / w, RESIZE_HEIGHT / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            if mode == "people":
                save_dir = OUT_DIR_PEOPLE
            else:
                save_dir = OUT_DIR_ANIMALS

            output_path = os.path.join(save_dir, f"result_{mode}_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        people_count = 0
        cats_count = 0
        dogs_count = 0

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if mode == "people" and cls == PERSON_CLASS_ID:
                    people_count += 1
                    label = f"Person {conf:.2f}"
                    color = (0, 255, 0)
                elif mode == "animals":
                    if cls == CAT_CLASS_ID:
                        cats_count += 1
                        label = f"Cat {conf:.2f}"
                        color = (255, 0, 0)
                    elif cls == DOG_CLASS_ID:
                        dogs_count += 1
                        label = f"Dog {conf:.2f}"
                        color = (0, 255, 0)
                    else:
                        continue
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt

        if mode == "people":
            cv2.putText(frame, f"People count: {people_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            total_animals = cats_count + dogs_count
            cv2.putText(frame, f"Cats: {cats_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Dogs: {dogs_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Total animals: {total_animals}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        writer.write(frame)
        cv2.imshow("YOLO", frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("0 - Webcam (people detection)")
    print("1 - Video (cats and dogs detection)")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        filename = input("Enter video filename (from videos folder): ")
        main(choice, filename)
    else:
        main(choice)