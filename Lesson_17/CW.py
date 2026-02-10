import cv2
import os
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "videos")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def main(choice, log, filename=None):
    if choice == 0:
        cap = cv2.VideoCapture(0)
    elif choice == 1:
        VIDEO_PATH = os.path.join(VIDEO_DIR, filename)
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        print("Invalid input")
        return
    
    if log == "0":
        log = False
    elif log == "1":
        log = True
    else:
        print("Invalid input")

    if not cap.isOpened():
        print("Cannot open video source")
        return

    MODEL_PATH = 'yolov8n.pt'
    CONF_THRESH = 0.5

    TRACKER = "bytetrack.yaml"

    model = YOLO(MODEL_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30
    
    writer = None

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(OUT_DIR, f"result_{int(time.time())}.mp4")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    seen_id_total = set()
    seen_id_class = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=log)
        r = result[0]

        if r.boxes is None or len(r.boxes) == 0:
            cv2.imshow("Video", frame)
            if writer is not None:
                writer.write(frame)
            continue
        
        boxes = r.boxes

        xyxy = boxes.xyxy.cpu().numpy()

        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        track_id = None
        if boxes.id is not None:
            track_id = boxes.id.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            class_id = int(cls[i])
            class_name = model.names[class_id]
            score = conf[i]

            tid = -1
            if track_id is not None:
                tid = int(track_id[i])

            if tid != -1:
                seen_id_total.add(tid)
                
                if class_name not in seen_id_class:
                    seen_id_class[class_name] = set()

                seen_id_class[class_name].add(tid)

            label = (f'{class_name} {score:.2f}')

            if tid != -1:
                label += f'ID {tid}'
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color = (0, 255, 0), thickness=2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.putText(frame, label, (x1 - tw - 10, y1 - th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            total = len(seen_id_total)
            cv2.putText(frame, f"Unique objects: {total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('Video', frame)
            if writer is not None:
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = int(input("Would you like to use camera or video file? (0 - camera, 1 - video): "))
    log = int(input("Would you like to see the logs in terminal? (0 - no, 1 - yes): "))
    if choice == 0:
        main(choice, log)
    elif choice == 1:
        video = str(input("Enter name of the video: "))
        main(choice, log, video)
    else:
        print("Invalid input")