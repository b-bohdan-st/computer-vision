import os
import  cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "output")
VIDEO_DIR = os.path.join(PROJECT_DIR, "videos")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

CAR_CLASS_ID = 2
BUS_CLASS_ID = 5
TRUCK_CLASS_ID = 7

def is_new_object(cx, cy, objects, threshold):
    for ox, oy in objects:
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if dist < threshold:
            return False
    return True

def main(video_name):
    video_path = os.path.join(VIDEO_DIR, video_name)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Video file not found")
        return
    out_fps = video.get(cv2.CAP_PROP_FPS) or 25

    model = YOLO("yolov8n.pt")
    CONF_THRESHOLD = 0.4

    RESIZE_WIDTH = 960 # 640
    RESIZE_HEIGHT = 720 # 480

    prev_time = time.time()
    fps = 0.0

    writer = None

    car_count = 0
    bus_count = 0
    truck_count = 0

    counted_objects = []
    DIST_THRESHOLD = 50

    while True:
        ret, frame = video.read()
        if not ret: break
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        if RESIZE_WIDTH is not None and RESIZE_HEIGHT is not None:
            h, w = frame.shape[:2]
            scale = min(RESIZE_WIDTH / w, RESIZE_HEIGHT / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = os.path.join(OUT_DIR, f"result_{int(time.time())}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if cls == CAR_CLASS_ID:
                    if is_new_object(cx, cy, counted_objects, DIST_THRESHOLD):
                        car_count += 1
                        counted_objects.append((cx, cy))
                    label = f"Car {conf:.2f}"
                    color = (0, 255, 0)
                elif cls == BUS_CLASS_ID:
                    if is_new_object(cx, cy, counted_objects, DIST_THRESHOLD):
                        bus_count += 1
                        counted_objects.append((cx, cy))
                    label = f"Bus {conf:.2f}"
                    color = (255, 255, 0)
                elif cls == TRUCK_CLASS_ID:
                    if is_new_object(cx, cy, counted_objects, DIST_THRESHOLD):
                        truck_count += 1
                        counted_objects.append((cx, cy))
                    label = f"Truck {conf:.2f}"
                    color = (255, 0, 0)
                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt

        total_count = car_count + bus_count + truck_count

        cv2.putText(frame, f"Car count: {car_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Bus count: {bus_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Truck count: {truck_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Total count: {total_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        writer.write(frame)
        cv2.imshow("YOLO", frame)

    video.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_name = str(input("Enter video name: "))
    main(video_name)