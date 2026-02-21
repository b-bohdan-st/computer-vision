import cv2
import os
import time
import yt_dlp
import numpy as np
from ultralytics import YOLO
import torch
import ast

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "output")
SAVES_DIR = os.path.join(PROJECT_DIR, "saves")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SAVES_DIR, exist_ok=True)

MODEL_PATH = "yolov8m.pt"
CONF_THRESH = 0.3
TRACKER = "bytetrack.yaml"

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

vehicle_classes = {"car", "truck", "bus"}
POINT_RADIUS = 10

BUTTON_ADD_ZONE = [10, 10, 160, 40]
BUTTON_SAVE_ZONES = [180, 10, 330, 40]

selected_zone = None
selected_point = None

def get_next_save_number():
    files = [f for f in os.listdir(SAVES_DIR) if f.startswith("zones_") and f.endswith(".txt")]
    numbers = []
    for f in files:
        try:
            numbers.append(int(f.replace("zones_", "").replace(".txt", "")))
        except:
            pass
    return max(numbers) + 1 if numbers else 1

def save_zones(zones):
    number = get_next_save_number()
    filename = f"zones_{number}.txt"
    path = os.path.join(SAVES_DIR, filename)
    with open(path, "w") as f:
        f.write(str(zones))
    print(f"Zones saved to {path}")

def load_zones_from_file(filename):
    path = os.path.join(SAVES_DIR, filename)
    with open(path, "r") as f:
        return ast.literal_eval(f.read())

def is_inside_zone(x, y, polygon):
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def get_stream_url(youtube_url):
    ydl_opts = {"format": "best[ext=mp4]"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def main(source_type, source_value, zones_file, auto_save):
    global selected_point, selected_zone

    if zones_file is not None:
        zones = load_zones_from_file(zones_file)
    else:
        zones = [[[400, 300], [900, 320], [1000, 500], [350, 480]]]

    if source_type == "camera":
        cap = cv2.VideoCapture(0)
    elif source_type == "video":
        cap = cv2.VideoCapture(source_value)
    elif source_type == "live":
        stream_url = get_stream_url(source_value)
        cap = cv2.VideoCapture(stream_url)
    else:
        return

    if not cap.isOpened():
        return

    model = YOLO(MODEL_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.fuse()

    car_states = {}

    def mouse_callback(event, x, y, flags, param):
        nonlocal zones
        scale_x = PROCESS_WIDTH / DISPLAY_WIDTH
        scale_y = PROCESS_HEIGHT / DISPLAY_HEIGHT
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)

        bx1, by1, bx2, by2 = BUTTON_ADD_ZONE
        sx1, sy1, sx2, sy2 = BUTTON_SAVE_ZONES

        if event == cv2.EVENT_LBUTTONDOWN:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                zones.append([[100, 100], [200, 100], [200, 200], [100, 200]])
                return
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                save_zones(zones)
                return
            for z_idx, zone in enumerate(zones):
                for p_idx, point in enumerate(zone):
                    px, py = point
                    if abs(px - real_x) < POINT_RADIUS and abs(py - real_y) < POINT_RADIUS:
                        selected_zone = z_idx
                        selected_point = p_idx
                        globals()["selected_zone"] = z_idx
                        globals()["selected_point"] = p_idx
                        return
        elif event == cv2.EVENT_MOUSEMOVE:
            if globals()["selected_zone"] is not None and globals()["selected_point"] is not None:
                zones[globals()["selected_zone"]][globals()["selected_point"]] = [real_x, real_y]
        elif event == cv2.EVENT_LBUTTONUP:
            globals()["selected_zone"] = None
            globals()["selected_point"] = None

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        process_frame = frame.copy()

        with torch.no_grad():
            result = model.track(
                process_frame,
                imgsz=1280,
                device="cuda" if torch.cuda.is_available() else "cpu",
                half=True,
                conf=CONF_THRESH,
                tracker=TRACKER,
                persist=True,
                verbose=False
            )

        r = result[0]

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

            for i in range(len(xyxy)):
                class_id = int(cls[i])
                class_name = model.names[class_id]
                if class_name not in vehicle_classes or track_id is None:
                    continue

                tid = int(track_id[i])
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)

                if tid not in car_states:
                    car_states[tid] = {"saved_zone": [False] * len(zones)}

                state = car_states[tid]

                while len(state["saved_zone"]) < len(zones):
                    state["saved_zone"].append(False)

                for z_idx, zone in enumerate(zones):
                    if not state["saved_zone"][z_idx] and is_inside_zone(cx, cy, zone):
                        vehicle_crop = frame[y1:y2, x1:x2].copy()
                        if vehicle_crop.size > 0:
                            filename = f"car_{tid}_zone{z_idx}_{int(time.time())}.jpg"
                            save_path = os.path.join(OUT_DIR, filename)
                            cv2.imwrite(save_path, vehicle_crop)
                        state["saved_zone"][z_idx] = True

                cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(process_frame, f"{class_name} ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for zone in zones:
            zone_pts = np.array(zone, np.int32).reshape((-1, 1, 2))
            overlay = process_frame.copy()
            cv2.fillPoly(overlay, [zone_pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.2, process_frame, 0.8, 0, process_frame)
            cv2.polylines(process_frame, [zone_pts], True, (0, 0, 255), 2)
            for point in zone:
                cv2.circle(process_frame, tuple(point), POINT_RADIUS, (0, 255, 255), -1)

        cv2.rectangle(process_frame, (BUTTON_ADD_ZONE[0], BUTTON_ADD_ZONE[1]),
                      (BUTTON_ADD_ZONE[2], BUTTON_ADD_ZONE[3]), (50, 50, 50), -1)
        cv2.putText(process_frame, "Add new zone", (BUTTON_ADD_ZONE[0] + 10, BUTTON_ADD_ZONE[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(process_frame, (BUTTON_SAVE_ZONES[0], BUTTON_SAVE_ZONES[1]),
                      (BUTTON_SAVE_ZONES[2], BUTTON_SAVE_ZONES[3]), (80, 80, 80), -1)
        cv2.putText(process_frame, "Save zones", (BUTTON_SAVE_ZONES[0] + 15, BUTTON_SAVE_ZONES[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        display_frame = cv2.resize(process_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Video", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if auto_save:
        save_zones(zones)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1 - camera\n2 - video\n3 - live")
    source = input("Select source: ")

    source_type = None
    source_value = None

    if source == "1":
        source_type = "camera"
    elif source == "2":
        source_type = "video"
        source_value = input("Enter video path: ")
    elif source == "3":
        source_type = "live"
        source_value = input("Enter live stream URL: ")

    print("1 - Don't load zones\n2 - Load zones from file")
    load_choice = input("Select option: ")

    zones_file = None

    if load_choice == "2":
        files = os.listdir(SAVES_DIR)
        if not files:
            print("No files in saves directory")
            print("Starting in mode without saved zones")
        else:
            for f in files:
                print(f)
            zones_file = input("Enter file name: ")

    print("1 - Don't save zones after exit\n2 - Save zones after exit")
    save_choice = input("Select option: ")
    auto_save = save_choice == "2"

    main(source_type, source_value, zones_file, auto_save)

# https://www.youtube.com/live/rnXIjl_Rzy4
# https://www.youtube.com/live/H0Z6faxNLCI

