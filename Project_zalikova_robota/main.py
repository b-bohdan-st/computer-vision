import cv2
import os
import time
import yt_dlp
import numpy as np
from ultralytics import YOLO
import torch

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "yolov8m.pt"
CONF_THRESH = 0.3
TRACKER = "bytetrack.yaml"

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

vehicle_classes = {"car", "truck", "bus"}
POINT_RADIUS = 10
selected_zone = None
selected_point = None
BUTTON_ADD_ZONE = [10, 10, 160, 40]

zones = [
    [[400, 300], [900, 320], [1000, 500], [350, 480]]
]

if not torch.cuda.is_available():
    print("WARNING: GPU not available, using CPU!")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def is_inside_zone(x, y, polygon):
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def mouse_callback(event, x, y, flags, param):
    global selected_point, selected_zone, zones
    scale_x = PROCESS_WIDTH / DISPLAY_WIDTH
    scale_y = PROCESS_HEIGHT / DISPLAY_HEIGHT
    real_x = int(x * scale_x)
    real_y = int(y * scale_y)
    bx1, by1, bx2, by2 = BUTTON_ADD_ZONE
    if event == cv2.EVENT_LBUTTONDOWN:
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            new_zone = [[100, 100], [200, 100], [200, 200], [100, 200]]
            zones.append(new_zone)
            print(f"Added new zone #{len(zones)}")
            return
        for z_idx, zone in enumerate(zones):
            for p_idx, point in enumerate(zone):
                px, py = point
                if abs(px - real_x) < POINT_RADIUS and abs(py - real_y) < POINT_RADIUS:
                    selected_zone = z_idx
                    selected_point = p_idx
                    return
    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_zone is not None and selected_point is not None:
            zones[selected_zone][selected_point] = [real_x, real_y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_zone = None
        selected_point = None

def get_stream_url(youtube_url):
    ydl_opts = {"format": "best[ext=mp4]"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def main(source_type, source_value):
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
                    car_states[tid] = {"saved_zone": [False]*len(zones)}
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
            zone_pts = np.array(zone, np.int32).reshape((-1,1,2))
            overlay = process_frame.copy()
            cv2.fillPoly(overlay, [zone_pts], (0,0,255))
            cv2.addWeighted(overlay,0.2,process_frame,0.8,0,process_frame)
            cv2.polylines(process_frame,[zone_pts],True,(0,0,255),2)
            for point in zone:
                cv2.circle(process_frame, tuple(point), POINT_RADIUS, (0,255,255), -1)
        cv2.rectangle(process_frame, (BUTTON_ADD_ZONE[0], BUTTON_ADD_ZONE[1]),
                      (BUTTON_ADD_ZONE[2], BUTTON_ADD_ZONE[3]), (50,50,50), -1)
        cv2.putText(process_frame, "Add new zone", (BUTTON_ADD_ZONE[0]+10, BUTTON_ADD_ZONE[1]+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        display_frame = cv2.resize(process_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Video", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1 - camera\n2 - video\n3 - live")
    source = input("Select source: ")
    if source == "1":
        main("camera", None)
    elif source == "2":
        main("video", input("Enter video path: "))
    elif source == "3":
        main("live", input("Enter live stream URL: "))

# https://www.youtube.com/live/rnXIjl_Rzy4
# https://www.youtube.com/live/H0Z6faxNLCI