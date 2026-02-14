import cv2
import os
import time
import csv
import yt_dlp
from ultralytics import YOLO
from datetime import timedelta
import torch

PROJECT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "yolov8s.pt"
CONF_THRESH = 0.5
TRACKER = "bytetrack.yaml"

DISTANCE_METERS = 1.8
DIST5TO3 = 1.0
DIST4TO3 = 1.0
DIST1TO2 = 1.0

TARGET_WIDTH = 1246
TARGET_HEIGHT = 701

vehicle_classes = {"car","truck","bus"}

LINE1_P1 = (10,445)
LINE1_P2 = (735,580)

LINE2_P1 = (605,275)
LINE2_P2 = (1025,315)

LINE3_P1 = (705,240)
LINE3_P2 = (1065,270)

LINE4_P1 = (930,175)
LINE4_P2 = (1145,190)

LINE5_P1 = (300,350)
LINE5_P2 = (900,420)

def point_side(px, py, A, B):
    return (px - A[0])*(B[1]-A[1]) - (py - A[1])*(B[0]-A[0])

def get_stream_url(youtube_url):
    ydl_opts = {"format": "best[ext=mp4]"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def main(source_type, source_value, speed_mode):
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
    model.to("cuda")
    model.fuse()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(OUT_DIR, f"result_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    csv_path = os.path.join(OUT_DIR, "stats.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time","Total","AvgSpeed"])

    car_states = {}
    car_speeds = []
    total_crossed = 0
    frame_index = 0
    last_csv_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(TARGET_WIDTH,TARGET_HEIGHT))
        frame_index += 1

        with torch.no_grad():
            result = model.track(frame, device="cuda", half=True, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)

        r = result[0]

        if r.boxes is not None and len(r.boxes)>0:
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
                x1,y1,x2,y2 = xyxy[i].astype(int)
                cx = (x1+x2)//2
                cy = y2

                if tid not in car_states:
                    car_states[tid] = {
                        "prev1": point_side(cx,cy,LINE1_P1,LINE1_P2),
                        "prev2": point_side(cx,cy,LINE2_P1,LINE2_P2),
                        "prev3": point_side(cx,cy,LINE3_P1,LINE3_P2),
                        "prev4": point_side(cx,cy,LINE4_P1,LINE4_P2),
                        "prev5": point_side(cx,cy,LINE5_P1,LINE5_P2),
                        "line1_time":None,
                        "line2_time":None,
                        "line3_time":None,
                        "line4_time":None,
                        "line5_time":None,
                        "counted":False,
                        "speed":None
                    }

                state = car_states[tid]

                s1 = point_side(cx,cy,LINE1_P1,LINE1_P2)
                s2 = point_side(cx,cy,LINE2_P1,LINE2_P2)
                s3 = point_side(cx,cy,LINE3_P1,LINE3_P2)
                s4 = point_side(cx,cy,LINE4_P1,LINE4_P2)
                s5 = point_side(cx,cy,LINE5_P1,LINE5_P2)

                current_time = frame_index / fps

                if speed_mode=="crosswalk":
                    if state["line2_time"] is None and state["prev2"]*s2<0: state["line2_time"]=current_time
                    if state["line3_time"] is None and state["prev3"]*s3<0: state["line3_time"]=current_time
                else:
                    if state["line1_time"] is None and state["prev1"]*s1<0: state["line1_time"]=current_time
                    if state["line2_time"] is None and state["prev2"]*s2<0: state["line2_time"]=current_time
                    if state["line3_time"] is None and state["prev3"]*s3<0: state["line3_time"]=current_time
                    if state["line4_time"] is None and state["prev4"]*s4<0: state["line4_time"]=current_time
                    if state["line5_time"] is None and state["prev5"]*s5<0: state["line5_time"]=current_time

                if not state["counted"]:
                    speed_kmh=None

                    if speed_mode=="crosswalk":
                        if state["line2_time"] and state["line3_time"]:
                            t=abs(state["line3_time"]-state["line2_time"])
                            if t>0: speed_kmh=(DISTANCE_METERS/t)*3.6
                    else:
                        if state["line5_time"] and state["line3_time"]:
                            t=abs(state["line3_time"]-state["line5_time"])
                            if t>0: speed_kmh=(DIST5TO3/t)*3.6
                        elif state["line4_time"] and state["line3_time"]:
                            t=abs(state["line3_time"]-state["line4_time"])
                            if t>0: speed_kmh=(DIST4TO3/t)*3.6
                        elif state["line1_time"] and state["line2_time"]:
                            t=abs(state["line2_time"]-state["line1_time"])
                            if t>0: speed_kmh=(DIST1TO2/t)*3.6

                    if speed_kmh:
                        state["speed"]=speed_kmh
                        car_speeds.append(speed_kmh)
                        total_crossed+=1
                        state["counted"]=True

                state["prev1"]=s1
                state["prev2"]=s2
                state["prev3"]=s3
                state["prev4"]=s4
                state["prev5"]=s5

                speed_text=f"; Speed: {state['speed']:.2f} km/h" if state["speed"] else ""

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{class_name} ID {tid}{speed_text}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        avg_speed=sum(car_speeds)/len(car_speeds) if car_speeds else 0

        cv2.putText(frame,f"Total: {total_crossed}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(frame,f"A/S: {avg_speed:.2f} km/h",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        if speed_mode=="crosswalk":
            cv2.line(frame,LINE2_P1,LINE2_P2,(255,0,0),2)
            cv2.line(frame,LINE3_P1,LINE3_P2,(0,0,255),2)
        else:
            cv2.line(frame,LINE1_P1,LINE1_P2,(0,255,255),2)
            cv2.line(frame,LINE2_P1,LINE2_P2,(255,0,0),2)
            cv2.line(frame,LINE3_P1,LINE3_P2,(0,0,255),2)
            cv2.line(frame,LINE4_P1,LINE4_P2,(255,0,255),2)
            cv2.line(frame,LINE5_P1,LINE5_P2,(0,255,0),2)

        cv2.imshow("Video",frame)
        writer.write(frame)

        if int(frame_index/fps)>last_csv_time:
            last_csv_time=int(frame_index/fps)
            csv_writer.writerow([str(timedelta(seconds=last_csv_time)),total_crossed,avg_speed])

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    csv_file.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    print("1 - camera\n2 - video\n3 - live")
    source=input("Select source: ")
    print("Speed mode:\n1 - crosswalk\n2 - before crosswalk")
    sm=input("Select speed mode: ")
    speed_mode="crosswalk" if sm=="1" else "before"

    if source=="1": main("camera",None,speed_mode)
    elif source=="2": main("video",input("Enter video path: "),speed_mode)
    elif source=="3": main("live",input("Enter live stream URL: "),speed_mode)