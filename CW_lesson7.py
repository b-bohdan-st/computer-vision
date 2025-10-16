import cv2
import numpy as np

def rgb(r,g,b):
    bgr = (b,g,r)
    return bgr

cap = cv2.VideoCapture(0)

lower_red_1 = np.array([0,100,100])
upper_red_1 = np.array([10,255,255])

lower_red_2 = np.array([160,100,100])
upper_red_2 = np.array([180,255,255])

points = []

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    is_detected = False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, rgb(255, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, rgb(255, 0, 0), 2)

                points.append((cx, cy))
            for i in range(1, len(points)):
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), rgb(0, 255, 0), 2)
                cv2.putText(frame, "Object detected", (frame.shape[1] - 150, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            # cv2.line(frame, points[i - 1], points[i], rgb(255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), rgb(255, 0, 0), 2)
            cv2.putText(frame, "Object not detected", (frame.shape[1] - 180, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)




    if not ret: break
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.imshow('Video', frame)
    cv2.imshow('Mask', mask)

cap.release()
cv2.destroyAllWindows()