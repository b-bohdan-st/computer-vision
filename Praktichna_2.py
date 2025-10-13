import cv2
import numpy as np

shape = 3

img = cv2.imread("images/photo.jpg")
img = cv2.resize(img, (img.shape[1] // shape, img.shape[0] // shape))
img_copy = img.copy()

img = cv2.GaussianBlur(img, (5, 5), 2)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red, upper_red = np.array([170, 0, 0]), np.array([179, 255, 255])
lower_green, upper_green = np.array([54, 0, 61]), np.array([84, 255, 163])
lower_blue, upper_blue = np.array([104, 54, 94]), np.array([119, 255, 255])
lower_yellow, upper_yellow = np.array([0, 71, 95]), np.array([38, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        perimetr = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True)
        # h_hsv = hsv[:, :, 0]
        # if 0 <= h_hsv <= 10 or 160 <= h_hsv <= 179:
        #     color = "red"
        # elif 36 <= h_hsv <= 85:
        #     color = "green"
        # elif 101 <= h_hsv <= 130:
        #     color = "blue"
        # elif 26 <= h_hsv <= 35:
        #     color = "yellow"
        # else:
        #     color = "Other color"
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        if len(approx) == 4:
            shape = "rectangle"
        elif len(approx) == 3:
            shape = "trictangle"
        elif len(approx) == 2:
            shape = "straight"
        elif len(approx) == 1:
            shape = "point"
        elif len(approx) == 10:
            shape = "star"
        elif len(approx) == 8:
            shape = "oval"
        else:
            shape = "Car (rectangle)"

        cv2.circle(img_copy, (cx, cy), 3, (0, 0, 255), 5)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(img_copy, f"S: {int(area)}", (x + 5, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Shape: {shape}", (x + 5, y + 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv2.putText(img_copy, f"X: {x}, Y: {y}", (x + 5, y + 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in blue_contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        color = "blue"
        cv2.putText(img_copy, f"Color: {color}", (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
for cnt in red_contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        color = "red"
        cv2.putText(img_copy, f"Color: {color}", (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
for cnt in green_contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        color = "green"
        cv2.putText(img_copy, f"Color: {color}", (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
for cnt in yellow_contours:
    area = cv2.contourArea(cnt)
    if area > 400:
        x, y, w, h = cv2.boundingRect(cnt)
        color = "yellow"
        cv2.putText(img_copy, f"Color: {color}", (x + 5, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


cv2.imshow("Original", img_copy)
cv2.imwrite("images/result.jpg", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()