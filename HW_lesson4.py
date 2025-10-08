import cv2
import numpy as np

img = cv2.imread('images/selfie.jpg')

scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)
img_copy = img.copy()
img_copy_color = img.copy()


img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)
img_copy = cv2.equalizeHist(img_copy)
img_copy = cv2.Canny(img_copy, 50, 120)
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 125, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f"x: {x}, y: {y}, S: {int(area)}"
        cv2.putText(img_copy_color, text, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 94, 255), 2)


cv2.imshow("Detected people", img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
