import cv2
import numpy as np

# image = cv2.imread('images/img1.jpg')

#photo
"""
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
image = cv2.flip(image, 0)
image = cv2.GaussianBlur(image, (9, 9), 5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.Canny(image, 150, 150)
image = cv2.dilate(image, None, iterations=2)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 150, 150)

kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)
"""

#video
video = cv2.VideoCapture('video/vid1.mp4')
# video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.imshow('duck', image)
# cv2.imshow('duck-new', image[0:250, 0:400])
#cv2.waitKey(0)
cv2.destroyAllWindows()