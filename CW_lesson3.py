import cv2
import numpy as np

img = np.ones((500,400, 3), np.uint8)

"""
img[:] = (255, 0, 0)
img[250:500, 0:400] = (0, 255, 255)

cv2.rectangle(img, (100, 100), (200, 200), (92, 232, 66), 100)
cv2.line(img, (100, 100), (200, 200), (0, 170, 255), 3)
cv2.line(img, (200, 100), (100, 200), (0, 170, 255), 3)

cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (0, 170, 255), 3)
"""

cv2.circle(img, (200, 250), 50, (0, 170, 255), 3)
cv2.putText(img, "Just a circle", (130, 170), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 170, 255))

"""
===rhombus===
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1] // 2, 0), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1], img.shape[0] // 2), (0, 170, 255), 3)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1] // 2, img.shape[0]), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 2, img.shape[0]), (img.shape[1], img.shape[0] // 2), (0, 170, 255), 3)
===rhombus in rhombus===
cv2.line(img, (img.shape[1] // 3, img.shape[0] // 2), (img.shape[1] // 2, img.shape[0] // 3), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 2, img.shape[0] // 3), (img.shape[1] // 3 * 2, img.shape[0] // 2), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 3, img.shape[0] // 2), (img.shape[1] // 2, img.shape[0] // 3 * 2), (0, 170, 255), 3)
cv2.line(img, (img.shape[1] // 2, img.shape[0] // 3 * 2), (img.shape[1] // 3 * 2, img.shape[0] // 2), (0, 170, 255), 3)
"""

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()