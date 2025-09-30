import cv2

photo = cv2.imread("images/photo.png")

cv2.rectangle(photo, (photo.shape[1] // 4, 1), (photo.shape[1] // 4 * 3, photo.shape[0] // 5 * 3), (92, 232, 66), 1)

cv2.putText(photo, "Bondar Bohdan", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)

cv2.imshow("Photo", photo)
cv2.waitKey(0)
cv2.destroyAllWindows()