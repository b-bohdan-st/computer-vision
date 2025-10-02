import cv2
import numpy as np

def dark_to_white_bg(img):
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    alpha_norm = alpha[:, :, None] / 255.0
    img = (bgr * alpha_norm + white_bg * (1 - alpha_norm)).astype(np.uint8)
    return img

photo = cv2.imread('images/photo.png', cv2.IMREAD_UNCHANGED)
photo = cv2.resize(photo, (120, 140))
photo = dark_to_white_bg(photo)

card = np.ones((400,600, 3), np.uint8)
card[0:400, 0:600] = (180, 180, 180)

ramka = cv2.rectangle(card, (10, 10), (590, 390), (103, 135, 143), 3)

card[25:165, 25:145] = photo

cv2.putText(card, "Bondar Bohdan", (180, 100), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
cv2.putText(card, "Computer Vision Student", (180, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (52, 144, 209), 2)

cv2.putText(card, "Email: bohdanschoolemail@gmail.com", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 79, 270))
cv2.putText(card, "Phone: +38(068)954-79-35", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 79, 270))
cv2.putText(card, "18/08/2009", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 79, 270))

qr_code = cv2.imread('images/qrcode.png')
qr_code = cv2.resize(qr_code, (90, 90))
card[270:360, 470:560] = qr_code

cv2.putText(card, "OpenCV Business Card", (card.shape[1] // 4, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# cv2.imwrite("images/business_card.png", card)
cv2.imshow("business_card", card)
cv2.waitKey(0)
cv2.destroyAllWindows()