import cv2
import numpy as np

img = cv2.imread('images/soldatu.jpg')

scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)
img_copy = img.copy() # копія img
img_copy_color = img.copy()


img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) # сірий колір
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2) # блюр для оптимізації
img_copy = cv2.equalizeHist(img_copy) # посилення контрасту
img_copy = cv2.Canny(img_copy, 50, 120) # контури

#==========ПОШУК КОНТУРІВ==========
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#RETR_EXTERNAL режим отримання контурів, знаходить крайній зовнішній контур
#CHAIN_APPROX_SIMPLE - процес наближеного вираження одних величин або об'єктівчерез інші

#==========МАЛЮВАННЯ КОНТУРІВ, ПРЯМОКУТНИКІВ ТА ТЕКСТУ==========
for cnt in contours:
    area = cv2.contourArea(cnt) # визначаємо площу контуру
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt) # boundingRect створює найменший прямокутник, який повністю в собі містить контур
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        #[cnt] список контурів, які малюємо
        #-1 - всі значення з масиву
        #(0, 255, 0) - колір контуру
        #2 - товщина контуру
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f"x: {x}, y: {y}, S: {int(area)}"
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)



cv2.imshow("Contours and coordinates", img)
cv2.imshow("Contours and 1", img_copy)
cv2.imshow("Copy", img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()