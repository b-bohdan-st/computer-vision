import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# створюємо функцію для генерації фігур
def generate_image(color, shape):
    image = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(image, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(image, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(image, [points], 0, color, -1)
    return image

# створюємо список міток
x = []
# створюємо список ознак
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (255, 255, 0),
    "pink": (255, 0, 255),
    "gray": (128, 128, 128),
    "orange": (255, 120, 0),
    "purple": (150, 0, 255),
}

shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for i in range(10):
            img = generate_image(color_name, shape)
            mean_color = cv2.mean(img)[:3] # B, G, R, alpha
