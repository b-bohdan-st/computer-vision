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
    "white": (255, 255, 255),
}

shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for i in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] # B, G, R, alpha
            features = [mean_color[0], mean_color[1], mean_color[2]]  # ознаки тут — середній колір по каналах B, G, R:
            x.append(features) # усі ознаки (features), тобто числові дані, за якими навчається модель
            y.append(f"{color_name}_{shape}") # усі мітки (labels), тобто правильні відповіді, які модель повинна передбачати

#3 розділяємо дані 70% даних — для навчання, 30% — для перевірки

# Бо якщо вона “вивчить” усе напам’ять — ми не знатимемо, чи справді
# вона вміє узагальнювати, чи просто запам’ятала приклади.
# Тому ми ділимо весь набір даних (dataset) на дві частини:
# x_train, y_train — дані для навчання моделі, x_test, y_test — дані для перевірки (тестування)
# x_train - ознаки (features) для навчання, модель бачить їх і вчиться
# y_train - правильні відповіді (labels) для навчання, щоб модель знала, що є правильним
# x_test - ознаки для перевірки, нові дані, яких модель "не бачила"
# y_test - правильні відповіді для перевірки, щоб оцінити, наскільки модель вгадує
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y)
# stratify=y — зберігаємо однакову пропорцію класів у train і test (важливо, щоб оцінка була чесною).
# Без stratify можна випадково отримати дисбаланс: наприклад,
# у тесті з'являться майже лише квадрати, і метрика спотвориться.

# створюємо та навчаємо модель. Вона вчиться порівнювати об'єкти за схожістю кольорів
model = KNeighborsClassifier(n_neighbors=3)# беремо 3 найближчих навчальних приклади
model.fit(x_train, y_train) # запам’ятали тренувальні приклади (і побудували структуру пошуку)

# перевіряємо точність
accuracy = model.score(x_test, y_test) # score для класифікатора — це accuracy частка вірних відповідей на тесті
print("Точність моделі:", round(accuracy * 100, 2), "%")

# тестуємо модель на новому зображенні
# список з кольором та формою для тесту
test_cases = [
    ((0, 255, 0), "circle"),
    ((128, 128, 128), "triangle"),
    ((255, 255, 255), "square"),
]

# цикл для тестування на всіх фігурах з test_cases
for i in test_cases:
    color, shape = i
    test_img = generate_image(color, shape)
    cv2.imshow(f"Test image {shape}", test_img)
    mean_color = cv2.mean(test_img)[:3]
    prediction = model.predict([mean_color])
    print("Передбачення:", prediction[0])

cv2.waitKey(0)
cv2.destroyAllWindows()