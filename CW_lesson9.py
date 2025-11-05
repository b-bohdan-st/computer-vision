import cv2

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)      # ділимо тільки на 2 частини: id і назва
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread('images/cat.jpg')
image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))

# Готуємо зображення для мережі
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

# відправляємо дані в мережу і запускаємо
net.setInput(blob)
preds = net.forward()   # preds — це вектор ймовірностей для 1000 класів

# знаходимо індекс класу з найбільшою ймовірністю
idx = preds[0].argmax()

# дістаємо назву класу і впевненість (у відсотках)
label = classes[idx] if idx < len(classes) else "unknown"
conf = float(preds[0][idx]) * 100

# пишемо результат у консоль
print("Клас:", label)
print("Ймовірність:", round(conf, 2), "%")

# підписуємо результат на зображенні простим текстом
text = label + ": " + str(int(conf)) + "%"   
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# показуємо вікно з результатом
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
