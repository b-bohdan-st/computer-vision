import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.layers.normalization import normalization

import os
import random
def test_train_photos(folder, folder_dest):
    classes = os.listdir(folder)
    for cls in classes:
        src_path = os.path.join(folder, cls)
        if not os.path.isdir(src_path):
            continue
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

        random.shuffle(files)
        split_idx = int(len(files) * 0.7)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        train_dst = os.path.join(folder_dest, "train")
        test_dst = os.path.join(folder_dest, "test")

        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)

        for f in train_files:
            os.replace(os.path.join(src_path, f), os.path.join(train_dst, f))

        for f in test_files:
            os.replace(os.path.join(src_path, f), os.path.join(test_dst, f))



# завантаження даних
train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                                    image_size=(128,128),
                                                                    batch_size=30,
                                                                    label_mode='categorical',)
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                                    image_size=(128,128),
                                                                    batch_size=30,
                                                                    label_mode='categorical',)

# нормалізація зображень
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x,y: (normalization_layer(x), y))

# побудова моделі
model = models.Sequential()

# вхідні дані

# 1-й фільтр, 32 - визначаємо краї та лінії
model.add(layers.Conv2D(filter = 32, kernel_size = (3,3),
                        activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# 2-й, 64 - контури
model.add(layers.Conv2D(
    filter = 64,
    kernel_size = (3,3),
    activation='relu'
))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# 3-й, 128 - фігури
model.add(layers.Conv2D(
    filter = 128,
    kernel_size = (3,3),
    activation='relu'
))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

# внутрішній шар

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(3, activation='softmax'))

# компіляція моделі
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# навчаємо модель
history = model.fit(
    train_ds,
    epochs=15,
    validation_data=test_ds,
)

test_lost, test_accuracy = model.evaluate(test_ds)
print(f"Якість: {test_accuracy}")

# перевірка
class_name = ['cars', 'cats', 'dogs']

image_test = image.load_img('images/', target_size=(128,128))
img_array = image.img_to_array(image_test)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])

print(f"Імовірність по класам: {class_name[0]}")
print(f"Модель визначила: {class_name[predicted_index]}")