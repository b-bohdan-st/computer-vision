import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.python.layers.normalization import normalization
import os
import random

def copy_file(src, dst):
    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

def test_train_photos(folder, folder_dest):
    subsets = ["train", "test"]

    for subset in subsets:
        os.makedirs(os.path.join(folder_dest, subset), exist_ok=True)

    classes = os.listdir(folder)

    for cls in classes:
        src_path = os.path.join(folder, cls)
        if not os.path.isdir(src_path):
            continue

        train_dir = os.path.join(folder_dest, "train", cls)
        test_dir = os.path.join(folder_dest, "test", cls)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = [
            f for f in os.listdir(src_path)
            if os.path.isfile(os.path.join(src_path, f))
        ]

        random.shuffle(files)
        split_idx = int(len(files) * 0.7)

        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            copy_file(os.path.join(src_path, f),
                      os.path.join(train_dir, f))

        for f in test_files:
            copy_file(os.path.join(src_path, f),
                      os.path.join(test_dir, f))

# назви класів
class_name = ['car', 'cat', 'dog']

def train_model():
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
    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),
                            activation='relu', input_shape=(128,128,3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # 2-й, 64 - контури
    model.add(layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # 3-й, 128 - фігури
    model.add(layers.Conv2D(
        filters = 128,
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
    predict_image("images/luna.jpg", model) # Модель визначила: dog

    # збереження моделі
    model.save('models/model_cw.h5')

def predict_image(img, model_name):
    image_test = image.load_img(img, target_size=(128, 128))
    img_array = image.img_to_array(image_test)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    predictions = model_name.predict(img_array)

    predicted_index = np.argmax(predictions[0])

    print(f"Модель визначила: {class_name[predicted_index]}")

# перевірка збереженої моделі
if __name__ == '__main__':
    # train_model()
    predict_image('images/bmw-420d-2.jpg', load_model('models/model_cw.h5')) # Модель визначила: car