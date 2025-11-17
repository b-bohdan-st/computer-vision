import pandas as pd # робота з csv
import numpy as np # математичні операції
import tensorflow as tf # навчання нейромереж
from tensorflow import keras # частинка для tensorflow
from tensorflow.keras import layers # для створення шарів
from sklearn.preprocessing import LabelEncoder # перетворює текстові мітки на числа
import matplotlib.pyplot as plt # діаграми, графіки

# читаємо csv
df = pd.read_csv('data/csv/figures_hw.csv')

# перетворюємо значення ст. label з csv
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# додаємо нову ознаку, співвідношення площі до периметра
df['area_to_perimeter'] = df['area'] / df['perimeter']

# обираємо елементи для навчання
X = df[['area', 'perimeter', 'corners', 'area_to_perimeter']]
y = df['label_enc']

# створюємо модель
model = keras.Sequential([layers.Dense(16, activation='relu', input_shape=(4,)),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(16, activation='softmax')])
# шари (layers): 8 нейронів, активація, кількість параметрів для навчання (у X - 3)
# потім створюємо 2 шар, input_shape вже не потрібно
# та останній 3, змінюємо activation на softmax

# компіляція моделі
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# навчаємо моделі
history = model.fit(X, y, epochs=500, verbose=0)

# візуалізація навчання
plt.plot(history.history['loss'], label='Loss_percent')
plt.plot(history.history['accuracy'], label='Accuracy_percent')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Learning process')
plt.legend()
plt.show()

# тестування
test = np.array([18, 16, 0, 18/16])
pred = model.predict(test)
print(f"Імовірність по кожному класу: {pred}")
print(f"Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}")