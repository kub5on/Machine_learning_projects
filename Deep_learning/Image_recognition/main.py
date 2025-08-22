import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

"""normalizacja wartości do przedziału od 0 do 1"""
X_train = X_train / 255.0
X_test = X_test / 255.0
X_val = X_val / 255.0

# (liczba_obrazów, wysokość, szerokość, liczba_kanałów)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

"""wizualizacja zbioru danych fashion_mnist"""
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

"tworzenie modelu sekwencyjnego z CNN"
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='leaky_relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='leaky_relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='leaky_relu'),
    layers.Dense(10, activation='softmax')
])
#
# """kompilacja i trening modelu"""
# model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')
# model.fit(
#     X_train,
#     y_train,
#     epochs=3,
#     validation_data=(X_val, y_val)
# )
# model.save('model.h5')

model = tf.keras.models.load_model('best_model.h5')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

predictions = model.predict(X_test)
