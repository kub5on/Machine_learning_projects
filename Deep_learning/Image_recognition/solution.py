import keras
import keras_tuner
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras_tuner.tuners import RandomSearch

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

def build_model(hp):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            filters=hp.Choice("conv_1_filter", values=[32, 64, 128]),
            kernel_size=hp.Choice("conv_1_kernel", values=[3, 5]),
            activation='leaky_relu',
            input_shape=(28, 28, 1))
    )
    model.add(
        layers.MaxPooling2D(
            (2, 2))
    )
    model.add(
        layers.Conv2D(
            filters=hp.Choice("conv_2_filter", values=[32, 64]),
            kernel_size=hp.Choice("conv_2_kernel", values=[3, 5]),
            activation='leaky_relu')
    )
    model.add(
        layers.MaxPooling2D(
            (2, 2))
    )
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation='leaky_relu')
    )
    model.add(
        layers.Dropout(hp.Float("drop", min_value=0.2, max_value=0.5, step=0.1))
    )
    model.add(
        layers.Dense(
            units=10,
            activation='softmax')
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_dir',
    project_name='fashion_mnist_cnn'
)

# tuner.search(X_train, y_train, epochs=3, validation_data=(X_val, y_val))
# best_model = tuner.get_best_models(1)[0]
# best_model.save('best_model.h5')

best_model = tf.keras.models.load_model('best_model.h5')

test_loss, test_acc = best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)
# print(test_loss)
# print(test_acc)
