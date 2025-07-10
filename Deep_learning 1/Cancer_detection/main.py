import keras
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

'''podział danych na X, y'''
dataset = pd.read_csv('cancer.csv')
X = dataset.drop(columns=['id', 'diagnosis'], axis=1)
y = dataset['diagnosis']

"""Nadanie etykiet danym objaśnianym przy pomocy LabelEncoder"""
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""tworzenie modelu deep learning"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

"""kompilacja modelu"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

"""trening modelu"""
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_split=0.1,
    verbose=1
)

"""predykcje"""
preds = model.predict(X_test).reshape(-1)
preds_binary = (preds > 0.5).astype(int)

"""ocena modelu"""
cr = classification_report(y_test, preds_binary, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test, preds_binary)
