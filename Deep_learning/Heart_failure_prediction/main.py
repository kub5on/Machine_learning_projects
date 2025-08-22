import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('heart.csv')
df = pd.get_dummies(df, columns=['Sex', 'ST_Slope', 'ExerciseAngina', 'RestingECG', 'ChestPainType'])

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""skalowanie"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""budowa modelu sekwencyjnego"""
model = keras.Sequential([
    keras.layers.Dense(6, activation='relu', input_shape=(X_train.shape[1], )),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy'
)

# """trening modelu"""
# model.fit(
#     X_train,
#     y_train,
#     validation_split=0.33,
#     batch_size=10,
#     epochs=100
# )
#
# model.save('model.h5')

model = keras.models.load_model('model.h5')

y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
