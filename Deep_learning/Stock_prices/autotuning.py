import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from keras_tuner.tuners import RandomSearch


keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


"""tworzenie modeli"""


def build_lstm_model(hp):
    model = keras.Sequential()
    model.add(LSTM(hp.Int('lstml', 32, 128, step=32), return_sequences=True, input_shape=(seq_length, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(hp.Int('lstml', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model


def build_gru_model(hp):
    model = keras.Sequential()
    model.add(GRU(hp.Int('gru1', 32, 128, step=32), return_sequences=True, input_shape=(seq_length, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(hp.Int('gru2', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

df = pd.read_csv('sp500.txt')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

data = df["Close"].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled = scaler.transform(test_data.reshape(-1, 1))

seq_length = 60
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

"""podział na dane walidacyjne"""
val_split = int(len(X_train) * 0.1)
X_val, y_val = X_train[-val_split:], y_train[-val_split:]
X_train2, y_train2 = X_train[:-val_split], y_train[:-val_split]

tuner_lstm = RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=3,
    executions_per_trial=1,
    directory='lstm',
    project_name='lstm',
    seed=42
)


tuner_gru = RandomSearch(
    build_gru_model,
    objective='val_loss',
    max_trials=3,
    executions_per_trial=1,
    directory='gru',
    project_name='gru',
    seed=42
)

best_lstm = tf.keras.models.load_model("best_lstm.h5")
best_gru = tf.keras.models.load_model("best_gru.h5")

# tuner_lstm.search(X_train2, y_train2, validation_data=(X_val, y_val), epochs=10, batch_size=32)
# best_lstm = tuner_lstm.get_best_models(1)[0]
# best_lstm.save('best_lstm.h5')

# tuner_gru.search(X_train2, y_train2, validation_data=(X_val, y_val), epochs=10, batch_size=32)
# best_gru = tuner_gru.get_best_models(1)[0]
# best_gru.save('best_gru.h5')

"""predykcje"""
best_pred_lstm = scaler.inverse_transform(best_lstm.predict(X_test))
best_pred_gru = scaler.inverse_transform(best_gru.predict(X_test))
y_test_inv = scaler.inverse_transform(y_test)

rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, best_pred_lstm))
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, best_pred_gru))
print(f'RMSE_lstm: {rmse_lstm}, RMSE_gru: {rmse_gru}')
