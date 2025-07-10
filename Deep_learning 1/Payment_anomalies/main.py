import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

"""wstępne przygotowanie danych"""
df = pd.read_csv('creditcard.csv')
X = df.drop(columns=['Time', 'Class'])
y = df['Class']

"""skalowanie zmiennych objaśniających"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""podział na dane testowe i treningowe"""
X_train = X_scaled[y == 0]
X_test = X_scaled
