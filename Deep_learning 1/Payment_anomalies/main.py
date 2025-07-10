import pandas as pd
from sklearn.preprocessing import StandardScaler
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
