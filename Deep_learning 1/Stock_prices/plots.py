import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('sp500.txt')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

data = df["Close"].values

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_scaled = scaler.transform(test_data.reshape(-1, 1))
