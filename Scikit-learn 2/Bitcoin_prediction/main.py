import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


bitcoin = pd.read_csv('bitcoin.csv', parse_dates=['date'], dayfirst=True, index_col='date')

bitcoin['value'] = pd.to_numeric(bitcoin['value'].str.replace(",", "."))
bitcoin['year'] = bitcoin.index.year
bitcoin['month'] = bitcoin.index.month
bitcoin['day'] = bitcoin.index.day
bitcoin['weekday'] = bitcoin.index.weekday

'''Reprezantacja danych za pomocą okręgu'''
bitcoin['month_sin'] = np.sin(2 * np.pi * bitcoin['month'] / 12)
bitcoin['month_cos'] = np.cos(2 * np.pi * bitcoin['month'] / 12)
bitcoin['weekday_sin'] = np.sin(2 * np.pi * bitcoin['weekday'] / 7)
bitcoin['weekday_cos'] = np.cos(2 * np.pi * bitcoin['weekday'] / 7)

delays = [1, 3, 7, 14]

for lag in delays:
    bitcoin[f'value_lag_{lag}'] = bitcoin['value'].shift(lag)
    bitcoin[f'value_roll_mean_{lag}'] = bitcoin['value'].rolling(window=lag).mean().shift(lag)

bitcoin.dropna(inplace=True)

X = bitcoin.drop(columns='value')
y = bitcoin['value']

ts_split = TimeSeriesSplit(n_splits=5)
models = {
    'LinearRegression': LinearRegression(),
    'Lasso (alpha: 0.1)': Lasso(alpha=0.1, random_state=42),
    'SVR': SVR(C=3, kernel='rbf', epsilon=0.1),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'XGBoost': XGBRegressor(max_depth=4, subsample=0.8, random_state=42),
    'RandomForest': RandomForestRegressor(max_depth=4, random_state=42)
}

results = {name: {"R²": [], "MAE": [], "RMSE": []} for name in models}
predictions = {name: [] for name in models}
scaler = StandardScaler()
#
# Trenowanie modeli i ewaluacja
for name, model in models.items():

    for i, (train_index, test_index) in enumerate(ts_split.split(X)):
        # Podział na dane treningowe i testowe
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if i == 0:
            X_train_scaled = scaler.fit_transform(X_train)
            continue
        else:
            X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[name]["R²"].append(r2_score(y_test, y_pred))
        results[name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        results[name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        predictions[name].append((bitcoin.index[test_index], y_pred))

final_results = {name: {metric: np.mean(values) for metric, values in metrics.items()} for name, metrics in results.items()}
results_df = pd.DataFrame(final_results).T


