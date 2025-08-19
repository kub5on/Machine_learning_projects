import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""pobranie danych"""
houses = pd.read_csv('houses.csv')
houses_copy = houses.copy()

"""uzupełnienie brakujących danych przy pomocy SimpleImputer"""
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

# print(houses.isnull().sum())
# print(houses_copy.isnull().sum())


"""usuniecie wartosci odstajacych (1.5IQR ponizej pierwszego kwartyla i 1.5IQR powyzej trzeciego kwartyla)"""
Q1 = houses["Cena"].quantile(0.25)
Q3 = houses["Cena"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

houses[(houses["Cena"] <= lower_bound) | (houses["Cena"] >= upper_bound)].shape[0]

houses = houses[(houses["Cena"] >= lower_bound) & (houses["Cena"] <= upper_bound)]


"""enkodowanie zmiennych kategorycznych metodą OneHotEncoder"""
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index
encoded.columns = encoder.get_feature_names_out(cat_features)

houses = houses.drop(columns=cat_features).join(encoded)
# print(houses.iloc[0])


"""podzał danych na dane treningowe (X_train, y_train) i dane testowe (X_test, y_test) w proporcjach 80/20"""
X = houses.drop(columns=["Cena"])
y = houses["Cena"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""standaryzacja danych wykorzystując StandardScaler"""
scaler = StandardScaler()
num_features.remove("Cena")
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# Lista modeli do przetestowania
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    metrics["Model"].append(name)
    metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
    metrics["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics["R² Score"].append(r2_score(y_test, y_pred))

    if hasattr(model, "coef_"):
        coefficients[name] = model.coef_

results_df = pd.DataFrame(metrics)
print(results_df)
input("Naciśnij Enter, aby zakończyć...")

importances = models["XGBoost"].feature_importances_

"""Wykres ważności cech"""
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), [X_train.columns[i] for i in range(len(importances))], rotation=42, fontsize=6)
plt.title("Wykres ważności cech z modelu XGBoost")
plt.xlabel('Cechy')
plt.ylabel('Importances')
plt.savefig('Wykres_cech_xgboost')
plt.show()


def plot_model_vs_prediction(ax, model_name, y_true, y_pred):
    ax.scatter(y_true, y_pred, color='red', alpha=0.7)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='yellow', linestyle='--')
    ax.set_title(f"{model_name} - rzeczywiste vs przewidywane")
    ax.set_xlabel("wartości rzeczywiste")
    ax.set_ylabel("wartości przewidziane")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_model_vs_prediction(axes[0, 0], "Linear Regression", y_test, predictions["Linear Regression"])
plot_model_vs_prediction(axes[0, 1], "Ridge Regression", y_test, predictions["Ridge Regression"])
plot_model_vs_prediction(axes[1, 0], "Lasso Regression", y_test, predictions["Lasso Regression"])
plot_model_vs_prediction(axes[1, 1], "XGBoost", y_test, predictions["XGBoost"])
plt.savefig('predicted_vs_real')
plt.show()
