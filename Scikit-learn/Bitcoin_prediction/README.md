## Bitcoin prices prediction

This project predicts future Bitcoin prices using historical data from 2019–2020 (_bitcoin.csv_) using machine learning algorithms. It compares six models and shows which predicts Bitcoin most accurately.

### Features
Converts dates into cyclical features (month & weekday) for better predictions.
Compares multiple models:
  - Linear Regression
  - Lasso Regression
  - SVR
  - K-Nearest Neighbors
  - Random Forest
  - XGBoost.

Evaluates performance using R², MAE, and RMSE.
Time series cross-validation ensures proper sequential testing.
