## Bitcoin prices prediction

This project predicts future Bitcoin prices using historical data from 2019–2020 (_bitcoin.csv_) using machine learning algorithms. It compares six models and shows which predicts Bitcoin most accurately.

### Features
Program converts dates into cyclical features (month & weekday) for better predictions.
Models used in comparison:
  - Linear Regression
  - Lasso Regression
  - SVR
  - K-Nearest Neighbors
  - Random Forest
  - XGBoost.

Evaluates performance using R², MAE, and RMSE.
Time series cross-validation ensures proper sequential testing.

### Dependencies
Python 3.8+, Pandas, Numpy, Matplotlib, Scikit-learn, Xgboost

### How to run
1. Add _bitcoin.csv_ to the project folder.
2. Install required packages from _requirements.txt_:
   ```pip install -r requirements.txt```
3. Run : ```python main.py```
4. Run: ```python plots.py```
5. Check the output metrics and prediction plots for model performance comparison.
