## Predicting stock prices with hyperparameter tuning

This project predicts future S&P 500 stock prices using **LSTM and GRU models** with **hyperparameter tuning** using _Keras Tuner_. It is used to find the optimal number of units and dropout rates for each model to improve forecasting accuracy.

## 🚀 Highlights
- **LSTM & GRU Models:** Sequential models with multiple layers, Dropout, and Batch Normalization.
- **Hyperparameter Tuning:** Uses Keras Tuner (RandomSearch) to optimize model parameters.
- **Sequence Data:** Sliding windows of 60 days capture temporal dependencies in stock prices.
- **Data Preprocessing:** Normalization with MinMaxScaler and train-test split (80/20).
- **Evaluation Metrics:** Root Mean Squared Error (RMSE) to compare model performance.

## Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Pandas`, `NumPy`, `Scikit-learn`, `Keras Tuner`

## ⚡ Quick Start
1. Add `sp500.txt` (historical S&P 500 data) to the project folder.
2. Install dependencies: ```pip install -r requirements.txt```
3. 

