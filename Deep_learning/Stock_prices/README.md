## Predicting stock prices
This project predicts future S&P 500 stock prices using **sequence-based deep learning models**: LSTM and GRU. The models learn patterns from historical closing prices to forecast future trends.

## Features
- **LSTM & GRU Models:** Sequential models with multiple layers, Dropout, and Batch Normalization.
- **Sequence Data:** Uses sliding windows of 60 days to capture temporal dependencies.
- **Data Preprocessing:** Normalization with MinMaxScaler and train-test split (80/20).
- **Evaluation Metrics:** Root Mean Squared Error (RMSE) for model performance comparison.

## Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Pandas`, `NumPy`, `Scikit-learn`

## ⚡ Quick Start
1. Add `sp500.txt` (historical S&P 500 data) to the project folder.
2. Install dependencies: ```pip install -r requirements.txt```
3. 
