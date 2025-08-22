## Predicting stock prices with hyperparameter tuning

"This project predicts future S&P 500 stock prices using LSTM and GRU models, with hyperparameter tuning using Keras Tuner to find the optimal number of units and dropout rates for each model. Pretrained models (`model_lstm.h5` and `model_gru.h5`) and tuned models (`best_lstm.h5` and `best_gru.h5`) are provided. In `main.py` and `autotuning.py`, the training sections are commented out.

## Features
- **LSTM & GRU Models:** Sequential models with multiple layers, Dropout, and Batch Normalization.
- **Hyperparameter Tuning:** Uses Keras Tuner (_RandomSearch_) to optimize model parameters.
- **Sequence Data:** Sliding windows of 60 days capture temporal dependencies in stock prices.
- **Data Preprocessing:** Normalization with MinMaxScaler and train-test split (80/20).
- **Evaluation Metrics:** Root Mean Squared Error _(RMSE_) to compare model performance.

## Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Pandas`, `NumPy`, `Scikit-learn`, `Keras Tuner`

## ⚡ Quick Start
1. Add `sp500.txt`, `model_gru.h5`, `model_lstm.h5`, `best_gru.h5`, `best_lstm.h5` to the project folder.
2. Install dependencies: ```pip install -r requirements.txt```.
3. Run: ```python main.py``` to load manually tuned models.
4. Run: ```python autotuning.py``` to load keras tuned models.
6. Run: ```python plots.py``` to compare the predictions of the models.





