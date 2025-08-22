## Detecting payment anomalies with autoencoder

This project detects anomalies in credit card transactions using an **autoencoder-based deep learning model**. The model learns normal transaction patterns and flags anomalies as potential frauds.

### Features
- **Autoencoder Model:** Encoder-decoder architecture with Dense layers, Batch Normalization, and Dropout.
- **Anomaly Detection:** Identifies unusual transactions based on reconstruction error.
- **Data Preprocessing:** Standard scaling of features and splitting into training/test sets.
- **Evaluation Metrics:** ROC curve, AUC score, classification report, and confusion matrix.

### Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`

### ⚡ Quick Start
1. Add `creditcard.csv` to the project folder.
2. Install dependencies: ```pip install -r requirements.txt```
3. Run: ```python main.py```
4. Run: ```python plots.py```



