## Breast cancer detection project 
This project predicts the type of breast cancer, classifying tumors as **Malignant (M)** or **Benign (B)**, based on the `cancer.csv` dataset.  
It uses a **sequential deep learning model** with 3 hidden layers (256 → 128 → 64 neurons) with ReLU activations and Dropout, and a single sigmoid output neuron for binary classification.

### Features
- **Deep Learning Model:** Sequential neural network with 3 hidden layers and Dropout to prevent overfitting.
- **Binary Classification:** Predicts malignant (1) or benign (0) tumors.
- **Data Preprocessing:** Label encoding, feature scaling, and train-test split.
- **Evaluation Metrics:** Accuracy, classification report (precision, recall, F1-score), and confusion matrix.

### Dependencies
`Python 3.9+`, `tensorflow`, `keras`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### How to run
1. Add _cancer.csv_ to the project folder.
2. Install required packages from requirements.txt:
```pip install -r requirements.txt```
3. Run : ```python main.py```
4. Run: ```python plots.py```
5. Check the output metrics and all plots for model performance comparison.

