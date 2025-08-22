## Fashion MNIST Classification with CNN and Hyperparameter Tuning

This project classifies images from the Fashion MNIST dataset into 10 categories using **Convolutional Neural Networks (CNN)**. It also includes **hyperparameter tuning** to optimize the CNN architecture and training parameters using _Keras Tuner_. In main.py and keras_tuning.py, the training sections are commented out.

### Features
- **CNN Model:** Sequential convolutional model with Conv2D, MaxPooling, Dense layers, and Dropout.
- **Hyperparameter Tuning:** Optimizes the number of filters, kernel sizes, dense units, dropout rate, and learning rate to improve accuracy.
- **Data Preprocessing:** Normalization of pixel values to [0,1] and reshaping to include channel dimension.
- **Train/Validation/Test Split:** 80% training, 10% validation, 10% testing.
- **Evaluation Metrics:** Accuracy and loss on the test set.
- **Pretrained Models:** `model.h5` for manually tuned CNN, `best_model.h5` for the Keras-tuned model.

### Technologies
`Python 3.9+`, `TensorFlow`, `Keras`, `Keras Tuner`, `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`

### ⚡ Quick Start
1. Add `model.h5`, `best_model.h5` and `fashion_mnist` dataset (automatically downloaded by TensorFlow) to the project folder if needed.
2. Install dependencies: ```pip install -r requirements.txt```.
3. Run: ```python main.py``` to load the pretrained CNN and evaluate it on the test set.
4. Run: ```python keras_tuning.py``` to load pretrained and tuned model and evaluate it on the test set.
5. Run: ```python plots.py``` to visualize dataset `fashion_mnist` and see the training results on the plots (confusion matrix, predictions plots).

