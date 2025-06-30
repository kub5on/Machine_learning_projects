from main import X_train, X_test, y_train, y_test, y_pred, scaled_X, encoded_iris
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


metrics = {"Model": [], "Accuracy": [], "Recall": [], "Precision": [], "F1": []}

metrics['Model'] = "KHNeighborsClassifier"
metrics['Accuracy'] = accuracy_score(y_test, y_pred)
metrics['Recall'] = recall_score(y_test, y_pred, average='macro')
metrics['Precision'] = precision_score(y_test, y_pred, average='macro')
metrics['F1'] = f1_score(y_test, y_pred, average='macro')

print(metrics)
