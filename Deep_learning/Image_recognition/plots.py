from main import X_train, y_train, class_names, X_test, y_test, predictions
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""wizualizacja zbioru danych fashion_mnist"""
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig('images.png')
plt.show()

"""wykresy z predykcjami"""
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'Prediction: {class_names[predictions[i].argmax()]}, Real: {class_names[y_test[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions.png')
plt.show()

"""confusion matrix"""
cm = confusion_matrix(y_test, predictions.argmax(axis=1))
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.xlabel("Predicted class")
plt.ylabel("Real class")
plt.savefig('confusion_matrix.png')
plt.show()
