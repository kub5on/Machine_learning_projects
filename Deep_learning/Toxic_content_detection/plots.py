import matplotlib.pyplot as plt
import json
from sklearn.metrics import multilabel_confusion_matrix
from main import y_pred, y_test

with open("training_history.json", mode='r', encoding='UTF-8') as file:
    history = json.load(file)

"""analiza przebiegu uczenia"""
epochs = range(1, len(history['accuracy']) + 1)
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']

plt.figure(figsize=(10, 8))
plt.plot(epochs, accuracy, 'o-', label='train_accuracy', color='red')
plt.plot(epochs, val_accuracy, 'o-', label='validation_accuracy', color='green')
plt.title('Analiza przebiegu uczenia')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.show()

'''confusion matrix'''
cm = multilabel_confusion_matrix(y_test, y_pred)

for i, matrix in enumerate(cm):
    with open('confusion_matrix.txt', mode='a', encoding='UTF-8') as f:
        f.write(f"Klasa {i}:\n{matrix}\n")
